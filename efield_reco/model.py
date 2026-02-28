import nifty.re as jft
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import detector_model
import copy

class SignalModel(jft.Model):
  def __init__(
      self, 
      n_channels,
      n_samples,
      sampling_rate,
      correlated_field_args,
      time_mean,
      time_std,
      n_padding,
      detector_response
      ):
    self.__n_channels = n_channels
    self.__n_samples = n_samples
    self.__n_padding = n_padding
    self.__n_samples_spec = n_samples // 2 + 1
    self.__freqs = jnp.fft.rfftfreq(self.__n_samples, 1./sampling_rate)
    self.__cfm = jft.CorrelatedFieldMaker('_')
    self.__correlated_field_args = copy.copy(correlated_field_args)
    self.__cfm.set_amplitude_total_offset(
      offset_mean=self.__correlated_field_args['offset_mean'],
      offset_std=self.__correlated_field_args['offset_std']
    )
    self.__correlated_field_args.pop('offset_mean')
    self.__correlated_field_args.pop('offset_std')
    self.__cfm.add_fluctuations(
      self.__n_samples_spec+self.__n_padding,
      1./sampling_rate,
      **self.__correlated_field_args
    )
    self.__detector_response = jnp.array(detector_response)

    self.__cfm_model = self.__cfm.finalize()
    self.__time_prior = jft.NormalPrior(
      time_mean,
      std=time_std,
      name='time_model'
    )
    self.__phase_prior = jft.NormalPrior(
      0,
      4.,
      name='phase_model'
    )
    self.__polarization_prior = jft.NormalPrior(
      0,
      10.,
      name='polarization_model'
    )
    super().__init__(
      domain=self.__cfm_model.domain | self.__time_prior.domain | self.__phase_prior.domain | self.__polarization_prior.domain
    )

      
  def __call__(self, x):
    return self.get_voltage_trace(x)
    #  val =  self.__harmonic_dvol * self.__ht(self.__sqrt_harmonic_cov * x)
    #  return val - jnp.min(val)
  
  def get_power_spectrum(self, x):
    return self.__cfm.power_spectrum(x)
  
  def get_voltage_spectrum(self, x):
    return  jnp.reshape(jnp.array([
      jnp.cos(self.__polarization_prior(x)) * self.get_efield_spectrum(x),
      jnp.sin(self.__polarization_prior(x)) * self.get_efield_spectrum(x)
    ]), (2, self.__n_samples_spec, 1)) * self.__detector_response

  def get_efield_spectrum(self, x):
    spec = jnp.exp(jnp.log(10)*self.__cfm_model(x)[:-self.__n_padding]) * jnp.exp(
      -2.j * jnp.pi * (self.__time_prior(x)*self.__freqs + self.__phase_prior(x))
      )
    spec.at[0].set(0)
    spec.at[self.__n_samples].set(0)
    return spec
  def get_efield_trace(self, x):
    return jnp.fft.irfft(self.get_efield_spectrum(x))
  
  def get_voltage_trace(self, x):
    return jnp.fft.irfft(self.get_voltage_spectrum(x), axis=1)
  
  def get_time(self, x):
    return self.__time_prior(x)
  
  def get_phase(self, x):
    return self.__phase_prior(x)
  
  def get_k_vectors(self):
    return self.__cfm_model.target_grids[0].harmonic_grid.mode_lengths
  