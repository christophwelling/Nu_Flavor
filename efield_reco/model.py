import nifty.re as jft
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import copy
import numpy as np

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
      detector_response,
      polarization_projections=None
      ):
    self.__n_channels = n_channels
    self.__n_samples = n_samples
    self.__n_padding = n_padding
    self.__n_samples_spec = n_samples // 2 + 1
    print('Model:', self.__n_samples, sampling_rate)
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
    if detector_response is not None:
      self.__detector_response = jnp.array(detector_response)
      self.__detector_response2 = jnp.transpose(detector_response, (0, 2, 1))
    else:
      self.__detector_response2 = jnp.ones((2, self.__n_channels, self.__n_samples_spec))
    if polarization_projections is None:
      projections = np.zeros((n_channels, 2, 2))
      projections[:, 0, 0] = 1
      projections[:, 1, 1] = 1
      self.__pol_projections = jnp.array(projections)
    else:
      self.__pol_projections = polarization_projections
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
      3.,
      name='polarization_model'
    )
    super().__init__(
      domain=self.__cfm_model.domain | self.__time_prior.domain | self.__phase_prior.domain | self.__polarization_prior.domain,
      white_init=True
    )
      
  def __call__(self, x):
    return self.get_voltage_trace(x)

  def get_power_spectrum(self, x):
    return self.__cfm.power_spectrum(x)
  
  def get_voltage_spectrum(self, x):
    #pol_sig = jnp.reshape(jnp.array([
    #  jnp.sin(self.__polarization_prior(x)) * self.get_efield_spectrum(x),
    #  jnp.cos(self.__polarization_prior(x)) * self.get_efield_spectrum(x)
    #]), (2, self.__n_samples_spec, 1))
    pol_sig2 = jnp.expand_dims(jnp.array([
      (jnp.sin(self.__polarization_prior(x))*self.__pol_projections[:, 0, 0] + jnp.cos(self.__polarization_prior(x))*self.__pol_projections[:, 0, 1]),
      (jnp.cos(self.__polarization_prior(x))*self.__pol_projections[:, 1, 1] + jnp.sin(self.__polarization_prior(x))*self.__pol_projections[:, 1, 0])
    ]), axis=2) * self.get_efield_spectrum(x)
    ret = pol_sig2 * self.__detector_response2
    return  jnp.transpose(ret, (0, 2, 1))

  def get_efield_spectrum(self, x):
    spec = jnp.exp(jnp.log(10)*self.__cfm_model(x)[:self.__n_samples_spec]) * jnp.exp(
      -2.j * jnp.pi * (self.__time_prior(x)*self.__freqs + self.__phase_prior(x))
      )
    spec.at[0].set(0)
    spec.at[self.__n_samples].set(0)
    return spec
  def get_abs_efield_spectrum(self, x):
    return jnp.exp(jnp.log(10)*self.__cfm_model(x)[:self.__n_samples_spec])

  def get_full_abs_efield_spectrum(self, x):
    return jnp.exp(jnp.log(10) * self.__cfm_model(x))

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
  