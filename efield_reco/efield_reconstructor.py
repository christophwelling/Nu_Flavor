import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import nifty.re as jft
import model
import Nu_Flavor.helpers.antenna_helper
import Nu_Flavor.efield_reco.model
import scipy.signal
import scipy.linalg
import jax.random
import pickle

class NiftyEfieldReco:
  def __init__(
      self,
      n_samples,
      sampling_rate,
      time_mean,
      time_std,
      correlated_field_args=None,
      n_padding=None,
      probability_samples=20,
      n_repeats=3
  ):
    self.__n_samples = n_samples
    self.__sampling_rate = sampling_rate
    self.__times = np.arange(n_samples) / sampling_rate
    self.__freqs = np.fft.rfftfreq(n_samples, 1./sampling_rate)
    self.__n_freqs = self.__freqs.shape[0]
    self.__time_mean = time_mean
    self.__time_std = time_std
    self.__probability_samples=probability_samples
    if correlated_field_args == None:
      self.__correlated_field_args = {
       "offset_mean": (-4.2),
        "offset_std": (1.7, 5e-1),
        "fluctuations": (1.2, 1.6),
        "loglogavgslope": (-2.3, 0.5),
        "flexibility": (.5, 1.),
        "asperity": (.5, .8)
      }
    else:
      self.__correlated_field_args = correlated_field_args
    if n_padding is None:
      self.__n_padding = self.__n_samples
    else:
      self.__n_padding = n_padding
    self.__model = None
    self.__det_response = None
    self.__waveforms = None
    self.__noiseless_waveforms = None
    self.__det = Nu_Flavor.helpers.antenna_helper.AntennaHelper()
    self.__template = np.fft.irfft(
      self.__det.get_antenna_response(0, 0, 1) * self.__det.get_amp_response()
    )
    self.__i_template_max = np.argmax(self.__template)
    self.__template = np.roll(
      self.__template,
      -self.__i_template_max+self.__n_samples//10)[:self.__n_samples]
    self.__rng_key = jax.random.PRNGKey(42)
    self.__noise_rms = 0
    self.__noise_covariance_data = pickle.load(open('noise_covariance.pkl', 'rb'))
    self.__noise_covariance = None
    self.__n_repeats = n_repeats

  def prepare_waveforms(
      self,
      waveforms,
      time_offsets,
      pulse_time,
      noiseless_waveforms=None
  ):
    wfs = np.zeros_like(waveforms)
    corrs = np.zeros((waveforms.shape[1], waveforms.shape[2]+self.__template.shape[0]-1))
    i_target = np.argmin(self.__times-pulse_time) - self.__i_template_max
    i_offsets = np.round(time_offsets * self.__sampling_rate).astype(int)
    i_offsets -= np.min(i_offsets)
    for i_channel in range(waveforms.shape[1]):
      for i_pol in range(2):
        corrs[i_channel] += np.abs(scipy.signal.correlate(
          np.roll(waveforms[i_pol, i_channel], -i_offsets[i_channel]),
          self.__template,
          mode='full'
        ))
    corr_shift = scipy.signal.correlation_lags(
      waveforms.shape[2],
      self.__template.shape[0],
      mode='full'
    )[np.argmax(np.sum(corrs, axis=0))]
    if noiseless_waveforms is not None:
      wf_noiseless = np.zeros_like(wfs)
    for i_channel in range(waveforms.shape[1]):
      for i_pol in range(2):
        wfs[i_pol, i_channel] = np.roll(waveforms[i_pol, i_channel], -corr_shift-i_offsets[i_channel])
        if noiseless_waveforms is not None:
          wf_noiseless[i_pol, i_channel] = np.roll(noiseless_waveforms[i_pol, i_channel], -corr_shift-i_offsets[i_channel])
    self.__noise_rms = np.sqrt(np.mean(wfs[:, :, wfs.shape[2]//2:]**2))
    self.__waveforms = wfs[:, :, :self.__n_samples]
    self.__noiseless_waveforms = wf_noiseless[:, :, :self.__n_samples]
    self.__actual_noise_covariance = np.matrix(self.__noise_covariance_data[0, :self.__n_samples, :self.__n_samples]
    )
    
    self.__noise_covariance = np.matrix(self.__noise_covariance_data[2, :self.__n_samples, :self.__n_samples]
    )
    self.__sqrt_noise_covariance = scipy.linalg.sqrtm(self.__noise_covariance)

  def get_times(self):
    return self.__times

  def get_freqs(self):
    return self.__freqs

  def get_waveforms(self, noiseless=False):
    if noiseless:
      return self.__noiseless_waveforms
    else:
      return self.__waveforms
  def get_noise_rms(self):
    return self.__noise_rms

  def build_model(
      self,
      antenna_indices,
      signal_direction
  ):
    det_responses_td = np.zeros((2, len(antenna_indices), 1024))
    for i_ant, antenna_index in enumerate(antenna_indices):
      for i_pol in range(2):
        det_responses_td[i_pol, i_ant] = np.fft.irfft(
          self.__det.get_antenna_response_for_antenna(
            signal_direction,
            antenna_index,
            i_pol,
            False
          ) * self.__det.get_amp_response()
        )
    max_sample = np.argmax(np.max(np.max(np.abs(det_responses_td), axis=0), axis=0))
    det_responses_td = np.roll(det_responses_td, -max_sample+20, axis=2)
    self.__det_response = np.fft.rfft(det_responses_td[:, :, :self.__n_samples], axis=2)
    self.__model = Nu_Flavor.efield_reco.model.SignalModel(
      self.__waveforms.shape[1],
      self.__waveforms.shape[2],
      self.__sampling_rate,
      self.__correlated_field_args,
      self.__time_mean,
      self.__time_std,
      self.__n_padding,
      np.transpose(self.__det_response, (0, 2, 1))
    )

  def generate_prior_sample(
      self
  ):
      self.__rng_key, subkey = jax.random.split(self.__rng_key)
      latent_sample = jft.random_like(subkey, self.__model.domain)
      sample = self.__model(latent_sample)
      return sample
  def generate_efield_prior(
      self
    ):
      self.__rng_key, subkey = jax.random.split(self.__rng_key)
      latent_sample = jft.random_like(subkey, self.__model.domain)
      sample = self.__model.get_efield_trace(latent_sample)
      return sample
  def noise_cov_inv(self, x):
    ret = jnp.zeros_like(x)
    for i_pol in range(2):
      for i_ch in range(x.shape[2]):
        ret = ret.at[i_pol, :, i_ch].set(self.__noise_covariance @ x[i_pol, :, i_ch])
    return ret
  def noise_std_inv(self, x):
    ret = jnp.zeros_like(x)
    for i_pol in range(2):
      for i_ch in range(x.shape[2]):
        ret = ret.at[i_pol, :, i_ch].set(self.__sqrt_noise_covariance @ x[i_pol, :, i_ch])
    return ret

  def run_reco(
      self
  ):
    # noise_std_inv = lambda x: 1/self.__noise_rms

    likelihood = jft.Gaussian(
      np.transpose(self.__waveforms, (0, 2, 1)),
      noise_cov_inv=self.noise_cov_inv,
      noise_std_inv=self.noise_std_inv
      ).amend(self.__model)
    self.__rng_key, sampling_key  = jax.random.split(self.__rng_key, 2)
    key2, subkey2  = jax.random.split(self.__rng_key, 2)
    start_pos = jft.Vector(jft.random_like(subkey2, self.__model.domain))
    self.__posterior_samples, self.__posterior_state = jft.optimize_kl(
      likelihood=likelihood,
      position_or_samples=start_pos,
      key=sampling_key,
      n_total_iterations=self.__n_repeats,
      n_samples=self.__probability_samples
    )
  
  def get_posterior_voltage_spectrum_samples(self, percentiles=None):
    samples = np.zeros((len(self.__posterior_samples), 2, self.__n_freqs, self.__waveforms.shape[1]), dtype=complex)
    for i_sample, sample in enumerate(self.__posterior_samples): 
      samples[i_sample] = self.__model.get_voltage_spectrum(sample)
    samples = np.transpose(samples, (0, 1, 3, 2))
    if percentiles is None:
      return samples
    else:
      return np.percentile(np.abs(samples), percentiles, axis=0)

  def get_rec_voltage_spectrum(self):
    samples = self.get_posterior_voltage_spectrum_samples()
    return np.mean(np.abs(samples), axis=0)
  
  def get_posterior_voltage_waveforms(self, percentiles=None):
    samples = np.zeros((len(self.__posterior_samples), 2, self.__n_samples, self.__waveforms.shape[1]))
    for i_sample, sample in enumerate(self.__posterior_samples): 
      samples[i_sample] = self.__model.get_voltage_trace(sample)
    samples = np.transpose(samples, (0, 1, 3, 2))
    if percentiles is None:
      return samples
    else:
      return np.percentile(samples, percentiles, axis=0)
  
  def get_rec_voltage_waveform(self):
    samples = self.get_posterior_voltage_waveforms()
    return np.mean(samples, axis=0)
  
  def get_posterior_efield_spectrum(self, percentiles=None):
    samples = np.zeros((len(self.__posterior_samples), self.__n_freqs), dtype=complex)
    for i_sample, sample in enumerate(self.__posterior_samples):
      samples[i_sample] = self.__model.get_efield_spectrum(sample)
    if percentiles is None:
      return samples
    else:
      print('111', samples.shape)
      return np.percentile(np.abs(samples), percentiles, axis=0)
  
  def get_rec_efield_spectrum(self):
    return np.mean(np.abs(self.get_posterior_efield_spectrum()), axis=0)


  def get_rec_power_spectrum(self, percentiles=None):
    samples = np.zeros((len(self.__posterior_samples), len(self.get_model_k_vectors())))
    for i_sample, sample in enumerate(self.__posterior_samples):
      samples[i_sample] = self.__model.get_power_spectrum(sample)
    if percentiles is None:
      return samples
    else:
      return np.percentile(samples, percentiles, axis=0)
  
  def get_model_k_vectors(self):
    return self.__model.get_k_vectors()
  
  def get_posterior_times(self):
    times = np.zeros(len(self.__posterior_samples))
    for i_sample, sample in enumerate(self.__posterior_samples):
      times[i_sample] = self.__model.get_time(sample)
    return times