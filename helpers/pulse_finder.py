import numpy as np
import scipy.signal
import NuRadioReco.utilities.signal_processing
import matplotlib.pyplot as plt
import helpers.pulse_counter
import os
class pulseFinder:
  def __init__(
      self,
      positions_file,
      upsampling_factor,
      filter_band,
      template_helper,
      quantile_data
  ):
    antenna_data = np.genfromtxt(
      positions_file,
      skip_header=2,
      delimiter=','
    )
    raddeg = np.pi / 180.
    self.__antenna_positions = antenna_data[:, 2:5]
    self.__antenna_positions -= .3 * np.array([
      np.cos(antenna_data[:, 6]*raddeg) * np.cos(antenna_data[:, 5]*raddeg),
      np.sin(antenna_data[:, 6]*raddeg) * np.cos(antenna_data[:, 5]*raddeg),
      np.sin(antenna_data[:, 5] * raddeg)
    ]).T
    self.__upsampling_factor = int(upsampling_factor)
    self.__sampling_rate = 3. * self.__upsampling_factor
    self.__filter_band = filter_band
    self.__channel_indices = []
    self.__waveforms = np.zeros(1)
    self.__beamformed_waveforms = np.zeros(1)
    self.__times = np.zeros(1)
    self.__frequencies = np.zeros(1)
    pueo_util_install_dir = os.environ['PUEO_UTIL_INSTALL_DIR']
    amp_data = np.genfromtxt(
      '{}/share/pueo/responses/signalChainMI/PUEO_SignalChainMI_0.csv'.format(pueo_util_install_dir),
      delimiter=','
    )
    self.__amp_response =  np.exp(-1.j*amp_data[:, 2])
    self.__amp_response = np.fft.irfft(self.__amp_response)
    self.__amp_response = scipy.signal.resample(self.__amp_response, 1024 * self.__upsampling_factor)
    self.__amp_response = np.fft.rfft(self.__amp_response)
    self.__dedispersed_waveforms = np.zeros(1)
    self.__template_helper = template_helper
    self.__correlations = np.zeros(1)
    self.__probabilities = np.zeros(1)
    self.__correlation_quantiles = np.genfromtxt(
      quantile_data,
      delimiter=','
    )
    self.__pulse_counter = helpers.pulse_counter.pulseCounter(
      int(5 * upsampling_factor)
    )


  def set_waveforms(
      self,
      waveforms,
      channel_indices
  ):
    self.__waveforms = np.zeros((2, waveforms.shape[1], waveforms.shape[2] * self.__upsampling_factor))
    self.__channel_indices = channel_indices
    self.__frequencies = np.fft.rfftfreq(self.__waveforms.shape[2] , 1. / self.__sampling_rate)
    for i_pol in range(waveforms.shape[0]):
      self.__waveforms[i_pol] = scipy.signal.resample(
        waveforms[i_pol],
        self.__waveforms.shape[2],
        axis=1
      )
    self.__times = np.arange(self.__waveforms.shape[2]) / 3. / self.__upsampling_factor
    if self.__filter_band is not None:
      filter_response = NuRadioReco.utilities.signal_processing.get_filter_response(
        self.__frequencies,
        self.__filter_band,
        'butterabs',
        5
      )
      for i_pol in range(waveforms.shape[0]):
        self.__waveforms[i_pol] = np.fft.irfft(
          np.fft.rfft(self.__waveforms[i_pol], axis=1) * filter_response,
          axis=1
        )


  def beamform(
      self,
      direction,
      ref_channel,
      dedispersed=False
  ):
    beams = np.zeros((2, self.__waveforms.shape[2]))
    for i_channel in range(self.__channel_indices.shape[0]):
      delta_t = np.dot(
        self.__antenna_positions[ref_channel] - self.__antenna_positions[self.__channel_indices[i_channel]],
        direction
      ) / .3
      sample_offset = int(np.round(delta_t * self.__sampling_rate))
      for i_pol in range(2):
        if dedispersed:
          beams[i_pol] += np.roll(self.__dedispersed_waveforms[i_pol, i_channel], sample_offset)
        else:
          beams[i_pol] += np.roll(self.__waveforms[i_pol, i_channel], sample_offset)
    beams /= self.__channel_indices.shape[0]
    return beams

  def get_times(self):
    return self.__times
  
  def get_frequenceis(self):
    return self.__frequencies
      
  def dedisperse(
      self
  ):
    self.__dedispersed_waveforms = np.zeros_like(self.__waveforms)
    for i_pol in range(2):
      self.__dedispersed_waveforms[i_pol] = np.fft.irfft(
        np.fft.rfft(self.__waveforms[i_pol], axis=1) * self.__amp_response,
        axis=1
      )
    return self.__dedispersed_waveforms
    
  def correlate(
      self,
      waveforms,
      pol_guess
  ):
    template, threshold, i_template = self.__template_helper.pick_template(
      waveforms[pol_guess]
    )
    self.__correlations = np.zeros((2, waveforms.shape[1] + template.shape[0]-1))
    self.__probabilities = np.zeros_like(self.__correlations)
    for i_pol in range(2):
      self.__correlations[i_pol] = np.abs(scipy.signal.hilbert(np.correlate(
        waveforms[i_pol],
        template,
        'full'
      )))
      for ii in range(self.__correlations.shape[1]):
        self.__probabilities[i_pol, ii] = self.__correlation_quantiles[i_template+1, np.argmin(
          np.abs(self.__correlations[i_pol, ii]-self.__correlation_quantiles[0])
          )]
    return self.__correlations, self.__probabilities
  
  def get_pulses(
      self,
      correlations,
      probability,
      thresholds
  ):
    return self.__pulse_counter.count_pulses(
      correlations,
      probability,
      thresholds
    )
