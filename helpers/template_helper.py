import numpy as np
import scipy.signal
import radiotools.helper
import NuRadioReco.utilities.signal_processing
import matplotlib.pyplot as plt

class templateHelper:
  def __init__(
      self,
      filename,
      background_file,
      upsampling_factor,
      filter_band=None,
      plot_templates=False
  ):
    self.__data = np.genfromtxt(
      filename,
      delimiter=','
    )
    if background_file is not None:
      self.__background_data = np.genfromtxt(
        background_file,
        delimiter=','
      )
      if self.__background_data.ndim == 1:
        self.__background_data = np.reshape(self.__background_data, (self.__background_data.shape[0], 1))
    else:
      self.__background_data = np.ones((self.__data.shape[0], 1))
    self.__templates = np.zeros((self.__data.shape[0], int(self.__data.shape[1] * upsampling_factor)))
    self.__upsampling_factor = upsampling_factor
    self.__sampling_rate = 3. * self.__upsampling_factor
    self.__filter_response = None
    if filter_band is not None:
      self.__filter_response = NuRadioReco.utilities.signal_processing.get_filter_response(
        np.fft.rfftfreq(self.__templates.shape[1], 1./self.__sampling_rate),
        filter_band,
        'butterabs',
        5
      )
    for i_tmp in range(self.__templates.shape[0]):
      self.__templates[i_tmp] = scipy.signal.resample(
        self.__data[i_tmp],
        self.__templates.shape[1]
      )
      if self.__filter_response is not None:
        self.__templates[i_tmp] = np.fft.irfft(
          np.fft.rfft(self.__templates[i_tmp]) * self.__filter_response
        )
    if plot_templates:
      fig1 = plt.figure(figsize=(8, 6))
      ax1_1 = fig1.add_subplot(111)
      for i_tmp in range(self.__templates.shape[0]):
        ax1_1.plot(
          np.arange(self.__templates.shape[1]) / 3. / self.__upsampling_factor,
          self.__templates[i_tmp]
        )
      ax1_1.grid()
      fig1.tight_layout()
      fig1.savefig('templates.png')

  def get_template(
      self,
      i_template
  ):
    return self.__templates[i_template]

  def pick_template(
      self,
      waveform
  ):
    tmp_corr_relative = np.zeros(self.__templates.shape[0])
    for ii in range(self.__templates.shape[0]):
      corr_threshold = self.__background_data[0, np.argmin(np.abs(.1 - self.__background_data[ii+1]))]
      tmp_max_corr = np.max(np.abs(scipy.signal.correlate(
        waveform,
        self.__templates[ii]
      )))
      tmp_corr_relative[ii] = tmp_max_corr / corr_threshold


    i_template = np.argmax(tmp_corr_relative)
    print(i_template, tmp_corr_relative)
    return self.__templates[i_template], self.__background_data[i_template], np.argmax(i_template)

  def get_template_size(
      self
  ):
    return self.__templates.shape[1]
  
  def get_n_templates(
      self
  ):
    return self.__templates.shape[0]