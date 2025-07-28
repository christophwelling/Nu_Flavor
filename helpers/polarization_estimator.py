import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import scipy.signal
sys.path.append('../')
import helpers.template_helper

class polarizationEstimator:
  def __init__(
      self,
      upsampling_factor=1
  ):
    self.__pol_steps = np.arange(0, 361, 2.) * np.pi / 180.
    self.__amp_steps = np.arange(0, 1.1, .01)
    self.__upsampling_factor = upsampling_factor

  def estimate_polarization_angle(
      self,
      waveforms,
      matched_filter_helper,
      signal_direction,
      antenna_ids,
      viewing_angle
  ):
    sample_shifts = self.find_sample_shift(
      waveforms,
      matched_filter_helper,
      signal_direction,
      antenna_ids,
      viewing_angle
    )
    template_fd = matched_filter_helper.generate_template(
      signal_direction,
      antenna_ids,
      45.*np.pi/180.,
      viewing_angle
    )
    template = np.fft.irfft(template_fd)
    template = template/ np.max(np.abs(template)) * np.max(np.abs(waveforms))
    chi2 = self.pol_scan(
      waveforms,
      np.roll(template, sample_shifts, axis=2)
    )
    rec_polarization = self.__pol_steps[np.argmin(np.min(chi2, axis=1))]
    return rec_polarization
  def get_pol_steps(self):
    return self.__pol_steps
  
  def get_amp_step(self):
    return self.__amp_steps
  
  def find_sample_shift(
      self,
      waveforms,
      matched_filter_helper,
      signal_direction,
      antenna_ids,
      viewing_angle
      ):
    corr_0 = np.zeros((2, waveforms.shape[2]))
    for pol in np.arange(0, 360, 45) * np.pi / 180.:
      template_0_fd = matched_filter_helper.generate_template(
        signal_direction,
        antenna_ids,
        pol,
        viewing_angle
      )
      corr_0[1] = np.abs(np.sum(matched_filter_helper.apply_matched_filter(
        template_0_fd,
        waveforms
      ), axis=(0, 1)))
      corr_0[0] = np.max(corr_0, axis=0)
    sample_shift = np.argmax(corr_0[0]) + int(380 * self.__upsampling_factor)
    return sample_shift
    

  def pol_scan(
      self,
      waveforms,
      template
  ):
    chi2 = np.zeros((self.__pol_steps.shape[0], self.__amp_steps.shape[0]))
    tmp = np.zeros_like(template)
    for i_pol, pol in enumerate(self.__pol_steps):
      for i_amp, amp in enumerate(self.__amp_steps):
        tmp[0] = -template[0] * amp * np.cos(pol)
        tmp[1] = -template[1] * amp * np.sin(pol)
        chi2[i_pol, i_amp] = np.sum(
          (tmp - waveforms)**2
        )
    return chi2