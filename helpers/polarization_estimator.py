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
      template_helper: helpers.template_helper.templateHelper,
  ):
    self.__tmpl_helper = template_helper
    self.__pol_steps = np.arange(0, 361, 1.) * np.pi / 180.
    self.__amp_steps = np.arange(0, 1.1, .01)

  def estimate_polarization(
      self,
      waveforms,
      pol_guess
  ):
    template, thr, i_template = self.__tmpl_helper.pick_template(
      waveforms[pol_guess]
    )
    corr = scipy.signal.correlate(
      waveforms[pol_guess],
      template
    )
    max_corr = np.argmax(np.abs(corr))
    if corr[max_corr] < 0:
      template *= -1.
    max_corr -= template.shape[0]
    likelihoods = np.zeros((self.__pol_steps.shape[0], self.__amp_steps.shape[0]))

    for i_pol in range(self.__pol_steps.shape[0]):
      v_component = np.cos(self.__pol_steps[i_pol])
      h_component = np.sin(self.__pol_steps[i_pol])
      wfs = np.zeros((2, self.__amp_steps.shape[0], template.shape[0]))
      wfs[:, :] = template * np.max(waveforms)
      wfs[0] = (-wfs[0].T * self.__amp_steps * h_component).T
      wfs[1] = (-wfs[1].T * self.__amp_steps * v_component).T
      wfs[0] += waveforms[0][max_corr:max_corr+template.shape[0]]
      wfs[1] += waveforms[1][max_corr:max_corr+template.shape[0]]
      likelihoods[i_pol] = np.sum(wfs[0]**2, axis=1) + np.sum(wfs[1]**2, axis=1)    
    return self.__pol_steps[np.argmin(np.min(likelihoods, axis=1))], self.__amp_steps[np.argmin(np.min(likelihoods, axis=0))]


