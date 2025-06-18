import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../')
import helpers.template_helper

class correlationHelper:
  def __init__(
    self,
    template_helper: helpers.template_helper.templateHelper
  ):
    self.__template_helper = template_helper
    return
  
  def correlate(
    self,
    waveform: np.array,
    i_template: int
  ):
    if i_template is None:
      template = self.__template_helper.pick_template(
        waveform
      )[0]
    else:
      template = self.__template_helper.get_template(i_template)
    correlation = np.correlate(
      waveform,
      template,
      'full'
    )
    return correlation[template.shape[0]-1:]

