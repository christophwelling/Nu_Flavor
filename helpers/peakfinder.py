import numpy as np
import matplotlib.pyplot as plt

class PeakFinder:
  def __init__(
      self
  ):
    return
  
  def find_peaks(
      self,
      data,
      tolerance
  ):
    peaks = []
    data_cp = np.copy(data)
    data_median = np.median(data)
    for i_step in range(data.shape[0]):
      i_peak = np.nanargmax(data_cp)
      peak_max = data_cp[i_peak]
      current_pos = i_peak
      peak_start = i_peak
      peak_end = i_peak
      valley_val = data_cp[i_peak]
      valley_pos = i_peak
      while True:
        if current_pos == 0:
          peak_start = 0
          break
        if np.isnan(data_cp[current_pos-1]):
          peak_start = current_pos
          break
        if data_cp[current_pos] < valley_val:
          valley_val = data_cp[current_pos]
          valley_pos = current_pos
        if data_cp[current_pos] - valley_val > tolerance * peak_max:
          peak_start = valley_pos
          break
        if data_cp[current_pos] < data_median:
          peak_start = current_pos
          break
        current_pos -= 1
      current_pos = i_peak
      valley_val = data_cp[i_peak]
      valley_pos = i_peak
      while True:
        if current_pos == data_cp.shape[0]-1:
          peak_end = current_pos
          break
        if np.isnan(data_cp[current_pos+1]):
          peak_end = current_pos
          break
        if data_cp[current_pos] < valley_val:
          valley_val = data_cp[current_pos]
          valley_pos = current_pos
        if data_cp[current_pos] - valley_val > tolerance * peak_max:
          peak_end = valley_pos
          break
        if data[current_pos] < data_median:
          peak_end = current_pos
          break
        current_pos += 1
      if peak_end - peak_start >= 5:
        peaks.append([peak_start, peak_end])
      data_cp[peak_start:peak_end+1] = np.nan
      if len(data_cp[~np.isnan(data_cp)]) == 0:
        return peaks
    return peaks
