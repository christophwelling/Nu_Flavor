import numpy as np

class pulseCounter:
  def __init__(
      self,
      edge_size
  ):
    self.__edge_size = edge_size
    return
  
  def count_pulses(
      self,
      correlations,
      probability,
      thresholds
  ):
    thrhlds = np.sort(thresholds)
    prob = np.copy(probability)
    corr = np.copy(correlations)
    peaks = []
    for i_thr in range(thrhlds.shape[0]):
      while np.min(prob) < thrhlds[i_thr]:
        peak = np.zeros(3, dtype=int)
        peak[:2] = self.__determine_peak_width(
          corr,
          prob
        )
        peak[2] = i_thr
        peaks.append(peak)
        prob[peak[0]:peak[1]+1] = 1.
        corr[peak[0]:peak[1]+1] = 0.
    results = np.zeros((len(peaks), 3), dtype=int)
    for i_peak in range(len(peaks)):
      results[i_peak] = peaks[i_peak][0:3]
    return results

  
  def __determine_peak_width(
      self,
      correlation,
      probability
  ):
    peak_center = np.argmax(correlation)
    peak_start = self.__crawl_to_edge(
      probability,
      peak_center,
      -1
    )
    start_found = False
    while not start_found:
      edge_start = np.max([0, peak_start - self.__edge_size])
      if peak_start == 0 or np.min(probability[edge_start:peak_start]) > .5:
        start_found = True
      else:
        peak_start = self.__crawl_to_edge(
          probability,
          edge_start,
          -1
        )

    peak_end = self.__crawl_to_edge(
      probability,
      peak_center,
      1
    )
    end_found = False
    while not end_found:
      edge_end = np.min([probability.shape[0]-1, peak_end + self.__edge_size])
      if peak_end == probability.shape[0] - 1 or np.min(probability[peak_end:edge_end]) > .5:
        end_found = True
      else:
        peak_end = self.__crawl_to_edge(
          probability,
          edge_end,
          1
        )
    return peak_start, peak_end

  def __crawl_to_edge(
      self,
      probability,
      start,
      direction
  ):
    i_pos = start
    while probability[i_pos] < .9 and i_pos > 0 and i_pos < probability.shape[0]-1:
      i_pos += direction
    return i_pos