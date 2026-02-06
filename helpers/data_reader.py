import numpy as np
import matplotlib.pyplot as plt
import ROOT
import scipy.signal
import radiotools.helper
import scipy.fft
import NuRadioReco.utilities.signal_processing
import os

cl = 3.e8
ROOT.gSystem.Load(os.environ['PUEO_UTIL_INSTALL_DIR'] + "/lib/libNiceMC.so")
ROOT.gSystem.Load(os.environ['PUEO_UTIL_INSTALL_DIR'] + "/lib/libpueoEvent.so")

class DataReader:
  def __init__(
      self,
      filename,
      antenna_pos_file,
      upsampling_factor=1,
      filter_band=None
  ):
    if antenna_pos_file is not None:
      self.__antenna_positions = np.genfromtxt(
        antenna_pos_file,
        skip_header=2,
        delimiter=','
      )[:, 2:5]
    else:
      self.__antenna_positions = None
    self.__upsampling_factor = int(upsampling_factor)
    self.__sampling_rate = 3. * self.__upsampling_factor
    self.__data_file = ROOT.TFile.Open(filename)
    self.__n_channels = 96
    self.__waveforms = np.zeros((2, self.__n_channels, 1))
    self.__waveforms_noiseless = np.zeros((2, self.__n_channels, 1))
    self.__signal_direction = np.zeros(3)
    self.__filter_band = filter_band
    self.__absorption_weight = 1.
    self.__direction_weight = 0.
    self.__position_weight = 0.
    self.__trigger_times = []
    self.__trace_start_time = 0
    self.__neutrino_energy = 0
    self.__polarization_angle = 0
    self.__viewing_angle = 0

  def get_n_events(
      self
  ):
    return self.__data_file.passTree.GetEntries()

  def get_interactions(
      self, 
      i_event
      ):
      self.__data_file.passTree.GetEntry(i_event)
      detEvt = getattr(self.__data_file.passTree, 'detectorEvents')[0]
      eventSummary = getattr(self.__data_file.passTree, 'eventSummary')

      '''
      for i in range(0, self.__num_interactions):
          print(eventSummary.shower[i].showerEnergy.eV)
          print('Is secondary? '+str(eventSummary.shower[i].secondary))
          print('Num interactions: '+str(eventSummary.shower[i].nInteractions))
          print('EM frac: '+str(eventSummary.shower[i].emFrac))
          print('Had frac: '+str(eventSummary.shower[i].hadFrac))
          print(self.__trigger_times)
          print(eventSummary.eventTime)
          print('xxxxxxxxxxxxxxxxxxxxxxxxxx')
      print(self.__num_interactions)
      #self.__interaction_energy = eventSummary.shower.showerEnergy
      '''
      
  def read_event(
      self,
      i_event
    ):
    self.__data_file.passTree.GetEntry(i_event)
    detEvt = getattr(self.__data_file.passTree, 'detectorEvents')[0]
    eventSummary = getattr(self.__data_file.passTree, 'eventSummary')
    self.__direction_weight = eventSummary.loop.directionWeight
    self.__absorption_weight = eventSummary.neutrino.path.getWeight()
    self.__neutrino_energy = eventSummary.neutrino.energy
    self.__position_weight = eventSummary.loop.positionWeight
    
    self.__num_interactions = len(eventSummary.shower)
    times = []
    energies = []
    secondaries = []
    had_fracs = []
    interaction_dists = []
    
    for i in range(0, self.__num_interactions):
      tof = 0
      energies.append(eventSummary.shower[i].showerEnergy)
      secondaries.append(eventSummary.shower[i].secondary)
      had_fracs.append(eventSummary.shower[i].hadFrac)
      for j in range(0, detEvt.rayTracingPath[i].size()-1):
        first_coordinate = np.zeros(3)
        second_coordinate = np.zeros(3)
        first_coordinate[:] = detEvt.rayTracingPath[i][j]
        second_coordinate[:] = detEvt.rayTracingPath[i][j+1]
        if (i == 0 and j == 0):
            starting_point = np.copy(first_coordinate)
        if j == 0:
            interaction_dists.append(np.sqrt(np.sum((first_coordinate-starting_point)**2))/1e3)
        n = detEvt.rayTracingRefractionIndices[i][j]
        dist = np.sqrt(np.sum((second_coordinate-first_coordinate)**2))/1e3
        tof += dist*n/(cl/1e3)
      times.append(tof+eventSummary.shower[i].interaction_time)

    self.__tof = np.array(times)
    self.__energies = np.array(energies)
    self.__secondaries = np.array(secondaries)
    self.__had_fracs = np.array(had_fracs)
    self.__int_dists = np.array(interaction_dists)
    
    if detEvt.RFdir_payload.size() > 0:
      self.__signal_direction[:] = detEvt.RFdir_payload[0]
      dir_global = np.array(detEvt.RFdir[0])
      dir_x = np.cross(dir_global, np.array([0, 0, 1]))
      dir_x /= np.linalg.norm(dir_x)
      dir_z = np.cross(dir_global, dir_x)
      pol = np.array(detEvt.polarizationAtDetector)
      self.__polarization_angle = -np.arctan2(np.dot(pol, dir_x), np.dot(pol, dir_z))
      self.__viewing_angle = detEvt.viewAngle[0]

    else:
      self.__signal_direction = np.zeros(3)
      self.__polarization_angle = 0
    n_samples = np.array(detEvt.waveformsV[0].GetY()).shape[0] * self.__upsampling_factor
    self.__trace_start_time = detEvt.waveformsV[0].GetPointX(0) * 1.e9
    self.__trigger_times = np.array(detEvt.triggerResults.triggeredSamples) / 3. / self.__upsampling_factor + self.__trace_start_time
    self.__waveforms = np.zeros((2, self.__n_channels, n_samples))
    self.__waveforms_noiseless = np.zeros((2, self.__n_channels, n_samples))
    freqs = scipy.fft.rfftfreq(n_samples, 1./self.__sampling_rate)
    for i_channel in range(self.__n_channels):
      if self.__upsampling_factor > 1:
        self.__waveforms[0, i_channel] = scipy.signal.resample(detEvt.waveformsH[i_channel].GetY(), n_samples)
        self.__waveforms[1, i_channel] = scipy.signal.resample(detEvt.waveformsV[i_channel].GetY(), n_samples)
      else:
        self.__waveforms[0, i_channel] = detEvt.waveformsH[i_channel].GetY()
        self.__waveforms[1, i_channel] = detEvt.waveformsV[i_channel].GetY()
      filter_response = None
      if self.__filter_band is not None:
        filter_response = NuRadioReco.utilities.signal_processing.get_filter_response(
          freqs,
          self.__filter_band,
          'butterabs',
          5
        )
        self.__waveforms[0, i_channel] = scipy.fft.irfft(scipy.fft.rfft(self.__waveforms[0, i_channel])*filter_response)
        self.__waveforms[1, i_channel] = scipy.fft.irfft(scipy.fft.rfft(self.__waveforms[1, i_channel])*filter_response)
      if detEvt.waveformsH_noiseless.size() > 0:
        if self.__upsampling_factor > 1:
          self.__waveforms_noiseless[0, i_channel] = scipy.signal.resample(detEvt.waveformsH_noiseless[i_channel].GetY(), n_samples)
          self.__waveforms_noiseless[1, i_channel] = scipy.signal.resample(detEvt.waveformsV_noiseless[i_channel].GetY(), n_samples)
        else:
          self.__waveforms_noiseless[0, i_channel] = detEvt.waveformsH_noiseless[i_channel].GetY()
          self.__waveforms_noiseless[1, i_channel] = detEvt.waveformsV_noiseless[i_channel].GetY()
        if filter_response is not None:
          self.__waveforms_noiseless[0, i_channel] = scipy.fft.irfft(scipy.fft.rfft(self.__waveforms_noiseless[0, i_channel])*filter_response)
          self.__waveforms_noiseless[1, i_channel] = scipy.fft.irfft(scipy.fft.rfft(self.__waveforms_noiseless[1, i_channel])*filter_response)
    
  def dedisperse(
      self,
      response
      ):
    if self.__upsampling_factor > 1:
      resp = scipy.signal.resample(response, int(response.shape[0] * self.__upsampling_factor))
    else:
      resp = response
    time_domain = np.zeros(self.__waveforms.shape[2])
    time_domain[:resp.shape[0]] = resp
    freq_domain = np.exp(1.j * np.unwrap(np.angle(np.fft.rfft(time_domain))))
    freq_domain /= np.abs(freq_domain)
    for i_pol in range(2):
      for i_channel in range(self.__waveforms.shape[1]):
        self.__waveforms[i_pol, i_channel] = np.fft.irfft(np.fft.rfft(self.__waveforms[i_pol, i_channel]) / freq_domain)
        self.__waveforms_noiseless[i_pol, i_channel] = np.fft.irfft(np.fft.rfft(self.__waveforms_noiseless[i_pol, i_channel]) / freq_domain)

  def get_times(
      self
  ):
    return np.arange(self.__waveforms.shape[2]) / self.__sampling_rate  + self.__trace_start_time
  
  def get_max_channel(
      self,
      noiseless=True
  ):
    if noiseless:
      wf = self.__waveforms_noiseless
    else:
      wf = self.__waveforms

    ch_max = np.max(wf, axis=2)
    i_max_channel = np.argmax(ch_max, axis=1)
    i_max_pol = np.argmax(np.max(ch_max, axis=1))
    return i_max_channel[i_max_pol], i_max_pol

  def get_waveform(
      self,
      i_channel,
      i_pol,
      noiseless=False
  ):
    if noiseless:
      return self.__waveforms_noiseless[i_pol, i_channel]
    else:
      return self.__waveforms[i_pol, i_channel]
  
  def get_close_channels(
      self,
      max_angle_diff,
      ref_channel
  ):
    ref_antenna_orientation = np.zeros(3)
    ref_antenna_orientation[:] = self.__antenna_positions[ref_channel, :]
    ref_antenna_orientation[2] = 0
    ref_antenna_orientation /= np.sqrt(np.sum(ref_antenna_orientation**2))
    channel_ids = []
    for i_channel in range(self.__n_channels):
      ant_orientation = np.zeros(3)
      ant_orientation[:] = self.__antenna_positions[i_channel, :]
      ant_orientation[2] = 0
      ant_orientation /= np.sqrt(np.sum(ant_orientation**2))
      if np.dot(ref_antenna_orientation, ant_orientation) > np.cos(max_angle_diff):
        channel_ids.append(i_channel)
    return np.array(channel_ids)

  def get_waveforms(
      self,
      max_angle_diff,
      rf_dir,
      polarization,
      channel=None,
      roll=True
  ):
    if channel is None:
      ref_channel = self.get_max_channel()[0]
    else:
      ref_channel = channel
    channel_ids = self.get_close_channels(
      max_angle_diff,
      ref_channel
    )
    waveforms = np.zeros((len(channel_ids), self.__waveforms.shape[2]))
    for i_channel, channel_id in enumerate(channel_ids):
      waveforms[i_channel] = self.__waveforms[polarization, channel_id]
      if roll:
        delta_t = np.dot(
          self.__antenna_positions[ref_channel] - self.__antenna_positions[channel_id],
          rf_dir
        ) / .3
        sample_offset = int(np.round(delta_t * self.__sampling_rate))
        waveforms[i_channel] = np.roll(waveforms[i_channel], sample_offset)

    return waveforms

  def beamform(
      self,
      max_angle_diff,
      rf_dir,
      polarization,
      channel=None
  ):
    if channel is None:
      ref_channel = self.get_max_channel()[0]
    else:
      ref_channel = channel
    beam_sum = np.zeros(self.__waveforms.shape[2])
    beam_sum += self.get_waveform(ref_channel, polarization)
    ref_antenna_orientation = np.zeros(3)
    ref_antenna_orientation[:] = self.__antenna_positions[ref_channel, :]
    ref_antenna_orientation[2] = 0
    ref_antenna_orientation /= np.sqrt(np.sum(ref_antenna_orientation**2))
    n_sum_channels = 1
    for i_channel in range(self.__n_channels):
      if i_channel == ref_channel:
        continue
      ant_orientation = np.zeros(3)
      ant_orientation[:] = self.__antenna_positions[i_channel, :]
      ant_orientation[2] = 0
      ant_orientation /= np.sqrt(np.sum(ant_orientation**2))
      if np.dot(ref_antenna_orientation, ant_orientation) < np.cos(max_angle_diff):
        continue
      delta_t = np.dot(
        self.__antenna_positions[ref_channel] - self.__antenna_positions[i_channel],
        rf_dir
      ) / .3
      sample_offset = int(np.round(delta_t * self.__sampling_rate))
      n_sum_channels += 1
      beam_sum += np.roll(self.__waveforms[polarization, i_channel], sample_offset)
    beam_sum /= n_sum_channels
    return beam_sum

  def get_signal_direction(
      self
  ):
    return self.__signal_direction

  def get_neutrino_absorption_weight(
      self
  ):
    return self.__absorption_weight
  
  def get_event_weight(
      self
  ):
    return self.__absorption_weight / self.__position_weight / self.__direction_weight

  def get_n_samples(
      self
  ):
    return self.__waveforms.shape[2]
  
  def get_trigger_times(
      self
  ):
    return self.__trigger_times
    
  def get_det_times(
      self
  ):
    return self.__tof

  def get_energies(
      self
  ):
    return self.__energies

  def get_secondaries(
      self
  ):
    return self.__secondaries
    
  def get_had_fracs(
      self
  ):
    return self.__had_fracs
    
  def get_int_dists(
      self
  ):
    return self.__int_dists
  
  def get_neutrino_energy(self):
    return self.__neutrino_energy
  
  def get_polarization_angle(self):
    return self.__polarization_angle

  def get_viewing_angle(self):
    return self.__viewing_angle
