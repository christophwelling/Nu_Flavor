import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import os


def plot_waveforms(
    i_event,
    i_trigger,
    times,
    wf,
    wf_noiseless,
    flavor,
    run
):
  plt.close('all')
  n_sectors = wf.shape[1]//4
  fig1, ax1 = plt.subplots(8, n_sectors, figsize=(8*n_sectors, 24), sharey=True, sharex=True, num=1, clear=True)
  for i_sector in range(n_sectors):
    for i_ring in range(4):
      for i_pol in range(2):
        ax1[i_ring*2+i_pol, i_sector].plot(
          times,
          wf[i_pol, i_sector*4+i_ring]
        )
        ax1[i_ring*2+i_pol, i_sector].plot(
          times,
          wf_noiseless[i_pol, i_sector*4+i_ring],
          alpha=.5
        )
        ax1[i_ring*2+i_pol, i_sector].grid()
  fig1.tight_layout()
  if not os.path.isdir('plots/{}'.format(flavor)):
    os.mkdir('plots/{}'.format(flavor))
  if not os.path.isdir('plots/{}/run{}'.format(flavor, run)):
    os.mkdir('plots/{}/run{}'.format(flavor, run))
  fig1.savefig('plots/{}/run{}/waveforms_{}_{}.png'.format(flavor, run, i_event, i_trigger))
  plt.close('all')
  return

def plot_correlation(
    i_event,
    i_trigger,
    times,
    correlation,
    waveforms,
    waveforms_noiseless,
    probabilities,
    noise_rms,
    flavor, 
    run,
    shower_times=None,
    shower_energies=None,
    shower_had_fracs=None,
    peaks=None
):
  plt.close('all')
  if not os.path.isdir('plots/{}'.format(flavor)):
    os.mkdir('plots/{}'.format(flavor))
  if not os.path.isdir('plots/{}/run{}'.format(flavor, run)):
    os.mkdir('plots/{}/run{}'.format(flavor, run))

  max_channel = np.argmax(np.max(np.max(np.abs(waveforms_noiseless), axis=0), axis=1))
  n_sectors = correlation.shape[1]//4
  # fig1, ax1 = plt.subplots(8, n_sectors, figsize=(8*n_sectors, 24), sharey=True, sharex=True, num=1, clear=True)
  fig2, ax2 = plt.subplots(6, 1, figsize=(20, 16), num=2, clear=True)
  # for i_sector in range(n_sectors):
  #   for i_ring in range(4):
  #     for i_pol in range(2):
  #       ax1[i_ring*2+i_pol, i_sector].plot(
  #         times,
  #         correlation[i_pol, i_sector*4+i_ring]
  #       )
  #       ax1[i_ring*2+i_pol, i_sector].grid()
  # fig1.tight_layout()
  # fig1.savefig('plots/{}/run{}/correlations_{}_{}.png'.format(flavor, run, i_event, i_trigger))
  for i_pol in range(2):
    ax2[i_pol].plot(
      times,
      waveforms[i_pol, max_channel]
    )
    ax2[i_pol].plot(
      times,
      waveforms_noiseless[i_pol, max_channel],
      alpha=1.
    )
    ax2[i_pol].set_ylim([-3. * noise_rms, 3. * noise_rms])
    for i in range(1, 3):
      ax2[i_pol].axhline(
        i*noise_rms,
        color='k',
        linestyle=':'
      )
      ax2[i_pol].axhline(
        -i*noise_rms,
        color='k',
        linestyle=':'
      )
    ax2[i_pol].grid()
  ax2[2].plot(
    times,
    np.sum(correlation, axis=1)[0]
    )
  ax2[2].plot(
    times,
    np.sum(correlation, axis=1)[1],
    alpha=.5
    )  
  ax2[2].grid()
  ax2[3].plot(
    times,
    (np.sum(np.sum(correlation, axis=0), axis=0)),
    color='C0',
    alpha=.5
  )
  ax2[3].plot(
    times,
    np.abs(scipy.signal.hilbert(np.sum(np.sum(correlation, axis=0), axis=0))),
    color='k',
    alpha=.2
  )
  for i_peak, peak in enumerate(peaks):
    ax2[3].plot(
      times[peak[0]:peak[1]+1],
      (np.abs(scipy.signal.hilbert(np.sum(np.sum(correlation, axis=0), axis=0))))[peak[0]:peak[1]+1],
      color='C{}'.format(i_peak%6)
    )
    # ax2[3].axvline(times[peak[0]], color='k', alpha=.2)
    # ax2[3].axvline(times[peak[1]], color='k', alpha=.2)
    if np.min(probabilities[peak[0]:peak[1]+1]) < 1.e-2:
      ax2[4].axvspan(
        times[peak[0]],
        times[peak[1]],
        color='C{}'.format(i_peak%6),
        alpha=.1
      )
      ax2[5].axvspan(
        times[peak[0]],
        times[peak[1]],
        color='C{}'.format(i_peak%6),
        alpha=.1
      )
  ax2[3].grid()
  ax2[4].plot(
    times,
    probabilities
  )
  ax2[4].grid()
  ax2[4].set_yscale('log')
  ax2[4].set_ylim([5.e-4, 1.1])
  ax2[4].grid(which='major', color='k', alpha=.5, linestyle='-')
  ax2[4].grid(which='minor', color='k', alpha=.2, linestyle='--')
  ax2[4].minorticks_on()
  if shower_times is not None:
    for i_shower, shower_time in enumerate(shower_times):
      if i_shower == 0:
        col = 'blue'
      else:
        col = 'red'
      if shower_had_fracs is None:
        linestyle=':'
      elif shower_had_fracs[i_shower] > .5:
        linestyle = '-'
      else:
        linestyle = '--'
      if np.max(times) > shower_time > np.min(times) and (shower_energies is None or shower_energies[i_shower] > 1.e16):
        for i_plot in [0, 1, 4]:
          ax2[i_plot].axvline(
            shower_time,
            color=col,
            linestyle=linestyle,
            alpha=.5 if i_plot==4 else .2
          )
        if shower_energies is not None:

          ax2[4].text(
            x=shower_time,
            y=np.power(10., - 3. * (i_shower+1) / len(shower_times)),
            s='E={:.2f}EeV'.format(shower_energies[i_shower]/1.e18),
            color=col
          )
          ax2[5].scatter(
            shower_times,
            shower_energies,
            color='C0'
          )
  ax2[5].set_yscale('log')
  ax2[5].set_ylim([1.e16, 1.e20])
  ax2[5].grid()
  for i_plot in range(6):
    ax2[i_plot].set_xlim([times[0], times[-1]])
  fig2.tight_layout()
  fig2.savefig('plots/{}/run{}/correlation_sum_{}_{}.png'.format(flavor, run, i_event, i_trigger))
  plt.close('all')
  return
  
def plot_found_pulses(
  i_event,
  i_trigger,
  times,
  correlation,
  waveforms,
  waveforms_noiseless,
  probabilities,
  noise_rms,
  flavor, 
  run,
  peaks,
  shower_times=None,
  shower_energies=None,
  shower_had_fracs=None
):
  samples_before = 32
  samples_after = 128 - samples_before
  max_channel = np.argmax(np.max(np.max(np.abs(waveforms_noiseless), axis=0), axis=1))
  if not os.path.isdir('plots/found_pulses/{}'.format(flavor)):
    os.mkdir('plots/found_pulses/{}'.format(flavor))
  if not os.path.isdir('plots/found_pulses/{}/run{}'.format(flavor, run)):
    os.mkdir('plots/found_pulses/{}/run{}'.format(flavor, run))
  n_peaks = 0
  freqs = np.fft.rfftfreq(128, 1. / 3.)
  for peak in peaks:
    if np.min(probabilities[peak[0]:peak[1]+1]) < .05:
      i_peak = peak[0] + np.argmin(probabilities[peak[0]:peak[1]])
      if i_peak < samples_before:
        i_peak = samples_before
      if i_peak > waveforms.shape[2] - samples_after:
        i_peak = waveforms.shape[2] - samples_after
      plt.close('all')
      fig1, ax1 = plt.subplots(4, 1, figsize=(10, 8))
      for i_pol in range(2):
        ax1[i_pol].plot(
          times[i_peak-samples_before:i_peak+samples_after],
          waveforms[i_pol, max_channel, i_peak-samples_before:i_peak+samples_after]
        )
        ax1[i_pol].plot(
          times[i_peak-samples_before:i_peak+samples_after],
          waveforms_noiseless[i_pol, max_channel, i_peak-samples_before:i_peak+samples_after]
        )
        ax1[i_pol+2].plot(
          freqs,
          np.abs(np.fft.rfft(waveforms[i_pol, max_channel, i_peak-samples_before:i_peak+samples_after]))
        )
        ax1[i_pol+2].plot(
          freqs,
          np.abs(np.fft.rfft(waveforms_noiseless[i_pol, max_channel, i_peak-samples_before:i_peak+samples_after]))
        )
        ax1[i_pol].grid()
        ax1[i_pol+2].grid()
      fig1.tight_layout()
      fig1.savefig('plots/found_pulses/{}/run{}/wf_{}_{}_{}.png'.format(flavor, run, i_event, i_trigger, n_peaks))
      n_peaks += 1