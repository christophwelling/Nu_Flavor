import matplotlib.pyplot as plt
import numpy as np


def plot_waveforms(
    i_event,
    i_trigger,
    times,
    wf,
    wf_noiseless
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
  fig1.savefig('plots/waveforms_{}_{}.png'.format(i_event, i_trigger))
  plt.close('all')
  return

def plot_correlation(
    i_event,
    i_trigger,
    times,
    correlation,
    waveforms,
    waveforms_noiseless
):
  plt.close('all')
  max_channel = np.argmax(np.max(np.max(np.abs(waveforms_noiseless), axis=0), axis=1))
  n_sectors = correlation.shape[1]//4
  fig1, ax1 = plt.subplots(8, n_sectors, figsize=(8*n_sectors, 24), sharey=True, sharex=True, num=1, clear=True)
  fig2, ax2 = plt.subplots(4, 1, figsize=(12, 12), num=2, clear=True)
  for i_sector in range(n_sectors):
    for i_ring in range(4):
      for i_pol in range(2):
        ax1[i_ring*2+i_pol, i_sector].plot(
          times,
          correlation[i_pol, i_sector*4+i_ring]
        )
        ax1[i_ring*2+i_pol, i_sector].grid()
  fig1.tight_layout()
  fig1.savefig('plots/correlations_{}_{}.png'.format(i_event, i_trigger))
  for i_pol in range(2):
    ax2[i_pol].plot(
      times,
      waveforms[i_pol, max_channel]
    )
    ax2[i_pol].plot(
      times,
      waveforms_noiseless[i_pol, max_channel],
      alpha=.5
    )
    ax2[i_pol].grid()
  ax2[2].plot(
    times,
    np.sum(correlation, axis=1)[0, ::-1]
    )
  ax2[2].plot(
    times,
    np.sum(correlation, axis=1)[1, ::-1],
    alpha=.5
    )  
  ax2[2].grid()
  ax2[3].plot(
    times,
    np.abs(np.sum(np.sum(correlation, axis=0), axis=0))[::-1]
  )
  ax2[3].grid()
  fig2.tight_layout()
  fig2.savefig('plots/correlation_sum_{}_{}.png'.format(i_event, i_trigger))
  plt.close('all')
  return
  