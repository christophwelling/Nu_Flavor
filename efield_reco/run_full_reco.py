import numpy as np
import matplotlib.pyplot as plt
import Nu_Flavor.helpers.data_reader
import Nu_Flavor.helpers.antenna_helper
import Nu_Flavor.efield_reco.efield_reconstructor
import argparse
import jax
jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('--event_id', type=int, default=-1)
args = parser.parse_args()

reader = Nu_Flavor.helpers.data_reader.DataReader(
  args.filename,
  None,
  1
)
antenna_helper = Nu_Flavor.helpers.antenna_helper.AntennaHelper()
efield_reconstructor = Nu_Flavor.efield_reco.efield_reconstructor.NiftyEfieldReco(
  n_samples=256,
  sampling_rate=3.,
  time_mean=2,
  time_std=1.,
  probability_samples=100,
  correlated_field_args={
    "offset_mean": (-4.),
    "offset_std": (.8, 5e-1),
    "fluctuations": (.5, 2.),
    "loglogavgslope": (-2.3, 0.5),
    "flexibility": (.2, 1.5),
    "asperity": (.2, 1.0)
  },
  n_repeats=4
)
for i_event in range(reader.get_n_events()):
  if args.event_id >= 0 and i_event != args.event_id:
    continue
  reader.read_event(i_event)
  signal_direction = reader.get_signal_direction()
  antenna_indices = antenna_helper.get_antenna_indices(
    signal_direction,
    30. * np.pi / 180.
  )
  waveforms = np.zeros((2, len(antenna_indices), 2048))
  waveforms_noiseless = np.zeros_like(waveforms)
  signal_travel_times = np.zeros(len(antenna_indices))
  for i_ant, antenna_index in enumerate(antenna_indices):
    signal_travel_times[i_ant] = antenna_helper.get_signal_travel_time(
      signal_direction,
      antenna_index
    ) * 1.e9
    for i_pol in range(2):
      waveforms[i_pol, i_ant] = reader.get_waveform(antenna_index, i_pol, noiseless=False)
      waveforms_noiseless[i_pol, i_ant] = reader.get_waveform(antenna_index, i_pol, noiseless=True)
  efield_reconstructor.prepare_waveforms(
    waveforms,
    signal_travel_times,
    20,
    waveforms_noiseless
  )
  prep_waveforms = efield_reconstructor.get_waveforms()
  prep_waveforms_noiseless = efield_reconstructor.get_waveforms(True)
  times = efield_reconstructor.get_times()
  freqs = efield_reconstructor.get_freqs()
  plt.close('all')
  fig1, ax1 = plt.subplots(prep_waveforms.shape[1], 2, figsize=(8, 2*prep_waveforms.shape[1]))
  for i_ant in range(prep_waveforms.shape[1]):
    for i_pol in range(2):
      ax1[i_ant, i_pol].plot(
        times,
        prep_waveforms[i_pol, i_ant],
        color='C0'
      )
      ax1[i_ant, i_pol].plot(
        times,
        prep_waveforms_noiseless[i_pol, i_ant],
        color='C1',
        alpha=.5
      )
      ax1[i_ant, i_pol].grid()
  fig1.tight_layout()
  fig1.savefig('plots/full_reco/prep_waveforms/waveforms_{}.png'.format(i_event))
  efield_reconstructor.build_model(
    antenna_indices,
    signal_direction
  )
  fig2, ax2 = plt.subplots(prep_waveforms.shape[1], 5, figsize=(30,3*prep_waveforms.shape[1]))
  for i_sample in range(10):
    prior_sample = efield_reconstructor.generate_prior_sample()
    efield_prior = efield_reconstructor.generate_efield_prior()
    prior_spec = np.abs(np.fft.rfft(efield_prior))

    for i_ant in range(prep_waveforms.shape[1]):
      for i_pol in range(2):
        ax2[i_ant, i_pol].plot(
          times,
          prior_sample[i_pol, :, i_ant],
          color='C{}'.format(i_sample)
        )
      ax2[i_ant, 2].plot(
        np.abs(np.fft.rfft(prior_sample[0, :, i_ant])),
        color='C{}'.format(i_sample)
      )
      ax2[i_ant, 3].plot(
        freqs,
        prior_spec / np.max(prior_spec),
        color='C{}'.format(i_sample)
      )
      ax2[i_ant, 4].plot(
        freqs,
        prior_spec,
        color='C{}'.format(i_sample)
      )
  for i_ant in range(prep_waveforms.shape[1]):
    for i_pol in range(2):
      ax2[i_ant, i_pol].plot(
        times,
        prep_waveforms_noiseless[i_pol, i_ant],
        color='k'
      )
      ax2[i_ant, i_pol].plot(
        times,
        prep_waveforms[i_pol, i_ant],
        color='k',
        marker='.',
        alpha=.5
      )
      wf_max = np.max(np.abs(prep_waveforms))
      ax2[i_ant, i_pol].set_ylim([-wf_max, wf_max])
      ax2[i_ant, i_pol].set_xlim([0, 20])
    for i_plot in range(5):
      ax2[i_ant, i_plot].grid()
    ax2[i_ant, 4].set_yscale('log')
  fig2.tight_layout()
  fig2.savefig('plots/full_reco/priors/priors_{}.png'.format(i_event))
  efield_reconstructor.run_reco()
  fig3, ax3 = plt.subplots(prep_waveforms.shape[1], 4, figsize=(24, 3*prep_waveforms.shape[1]))
  posterior_v_spec = efield_reconstructor.get_posterior_voltage_spectrum_samples([16, 84])
  posterior_v_trace = efield_reconstructor.get_posterior_voltage_waveforms([16, 84])
  rec_v_spec = efield_reconstructor.get_rec_voltage_spectrum()
  rec_v_trace = efield_reconstructor.get_rec_voltage_waveform()
  noise_rms = efield_reconstructor.get_noise_rms()
  for i_ant in range(prep_waveforms.shape[1]):
    for i_pol in range(2):
      ax3[i_ant, i_pol].plot(
        times,
        prep_waveforms[i_pol, i_ant],
        color='C0',
        alpha=.5,
        marker='.'
      )
      ax3[i_ant, i_pol+2].plot(
        freqs,
        np.abs(np.fft.rfft(prep_waveforms[i_pol, i_ant])),
        color='C0',
        alpha=.5,
        marker='.'
      )
      ax3[i_ant, i_pol].grid()
      ax3[i_ant, i_pol+2].grid()
      ax3[i_ant, i_pol].plot(
        times,
        prep_waveforms_noiseless[i_pol, i_ant],
        color='C1'
      )
      ax3[i_ant, i_pol+2].plot(
        freqs,
        np.abs(np.fft.rfft(prep_waveforms_noiseless[i_pol, i_ant])),
        color='C1'
      )
      ax3[i_ant, i_pol].axhline(
        noise_rms,
        color='k',
        linestyle='--',
        alpha=.3
      )
      ax3[i_ant, i_pol].axhline(
        -noise_rms,
        color='k',
        linestyle='--',
        alpha=.3
      )
      ax3[i_ant, i_pol].axhline(
        2*noise_rms,
        color='k',
        linestyle=':',
        alpha=.3
      )
      ax3[i_ant, i_pol].axhline(
        -2*noise_rms,
        color='k',
        linestyle=':',
        alpha=.3
      )


      ax3[i_ant, i_pol].fill_between(
        times,
        posterior_v_trace[0][i_pol, i_ant],
        posterior_v_trace[1][i_pol, i_ant],
        color='k',
        alpha=.2
      )
      ax3[i_ant, i_pol+2].fill_between(
        freqs,
        np.abs(posterior_v_spec[0][i_pol, i_ant]),
        np.abs(posterior_v_spec[1][i_pol, i_ant]),
        color='k',
        alpha=.1
      )
      ax3[i_ant, i_pol].plot(
        times,
        rec_v_trace[i_pol, i_ant],
        color='C2',
        linestyle='--'
      )
      ax3[i_ant, i_pol+2].plot(
        freqs,
        rec_v_spec[i_pol, i_ant],
        color='C2',
        linestyle='--'
      )
  fig3.tight_layout()
  fig3.savefig('plots/full_reco/rec_results/rec_result_{}.png'.format(i_event))

  fig4, ax4 = plt.subplots(1, 3, figsize=(16, 8))
  rec_efield_spec = efield_reconstructor.get_rec_efield_spectrum()
  efield_spec_posterior = efield_reconstructor.get_posterior_efield_spectrum([16, 84])
  sim_efield_data = reader.get_efields()
  sim_efields = np.zeros((sim_efield_data.shape[0], 256))
  for i_efield in range(sim_efields.shape[0]):
    i_start = np.argmax(sim_efield_data[i_efield]) - 50
    if i_start < 0:
      i_start = 0
    if i_start > sim_efield_data.shape[1] - 257:
      i_start = sim_efield_data.shape[1] - 257
    sim_efields[i_efield] = sim_efield_data[i_efield, i_start: i_start+256] / 3.
  efield_freqs = np.fft.rfftfreq(256, 1./3.)
  rec_power_spectrum = efield_reconstructor.get_rec_power_spectrum([5, 95, 16, 84, 50])
  power_spectrum_samples = efield_reconstructor.get_rec_power_spectrum()
  model_k_vectors = efield_reconstructor.get_model_k_vectors()
  efield_max = np.zeros(sim_efields.shape[0])
  for i_efield in range(sim_efields.shape[0]):
    efield_spec = np.abs(np.fft.rfft(sim_efields[i_efield]))
    ax4[0].plot(
      efield_freqs,
      efield_spec,
      color='C1'
    )
    efield_max[i_efield] = np.max(efield_spec)
  ax4[0].set_ylim([0, 1.2 * np.max(efield_spec)])
  ax4[0].set_xlabel('f [GHz]')
  ax4[0].set_ylabel('E')
  ax4[0].fill_between(
    freqs,
    efield_spec_posterior[0],
    efield_spec_posterior[1],
    alpha=.3,
    color='k'
  )
  ax4[0].plot(
    freqs,
    rec_efield_spec,
    color='C2',
    linestyle='--'
  )
  ax4[0].grid()
  ax4[0].axvline(
    .3,
    color='r',
    alpha=.5,
    linestyle=':'
  )
  ax4[0].axvline(
    1.3,
    color='r',
    alpha=.5,
    linestyle=':'
  )
  for i_range in range(2):
    ax4[1].fill_between(
      model_k_vectors,
      rec_power_spectrum[i_range*2],
      rec_power_spectrum[i_range*2+1],
      color='k',
      alpha=.2
    )
  for i_sample, sample in enumerate(power_spectrum_samples):
    ax4[1].plot(
      model_k_vectors,
      sample,
      color='C0',
      alpha=.05
    )
  ax4[1].plot(
    model_k_vectors,
    rec_power_spectrum[4],
    color='C2',
    linestyle='--'
  )
  ax4[1].grid()
  ax4[1].set_xscale('log')
  ax4[1].set_yscale('log')
  ax4[1].set_ylim([1.e-10, 1.e7])
  posterior_times = efield_reconstructor.get_posterior_times()
  ax4[2].hist(
    posterior_times,
    bins = np.arange(np.min(posterior_times)-.2, np.max(posterior_times)+.2, .1)
  )
  ax4[2].grid()
  ax4[2].set_xlabel('t [ns]')
  fig4.tight_layout()
  fig4.savefig('plots/full_reco/rec_results/rec_efield_{}.png'.format(i_event))
