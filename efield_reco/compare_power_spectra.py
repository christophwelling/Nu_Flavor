import numpy as np
import matplotlib.pyplot as plt
import glob
import json

def read_results(path):
    dirs = glob.glob(path + '/run*')
    results = np.array([])
    amplitudes = np.array([])
    n_events = 0
    for dir in dirs:
        filenames = glob.glob(dir + '/*')
        n_events += len(filenames)
        dir_amplitudes = np.zeros(len(filenames))
        dir_results = np.zeros(len(filenames))
        for i_file, filename in enumerate(filenames):
            with open(filename, 'r') as file:
                res = json.load(file)
                dir_results[i_file] = np.mean(res['loglogavgslope'])
                dir_amplitudes[i_file] = res['max_signal']
        results = np.append(results, dir_results)
        amplitudes = np.append(amplitudes, dir_amplitudes)
    print('N events: ', n_events)
    return results, amplitudes
path_nc = "/project/avieregg/welling/pueo/efield_reco/nc/results/rec_results/"
path_e = "/project/avieregg/welling/pueo/efield_reco/el/results/rec_results/"

results_nc, amplitudes_nc = read_results(path_nc)
results_e, amplitudes_e = read_results(path_e)
slope_bins = np.arange(-4, 0, .1)
fig1, ax1 = plt.subplots(2, 2, figsize=(12, 8))
n_nc, b_nc, p_nc = ax1[0, 0].hist(
    results_nc, 
    bins=slope_bins,
    alpha=.5,
    label='nc',
    density=True
)
n_el, b_el, p_el = ax1[0, 0].hist(
    results_e,
    bins=slope_bins,
    alpha=.5,
    label='e',
    density=True
)
nn_nc, bb_nc, pp_nc = ax1[0, 1].hist(
    results_nc,
    bins=slope_bins,
    alpha=1.,
    label='nc',
    cumulative=-1,
    density=True,
    histtype='step'
)
nn_el, bb_el, pp_el = ax1[0, 1].hist(
    results_e,
    bins=slope_bins,
    alpha=1.,
    label='e',
    cumulative=-1,
    density=True,
    histtype='step'
)
ax1[0, 0].grid()
ax1[0, 0].legend()
ax1[0, 1].grid()
ax1[0, 1].legend()
ax1[1, 0].fill_between(
    (b_el[1:] + b_el[:-1]) / 2,
    np.zeros_like(n_nc),
    n_nc / (n_nc + n_el),
    color='C0'
)
ax1[1, 0].fill_between(
    (b_el[1:] + b_el[:-1]) / 2,
    n_nc / (n_nc + n_el),
    np.ones_like(n_nc),
    color='C1'
)
ax1[1, 0].grid()
# ax1[1, 1].scatter(
#     amplitudes_nc,
#     results_nc,
#     label='nc',
#     marker='.'
# )
# ax1[1, 1].scatter(
#     amplitudes_e,
#     results_e,
#     label='e',
#     marker='.',
#     alpha=.5
# )
# ax1[1, 1].legend()
ax1[1, 1].grid()
ax1[1, 1].fill_between(
    (bb_el[1:] + bb_el[:-1]) / 2,
    np.zeros_like(nn_nc),
    nn_nc / (nn_nc + nn_el),
    color='C0'
)
ax1[1, 1].fill_between(
    (bb_el[1:] + bb_el[:-1]) / 2,
    nn_nc / (nn_nc + nn_el),
    np.ones_like(nn_nc),
    color='C1'
)
ax1[0, 0].set_xlabel('power spectrum slope')
ax1[0, 1].set_xlabel('power spectrum slope')
ax1[1, 0].set_xlabel('power spectrum slope')
ax1[1, 1].set_xlabel('power spectrum slope')
# ax1[1, 1].set_xlim([0, 1.5])
fig1.tight_layout()
fig1.savefig('plots/results_slope.png')
