import numpy as np
import matplotlib.pyplot as plt
import glob
import json

def read_results(path):
    dirs = glob.glob(path + '/*')
    results = np.array([])
    for dir in dirs:
        filenames = glob.glob(dir + '/*')
        dir_results = np.zeros(len(filenames))
        for i_file, filename in enumerate(filenames):
            with open(filename, 'r') as file:
                res = json.load(file)
                dir_results[i_file] = res['power_spectrum_fit'][1]
        results = np.append(results, dir_results)
    return results
path_nc = "/project/avieregg/welling/pueo/efield_reco/nc/results/rec_results/"
path_e = "/project/avieregg/welling/pueo/efield_reco/e/results/rec_results/"

results_nc = read_results(path_nc)
results_e = read_results(path_e)
slope_bins = np.arange(-8, -1, .25)
fig1, ax1 = plt.subplots(1, 2, figsize=(8, 6))
ax1[0].hist(
    results_nc,
    bins=slope_bins,
    alpha=.5,
    label='nc',
    density=True
)
ax1[0].hist(
    results_e,
    bins=slope_bins,
    alpha=.5,
    label='e',
    density=True
)
ax1[1].hist(
    results_nc,
    bins=slope_bins,
    alpha=1.,
    label='nc',
    cumulative=-1,
    density=True,
    histtype='step'
)
ax1[1].hist(
    results_e,
    bins=slope_bins,
    alpha=1.,
    label='e',
    cumulative=-1,
    density=True,
    histtype='step'
)
ax1[0].grid()
ax1[0].legend()
ax1[1].grid()
ax1[1].legend()
fig1.tight_layout()
fig1.savefig('plots/results_slope.png')
