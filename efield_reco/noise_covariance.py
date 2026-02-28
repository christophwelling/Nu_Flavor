import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import Nu_Flavor.helpers.data_reader
import argparse
import pickle
import sklearn.covariance
import scipy.linalg


parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('--n_events', type=int, default=-1)
args = parser.parse_args()

reader = Nu_Flavor.helpers.data_reader.DataReader(
  args.filename,
  None,
  1
)

if args.n_events < 0:
  n_events = reader.get_n_events()
else:
  n_events = min(reader.get_n_events(), args.n_events)

waveforms = np.zeros((2, 96, n_events, 1024))

for i_event in range(n_events):
  reader.read_event(i_event)
  for i_pol in range(2):
    for i_channel in range(96):
      waveforms[i_pol, i_channel, i_event] = reader.get_waveform(
        i_channel,
        i_pol
      )

fig1, ax1 = plt.subplots(3, 2, figsize=(18, 16))
covariance = np.zeros((4, 256, 256))

for i_pol in range(2):
  reshaped_array = waveforms[i_pol, :, :, :256].reshape(96*n_events, 256)
  cov_model = sklearn.covariance.GraphicalLassoCV()
  cov_model.fit(reshaped_array)
  covariance[i_pol] = cov_model.covariance_
  covariance[i_pol+2] = cov_model.precision_
sqrt_inv_covariance = scipy.linalg.sqrtm(covariance[3])
corr_max = np.max(np.abs(covariance[0]))
inv_corr_max = np.max(np.abs(covariance[2]))
sqrt_corr_max = np.max(np.abs(sqrt_inv_covariance))
cplot = ax1[0, 0].pcolormesh(
  covariance[0],
  vmax=corr_max,
  vmin=-corr_max,
  cmap='seismic'
)
plt.colorbar(cplot, ax=ax1[0, 0])
cplot2 = ax1[0, 1].pcolormesh(
  np.log10(np.abs(covariance[0])),
  vmax=np.log10(corr_max),
  vmin=np.log10(corr_max)-3,
  cmap='plasma'
)
plt.colorbar(cplot2, ax=ax1[0, 1])

cplot3 = ax1[1, 0].pcolormesh(
  covariance[2],
  vmax=inv_corr_max,
  vmin=-inv_corr_max,
  cmap='seismic'
)
plt.colorbar(cplot3, ax=ax1[1, 0])
cplot4 = ax1[1, 1].pcolormesh(
  np.log10(np.abs(covariance[2])),
  vmax=np.log10(inv_corr_max),
  vmin=np.log10(inv_corr_max)-3,
  cmap='plasma'
)
plt.colorbar(cplot4, ax=ax1[1, 1])
cplot5 = ax1[2, 0].pcolormesh(
  sqrt_inv_covariance,
  vmax=sqrt_corr_max,
  vmin=-sqrt_corr_max,
  cmap='seismic'
)
plt.colorbar(cplot5, ax=ax1[2, 0])
cplot6 = ax1[2, 1].pcolormesh(
  np.log10(np.abs(sqrt_inv_covariance)),
  vmax=np.log10(sqrt_corr_max),
  vmin=np.log10(sqrt_corr_max) - 3,
  cmap='plasma'
)
plt.colorbar(cplot6, ax=ax1[2,1])
fig1.tight_layout()
fig1.savefig('plots/covariance_matrix.png')
# with open('noise_covariance.pkl', 'wb') as output_file:
#   pickle.dump(covariance, output_file)
fig2, ax2 = plt.subplots(2, 1, figsize=(9, 16))
xx = np.arange(covariance.shape[1])
for i_shift in range(100):
  ax2[0].plot(
    xx - i_shift,
    covariance[0, i_shift],
    color='C0',
    alpha=.05
  )
  ax2[1].plot(
    xx - i_shift,
    covariance[2, i_shift],
    color='C0',
    alpha=.05
  )
ax2[0].set_xlim([-50, 50])
ax2[1].set_xlim([-50, 50])
ax2[0].grid()
ax2[1].grid()
ax2[0].set_ylabel('Covariance')
ax2[1].set_ylabel('Inv. Covariance')
ax2[0].set_xlabel(r'$i_x - i_y$')
ax2[1].set_xlabel(r'$i_x - i_y$')
fig2.tight_layout()
fig2.savefig('plots/covariance.png')