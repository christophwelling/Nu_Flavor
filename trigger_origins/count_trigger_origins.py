import numpy as np
import sys
sys.path.append('../')
import argparse
import helpers.data_reader
import matplotlib.pyplot as plt
import glob

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str)
args = parser.parse_args()
folder_content = glob.glob(args.folder + '*')
print(folder_content)
for folder in folder_content:
  filename = glob.glob(folder + '/IceFinal*')[0]
  dataReader = helpers.data_reader.DataReader(
    filename,
    None
  )
  make_plots = False
  dataReader.read_event(0)
  trig_list = [dataReader.get_neutrino_energy()]
  run_id = folder.split('/')[-1][3:]
  print(run_id)
  for i_event in range(dataReader.get_n_events()):
    dataReader.read_event(i_event)
    det_times = dataReader.get_det_times()
    times = dataReader.get_times()
    max_channel = dataReader.get_max_channel()
    shower_energies = dataReader.get_energies()
    wf = dataReader.get_waveform(*max_channel, True)
    if make_plots:
      plt.close('all')
      fig1 = plt.figure(figsize=(12, 6))
      ax1_1 = fig1.add_subplot(111)
      ax1_1.plot(
        times,
        wf
      )

      for i_t, t in enumerate(det_times):
        if i_t ==0:
          color='blue'
        else:
          color='r'
        ax1_1.axvline(
          t*1.e9 + 215,
          color=color,
          linestyle=':'
          )
        ax1_1.text(
          t*1.e9 + 220,
          .5 * np.max(wf),
          'E={:.2f}EeV'.format(shower_energies[i_t]/1.e18),
          color=color,
          rotation='vertical'
        )
      for i_trigger, trigger_time in enumerate(dataReader.get_trigger_times()):
        ax1_1.axvline(
          trigger_time,
          color='k',
          linestyle='--'
        )
        plt.show()
    for i_trigger, trigger_time in enumerate(dataReader.get_trigger_times()):
      if trigger_time < 0:
        continue
      primary_trigger = 1
      if np.abs(trigger_time - det_times[0]*1.e9 - 215) > 20:
        primary_trigger = 0
      if np.argmax(shower_energies[np.abs(trigger_time - det_times*1.e9-215) < 20]) > 0:
        primary_trigger = 0
      trig_list.append(primary_trigger)
  np.savetxt(
    'triggers_{}.txt'.format(run_id),
    trig_list,
    delimiter=', '
    )
