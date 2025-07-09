import os
import NuRadioReco
import data_reader as dr
import matplotlib.pyplot as plt
from labellines import *
import numpy as np
from scipy.signal import fftconvolve

data_reader = dr.DataReader('IceFinal_19502_allTree.root', '/home/austin/pueo_build/install/share/pueo/geometry/dec23/qrh.dat')
n_events = data_reader.get_n_events()

# Loop over events
for i in range(0, n_events):

    # Initializing
    data_reader.read_event(i)
    times = data_reader.get_times()
    det_times = data_reader.get_det_times()*1e9
    trigger_times= data_reader.get_trigger_times()
    energies = data_reader.get_energies()
    secondaries = data_reader.get_secondaries()
    had_fracs = data_reader.get_had_fracs()

    # Loading data
    signal_chain_data = np.loadtxt('PUEO_SignalChainMI_0.csv', delimiter = ',')
    signal_freq, signal = signal_chain_data[:, 0], signal_chain_data[:, 1]*np.cos(signal_chain_data[:, 2])+signal_chain_data[:, 1]*1j*np.sin(signal_chain_data[:, 2])
    
    # Calculating group delay
    group_delay = -(1/(2*np.pi))*np.diff(np.unwrap(signal_chain_data[:, 2]))/np.diff(signal_chain_data[:, 0])
    group_delay = np.mean(group_delay)/1e-9

    # Signal chain response for dedispersion
    signal_times = (0.5/signal_freq[-1])*np.arange(0, len(signal_freq), 1)
    signal_ifft = np.fft.irfft(signal, n = 1024, norm = 'forward')
    
    # Best channel
    max_channel, pol = data_reader.get_max_channel()

    # Initialize figure
    fig = plt.figure()
    
    # Get noiseless waveform
    waveform_noiseless = data_reader.get_waveform(max_channel, pol, True)
    plt.plot(times, waveform_noiseless, color = 'red', label = 'Dispersed')
    
    # Timing information pre-dedispersion
    est_idx = np.argmin(np.abs(times - det_times[0]))
    threshold = np.max(waveform_noiseless)*0.005
    pulse_indices = np.where(np.abs(waveform_noiseless) > threshold)[0]
    first_pulse_idx = pulse_indices[0]
    first_pulse_time = times[first_pulse_idx]
    dt = first_pulse_time - np.sort(det_times)[0]
    det_times = det_times+dt-group_delay
    
    # Dedisperse waveform with signal chain respons
    data_reader.dedisperse(signal_ifft)
    
    # Reset noiseless waveform
    waveform_noiseless = data_reader.get_waveform(max_channel, pol, True)
    plt.plot(times, waveform_noiseless, color = 'blue', label = 'Dedispersed')
    plt.title('Dedispersion results, average group delay: '+str(np.round(group_delay, decimals = 0))+'ns')
    plt.legend()
    plt.grid(which = 'both')
    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage')
    
    # Getting all waveforms
    waveform = data_reader.get_waveform(max_channel, pol, False)
    waveform_noiseless = data_reader.get_waveform(max_channel, pol, True)
    waveform_H = data_reader.get_waveform(max_channel, 0, False)
    waveform_noiseless_H = data_reader.get_waveform(max_channel, 0, True)
    waveform_V = data_reader.get_waveform(max_channel, 1, False)
    waveform_noiseless_V = data_reader.get_waveform(max_channel, 1, True)
    
    # Plotting single channel waveform with noise
    fig = plt.figure()
    plt.plot(times, waveform, color = 'black', alpha = 0.25)
    
    # My attempt at aligning the calculated "det_times" via time of flight with the maximums seen in the waveform
    # I do this by looking in a small window around the calculated time and finding the maximum within that window
    # If there is a second pulse too close in time to the pulse that was just corrected, I use that correction on the second pulse as well
    fudge_time = 5 #ns
    fudge_index = int(fudge_time/0.33)
    
    time_shift = 0
    for j in range(0, len(det_times)):
        if j > 0 and np.abs(det_times[j]-det_times[j-1])<fudge_time:
            det_times[j] += time_shift
        else:
            min_index = np.argmin(np.abs(times-det_times[j]))-fudge_index
            max_index = np.argmin(np.abs(times-det_times[j]))+fudge_index
            max_time = times[np.argmax(np.abs(np.where(np.logical_and(times > times[min_index], times < times[max_index]), waveform_noiseless, 0)))]
            time_shift = max_time-det_times[j]
            det_times[j] += time_shift
    
    # Plotting single channel waveform without noise and adding interaction identifiers
    # Red lines indicate primary interactions while blue lines indicate secondaries
    # Solid lines are hadronic interactions, dashed lines are EM interactions
    plt.plot(times, waveform_noiseless, color = 'black')
    for j in range(0, len(det_times)):
        if had_fracs[j] == 0:
            linestyle = '--'
        else:
            linestyle = '-'
        if secondaries[j] == 0:
            color = 'red'
        else:
            color = 'blue'
        plt.axvline(x = det_times[j], color = color, linestyle = linestyle, label = str(np.round(energies[j]/1e18, decimals = 1))+' EeV')
    
    # Labelling lines and adding a title
    labelLines(plt.gca().get_lines(),zorder=2.5)
    plt.grid(which = 'both')
    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage (V)')
    plt.title('Event Number: '+str(i))

    ######################################################################################################################### 
    
    # Matched Filtering Starts Here
    
    # Calculating noise
    noise = waveform-waveform_noiseless
    RMS_Noise = np.sqrt(np.mean(noise**2))
    noise_H = waveform_H-waveform_noiseless_H
    noise_V = waveform_V-waveform_noiseless_V
    
    # Very stupid beamforming, where I assume we beamform perfectly in 16 antennas, scaling SNR by a factor 4
    sig_dir = data_reader.get_signal_direction()
    beam = noise+4*waveform_noiseless
    beam_H = noise_H+4*waveform_noiseless_H
    beam_V = noise_V+4*waveform_noiseless_V

    # Define window around maximum pulse in waveform and set as the template to search for
    num_lower = 20
    num_upper = 20
    max_E_loc = np.argmax(waveform_noiseless)
    first_wfm = waveform_noiseless[(max_E_loc-num_lower):(max_E_loc+num_upper)]
    
    # Add the template window to the previous plot for diagnostic purposes 
    plt.axvspan(times[max_E_loc-num_lower], times[max_E_loc+num_upper], color='purple', alpha=0.25, label='Template region')

    # Matched filter: convolve with time-reversed template
    matched_filter_output = fftconvolve(beam, first_wfm[::-1], mode='same')
    matched_filter_output_H = fftconvolve(beam_H, first_wfm[::-1], mode='same')
    matched_filter_output_V = fftconvolve(beam_V, first_wfm[::-1], mode='same')

    # From the estimated det_times above, plot the frequency spectra within a window around the pulse
    # This does not always work because the det_times aren't always perfectly positioned
    fig = plt.figure()
    W = np.fft.rfftfreq(len(times), d=(times[-1]-times[0])/len(times)*1e-9)
    for j in range(0, len(det_times)):
        if had_fracs[j] == 0:
            linestyle = '--'
        else:
            linestyle = '-'
        if secondaries[j] == 0:
            color = 'red'
        else:
            color = 'blue'
        plt.plot(W, np.abs(np.fft.rfft(np.where(np.logical_or(times<(det_times[j]-num_lower), times>(det_times[j]+num_upper)), 0, beam-noise))), linestyle = linestyle, color = color)
    plt.plot(W, np.abs(np.fft.rfft(np.where(np.logical_or(times<(det_times[j]-num_lower), times>(det_times[j]+num_upper)), 0, noise))), linestyle = '-', color = 'black', label = 't = '+str(np.round(det_times[j], decimals = 0)))
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which = 'both')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('V/Hz')

    # Plotting beamformed signal
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.title("Beamformed Signal, Noise RMS: "+str(np.round(RMS_Noise, decimals = 4))+' V')
    plt.plot(times, beam, color = 'black')
    plt.plot(times, beam-noise, color = 'red')
    plt.ylabel('Voltage (V)')
    plt.grid()

    # Plotting matched milter output
    plt.subplot(2, 1, 2)
    plt.title("Matched Filter Output")
    plt.plot(times, np.abs(matched_filter_output), color = 'black') 
    plt.grid()
    plt.tight_layout()
    plt.xlabel('Time (ns)')
    plt.ylabel('Correlation (A.U.)')
    plt.show()
