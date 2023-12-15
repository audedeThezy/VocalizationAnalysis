
### Import

import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import glob
import pandas as pd
import noisereduce as nr
import time
import libf0
import statistics
from scipy import signal
from skimage.restoration import denoise_tv_chambolle


### Files

basepath = "D:\\data\\PlaybackT01\\ch2"
listRecordingsPath = glob.glob(os.path.join(basepath, '*.wav'))


### Calculation
def find_steps(data) :
    dary = np.array(data)
    dary -= np.average(dary)
    step = np.hstack((np.ones(len(dary)), -1*np.ones(len(dary))))
    dary_step = np.convolve(dary, step, mode='valid')
    # Get the peaks of the convolution
    peaks = signal.find_peaks(dary_step, width=5)[0]

    # plots
    # plt.figure()
    #
    # plt.plot(dary)
    #
    # plt.plot(dary_step/10)
    #
    # for ii in range(len(peaks)):
    #     plt.plot((peaks[ii], peaks[ii]), (-1500, 1500), 'r')
    #
    # plt.ylim([-20,20])
    # plt.show()
    return peaks

#, h=, n=
list_Name = []
list_F0 = []
list_Duration = []

start = time.perf_counter()
for VocName in listRecordingsPath :
    print(VocName)
    y, sr = librosa.load(VocName, sr = 300000) # important, garder cette frequence d'echantillognage, sinon faux.

    #y = nr.reduce_noise(y1, sr, hop_length = 512, freq_mask_smooth_hz = 1000)
    f0,_,_= libf0.salience(y, Fs=sr, N=1100, H=1100, F_min=38141, F_max=90000, R=10.0, num_harm=10,                  freq_smooth_len=11, alpha=0.9, gamma=0.0, constraint_region=None, tol=9, score_low=0.01, score_high=1.0)
    times = librosa.times_like(f0, sr = sr,  hop_length = 1100)

    #very depend of the F thresholds

    # Convolution part
    x = np.array(f0)
    plt.plot(f0)
    x_std = (x - x.mean()) / x.std()
    x_denoiseP = (denoise_tv_chambolle(x_std, weight= 0.5))**2
    x_denoiseN = -(denoise_tv_chambolle(x_std, weight= 0.5))**2
    stepsP = find_steps(x_denoiseP)
    stepsN = find_steps(x_denoiseN)

    vocf0 = []
    voctime = []
    begvoctime = [0]
    endvoctime = [0]
    vocduration = [0]
    for i in range(min(len(stepsP), len(stepsN))):
        if np.mean(f0[stepsP[i]:stepsN[i]]) < 45000 :
            continue
        vocf0.append(list(f0[stepsP[i]:stepsN[i]]))
        voctime.append(list(times[stepsP[i]:stepsN[i]]))
        begvoctime.append(f0[stepsP[i]])
        endvoctime.append(f0[stepsN[i]-1])
        vocduration.append((times[stepsN[i]]) - times[stepsP[i]])

    Name = VocName.split("_")[-1].split(".")[0]
    save_librosa_plot(Name, vocf0, voctime, y, sr, begvoctime[-1], endvoctime[-1])
    list_Name.append(Name)
    #list_F0.append(np.mean(vocf0))
    #list_Duration.append(duration)

finish = time.perf_counter()

print(f'It took {finish-start:.2f} second(s) to finish')

###

DF = pd.DataFrame({'Name' : list_Name, "F0" : list_F0, 'Duration' : list_Duration})
DF.to_csv(basepath + "\\DF")

### Signal to noise ratio

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

##### plot

import matplotlib.pyplot as plt

def save_librosa_plot(Name, f0, times, y, sr, beg, end) :

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    fig, ax = plt.subplots(dpi=200)

    img = librosa.display.specshow(D, x_axis='ms', y_axis='linear', ax=ax,  sr=sr, fmin=20000, fmax=100000)

    ax.set(title='fundamental frequency estimation')

    ax.set_ylim(20000, 100000)

    fig.colorbar(img, ax=ax, format="%+2.f dB")

    for vocnumber in range(len(vocf0)) :

        ax.plot(times[vocnumber], f0[vocnumber], label='f0', color='cyan', linewidth=2)

    ax.legend(loc='upper right')

    plt.savefig(basepath + '\\plotParam\\' + Name, transparent = True )
    plt.close()

def show_librosa_plot(Name, f0, times, y, sr) :

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    fig, ax = plt.subplots()

    img = librosa.display.specshow(D, x_axis='ms', y_axis='linear', ax=ax,  sr=sr, fmin=20000, fmax=100000)

    ax.set(title='pYIN fundamental frequency estimation')

    ax.set_ylim(40000, 90000)

    fig.colorbar(img, ax=ax, format="%+2.f dB")

    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)

    ax.legend(loc='upper right')
    plt.show()
###
def ExtractRecordingsPaths(playback_time_str, basepath) :
    """return a list of wavefiles paths"""
    if not os.path.isdir(basepath) :
        print("This path doesn't exist")

    playback_time = time.strptime(playback_time_str, '%H-%M-%S')
    listRecordingsPaths = glob.glob(os.path.join(basepath, '*.wav'))

    listgoodtime = []
    for RecordingPath in listRecordingsPaths :
        recording_time_str = RecordingPath.split('_')[-2]
        try :
            recording_time = time.strptime(recording_time_str, '%H-%M-%S')
        except :
                print('Audio files names must be of the form TestSourisAude2023-11-28_15-21-11_0000311')
        if playback_time < recording_time :
            listgoodtime.append(RecordingPath)
    return listgoodtime

### Test efficiency
basepath = "D:\\data\\playback2T01\\ch2"
listRecordingsPath = glob.glob(os.path.join(basepath, '*.wav'))
basepath2 = "D:\\data\\playbackT01\\ch2"
listRecordingsPath += glob.glob(os.path.join(basepath2, '*.wav'))
basepath3 = "D:\\data\\1MouseT05\\ch2"
listRecordingsPath += glob.glob(os.path.join(basepath3, '*.wav'))


playback_time_str = '15-00-00'
file_paths = ExtractRecordingsPaths(playback_time_str, basepath2)
all_accuracies = []
confusion_accuracy = []
confusion_detection = []
confusion_precision = []
parameter_list = [30000,31000,32000,33000,34000,35000,36000,37000,38000,39000,40000,41000,42000, 43000, 44000,45000] #for frequency
#parameter_list = [40000,41000]
#parameter_list = [i*100 for i in range(1,11)]
#parameter_list = [0.05,0.1,0.5,1,1.5]
#parameter_list = [i for i in range(0,20)]
#parameter_list = [i*2 for i in range(0,10)]
#parameter_list = [i*100 for i in range(1,11)]
execution_time = []
execution_time_std = []
width = 10
weight = 0.5
fmin = 38000
h = 600
for n in parameter_list :
    one_execution_time=[]
    for _ in range(0,2) :
        start=time.perf_counter()
        generated_df = ExtractFeaturesRecordings(file_paths, h=h,n=n, fmin=fmin, weight=weight, width=width, tol=tol)
        finish = time.perf_counter()
        one_execution_time.append(finish-start)
    print(one_execution_time)
    execution_time.append(np.mean(one_execution_time))
    execution_time_std.append(np.std(one_execution_time))

    generated_df['Filename'] = generated_df['Filename'].astype(int)
    generated_df['VocNumber'] = generated_df.groupby('Filename').cumcount()
    # Merge with reference_df to fill missing lines with zero values
    merged_df = pd.merge(reference_df, generated_df, on=['Filename', 'VocNumber'], how='left').fillna(0)

    # Add 'actual' column
    merged_df['actual'] = merged_df['True_Duration'] > 0
    # Add 'prediction' column
    merged_df['prediction'] = merged_df['Duration'] > 0
    true_negatives = merged_df[(merged_df['actual'] == False) & (merged_df['prediction'] == False)]
    true_positives = merged_df[(merged_df['actual'] == True) & (merged_df['prediction'] == True)]
    false_negatives = merged_df[(merged_df['actual'] == True) & (merged_df['prediction'] == False)]
    false_positives = merged_df[(merged_df['actual'] == False) & (merged_df['prediction'] == True)]

    confusion_accuracy.append((len(true_positives)+len(true_negatives))/len(merged_df))
    confusion_detection.append(len(true_positives)/(len(true_positives)+len(false_negatives)))
    confusion_precision.append(len(true_positives)/(len(true_positives)+len(false_positives)))
    # Calculate accuracy
    true_positives['Accuracy'] = 1 - np.abs(true_positives['Duration'] - true_positives['True_Duration']) / true_positives['True_Duration']
    true_positives = true_positives.fillna(1)
    true_positives.replace(-np.inf, 0, inplace=True)

    # Append accuracy column to the list
    all_accuracies.append(list(true_positives['Accuracy'].values))
