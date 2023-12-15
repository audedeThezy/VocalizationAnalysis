import pandas as pd
import librosa
from multiprocessing import Pool
import sys
import time
import glob
import os
import libf0
import numpy as np
import os
import math
import glob
import librosa
import time
import libf0
import statistics
import numpy as np
import pandas as pd
from scipy import signal
from skimage.restoration import denoise_tv_chambolle
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import ttest_1samp
from scipy.stats import t

def ExtractRecordingsPathsOK(playback_time_str, basepath) :
    """return a list of wavefiles paths"""
    # TODO : check that it's working
    playback_time = time.strptime(playback_time_str, '%Y-%m-%d %H-%M-%S')
    listRecordingsPaths = glob.glob(os.path.join(basepath, '*.wav'))
    listgoodtime = []
    for RecordingPath in listRecordingsPaths :
        recording_time_str = " ".join([RecordingPath.split('_')[-3], RecordingPath.split('_')[-2]])
        recording_time = time.strptime(recording_time_str, '%Y-%m-%d %H-%M-%S')
        if playback_time < recording_time : #on peut aussi ajouter un seuil pour que ce ne soit pas trop longtemps après ??
            listgoodtime.append(RecordingPath)
    return listgoodtime

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




def find_steps(data, width) :
    dary = np.array(data)
    dary -= np.average(dary)
    step = np.hstack((np.ones(len(dary)), -1*np.ones(len(dary))))
    dary_step = np.convolve(dary, step, mode='valid')
    # Get the peaks of the convolution
    peaks = signal.find_peaks(dary_step, width=width)[0]
    return peaks


def process_single_audio(file_path, n, h, fmin, weight, width, tol):

    try:
        y, sr = librosa.load(file_path, sr = 300000) # important, garder cette frequence d'echantillognage, sinon faux.

        #y = nr.reduce_noise(y1, sr, hop_length = 512, freq_mask_smooth_hz = 1000)
        f0,_,_= libf0.salience(y, Fs=sr, N=n, H=n, F_min=fmin, F_max=90000, R=10.0, num_harm=1,                  freq_smooth_len=11, alpha=0.9, gamma=0.0, constraint_region=None, tol=tol, score_low=0.01, score_high=1.0)
        times = librosa.times_like(f0, sr = sr,  hop_length = h)

        #very depend of the F thresholds

        # Convolution part
        # denoising the extimated f0
        x = np.array(f0)
        x_std = (x - x.mean()) / x.std()
        x_denoiseP = (denoise_tv_chambolle(x_std, weight= weight))**6
        x_denoiseN = -(denoise_tv_chambolle(x_std, weight=weight))**6
        # find the steps
        stepsP = find_steps(x_denoiseP, width)
        stepsN = find_steps(x_denoiseN, width)

        vocf0 = []
        #voctime = []
        begvoctime = []
        endvoctime = []
        vocduration = []
        # for each vocalization found in the audio
        for i in range(min(len(stepsP), len(stepsN))):
            if np.mean(f0[stepsP[i]:stepsN[i]]) < 45000 :
                continue
            vocf0.append(np.mean(f0[stepsP[i]:stepsN[i]-1]))
            begvoctime.append(f0[stepsP[i]])
            endvoctime.append(f0[stepsN[i]-1])
            vocduration.append(times[stepsN[i]] - times[stepsP[i]])
        if not vocf0 :
            return None
        return {
            'Filename': [file_path.split("_")[-1].split(".")[0]]*len(vocf0),
            'F0' : vocf0,
            'beginning F0' : begvoctime,
            'end F0' : endvoctime,
            'Duration' : vocduration
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def ExtractFeaturesRecordings(file_paths, n, h, fmin, weight, width, tol):
    results = {
            'Filename': [],
            'Frequency' : [],
            'Duration' : [],
            'BeginningFrequency' : [],
            'EndFrequency' : []
        }

    for file_path in file_paths:
        result = process_single_audio(file_path, n, h, fmin, weight, width, tol)
        if result is not None:
            results['Filename'].extend(result['Filename'])
            results['Frequency'].extend(result['F0'])
            results['BeginningFrequency'].extend(result['beginning F0'])
            results['EndFrequency'].extend(result['end F0'])
            results['Duration'].extend(result['Duration'])

    df = pd.DataFrame(results)
    df = df.dropna(axis=0, how='any')
    return df


#### Execution function

#can be change deoending on experimental setup (not. any -> all, meanf0 could be max or min)

def VocAnalysis(playback_time_str, basepath, Feature = "VocNumber", thresholds = [0, 100]) :
    if not Feature in ["VocNumber", "Duration", "Frequency"] :
        raise Exception("Possible methods are VocNumber, Duration, Frequency")
    listRecordingsPaths = ExtractRecordingsPaths(playback_time_str, basepath)
    if len(listRecordingsPaths) == 0 :
        return False
    FeaturesDF = ExtractFeaturesRecordings(listRecordingsPaths)
    if Feature == "VocNumber" :
        if (len(FeaturesDF) >= min(thresholds) and len(FeaturesDF) < max(thresholds)) :
            return True
    elif Feature == "Duration" or Feature == "Frequency":
        if any(FeaturesDF[Feature] >= min(thresholds)) and any(FeaturesDF[Feature] < max(thresholds)) :
            return True
    return False

#### Execution
# start = time.perf_counter()
#
# print(VocAnalysis(playback_time_str='14-50-21', basepath="D:\\data\\1MouseT05\\ch2", Feature = "Frequency", thresholds = [60000, 70000]))
#
# finish = time.perf_counter()
# print(f'It took {finish-start:.2f} second(s) to finish')
#### Execution without function
#
# #  Process the audio files and get the DataFrame
# start = time.perf_counter()
#
# result_df = ExtractFeaturesRecordings(file_paths)
#
# finish = time.perf_counter()
# print(f'It took {finish-start:.2f} second(s) to finish')
#
# # Get the total number of files
# total_files = len(result_df)
#
# print("Total number of files:", total_files)
# print(result_df)
# result_df.to_csv("D:\\data\\PlaybackT01\\result_df.csv")
playback_time_str = '15-00-00'
file_paths = ExtractRecordingsPaths(playback_time_str, "D:\\data\\playbackT01\\ch2")
ExtractFeaturesRecordings(file_paths, n=300, h=300, fmin=38000, weight=0.5, width=10, tol=5)
