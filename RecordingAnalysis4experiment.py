# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:19:49 2023

@author: de Witasse Thézy
"""

import pandas as pd
import librosa
from multiprocessing import Pool
import sys
import time
import glob
import os
import numpy as np
from scipy import signal
from skimage.restoration import denoise_tv_chambolle
import libf0

# Config of the computer: import note
# Pay attention when naming the file.

# File paths
def ExtractRecordingsPaths(playback_time_str, basepath):
    """Return a list of wavefiles paths"""
    if not os.path.isdir(basepath):
        print("This path doesn't exist")

    playback_time = time.strptime(playback_time_str, '%H-%M-%S')
    listRecordingsPaths = glob.glob(os.path.join(basepath, '*.wav'))

    listgoodtime = []
    for RecordingPath in listRecordingsPaths:
        recording_time_str = RecordingPath.split('_')[-2]
        try:
            recording_time = time.strptime(recording_time_str, '%H-%M-%S')
        except:
            print('Audio files names must be of the form TestSourisAude2023-11-28_15-21-11_0000311')
        if playback_time < recording_time:
            listgoodtime.append(RecordingPath)
    return listgoodtime


# Extract features for one audio
def find_steps(data):
    dary = np.array(data)
    dary -= np.average(dary)
    step = np.hstack((np.ones(len(dary)), -1 * np.ones(len(dary))))
    dary_step = np.convolve(dary, step, mode='valid')
    # Get the peaks of the convolution
    peaks = signal.find_peaks(dary_step, width=5)[0]
    return peaks


def process_single_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=300000)

        f0, _, _ = libf0.salience(y, Fs=sr, N=1100, H=1100, F_min=38141,
                                  F_max=90000, R=10.0, num_harm=1,
                                  freq_smooth_len=11, alpha=0.9, gamma=0.0,
                                  constraint_region=None, tol=5,
                                  score_low=0.01, score_high=1.0)
        times = librosa.times_like(f0, sr=sr, hop_length=500)

        # Convolution part
        # Denoising the estimated f0
        x = np.array(f0)
        x_std = (x - x.mean()) / x.std()
        x_denoiseP = (denoise_tv_chambolle(x_std, weight=0.5)) ** 2
        x_denoiseN = -(denoise_tv_chambolle(x_std, weight=0.5)) ** 2
        # Find the steps
        stepsP = find_steps(x_denoiseP)
        stepsN = find_steps(x_denoiseN)

        vocf0 = []
        vocduration = []
        # For each vocalization found in the audio
        for i in range(min(len(stepsP), len(stepsN))):
            if np.mean(f0[stepsP[i]:stepsN[i]]) < 45000:
                continue
            vocf0.append(np.mean(f0[stepsP[i]:stepsN[i]]))
            vocduration.append(times[stepsN[i]] - times[stepsP[i]])
        if not vocf0:
            return None
        return {
            'Filename': [file_path.split("_")[-1].split(".")[0]] * len(vocf0),
            'F0': vocf0,
            'Duration': vocduration
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# Processing
def ExtractFeaturesRecordings(file_paths):
    results = {
        'Filename': [],
        'Frequency': [],
        'Duration': [],
    }

    for file_path in file_paths:
        result = process_single_audio(file_path)
        if result is not None:
            results['Filename'].extend(result['Filename'])
            results['Frequency'].extend(result['F0'])
            results['Duration'].extend(result['Duration'])

    df = pd.DataFrame(results)
    df = df.dropna(axis=0, how='any')
    return df

# Execution function

# Can be changed depending on experimental setup

def VocAnalysis(playback_time_str, basepath, Feature="VocNumber", thresholds=[0, 100]):
    if not Feature in ["VocNumber", "Duration", "Frequency"]:
        raise Exception("Possible methods are VocNumber, Duration, Frequency")
    listRecordingsPaths = ExtractRecordingsPaths(playback_time_str, basepath)
    if len(listRecordingsPaths) == 0:
        return False
    FeaturesDF = ExtractFeaturesRecordings(listRecordingsPaths)
    if Feature == "VocNumber":
        if (len(FeaturesDF) >= min(thresholds) and len(FeaturesDF) < max(thresholds)):
            return True
    elif Feature == "Duration" or Feature == "Frequency":
        if any(FeaturesDF[Feature] >= min(thresholds)) and any(FeaturesDF[Feature] < max(thresholds)):
            return True
    return False

#%% Execution

basepath = "D:\\data\\PlaybackT01\\ch2"
playback_time_str = '15-21-21'
file_paths = ExtractRecordingsPaths(playback_time_str, basepath)

start = time.perf_counter()

print(VocAnalysis(playback_time_str='14-50-21', basepath="D:\\data\\1MouseT05\\ch2", Feature="Frequency",
                  thresholds=[60000, 70000]))

finish = time.perf_counter()
print(f'It took {finish - start:.2f} second(s) to finish')
#%% Execution without function

# Process the audio files and get the DataFrame
start = time.perf_counter()

result_df = ExtractFeaturesRecordings(file_paths)

finish = time.perf_counter()
print(f'It took {finish - start:.2f} second(s) to finish')

# Get the total number of files
total_files = len(result_df)

print("Total number of files:", total_files)
print(result_df)
# result_df.to_csv("D:\\data\\PlaybackT01\\result_df.csv")
