# Author: Constantin Vaillant-Tenzer

import numpy as np
import scipy.signal
from pydub import AudioSegment


def get_bpm(file_path):
    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    # Convert to mono and get raw data
    audio = audio.set_channels(1)
    samples = np.array(audio.get_array_of_samples())

    # Calculate the envelope of the signal
    envelope = np.abs(scipy.signal.hilbert(samples))

    # Downsample the envelope to reduce computation
    downsample_factor = 100
    envelope = scipy.signal.decimate(envelope, downsample_factor)

    # Calculate autocorrelation
    autocorr = np.correlate(envelope, envelope, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]

    # Find peaks in the autocorrelation
    peaks, _ = scipy.signal.find_peaks(autocorr, distance=audio.frame_rate // 2)

    # Calculate the intervals between peaks
    intervals = np.diff(peaks)

    # Calculate BPM
    bpm = 60.0 / (np.mean(intervals) * downsample_factor / audio.frame_rate)

    return bpm