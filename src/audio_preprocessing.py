import librosa
import noisereduce as nr
import scipy.signal

import numpy as np

def preprocess_audio(audio_path):
    """
    Preprocess audio with noise reduction and filtering.
    
    Args:
        audio_path: path to the audio file
        
    Returns:
        tuple of (preprocessed audio as float32 numpy array, sample rate)
    """
    # Load audio with standard 16kHz sample rate
    audio, sr = librosa.load(audio_path, sr=16000, dtype=np.float32)
    
    # Ensure float32 dtype
    audio = audio.astype(np.float32)
    
    # Normalize amplitude
    audio = librosa.util.normalize(audio)
    
    # Noise reduction using spectral gating
    audio_reduced_noise = nr.reduce_noise(
        y=audio, 
        sr=sr,
        stationary=True,
        prop_decrease=0.75
    )
    
    # Ensure float32 after noise reduction
    audio_reduced_noise = audio_reduced_noise.astype(np.float32)
    
    # Low-pass filter to reduce high frequency noise
    b, a = scipy.signal.butter(4, 3000, 'low', fs=sr)
    audio_filtered = scipy.signal.filtfilt(b, a, audio_reduced_noise)
    
    # Final ensure float32
    audio_filtered = audio_filtered.astype(np.float32)
    
    return audio_filtered, sr
