from typing import List, Dict, Any
import numpy as np
import torch
import librosa
import noisereduce as nr
import scipy.signal
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from .pipeline import AudioPreprocessor, SpeechDetector, SpeakerDiarizer, SpeechRecognizer

class LibrosaPreprocessor(AudioPreprocessor):
    """Concrete audio preprocessor using librosa"""
    def process(self, audio_path: str) -> tuple[np.ndarray[Any, np.dtype[np.float32]], int]:
        # Load audio with standard 16kHz sample rate
        audio, sr = librosa.load(audio_path, sr=16000, dtype=np.float32)
        sr = int(sr)  # Ensure sample rate is integer
        
        # Normalize and reduce noise
        audio = librosa.util.normalize(audio)
        audio_reduced_noise = nr.reduce_noise(
            y=audio, 
            sr=sr,
            stationary=True,
            prop_decrease=0.75
        )
        
        # Low-pass filter
        b, a = scipy.signal.butter(4, 3000, 'low', fs=sr)
        audio_filtered = scipy.signal.filtfilt(b, a, audio_reduced_noise)
        
        return audio_filtered.astype(np.float32), sr

class SileroVAD(SpeechDetector):
    """Concrete speech detector using Silero VAD"""
    def detect(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, float]]:
        model_and_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True,
            onnx=False,
            trust_repo=True
        )
        
        # Type-safe tuple unpacking
        if not isinstance(model_and_utils, (list, tuple)) or len(model_and_utils) < 2:
            raise ValueError("Invalid model loading result")
            
        model = model_and_utils[0]
        utils = model_and_utils[1]
        
        if not isinstance(utils, (list, tuple)) or len(utils) < 1:
            raise ValueError("Could not get speech_timestamps function")
            
        get_speech_timestamps = utils[0]
        
        audio_tensor = torch.from_numpy(np.ascontiguousarray(audio))
        timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=sample_rate,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            window_size_samples=1024,
            speech_pad_ms=30,
            return_seconds=True
        )
        
        return [{'start': float(ts['start']), 'end': float(ts['end'])} 
                for ts in timestamps]

class PyannoteDiarizer(SpeakerDiarizer):
    """Concrete speaker diarizer using pyannote.audio"""
    def __init__(self):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token="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"
        )
    
    def diarize(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        waveform = torch.from_numpy(audio)[None, :]
        diarization_result = self.pipeline({
            "waveform": waveform,
            "sample_rate": sample_rate
        })
        
        return [{
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end
        } for turn, _, speaker in diarization_result.itertracks(yield_label=True)]
    
    def split_audio(self, audio: np.ndarray, sample_rate: int, 
                   segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{
            "audio": audio[int(seg["start"] * sample_rate):int(seg["end"] * sample_rate)],
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"]
        } for seg in segments]

class WhisperASR(SpeechRecognizer):
    """Concrete speech recognizer using faster-whisper"""
    def __init__(self):
        self.model = WhisperModel(
            "medium",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float32"
        )
    
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> List[str]:
        segments, _ = self.model.transcribe(
            audio,
            language="ja",
            beam_size=5,
            word_timestamps=True,
            vad_filter=True
        )
        return [seg.text for seg in segments]
