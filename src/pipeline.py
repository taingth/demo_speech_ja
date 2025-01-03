from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from transformers import pipeline

class AudioPreprocessor(ABC):
    """Abstract base class for audio preprocessing"""
    @abstractmethod
    def process(self, audio_path: str) -> tuple[np.ndarray, int]:
        """Process audio file"""
        pass

class SpeechDetector(ABC):
    """Abstract base class for voice activity detection"""
    @abstractmethod
    def detect(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, float]]:
        """Detect speech segments"""
        pass

class SpeakerDiarizer(ABC):
    """Abstract base class for speaker diarization"""
    @abstractmethod
    def diarize(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Identify speakers"""
        pass

    @abstractmethod
    def split_audio(self, audio: np.ndarray, sample_rate: int, 
                   segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split audio by speaker segments"""
        pass

class SpeechRecognizer(ABC):
    """Abstract base class for speech recognition"""
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> List[str]:
        """Transcribe audio"""
        pass

class StepsByStepsConversationProcessor:
    """Main pipeline for processing conversations"""
    def __init__(self,
                 preprocessor: AudioPreprocessor,
                 detector: SpeechDetector,
                 diarizer: SpeakerDiarizer,
                 recognizer: SpeechRecognizer):
        self.preprocessor = preprocessor
        self.detector = detector
        self.diarizer = diarizer
        self.recognizer = recognizer

    def process_conversation(self, audio_path: str) -> Any:
        """Process a conversation audio file"""
        # 1. Preprocess audio
        audio, sr = self.preprocessor.process(audio_path)
        
        # 2. Detect speech segments
        speech_segments = self.detector.detect(audio, sr)
        if not speech_segments:
            return []
        
        # 3. Process speech segments
        all_results = []
        for segment in speech_segments:
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            speech_audio = np.asarray(audio[start_sample:end_sample])
            
            # Perform diarization
            speaker_segments = self.diarizer.diarize(speech_audio, sr)
            
            # Split audio by speakers
            audio_chunks = self.diarizer.split_audio(speech_audio, sr, speaker_segments)
            
            # Transcribe each chunk
            for chunk in audio_chunks:
                transcription = self.recognizer.transcribe(chunk['audio'], sr)
                all_results.append({
                    'speaker': chunk['speaker'],
                    'text': transcription,
                    'start_time': segment['start'] + chunk['start'],
                    'end_time': segment['start'] + chunk['end']
                })
        
        # Sort results by start time
        all_results.sort(key=lambda x: x['start_time'])
        return str(all_results)
