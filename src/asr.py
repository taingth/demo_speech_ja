import torch
from faster_whisper import WhisperModel

class ASRProcessor:
    """
    Handle speech recognition using faster-whisper.
    """
    def __init__(self):
        self.model = WhisperModel(
            "medium",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float32"
        )
    
    def transcribe_segment(self, audio_segment, sample_rate):
        """
        Transcribe an audio segment using the Whisper model.
        
        Args:
            audio_segment: numpy array containing audio data
            sample_rate: sample rate of the audio
            
        Returns:
            list of transcribed text segments
        """
        segments, _ = self.model.transcribe(
            audio_segment,
            language="ja",
            beam_size=5,
            word_timestamps=True,
            vad_filter=True
        )
        
        return [seg.text for seg in segments]
