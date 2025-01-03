from pyannote.audio import Pipeline
import numpy as np
import torch
import os

def perform_diarization(audio_array, sample_rate):
    """
    Perform speaker diarization using pyannote.audio.
    
    Args:
        audio_array: numpy array of audio samples
        sample_rate: sample rate of the audio
        
    Returns:
        list of speaker segments with start/end times
    """
    try:
        # Initialize pipeline
        try:
            diarization = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=os.getenv("HUGGINGFACE_TOKEN")  # Replace with actual token
            )
            
        except Exception as e:
            raise Exception(
                "Failed to initialize diarization pipeline. "
                "Make sure you have set up a valid HuggingFace token. "
                "Visit https://huggingface.co/pyannote/speaker-diarization "
                "to accept the user agreement and create a token. "
                f"Error: {str(e)}"
            )
        
        # Convert audio to float32 numpy array first
        audio_array = np.asarray(audio_array, dtype=np.float32)
        
        # Convert to torch tensor with shape (channel, time)
        if len(audio_array.shape) == 1:
            # Mono audio - add channel dimension
            waveform = torch.from_numpy(audio_array)[None, :]
        else:
            # Already has channel dimension
            waveform = torch.from_numpy(audio_array)
        
        # Run diarization
        diarization_result = diarization({
            "waveform": waveform,
            "sample_rate": sample_rate
        })
        
        # Extract speaker segments
        speaker_segments = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            speaker_segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end
            })
        
        return speaker_segments
        
    except Exception as e:
        raise Exception(f"Diarization failed: {str(e)}")

def split_audio_by_speakers(audio_array, sample_rate, speaker_segments):
    """
    Split audio into chunks based on speaker segments.
    
    Args:
        audio_array: numpy array of audio samples
        sample_rate: sample rate of the audio
        speaker_segments: list of speaker segments with start/end times
        
    Returns:
        list of audio chunks with speaker information
    """
    try:
        audio_chunks = []
        
        for segment in speaker_segments:
            start_sample = int(segment["start"] * sample_rate)
            end_sample = int(segment["end"] * sample_rate)
            
            # Extract audio chunk
            chunk = audio_array[start_sample:end_sample]
            
            audio_chunks.append({
                "audio": chunk,
                "speaker": segment["speaker"],
                "start": segment["start"],
                "end": segment["end"]
            })
        
        return audio_chunks
        
    except Exception as e:
        raise Exception(f"Audio splitting failed: {str(e)}")
