import torch
import numpy as np

SAMPLING_RATE = 16000  # Model expects 16kHz sampling rate

def detect_speech(audio_array, sample_rate):
    """
    Detect speech segments using Silero VAD.
    
    Args:
        audio_array: numpy array of audio samples
        sample_rate: sample rate of the audio
        
    Returns:
        list of dictionaries containing speech segments with 'start' and 'end' times in seconds
    """
    try:
        # Load model and utils (handle as tuple)
        model_and_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                       model='silero_vad',
                                       force_reload=True,
                                       onnx=False,
                                       trust_repo=True)
        
        # Unpack model and utils
        if isinstance(model_and_utils, (list, tuple)) and len(model_and_utils) >= 2:
            model = model_and_utils[0]
            utils = model_and_utils[1]
        else:
            raise ValueError("Unexpected model loading result")
        
        # Get speech_timestamps function from utils
        if isinstance(utils, (list, tuple)) and len(utils) >= 1:
            get_speech_timestamps = utils[0]
        else:
            raise ValueError("Could not get speech_timestamps function from utils")
        
        # Ensure float32 array and convert to tensor
        audio_array = np.asarray(audio_array, dtype=np.float32)
        audio_tensor = torch.from_numpy(np.ascontiguousarray(audio_array))
        
        # Get speech timestamps
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
        
        # Convert timestamps to our format
        speech_segments = []
        for ts in timestamps:
            speech_segments.append({
                'start': float(ts['start']),
                'end': float(ts['end'])
            })
        
        return speech_segments
        
    except Exception as e:
        print(f"Error in VAD processing: {str(e)}")
        return []
