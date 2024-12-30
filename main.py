import librosa
import noisereduce as nr
import scipy.signal
import torch
import numpy as np
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment

def preprocess_audio(audio_path):
    """
    Preprocess audio with noise reduction and filtering.
    """
    # Load audio with standard 16kHz sample rate
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Normalize amplitude
    audio = librosa.util.normalize(audio)
    
    # Noise reduction using spectral gating
    audio_reduced_noise = nr.reduce_noise(
        y=audio, 
        sr=sr,
        stationary=True,
        prop_decrease=0.75
    )
    
    # Low-pass filter to reduce high frequency noise
    b, a = scipy.signal.butter(4, 3000, 'low', fs=sr)
    audio_filtered = scipy.signal.filtfilt(b, a, audio_reduced_noise)
    
    return audio_filtered, sr

def detect_speech(audio_array, sample_rate):
    """
    Detect speech segments using Silero VAD.
    """
    # Download and load the model
    torch.hub.download_url_to_file(
        'https://raw.githubusercontent.com/snakers4/silero-vad/master/files/silero_vad.jit',
        'silero_vad.jit'
    )
    model = torch.jit.load('silero_vad.jit')
    
    # Convert to 16-bit
    audio_tensor = torch.FloatTensor(audio_array)
    
    # Initialize VAD iterator
    vad_iterator = model(audio_tensor, sample_rate)
    
    # Get speech timestamps
    speech_timestamps = []
    window_size_samples = 512  # number of samples per window
    
    for i in range(0, len(audio_tensor), window_size_samples):
        chunk = audio_tensor[i:i + window_size_samples]
        if len(chunk) < window_size_samples:
            break
            
        # Get the probability of speech in this window
        speech_prob = model(chunk, sample_rate).item()
        
        if speech_prob > 0.5:  # Adjust threshold as needed
            speech_timestamps.append({
                'start': i / sample_rate,
                'end': (i + window_size_samples) / sample_rate
            })
    
    # Merge consecutive speech segments
    merged_timestamps = []
    if speech_timestamps:
        current_segment = speech_timestamps[0]
        
        for segment in speech_timestamps[1:]:
            if segment['start'] - current_segment['end'] < 0.5:  # Merge if gap is less than 0.5s
                current_segment['end'] = segment['end']
            else:
                merged_timestamps.append(current_segment)
                current_segment = segment
        
        merged_timestamps.append(current_segment)
    
    return merged_timestamps

def perform_diarization(audio_array, sample_rate):
    """
    Perform speaker diarization using pyannote.audio.
    """
    diarization = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token="YOUR_AUTH_TOKEN"  # Replace with actual token
    )
    
    # Run diarization
    diarization_result = diarization({
        "waveform": audio_array,
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

def split_audio_by_speakers(audio_array, sample_rate, speaker_segments):
    """
    Split audio into chunks based on speaker segments.
    """
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

class ASRProcessor:
    """
    Handle speech recognition using faster-whisper.
    """
    def __init__(self):
        self.model = WhisperModel(
            "medium",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16"
        )
    
    def transcribe_segment(self, audio_segment, sample_rate):
        segments, _ = self.model.transcribe(
            audio_segment,
            language="ja",
            beam_size=5,
            word_timestamps=True,
            vad_filter=True
        )
        
        return [seg.text for seg in segments]

class JapaneseConversationProcessor:
    """
    Main pipeline for processing Japanese conversations.
    """
    def __init__(self):
        self.asr_processor = ASRProcessor()
    
    def process_conversation(self, audio_path):
        try:
            # 1. Preprocess audio
            print("Preprocessing audio...")
            audio, sr = preprocess_audio(audio_path)
            
            # 2. Detect speech segments
            print("Detecting speech segments...")
            speech_timestamps = detect_speech(audio, sr)
            
            # 3. Perform speaker diarization
            print("Performing speaker diarization...")
            speaker_segments = perform_diarization(audio, sr)
            
            # 4. Split audio by speakers
            print("Splitting audio by speakers...")
            audio_chunks = split_audio_by_speakers(
                audio, 
                sr, 
                speaker_segments
            )
            
            # 5. Transcribe each segment
            print("Transcribing segments...")
            results = []
            for i, chunk in enumerate(audio_chunks, 1):
                print(f"Transcribing segment {i}/{len(audio_chunks)}...")
                transcription = self.asr_processor.transcribe_segment(
                    chunk["audio"],
                    sr
                )
                
                results.append({
                    "speaker": chunk["speaker"],
                    "text": transcription,
                    "start_time": chunk["start"],
                    "end_time": chunk["end"]
                })
            
            return results
            
        except Exception as e:
            raise Exception(f"Error processing conversation: {str(e)}")

def main():
    """
    Main entry point of the application.
    """
    try:
        # Initialize processor
        processor = JapaneseConversationProcessor()
        
        # Get audio path from user
        audio_path = input("Enter the path to your audio file: ")
        
        # Process conversation
        print("\nProcessing conversation...")
        results = processor.process_conversation(audio_path)
        
        # Print results
        print("\nTranscription Results:")
        print("=====================")
        for result in results:
            print(f"\nSpeaker {result['speaker']}:")
            print(f"Time: {result['start_time']:.2f}s - {result['end_time']:.2f}s")
            print(f"Text: {' '.join(result['text'])}")
            print("-" * 50)

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
