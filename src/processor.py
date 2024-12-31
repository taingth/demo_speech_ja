import numpy as np
from .audio_preprocessing import preprocess_audio
from .vad import detect_speech
from .diarization import perform_diarization, split_audio_by_speakers
from .asr import ASRProcessor

class JapaneseConversationProcessor:
    """
    Main pipeline for processing Japanese conversations.
    """
    def __init__(self):
        self.asr_processor = ASRProcessor()
    
    def process_conversation(self, audio_path):
        """
        Process a Japanese conversation audio file.
        
        Args:
            audio_path: path to the audio file
            
        Returns:
            list of dictionaries containing:
                - speaker: speaker identifier
                - text: transcribed text
                - start_time: start time of segment
                - end_time: end time of segment
        """
        try:
            # 1. Preprocess audio
            print("Preprocessing audio...")
            audio, sr = preprocess_audio(audio_path)
            
            # 2. Detect speech segments
            print("Detecting speech segments using Silero VAD...")
            speech_segments = detect_speech(audio, sr)
            
            if not speech_segments:
                print("\nNo speech segments were detected in the audio file.")
                print("This could be due to:")
                print("- The audio file being too quiet")
                print("- No speech being present")
                print("- The audio format not being compatible")
                print("\nPlease check that:")
                print("- The audio file contains clear speech")
                print("- The audio file is not corrupted")
                print("- The audio format is supported (WAV recommended)")
                return []
            
            print(f"Found {len(speech_segments)} speech segments")
            
            # 3. Process only speech segments for diarization
            print("\nPerforming speaker diarization...")
            print("This may take a few minutes for longer audio files...")
            all_results = []
            
            for segment in speech_segments:
                start_sample = int(segment['start'] * sr)
                end_sample = int(segment['end'] * sr)
                
                # Extract speech segment and ensure it's a numpy array
                speech_audio = np.asarray(audio[start_sample:end_sample])
                
                # Perform diarization on speech segment
                speaker_segments = perform_diarization(speech_audio, sr)
                
                # Split audio by speakers
                print("Splitting audio by speakers...")
                audio_chunks = split_audio_by_speakers(
                    speech_audio,
                    sr,
                    speaker_segments
                )
                
                # Transcribe each chunk
                print(f"\nTranscribing segment {len(all_results) + 1}/{len(speech_segments)}...")
                print(f"Time range: {segment['start']:.2f}s - {segment['end']:.2f}s")
                for chunk in audio_chunks:
                    # Adjust timestamps to be relative to the full audio
                    absolute_start = segment['start'] + chunk['start']
                    absolute_end = segment['start'] + chunk['end']
                    
                    transcription = self.asr_processor.transcribe_segment(
                        chunk['audio'],
                        sr
                    )
                    
                    all_results.append({
                        'speaker': chunk['speaker'],
                        'text': transcription,
                        'start_time': absolute_start,
                        'end_time': absolute_end
                    })
            
            # Sort results by start time
            all_results.sort(key=lambda x: x['start_time'])
            return all_results
            
        except Exception as e:
            raise Exception(f"Error processing conversation: {str(e)}")
