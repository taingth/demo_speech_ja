import os
from src.processor import JapaneseConversationProcessor

def main():
    """
    Main entry point of the application.
    """
    try:
        print("\nJapanese Conversation Transcription System")
        print("========================================")
        
        # Initialize processor
        processor = JapaneseConversationProcessor()
        
        # Get audio path from user
        audio_path = input("\nEnter the path to your audio file: ")
        
        if not os.path.exists(audio_path):
            print(f"\nError: File not found: {audio_path}")
            print("Please check the file path and try again.")
            return 1
            
        # Process conversation
        print("\nProcessing conversation...")
        print("This may take several minutes depending on the audio length.")
        print("\nSteps:")
        print("1. Preprocessing audio")
        print("2. Detecting speech segments")
        print("3. Identifying speakers")
        print("4. Transcribing speech")
        print("\nProgress:")
        
        results = processor.process_conversation(audio_path)
        
        if not results:
            return 1
            
        # Print results
        print("\nTranscription Results:")
        print("=====================")
        
        if len(results) > 0:
            total_duration = results[-1]['end_time'] - results[0]['start_time']
            print(f"\nProcessed {len(results)} segments over {total_duration:.2f} seconds")
            print("\nDetailed Results:")
            print("-----------------")
            
            for result in results:
                print(f"\nSpeaker {result['speaker']}:")
                print(f"Time: {result['start_time']:.2f}s - {result['end_time']:.2f}s")
                print(f"Text: {' '.join(result['text'])}")
                print("-" * 50)
        else:
            print("\nNo transcription results available.")
            print("Please check that your audio file contains clear Japanese speech.")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nIf this error persists, please check:")
        print("- Your audio file is in a supported format (WAV recommended)")
        print("- You have sufficient disk space for models")
        print("- Your internet connection is stable (for first run)")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
