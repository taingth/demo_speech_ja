from .pipeline import ConversationProcessor
from .implementations import (
    LibrosaPreprocessor,
    SileroVAD,
    PyannoteDiarizer,
    WhisperASR
)

class JapaneseConversationProcessor(ConversationProcessor):
    """
    Main pipeline for processing Japanese conversations.
    """
    def __init__(self):
        super().__init__(
            preprocessor=LibrosaPreprocessor(),
            detector=SileroVAD(),
            diarizer=PyannoteDiarizer(),
            recognizer=WhisperASR()
        )
