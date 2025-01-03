from typing import Any, Dict, List
from .pipeline import StepsByStepsConversationProcessor, pipeline
from .implementations import (
    LibrosaPreprocessor,
    SileroVAD,
    PyannoteDiarizer,
    WhisperASR,
    ReazonASR,
    NueAsr
)
from .model import PipelineName
import torch


class JapaneseConversationProcessor(StepsByStepsConversationProcessor):
    """
    Main pipeline for processing Japanese conversations.
    """
    def __init__(self, pipeline_name: PipelineName):
        self.pipeline_name = pipeline_name
        
        # Add a new pipeline for whisper openai from huggingface
        if self.pipeline_name.name == "whisper_OpenAI":
            super().__init__(
            preprocessor=LibrosaPreprocessor(),
            detector=SileroVAD(),
            diarizer=PyannoteDiarizer(),
            recognizer=WhisperASR()
        )
        # Add a new pipeline for Kotoba model from huggingface
        elif self.pipeline_name.name == "kotoba":
            model_id = "kotoba-tech/kotoba-whisper-v2.2"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
            
            self.pipe = pipeline(
                model=model_id,
                torch_dtype=torch_dtype,
                device=device,
                model_kwargs=model_kwargs,
                batch_size=8,
                trust_remote_code=True,
            )
        # Add a new pipeline for reazonspeech
        elif self.pipeline_name.name == "reazonspeech":
            super().__init__(
                preprocessor=LibrosaPreprocessor(),
                detector=SileroVAD(),
                diarizer=PyannoteDiarizer(),
                recognizer=ReazonASR()
            )
        elif self.pipeline_name.name == "nue-asr":
            super().__init__(
                preprocessor=LibrosaPreprocessor(),
                detector=SileroVAD(),
                diarizer=PyannoteDiarizer(),
                recognizer=NueAsr()
            )
        # Add a new pipeline for symbl
        elif self.pipeline_name.name == "symbl":
            pass
        # Add a new pipeline for speechmatics
        elif self.pipeline_name.name == "speechmatics":
            pass
        elif self.pipeline_name.name == "whisperAPI":
            pass
        else:
            raise ValueError(f"Unknown pipeline name: {self.pipeline_name.name}")
    
    def process_conversation(self, audio_path: str) -> Any:
        
        # End-to-end pipeline
        if self.pipeline_name.name == "kotoba":
            return self.pipe(audio_path)
        # Step-by-step pipeline
        elif self.pipeline_name.name in ["whisper_OpenAI", "reazonspeech", "nue-asr"]:
            return super().process_conversation(audio_path)
        elif self.pipeline_name.name == "whisperAPI":
            pass
        elif self.pipeline_name.name == "speechmatics":
            pass
        elif self.pipeline_name.name == "symbl":
            pass
        else:
            raise ValueError(f"Unknown pipeline name: {self.pipeline_name.name}")