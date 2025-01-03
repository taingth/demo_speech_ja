from pydantic import BaseModel
from typing import Literal


class PipelineName(BaseModel):
    name: Literal['kotoba', 'nue-asr', 'reazonspeech', 'whisper_OpenAI', 'symbl', 'speechmatics', 'whisperAPI']