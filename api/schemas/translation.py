from pydantic import BaseModel

class TranslationRequest(BaseModel):
    text: str
    explain: bool = False