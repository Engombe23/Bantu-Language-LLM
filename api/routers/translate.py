from fastapi import APIRouter
from api.schemas.translation import TranslationRequest
from api.core.morph_rules import morph_preprocess, morph_postprocess

router = APIRouter(prefix="/translate", tags=["Translation"])

# Morphological translation endpoint
@router.post("/")
def translate(req: TranslationRequest):
  morphemes = morph_preprocess(req.text)
  translation = morph_postprocess(morphemes)

  response = {"translation": translation}
  if req.explain:
    response["morphemes"] = morphemes
  return response