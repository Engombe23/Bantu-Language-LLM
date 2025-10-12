from fastapi import FastAPI
from api.routers.translate import router as translate_app

app = FastAPI(title="Morphology Machine Translation API", version=0.1)

app.include_router(translate_app)

@app.get("/")
def home():
  return {"message": "Morphology MT API is running!"}