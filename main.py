from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import engine

app = FastAPI(title="ScriptTruth")

MAX_INPUT_WORDS = 500  # Fix 9 — explicit, visible limit; never silently truncated


class VerifyRequest(BaseModel):
    scene_text: str
    genre_mode: str = "modern"  # historical | modern | fictional | alternate_history


@app.post("/verify")
def verify(req: VerifyRequest):
    word_count = len(req.scene_text.split())
    if word_count == 0:
        raise HTTPException(status_code=400, detail="Please paste a scene to verify.")
    if word_count > MAX_INPUT_WORDS:
        raise HTTPException(
            status_code=400,
            detail=f"Please limit input to {MAX_INPUT_WORDS} words for this demo.",
        )

    results = engine.verify_scene(req.scene_text, req.genre_mode)
    return {"results": results}


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve the static frontend last so /verify and /health are matched first
app.mount("/", StaticFiles(directory="static", html=True), name="static")
