from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.routers import generate

app = FastAPI(title="Chord Transformer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

app.include_router(generate.router)
