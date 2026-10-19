"""
FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router as api_router
from app.websocket import router as ws_router
from app.inference import get_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm model on startup
    log.info("Loading inference model...")
    get_session()
    log.info("Model ready.")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="DeepFake Voice Scam Detector",
    description=(
        "Real-time API for detecting AI-generated / deepfake voices in phone calls. "
        "Trained on ASVspoof 2019 LA + WaveFake."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from the frontend dev server and production domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(ws_router)


@app.get("/")
async def root():
    return {
        "service": "DeepFake Voice Detector",
        "version": "1.0.0",
        "docs":    "/docs",
    }
