"""
WebSocket endpoint for real-time (live microphone) deepfake detection.
Client sends raw PCM chunks; server responds with JSON predictions.

Protocol:
  Client → Server : binary message containing raw PCM bytes
                    (16 kHz, 16-bit signed, mono)
  Server → Client : JSON  { chunk_id, is_fake, confidence, risk_level }
  Client → Server : text "close" to end session
"""

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.inference import analyze_chunk

log    = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/live")
async def live_detection(ws: WebSocket):
    await ws.accept()
    chunk_id = 0
    log.info("WebSocket connection opened")

    try:
        while True:
            msg = await ws.receive()

            # Text "close" signals client disconnect
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text") == "close":
                break

            raw_bytes = msg.get("bytes")
            if not raw_bytes:
                continue

            result = analyze_chunk(raw_bytes, chunk_id)
            await ws.send_text(json.dumps(result))
            chunk_id += 1

    except WebSocketDisconnect:
        pass
    finally:
        log.info(f"WebSocket closed after {chunk_id} chunks")
