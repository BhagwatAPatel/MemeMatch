# MemeMatch

A real-time computer-vision app that reads your facial expression from your webcam
and overlays a matching meme that follows your face. Runs 100% locally.

## Status
Phase 1: live camera preview with FPS counter. See `docs/SPEC.md` for the full specification and roadmap.

## Run
```bash
python -m app.main            # default camera, press q to quit
python -m app.main --camera 1 # another camera
```

## Setup (macOS, Apple Silicon)
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```