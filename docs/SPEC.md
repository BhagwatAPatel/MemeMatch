# MemeMatch - Product Specification & Development Roadmap

Version 0.1 - September 2026
Target platform: macOS (Apple Silicon) first

---

## 0. Research Summary & Key Decisions

| Area | Decision | Why |
|---|---|---|
| Language | Python 3.12 | Best library support; MediaPipe wheels cover 3.9-3.12 |
| Face detection + landmarks | MediaPipe **Face Landmarker** task (`face_landmarker.task` model bundle) | One model does detection, 478 landmarks AND 52 blendshape scores, runs ~5-15 ms/frame on CPU on Apple Silicon |
| Expression features | 52 blendshape scores (+ a handful of normalised landmark distances later if needed) | Already normalised 0-1, already invariant to face position/size/lighting because MediaPipe handles that. No need to design geometry from scratch |
| Classifier | Small PyTorch MLP (52 -> 64 -> 32 -> N classes), with a scikit-learn logistic regression as a baseline | Tiny, trains in seconds on a laptop, real-time inference is sub-millisecond, and you learn a real training loop |
| Dataset | **Self-recorded** via a built-in collection tool (record labelled blendshape vectors from your own webcam), stored as CSV | Public FER datasets (FER2013, AffectNet) are low-res images and license-restricted. Your MVP needs to work on *your* face, on *your* camera. Self-collection is fast (10 min), private, and teaches the full ML loop |
| Meme library | Folder per category, PNG with alpha (RGBA). No metadata for MVP | Simplest thing that works. JSON manifest is a V1 option |
| Overlay | OpenCV/NumPy alpha compositing, anchored above the head, scaled to face width, exponential smoothing | Simple and reliable; rotation is V1 |
| Temporal logic | Rolling majority vote + confidence threshold + minimum hold time + cooldown | Prevents meme flicker |
| GUI | Plain OpenCV window until Phase 9, then PySide6 | Don't build UI before the pipeline works |
| Virtual camera | `pyvirtualcam` + OBS 30+ (Post-MVP) | Only reliable Python route on macOS 13+ |
| Tests | pytest, introduced at Phase 3 (features are pure functions) | Test what's testable; don't mock the camera |
| Packaging | PyInstaller (Phase 12, known-risky with MediaPipe) | Fallback: `pip install` + run script |

### What we are NOT using and why
- **No image-based CNN**: training one needs tens of thousands of labelled face images and a GPU; blendshapes already encode the expression.
- **No database, API, cloud, containers**: memes are files, samples are CSV, everything runs locally.
- **No multi-face tracking in MVP**: `num_faces=1`. Multi-face is Future.
- **No animated GIFs in MVP**: static PNG only. GIF frame cycling is V1.

### Expression classes
| Tier | Classes |
|---|---|
| **MVP (5)** | neutral, happy, surprised, angry, sad |
| V1 (+3) | laughing, unimpressed, confused |
| V2 / experimental | smug, suspicious, shocked, crying |

Reason: the MVP five map cleanly onto distinct blendshape patterns (mouthSmile, jawOpen+browInnerUp, browDown, mouthFrown). "Smug" and "suspicious" are subtle and person-specific; we add them once the pipeline is proven and you can measure how well they separate.

---

## 1. Product Overview

**Product name:** MemeMatch

**Purpose:** A real-time computer-vision desktop app that reads the user's facial expression from a webcam and overlays a matching meme image that follows their face.

**Problem / opportunity:** Video calls are visually boring. Snapchat-style filters exist on phones, but there is no simple, local, open, hackable desktop tool that reacts to *what expression you're making* rather than just tracking your face.

**Target users:** The developer (learning project), friends on Discord/Zoom, and anyone who wants a funny local video filter. Also a portfolio piece demonstrating CV + ML + real-time engineering.

**Core user experience:**
1. Open MemeMatch, pick a camera.
2. See a live preview.
3. Make a face. Within ~0.5 s a meme appears near your head and follows you.
4. Change expression, meme changes (after a short cooldown so it doesn't flicker).
5. (Post-MVP) Turn on virtual camera and use it in Discord/Zoom/FaceTime.

**Product vision:** "An AI meme filter for video calls" that is fast, automatic, funny, and 100% local.

---

## 2. Functional Requirements

| ID | Requirement | Tier |
|---|---|---|
| FR-001 | The application shall list available cameras and allow the user to select one. | MVP (index-based on CLI; dropdown in UI phase) |
| FR-002 | The application shall display a live camera preview at a minimum of 15 FPS. | MVP |
| FR-003 | The application shall detect a single face in real time. | MVP |
| FR-004 | The application shall extract facial landmarks and blendshape scores for the detected face. | MVP |
| FR-005 | The application shall convert landmarks/blendshapes into a fixed-length numeric feature vector. | MVP |
| FR-006 | The application shall provide a data-collection mode that records labelled feature vectors from the webcam to a CSV file. | MVP |
| FR-007 | The application shall include a training script that trains an expression classifier from the collected CSV and saves the model to disk. | MVP |
| FR-008 | The application shall include an evaluation script that reports accuracy and a confusion matrix on a held-out test split. | MVP |
| FR-009 | The application shall classify the live expression into one of the MVP classes with a confidence score every frame. | MVP |
| FR-010 | The application shall apply temporal smoothing so the displayed expression does not change more than once per configured hold time. | MVP |
| FR-011 | The application shall load memes from a local folder structure (`memes/<class>/*.png`). | MVP |
| FR-012 | The application shall select a meme from the category matching the confirmed expression (random within category). | MVP |
| FR-013 | The application shall overlay the selected meme with transparency, positioned relative to the face. | MVP |
| FR-014 | The overlay shall scale with face size and follow face movement with smoothing. | MVP |
| FR-015 | The application shall show the current expression, confidence, and FPS on screen. | MVP |
| FR-016 | The application shall not display a meme when no face is detected or confidence is below threshold. | MVP |
| FR-017 | The application shall provide a PySide6 GUI with live preview, Start/Stop, camera selector, and expression info panel. | V1 |
| FR-018 | The overlay shall rotate with head roll. | V1 |
| FR-019 | The application shall support animated GIF memes. | V1 |
| FR-020 | The application shall support a per-meme JSON manifest (anchor position, scale multiplier). | V1 |
| FR-021 | The application shall expose processed video as a virtual camera usable in Zoom/Discord/FaceTime. | V2 |
| FR-022 | The application shall support multiple faces. | Future |
| FR-023 | The application shall support an online meme source. | Future |

## 3. Non-Functional Requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-001 | Real-time responsiveness | >= 20 FPS end-to-end at 640x480 on an Apple Silicon Mac; expression-to-meme latency < 1 s including smoothing |
| NFR-002 | Local processing | No frame, landmark, or feature ever leaves the machine. No network calls at runtime |
| NFR-003 | CPU usage | Single process, no GPU required; should not peg all cores |
| NFR-004 | Reliability | Camera loss, missing model file, empty meme folder produce a clear message, not a crash |
| NFR-005 | Maintainability | One module per pipeline stage; each stage testable in isolation |
| NFR-006 | Usability | Runs with one command; no config editing needed for defaults |
| NFR-007 | Privacy | Collected training data stored only as numeric CSV (no images); logs contain no frames |
| NFR-008 | Reproducibility | `requirements.txt` pinned; fresh clone + venv + install works |

---

## 4. MVP Definition

MVP is complete when, from a fresh clone, you can:

1. Run a collection tool and record 5 expressions from your webcam.
2. Train a model and see >= 85% test accuracy on your own data.
3. Run the app, see yourself, make a face, and see a correct meme appear above your head and follow you at >= 20 FPS.
4. All in an OpenCV window (no PySide6 yet), single face, static PNGs, no virtual camera.

## 5. Release Tiers

| Tier | Contents |
|---|---|
| **MVP** | Phases 0-8: pipeline + CLI app in an OpenCV window |
| **V1** | PySide6 GUI, head-roll rotation, GIF memes, JSON manifest, 3 extra classes, tests + packaging |
| **V2** | Virtual camera via OBS/pyvirtualcam |
| **Future** | Multi-face, online meme source, personalised fine-tuning, non-Mac builds |

---

## 6. Technical Architecture

### 6.1 Runtime data flow

```text
 Physical camera (OpenCV VideoCapture)
        |  BGR frame (numpy H x W x 3)
        v
 face_tracker.py  ---- MediaPipe Face Landmarker (VIDEO mode)
        |  478 landmarks + 52 blendshapes + face box
        v
 features.py      ---- blendshapes -> np.float32[52]
        |
        v
 expression_classifier.py ---- PyTorch MLP -> (class, confidence)
        |
        v
 smoothing.py     ---- majority vote / threshold / hold / cooldown
        |  confirmed expression (or None)
        v
 meme_engine.py   ---- category -> pick PNG (RGBA)
        |
        v
 overlay.py       ---- scale, position (smoothed), alpha-blend onto frame
        |
        v
 display (cv2.imshow)   -->  [V2] virtual_camera.py -> OBS Virtual Camera -> Zoom/Discord
```

### 6.2 Threading model (V1 UI phase)
```text
Main thread (Qt event loop)  <--signal(frame)--  Worker QThread: capture -> track -> classify -> overlay
```
MVP runs everything in one loop; it is fast enough. Threading is introduced only when the GUI needs it.

### 6.3 Dependencies (pinned in requirements.txt as installed)
- opencv-python, numpy
- mediapipe
- torch, scikit-learn
- PySide6 (V1)
- pyvirtualcam (V2, needs OBS 30+)
- pytest (dev)

### 6.4 Project structure

```text
MemeMatch/
├── app/                         # the runtime application (a Python package)
│   ├── __init__.py
│   ├── main.py                  # entry point: python -m app.main
│   ├── camera.py                # open camera, read frames, handle errors, FPS counter
│   ├── face_tracker.py          # MediaPipe Face Landmarker wrapper -> FaceResult
│   ├── features.py              # FaceResult -> feature vector (shared with training)
│   ├── expression_classifier.py # loads model, predicts (class, confidence)
│   ├── smoothing.py             # temporal decision logic
│   ├── meme_engine.py           # loads memes/ folder, picks a meme
│   ├── overlay.py               # positions and alpha-blends meme onto frame
│   ├── ui.py                    # [V1] PySide6 window
│   └── virtual_camera.py        # [V2] pyvirtualcam output
├── training/
│   ├── collect.py               # webcam -> labelled CSV rows  (python -m training.collect)
│   ├── dataset.py               # load CSV, split, tensors
│   ├── model.py                 # MLP definition (imported by app too)
│   ├── train.py                 # training loop, saves models/expression_model.pt
│   └── evaluate.py              # accuracy, confusion matrix
├── models/
│   ├── face_landmarker.task     # downloaded MediaPipe bundle (~3.7 MB)
│   └── expression_model.pt      # your trained classifier
├── memes/
│   ├── happy/  angry/  surprised/  sad/  neutral/
├── data/                        # collected CSV samples (gitignored)
├── tests/
├── docs/
│   └── SPEC.md                  # this file
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

**Why `app/` and `training/` are separate packages:** training is offline and runs once; the app is real-time and runs always. `features.py` and `model.py` are shared so the live app computes features exactly the same way the training data was made. That "train/serve skew" bug is one of the most common real-world ML failures, and the structure prevents it.

---

## 7. ML Design

| Item | Decision |
|---|---|
| Input | 52 blendshape scores (float32, each 0-1) from MediaPipe |
| Output | Softmax over N classes (N=5 for MVP) |
| Architecture | `Linear(52,64) -> ReLU -> Dropout(0.2) -> Linear(64,32) -> ReLU -> Linear(32,N)` |
| Baseline | scikit-learn `LogisticRegression` on the same features (to prove the MLP is worth it) |
| Dataset | ~300-500 samples per class, recorded at ~10 samples/s while holding each expression, in at least 3 sessions (different lighting, distance, head angle). A session is one `collect` run; its id is a timestamp, with an optional `--session` override. Every session must contain every label |
| Preprocessing | None needed beyond float32 cast; blendshapes are already normalised. (If we later add landmark distances we standardise them.) |
| Split | **Whole sessions** are assigned to splits, never individual samples. Requires >= 3 sessions. By default the last session in file order is test, the one before it is validation, the rest are train; `load_dataset(test_session=, val_session=)` overrides. Fewer than 3 sessions, or a label missing from any session, is an error with a per-session count table |
| Training | Fixed seed, Adam, lr 1e-3, cross-entropy (no class weighting), batch 64, 50 epochs, keep the epoch with best val accuracy (ties broken by lower val loss). Always saves; the shipped model is trained on the train sessions only |
| Overfitting control | Dropout, best-epoch selection on val accuracy, small model, session-based split |
| Metrics | Accuracy, per-class precision/recall, confusion matrix, and confidence for correct vs. wrong test predictions (used to choose the live threshold). `evaluate` exits non-zero if test accuracy < 85% or any class recall < 70% |
| Persistence | One `.pt` file holding the state dict, the label names in order and the feature names. `load_model` raises a clear error if the feature contract differs from `FEATURE_NAMES`; there is no remapping |
| Inference | `model.eval()`, `torch.no_grad()`, single forward pass per frame (< 1 ms) |

**Why not just threshold blendshapes by hand?** You could (mouthSmile > 0.5 => happy). But rules break on "surprised vs angry-with-open-mouth" and the classifier learns your face. The MLP is also the educational point of the project. We *will* write a rule-based fallback in Phase 6 as a comparison, which is a great 10-line exercise.

---

## 8. Meme Matching & Overlay Design

| Concern | MVP decision |
|---|---|
| Mapping | class name == folder name |
| Selection | random within folder; avoid repeating the last meme |
| Format | PNG with alpha channel; loaded once at startup with `cv2.IMREAD_UNCHANGED` |
| Position | anchored above the head: centre-x = face centre, bottom = top of face box minus 10% face height |
| Scale | meme width = 1.2 x face width, aspect preserved |
| Rotation | none (V1: rotate by roll angle from the eye line) |
| Tracking | face box every frame, smoothed with EMA (alpha ~0.3) for position and size |
| Confidence threshold | 0.70 |
| Vote window | last 10 predictions must agree by majority |
| Hold time | minimum 1.5 s before a switch |
| Cooldown | 0.5 s of "no meme" between switches (optional, tunable) |
| No face | fade out / hide immediately |

---

## 9. Real-time Pipeline & Performance

Per-frame budget at 30 FPS is 33 ms. Expected on an M-series Mac at 640x480:

| Stage | Expected |
|---|---|
| Camera read | 1-3 ms (limited by camera) |
| Face Landmarker | 5-15 ms |
| Features + MLP | < 1 ms |
| Overlay compositing | 1-3 ms |
| Display | 1-2 ms |
| **Total** | ~10-25 ms => 30 FPS achievable, 20 FPS comfortable |

Mitigations if needed: run at 640x480 not 1080p; run landmarker every 2nd frame and reuse the box; never resize the meme every frame (cache per size bucket). CPU only; no GPU work required.

---

## 10. Privacy Design

- All processing local. No sockets opened at runtime.
- Training data = CSV of 52 floats + label + session id. **No images are stored.**
- `data/` is gitignored; you decide if you ever share it.
- Logs print class names and FPS, never frames or landmarks.
- Note: MediaPipe's own privacy notice states input data stays on device but that it may send *usage metrics* to Google. We will document this in the README and, if a disable option exists in the installed version, use it.

---

## 11. Virtual Camera (V2) - honest assessment

- macOS 12.3+ replaced the old DAL plugin system with **Camera Extensions** (system extensions, must be signed, Swift/Obj-C, installed inside a .app bundle). Pure Python cannot create one.
- **OBS Studio 30+** ships a signed Camera Extension. `pyvirtualcam` (0.14+) on macOS 13+ sends frames into OBS's virtual camera. One-time setup: install OBS, click Start/Stop Virtual Camera once.
- Result: "OBS Virtual Camera" appears as a camera in Zoom, Discord, Meet, and FaceTime (Camera Extensions are system-wide).
- Limitations: OBS must be installed (not running); only one virtual camera instance; RGB only (no alpha); user must grant Camera permission to the Python process (Terminal).
- Alternative (Future): write a native Swift Camera Extension and feed it from Python over shared memory. Legitimate, but a separate project.

---

## 12. Testing Requirements

| Type | What | When introduced |
|---|---|---|
| Unit | `features.py` (vector length, ordering), `smoothing.py` (vote/hold/cooldown logic with fake timestamps), `overlay.py` (blend math on synthetic images), `meme_engine.py` (folder loading, no-repeat) | Phase 3 onward, as each module is written |
| ML eval | `training/evaluate.py` on held-out split; confusion matrix must show no class < 70% recall | Phase 5 |
| Integration | Run the pipeline on a short recorded video file instead of a live camera; assert a face is found and a class is produced | Phase 6 |
| Performance | FPS counter on screen; log average frame time over 30 s | Phase 6 |
| Manual | Checklist per phase (see Success Checks) | Every phase |

---

## 13. Acceptance Criteria (MVP requirements)

**FR-001/002 (camera)**: `python -m app.main` opens the default camera, shows live video >= 15 FPS, `q` quits, and a missing camera prints a readable error.

**FR-003/004 (face + landmarks)**: landmarks drawn on your face; follow you as you move, lean, turn ~30 degrees; drop out cleanly when you leave frame; FPS still >= 20.

**FR-005 (features)**: `features.extract()` returns shape `(52,)` float32; unit test passes; printing a few values changes visibly when you smile.

**FR-006 (collect)**: pressing a key records ~10 rows/s to `data/samples_<session>.csv` with a label; file opens in a spreadsheet and looks sane.

**FR-007/008 (train/eval)**: `python -m training.train` finishes in < 1 minute, saves `models/expression_model.pt`; `python -m training.evaluate` prints test accuracy >= 85% and a 5x5 confusion matrix; logistic-regression baseline printed for comparison.

**FR-009/010 (live inference + smoothing)**: expression label and confidence drawn on screen; the label does not flicker frame-to-frame; a held expression is confirmed within ~1 s.

**FR-011/012 (meme engine)**: app starts with 2+ PNGs in each of 5 folders; a warning (not a crash) if a folder is empty.

**FR-013/014 (overlay)**: meme appears above your head with clean transparent edges, scales when you move closer/further, follows you smoothly, disappears when no face.

**FR-015/016**: HUD shows expression, confidence, FPS; nothing overlays when confidence < 0.7.

---

## 14. Development Roadmap

Each phase becomes one or more numbered Steps in the mentoring sessions. Every step ends with a commit.

| Phase | Name | Deliverable | Git milestone |
|---|---|---|---|
| 0 | Environment & project setup | repo skeleton, venv, first deps, GitHub push | `chore: initial project skeleton` |
| 1 | Camera | `camera.py`, `main.py` shows live video with FPS | `feat: live camera preview with FPS counter` |
| 2 | Face detection | `face_tracker.py`, landmarks drawn, model bundle downloaded | `feat: MediaPipe face landmarker integration` |
| 3 | Feature extraction | `features.py` + first pytest | `feat: blendshape feature extraction with tests` |
| 4 | Data collection | `training/collect.py`, record 5 classes | `feat: expression data collection tool` |
| 5 | Model | `model.py`, `dataset.py`, `train.py`, `evaluate.py`, saved model | `feat: train and evaluate expression MLP` |
| 6 | Live inference | `expression_classifier.py`, `smoothing.py`, HUD | `feat: real-time expression classification` |
| 7 | Meme engine | `meme_engine.py`, meme folders | `feat: local meme library and selection` |
| 8 | Overlay & tracking | `overlay.py`, EMA smoothing | `feat: meme overlay tracking the face` (MVP DONE) |
| 9 | GUI | `ui.py` PySide6, worker thread | `feat: PySide6 interface` |
| 10 | Virtual camera | `virtual_camera.py` via pyvirtualcam/OBS | `feat: virtual camera output` |
| 11 | Testing & performance | fill in tests, video-file integration test, perf log | `test: integration and performance tests` |
| 12 | Packaging & docs | PyInstaller attempt, README, architecture doc, portfolio material | `docs: final documentation` |

Git teaching schedule:
- Phase 0: init, status, add, commit, remote, push, .gitignore
- Phase 1-2: log, diff, amend a message, `git restore` to undo an uncommitted change
- Phase 3: first branch (`feature/features`) + merge, because tests + code arrive together
- Phase 5: tags (`v0.1-mvp-model`), `git revert` a bad commit
- Phase 8: tag `v0.1-mvp`
- Phase 9+: branches for every feature, PR-style merges on GitHub

---

## 15. Open Questions to Verify During Build

1. MediaPipe 1.0 renamed some Python namespaces. We verify the import path in Phase 2 and adjust.
2. Whether the installed MediaPipe exposes a metrics opt-out.
3. Exact OBS/pyvirtualcam behaviour on the current macOS version (Phase 10).
4. PyInstaller + MediaPipe bundling (Phase 12) - may require hidden-import flags or fall back to a pip-install distribution.
