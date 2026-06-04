# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

Single Python batch application: `face_head.py` splits classroom videos (≤120s segments), runs YOLOv10 face/head detection, logs head-up rate CSVs, and writes annotated MP4s. There is no web server, database, or Docker stack. `A.tex` is a separate LaTeX report and is not required to run the pipeline.

### System dependencies

- **ffmpeg** and **ffprobe** must be on `PATH` (preinstalled on the Cloud VM at `/usr/bin/`).
- **Python 3.12+** with pip packages: `opencv-python`, `numpy`, `Pillow`, `ultralytics` (installed via the VM update script into `~/.local`).

### Running the pipeline

1. Place input videos in `data/` (gitignored): `*.mp4`, `.mov`, `.mkv`, `.avi`.
2. From repo root: `python3 face_head.py`
   - `--data-dir PATH` — alternate input folder
   - `--force` — reprocess all segments (ignore `output/progress.json`)
3. **First run** downloads YOLO weights into `weights/` (~12 MB). The primary head-model URL on GitHub may 404; the script automatically falls back to jsDelivr.
4. Outputs (gitignored except sample logs under `output/logs/`):
   - `output/segments/` — split MP4s
   - `output/annotated/` — overlay videos
   - `output/logs/*.log` — CSV: `segment,timestamp_sec,face_count,head_count,head_up_rate`
   - `output/progress.json` — resume state

### Lint and tests

No linter or test suite is configured in-repo. For a quick sanity check: `python3 -m py_compile face_head.py`.

### Optional

- **GPU**: Ultralytics uses CUDA when available; CPU inference works but is slower on long videos.
- **Chinese HUD labels**: install `fonts-noto-cjk` if `/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc` is missing (Pillow falls back to a default font).
- **`A.tex`**: requires XeLaTeX/LuaLaTeX, `ctex`, and external `BNUpapers` / `figures/` assets not shipped in this repo.

### Gotchas

- Empty or missing `data/` causes exit code 1 with a clear log message.
- Synthetic test patterns (e.g. ffmpeg `testsrc`) produce zero face detections; use real classroom footage for meaningful rates.
- Re-running without `--force` skips segments already listed in `output/progress.json`.
