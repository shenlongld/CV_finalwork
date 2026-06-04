#!/usr/bin/env python3
"""Split classroom videos, detect faces/heads with YOLOv10, log head-up rate."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
WEIGHTS_DIR = SCRIPT_DIR / "weights"
OUTPUT_DIR = SCRIPT_DIR / "output"
SEGMENTS_DIR = OUTPUT_DIR / "segments"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
LOGS_DIR = OUTPUT_DIR / "logs"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

MAX_SEGMENT_SECONDS = 120
DETECT_INTERVAL_SEC = 1.0
DEFAULT_FACE_MODEL = WEIGHTS_DIR / "yolov10n-face.pt"
DEFAULT_HEAD_MODEL = WEIGHTS_DIR / "yolov10n-head.pt"

FACE_MODEL_URL = (
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov10n-face.pt"
)
HEAD_MODEL_URLS = (
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov10n-head.pt",
    "https://cdn.jsdelivr.net/gh/Abcfsa/YOLOv8_head_detector@main/nano.pt",
)
CN_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
COLOR_HEAD_UP = (0, 200, 0)
COLOR_HEAD_DOWN = (0, 80, 255)
COLOR_FACE = (0, 255, 120)


@dataclass
class DetectionStats:
    face_count: int
    head_count: int
    head_up_count: int
    head_up_rate: float


@dataclass
class OverlayState:
    stats: DetectionStats
    face_boxes: np.ndarray
    head_boxes: np.ndarray
    head_up_flags: list[bool]
    matched_face_indices: set[int]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dirs() -> None:
    for path in (WEIGHTS_DIR, SEGMENTS_DIR, ANNOTATED_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path) -> None:
    logging.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        dest.write_bytes(response.read())


def ensure_model(path: Path, urls: tuple[str, ...]) -> None:
    if path.exists() and path.stat().st_size > 1000:
        return

    last_error: Exception | None = None
    for url in urls:
        try:
            download_file(url, path)
            if path.stat().st_size > 1000:
                if url != urls[0]:
                    logging.warning("Primary model URL unavailable, using fallback: %s", url)
                return
        except Exception as exc:  # noqa: BLE001 - try next mirror
            last_error = exc
            logging.warning("Failed to download %s: %s", url, exc)

    raise RuntimeError(f"Unable to download model to {path}") from last_error


def load_progress() -> dict:
    if not PROGRESS_FILE.exists():
        return {"videos": {}}
    return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))


def save_progress(progress: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")


def probe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def split_video_to_mp4(video_path: Path, video_key: str) -> list[Path]:
    duration = probe_duration(video_path)
    segment_count = max(1, int(np.ceil(duration / MAX_SEGMENT_SECONDS)))
    output_pattern = SEGMENTS_DIR / f"{video_key}_seg%03d.mp4"

    existing = sorted(SEGMENTS_DIR.glob(f"{video_key}_seg*.mp4"))
    if len(existing) >= segment_count:
        logging.info("Reusing %d existing segments for %s", len(existing), video_path.name)
        return existing[:segment_count]

    for old_segment in existing:
        old_segment.unlink(missing_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-f",
        "segment",
        "-segment_time",
        str(MAX_SEGMENT_SECONDS),
        "-reset_timestamps",
        "1",
        str(output_pattern),
    ]
    logging.info(
        "Splitting %s into <=%ds mp4 segments (duration=%.1fs, expected=%d)",
        video_path.name,
        MAX_SEGMENT_SECONDS,
        duration,
        segment_count,
    )
    subprocess.run(cmd, check=True)
    segments = sorted(SEGMENTS_DIR.glob(f"{video_key}_seg*.mp4"))
    if not segments:
        raise RuntimeError(f"No segments created for {video_path}")
    return segments


def count_boxes(result) -> int:
    boxes = result.boxes
    return 0 if boxes is None else len(boxes)


def boxes_xyxy(result) -> np.ndarray:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 4), dtype=np.int32)
    return boxes.xyxy.cpu().numpy().astype(np.int32)


def box_center(box: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_box(px: float, py: float, box: np.ndarray, margin: float = 0.05) -> bool:
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    return (
        x1 - margin * width <= px <= x2 + margin * width
        and y1 - margin * height <= py <= y2 + margin * height
    )


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_heads_to_faces(
    face_boxes: np.ndarray,
    head_boxes: np.ndarray,
) -> tuple[list[bool], set[int]]:
    """Match each head to at most one face; face center inside head box => head up."""
    if len(head_boxes) == 0:
        return [], set()

    candidates: list[tuple[float, int, int]] = []
    for head_idx, head in enumerate(head_boxes):
        for face_idx, face in enumerate(face_boxes):
            cx, cy = box_center(face)
            if point_in_box(cx, cy, head):
                candidates.append((box_iou(face, head), head_idx, face_idx))

    candidates.sort(reverse=True)
    matched_heads: set[int] = set()
    matched_faces: set[int] = set()
    head_up_flags = [False] * len(head_boxes)

    for _, head_idx, face_idx in candidates:
        if head_idx in matched_heads or face_idx in matched_faces:
            continue
        matched_heads.add(head_idx)
        matched_faces.add(face_idx)
        head_up_flags[head_idx] = True

    return head_up_flags, matched_faces


def get_cn_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if CN_FONT_PATH.exists():
        return ImageFont.truetype(str(CN_FONT_PATH), size)
    return ImageFont.load_default()


def draw_cn_texts(
    frame: np.ndarray,
    labels: list[tuple[str, tuple[int, int], tuple[int, int, int], int]],
) -> np.ndarray:
    """Draw Chinese labels with filled background. labels: (text, pos, bgr_color, font_size)."""
    if not labels:
        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    for text, (x, y), bgr_color, font_size in labels:
        font = get_cn_font(font_size)
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        bbox = draw.textbbox((x, y), text, font=font)
        pad = 3
        draw.rectangle(
            (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
            fill=(20, 20, 20),
        )
        draw.text((x, y), text, font=font, fill=rgb_color)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def detect_stats(face_model: YOLO, head_model: YOLO, frame: np.ndarray) -> tuple[DetectionStats, OverlayState]:
    face_result = face_model(frame, verbose=False)[0]
    head_result = head_model(frame, verbose=False)[0]

    face_boxes = boxes_xyxy(face_result)
    head_boxes = boxes_xyxy(head_result)
    face_count = len(face_boxes)
    head_count = len(head_boxes)
    head_up_flags, matched_faces = match_heads_to_faces(face_boxes, head_boxes)
    head_up_count = sum(head_up_flags)
    head_up_rate = head_up_count / head_count if head_count > 0 else 0.0

    stats = DetectionStats(
        face_count=face_count,
        head_count=head_count,
        head_up_count=head_up_count,
        head_up_rate=head_up_rate,
    )
    overlay = OverlayState(
        stats=stats,
        face_boxes=face_boxes,
        head_boxes=head_boxes,
        head_up_flags=head_up_flags,
        matched_face_indices=matched_faces,
    )
    return stats, overlay


def draw_overlay(frame: np.ndarray, overlay: OverlayState, timestamp_sec: float) -> np.ndarray:
    vis = frame.copy()
    cn_labels: list[tuple[str, tuple[int, int], tuple[int, int, int], int]] = []

    for head_idx, (x1, y1, x2, y2) in enumerate(overlay.head_boxes):
        is_head_up = overlay.head_up_flags[head_idx] if head_idx < len(overlay.head_up_flags) else False
        color = COLOR_HEAD_UP if is_head_up else COLOR_HEAD_DOWN
        status = "抬头" if is_head_up else "低头"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cn_labels.append((status, (x1, max(y1 - 26, 4)), color, 18))

    for face_idx, (x1, y1, x2, y2) in enumerate(overlay.face_boxes):
        if face_idx not in overlay.matched_face_indices:
            continue
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_FACE, 1)

    stats = overlay.stats
    hud_lines = [
        (f"时间: {timestamp_sec:6.1f}s", COLOR_HEAD_UP, 22),
        (f"抬头: {stats.head_up_count} / 总计: {stats.head_count}", COLOR_HEAD_UP, 22),
        (f"人脸检测: {stats.face_count}", COLOR_FACE, 20),
        (f"抬头率: {stats.head_up_rate:.1%}", COLOR_HEAD_UP, 22),
    ]
    for idx, (line, color, size) in enumerate(hud_lines):
        cn_labels.append((line, (16, 16 + idx * 34), color, size))

    return draw_cn_texts(vis, cn_labels)


def append_log_line(
    log_path: Path,
    segment_name: str,
    timestamp_sec: float,
    stats: DetectionStats,
) -> None:
    header = "segment,timestamp_sec,face_count,head_count,head_up_rate\n"
    line = (
        f"{segment_name},{timestamp_sec:.1f},{stats.face_count},"
        f"{stats.head_count},{stats.head_up_rate:.4f}\n"
    )
    if not log_path.exists():
        log_path.write_text(header + line, encoding="utf-8")
    else:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(line)


def process_segment(
    segment_path: Path,
    face_model: YOLO,
    head_model: YOLO,
    log_path: Path,
    output_path: Path,
) -> None:
    cap = cv2.VideoCapture(str(segment_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open segment: {segment_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detect_every = max(1, int(round(fps * DETECT_INTERVAL_SEC)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video: {output_path}")

    frame_idx = 0
    overlay = OverlayState(
        stats=DetectionStats(0, 0, 0, 0.0),
        face_boxes=np.empty((0, 4), dtype=np.int32),
        head_boxes=np.empty((0, 4), dtype=np.int32),
        head_up_flags=[],
        matched_face_indices=set(),
    )

    logging.info("Processing segment %s", segment_path.name)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_sec = frame_idx / fps
        if frame_idx % detect_every == 0:
            stats, overlay = detect_stats(face_model, head_model, frame)
            append_log_line(log_path, segment_path.stem, timestamp_sec, stats)
            logging.info(
                "%s t=%.1fs up=%d/%d faces=%d rate=%.2f%%",
                segment_path.name,
                timestamp_sec,
                stats.head_up_count,
                stats.head_count,
                stats.face_count,
                stats.head_up_rate * 100,
            )

        writer.write(draw_overlay(frame, overlay, timestamp_sec))
        frame_idx += 1

    cap.release()
    writer.release()
    logging.info("Finished segment %s (%d frames)", segment_path.name, frame_idx)


def video_key(video_path: Path) -> str:
    return video_path.stem.replace(" ", "_")


def process_video(
    video_path: Path,
    face_model: YOLO,
    head_model: YOLO,
    progress: dict,
    force: bool = False,
) -> None:
    key = video_key(video_path)
    progress.setdefault("videos", {}).setdefault(key, {"completed_segments": []})
    video_progress = progress["videos"][key]
    completed = set(video_progress.get("completed_segments", []))

    segments = split_video_to_mp4(video_path, key)
    log_path = LOGS_DIR / f"{key}.log"

    if force and log_path.exists():
        log_path.unlink()

    for segment_path in segments:
        segment_name = segment_path.stem
        output_path = ANNOTATED_DIR / f"{segment_name}_annotated.mp4"

        if not force and segment_name in completed and output_path.exists():
            logging.info("Skip completed segment: %s", segment_name)
            continue

        process_segment(segment_path, face_model, head_model, log_path, output_path)

        completed.add(segment_name)
        video_progress["completed_segments"] = sorted(completed)
        video_progress["log_file"] = str(log_path)
        save_progress(progress)


def collect_videos(data_dir: Path) -> list[Path]:
    patterns = ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.mkv", "*.MKV", "*.avi", "*.AVI")
    videos: list[Path] = []
    for pattern in patterns:
        videos.extend(sorted(data_dir.glob(pattern)))
    return sorted(set(videos))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split videos and compute head-up rate.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Input video directory")
    parser.add_argument("--face-model", type=Path, default=DEFAULT_FACE_MODEL, help="YOLOv10 face weights")
    parser.add_argument("--head-model", type=Path, default=DEFAULT_HEAD_MODEL, help="YOLOv10 head weights")
    parser.add_argument("--force", action="store_true", help="Reprocess all segments")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    ensure_dirs()
    args = parse_args()

    if not args.data_dir.exists():
        logging.error("Data directory not found: %s", args.data_dir)
        return 1

    videos = collect_videos(args.data_dir)
    if not videos:
        logging.error("No videos found in %s", args.data_dir)
        return 1

    ensure_model(args.face_model, (FACE_MODEL_URL,))
    ensure_model(args.head_model, HEAD_MODEL_URLS)

    logging.info("Loading models...")
    face_model = YOLO(str(args.face_model))
    head_model = YOLO(str(args.head_model))
    progress = load_progress()

    for video_path in videos:
        logging.info("=== Video: %s ===", video_path.name)
        process_video(video_path, face_model, head_model, progress, force=args.force)

    logging.info("All done. Logs: %s", LOGS_DIR)
    logging.info("Annotated videos: %s", ANNOTATED_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
