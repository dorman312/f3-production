#!/usr/bin/env python3
"""
stream.py — Standalone RTSP stream processor for Fan Flow & Fusion.

Fetches all cameras for a venue, connects to each RTSP stream, analyzes
one frame every 15 seconds with YOLOv8 person detection, posts results to
the F3 API, and writes a CSV audit log to logs/.

Usage:
    python3 stream.py --venue-id 7bd60213-8351-480e-9ff4-399130fd0c2f
    python3 stream.py --venue-id <ID> --api-url https://web-production-34b8b.up.railway.app
    python3 stream.py --venue-id <ID> --interval 15 --no-ingest
    python3 stream.py --venue-id <ID> --no-ingest     # test cameras without writing data

Options:
    --venue-id   (required) UUID of the venue whose cameras to process
    --api-url    Base URL of the F3 API
                 [default: https://web-production-34b8b.up.railway.app]
    --interval   Seconds between frame captures per camera  [default: 15]
    --no-ingest  Print and log results only; do not POST to /ingest/camera
    --model      YOLOv8 weights file  [default: yolov8n.pt]
"""

import argparse
import asyncio
import csv
import json
import logging
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────
API_URL        = "https://web-production-34b8b.up.railway.app"
MIN_CONFIDENCE = 0.40   # frames below this confidence are skipped
RECONNECT_WAIT = 30     # seconds between reconnect attempts
LOG_DIR        = Path(__file__).resolve().parent / "logs"

# ── ANSI colours (auto-disabled when not a TTY) ───────────────────
_TTY = sys.stdout.isatty()
def _c(code, t): return f"\033[{code}m{t}\033[0m" if _TTY else t
def _green(t):   return _c("32", t)
def _yellow(t):  return _c("33", t)
def _red(t):     return _c("31", t)
def _bold(t):    return _c("1",  t)
def _dim(t):     return _c("2",  t)
def _cyan(t):    return _c("36", t)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s %(name)s  %(message)s",
)

_CSV_HEADER = ["timestamp", "venue_id", "zone_id", "zone_name",
               "camera_id", "person_count", "avg_confidence", "density_score"]


# ── CSV ───────────────────────────────────────────────────────────

def _csv_path() -> Path:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return LOG_DIR / f"readings_{date_str}.csv"


def _append_csv(row: dict) -> None:
    path = _csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_HEADER)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ── API ───────────────────────────────────────────────────────────

def _api_get(base: str, path: str) -> list | dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"API error {exc.code} for {path}") from exc
    except Exception as exc:
        raise SystemExit(f"Cannot reach API at {base}: {exc}") from exc


def _api_post(base: str, path: str, body: dict, timeout: int = 5) -> int:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        base.rstrip("/") + path, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:
        return 0


# ── Analyzer loader ───────────────────────────────────────────────

def _load_analyzer(model_path: str):
    try:
        from ml.crowd_analyzer import CrowdDensityAnalyzer
        return CrowdDensityAnalyzer(model_path)
    except ImportError:
        raise SystemExit(
            "Cannot import CrowdDensityAnalyzer. "
            "Run this script from the f3-production directory."
        )


# ── Console output ────────────────────────────────────────────────

def _print_header(cameras: list, venue_name: str, api_url: str, interval: int) -> None:
    print()
    print(_bold("Fan Flow & Fusion — Live Stream Processor"))
    print(_dim(f"  Venue:     {venue_name}"))
    print(_dim(f"  API:       {api_url}"))
    print(_dim(f"  Cameras:   {len(cameras)}"))
    print(_dim(f"  Interval:  {interval}s per frame"))
    print(_dim(f"  Min conf:  {MIN_CONFIDENCE}"))
    print(_dim(f"  CSV log:   {_csv_path()}"))
    print(_dim("  Press Ctrl-C to stop"))
    print(_dim("─" * 72))
    print()


def _risk_col(density: float):
    if density >= 70: return _red
    if density >= 40: return _yellow
    return _green


def _print_reading(*, camera_id, zone_name, result, ts, posted, no_ingest):
    density = result["density_score"]
    col     = _risk_col(density)
    status  = _dim("(no ingest)") if no_ingest else (_green("✓ posted") if posted else _yellow("✗ ingest failed"))
    print(
        f"  {_dim(ts.strftime('%H:%M:%S'))}  "
        f"{_cyan(camera_id[:22]):<32}  "
        f"zone={zone_name[:20]:<22}  "
        f"people={_bold(str(result['person_count'])):>5}  "
        f"conf={result['avg_confidence']:.2f}  "
        f"density={col(f'{density:.1f}%')}  "
        f"{status}"
    )


def _print_skip(camera_id, zone_name, conf):
    print(
        f"  {_dim('--:--:--')}  "
        f"{_cyan(camera_id[:22]):<32}  "
        f"zone={zone_name[:20]:<22}  "
        f"{_yellow(f'skipped — low conf {conf:.2f} < {MIN_CONFIDENCE}')}"
    )


def _print_event(camera_id, msg, level="info"):
    icon = {"ok": "✓", "warn": "⚠", "error": "✗"}.get(level, "·")
    col  = {"ok": _green, "warn": _yellow, "error": _red}.get(level, _dim)
    print(f"  {col(icon)} {_cyan(camera_id[:22])}: {msg}")


# ── Per-camera async task ─────────────────────────────────────────

async def _camera_task(
    *,
    cam: dict,
    venue_id: str,
    api_url: str,
    interval: int,
    no_ingest: bool,
    model_path: str,
    executor: ThreadPoolExecutor,
) -> None:
    camera_id = cam["camera_id"]
    rtsp_url  = cam["rtsp_url"]
    zone_id   = cam.get("zone_id", "")
    zone_name = cam.get("name") or cam.get("camera_id") or zone_id[:12]

    analyzer = _load_analyzer(model_path)
    loop     = asyncio.get_running_loop()

    while True:
        # ── Connect ────────────────────────────────────────────────
        _print_event(camera_id, f"connecting ({rtsp_url.split('@')[-1]})", "info")

        def _open():
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        cap: cv2.VideoCapture = await loop.run_in_executor(executor, _open)
        is_open = await loop.run_in_executor(executor, cap.isOpened)

        if not is_open:
            await loop.run_in_executor(executor, cap.release)
            _print_event(camera_id, f"could not open stream — retry in {RECONNECT_WAIT}s", "warn")
            await asyncio.sleep(RECONNECT_WAIT)
            continue

        _print_event(camera_id, "connected ✓", "ok")

        # ── Read loop ──────────────────────────────────────────────
        try:
            while True:
                ret, frame = await loop.run_in_executor(executor, cap.read)
                if not ret or frame is None:
                    _print_event(camera_id, f"stream ended — reconnecting in {RECONNECT_WAIT}s", "warn")
                    break

                result: dict = await loop.run_in_executor(
                    executor, analyzer.analyze_frame, frame
                )
                ts   = datetime.now(timezone.utc)
                conf = result["avg_confidence"]

                # Confidence gate
                if conf < MIN_CONFIDENCE:
                    _print_skip(camera_id, zone_name, conf)
                    await asyncio.sleep(interval)
                    continue

                # Post to API
                posted = False
                if not no_ingest and zone_id:
                    status = await loop.run_in_executor(executor, lambda: _api_post(
                        api_url, "/ingest/camera", {
                            "venue_id":     venue_id,
                            "zone_id":      zone_id,
                            "person_count": result["person_count"],
                            "timestamp":    ts.isoformat(),
                            "camera_id":    camera_id,
                        }
                    ))
                    posted = status == 201

                # Console output
                _print_reading(
                    camera_id=camera_id,
                    zone_name=zone_name,
                    result=result,
                    ts=ts,
                    posted=posted,
                    no_ingest=no_ingest,
                )

                # CSV log (always, regardless of --no-ingest)
                await loop.run_in_executor(executor, _append_csv, {
                    "timestamp":     ts.isoformat(),
                    "venue_id":      venue_id,
                    "zone_id":       zone_id,
                    "zone_name":     zone_name,
                    "camera_id":     camera_id,
                    "person_count":  result["person_count"],
                    "avg_confidence":round(conf, 3),
                    "density_score": round(result["density_score"], 1),
                })

                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            await loop.run_in_executor(executor, cap.release)
            raise
        except Exception as exc:
            _print_event(camera_id, f"error: {exc}", "error")
        finally:
            await loop.run_in_executor(executor, cap.release)

        await asyncio.sleep(RECONNECT_WAIT)


# ── Main ──────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    api_url  = args.api_url.rstrip("/")
    venue_id = args.venue_id

    # Fetch venue name for display
    try:
        venue_info = _api_get(api_url, f"/venues/{venue_id}")
        venue_name = venue_info.get("name", venue_id)
    except SystemExit:
        venue_name = venue_id

    # Fetch cameras
    print(_dim(f"\nFetching cameras for venue {venue_id} …"))
    cameras: list = _api_get(api_url, f"/venues/{venue_id}/cameras")

    if not cameras:
        print(_yellow("No cameras registered for this venue."))
        print(f"\n  Register one first:  python3 register_camera.py")
        print(f"  Then retry:          python3 stream.py --venue-id {venue_id}\n")
        return

    for cam in cameras:
        cam.setdefault("venue_id", venue_id)

    _print_header(cameras, venue_name, api_url, args.interval)

    executor = ThreadPoolExecutor(
        max_workers=max(len(cameras) * 2, 4),
        thread_name_prefix="rtsp",
    )

    tasks = [
        asyncio.create_task(
            _camera_task(
                cam=cam,
                venue_id=venue_id,
                api_url=api_url,
                interval=args.interval,
                no_ingest=args.no_ingest,
                model_path=args.model,
                executor=executor,
            ),
            name=f"cam-{cam['camera_id']}",
        )
        for cam in cameras
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        executor.shutdown(wait=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="F3 standalone RTSP stream processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--venue-id", required=True, metavar="UUID",
                        help="Venue UUID to process cameras for")
    parser.add_argument("--api-url", default=API_URL, metavar="URL",
                        help=f"F3 API base URL  [default: {API_URL}]")
    parser.add_argument("--interval", type=int, default=15, metavar="SEC",
                        help="Seconds between frame captures  [default: 15]")
    parser.add_argument("--no-ingest", action="store_true",
                        help="Print and log only; do not POST to /ingest/camera")
    parser.add_argument("--model", default="yolov8n.pt", metavar="FILE",
                        help="YOLOv8 weights file  [default: yolov8n.pt]")

    args = parser.parse_args()

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print(_dim("\nStopped."))
        sys.exit(0)


if __name__ == "__main__":
    main()
