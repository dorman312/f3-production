#!/usr/bin/env python3
"""
stream.py — Standalone RTSP frame processor for Fan Flow & Fusion.

Fetches all cameras registered for a venue, opens their RTSP streams,
and continuously analyzes frames with YOLOv8 person detection, printing
live results to the console and posting readings to the F3 API.

Usage:
    python3 stream.py --venue-id <VENUE_ID>
    python3 stream.py --venue-id <VENUE_ID> --api-url http://localhost:8000
    python3 stream.py --venue-id <VENUE_ID> --interval 15 --no-ingest

Options:
    --venue-id   (required) UUID of the venue whose cameras to process
    --api-url    Base URL of the F3 API  [default: http://localhost:8000]
    --interval   Seconds between frame captures per camera  [default: 30]
    --no-ingest  Print results only; do not POST to /ingest/camera
    --model      YOLOv8 model weights file  [default: yolov8n.pt]
    --verbose    Show per-frame bounding-box details
"""

import argparse
import asyncio
import json
import logging
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

# ── Optional ANSI colour codes ────────────────────────────────────
_USE_COLOUR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text

def _green(t):  return _c("32", t)
def _yellow(t): return _c("33", t)
def _red(t):    return _c("31", t)
def _bold(t):   return _c("1",  t)
def _dim(t):    return _c("2",  t)
def _cyan(t):   return _c("36", t)


logging.basicConfig(
    level=logging.WARNING,          # silence third-party noise
    format="%(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger("f3.stream")


# ── API helpers ───────────────────────────────────────────────────

def api_get(base_url: str, path: str, timeout: int = 10) -> dict | list:
    url = base_url.rstrip("/") + path
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"API error {exc.code} for {url}: {exc.reason}") from exc
    except Exception as exc:
        raise SystemExit(f"Cannot reach API at {url}: {exc}") from exc


def api_post(base_url: str, path: str, payload: dict, timeout: int = 5) -> int:
    """Returns HTTP status code. Logs failures but does not raise."""
    url  = base_url.rstrip("/") + path
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        logger.warning("POST %s → HTTP %d %s", path, exc.code, exc.reason)
        return exc.code
    except Exception as exc:
        logger.warning("POST %s failed: %s", path, exc)
        return 0


# ── YOLO analyzer (local import so users without ultralytics still see errors) ─

def _load_analyzer(model_path: str):
    try:
        from ml.crowd_analyzer import CrowdDensityAnalyzer
        return CrowdDensityAnalyzer(model_path)
    except ImportError:
        raise SystemExit(
            "Cannot import CrowdDensityAnalyzer. "
            "Run this script from the f3-production directory."
        )


# ── Console output ─────────────────────────────────────────────────

def _risk_colour(density: float) -> callable:
    if density >= 70: return _red
    if density >= 40: return _yellow
    return _green


def _print_header(cameras: list[dict]) -> None:
    print()
    print(_bold("Fan Flow & Fusion — Live RTSP Frame Processor"))
    print(_dim(f"  Cameras: {len(cameras)}   Press Ctrl-C to stop"))
    print(_dim("─" * 70))
    print()


def _print_result(
    *,
    camera_id: str,
    zone_name: str,
    result: dict,
    ts: datetime,
    posted: bool,
    verbose: bool,
) -> None:
    density  = result["density_score"]
    colour   = _risk_colour(density)
    wait_str = ""

    print(
        f"  {_dim(ts.strftime('%H:%M:%S'))}  "
        f"{_cyan(camera_id[:22]): <24}  "
        f"zone={zone_name[:20]: <22}  "
        f"people={_bold(str(result['person_count'])):>5}  "
        f"density={colour(f'{density:5.1f}%')}  "
        f"conf={result['avg_confidence']:.2f}  "
        f"{'✓ posted' if posted else _dim('(no ingest)')}"
    )

    if verbose and result["bounding_box_count"]:
        print(
            _dim(f"         └ boxes={result['bounding_box_count']}  "
                 f"coverage={result['zone_coverage_pct']:.1f}%  "
                 f"wait≈{result['density_score'] * 0.12:.1f}min")
        )


def _print_camera_event(camera_id: str, msg: str, level: str = "info") -> None:
    icon = {"info": "●", "warn": "⚠", "error": "✗", "ok": "✓"}.get(level, "·")
    col  = {"warn": _yellow, "error": _red, "ok": _green}.get(level, _dim)
    print(f"  {col(icon)} {_cyan(camera_id[:22])}: {msg}")


# ── Per-camera async task ─────────────────────────────────────────

async def _camera_task(
    *,
    cam: dict,
    api_url: str,
    interval: int,
    no_ingest: bool,
    model_path: str,
    verbose: bool,
    executor: ThreadPoolExecutor,
) -> None:
    """
    Runs the full lifecycle for one camera:
      connect → read frame → analyze → (post) → wait → repeat
    Reconnects with exponential back-off on failure.
    """
    camera_id = cam["camera_id"]
    rtsp_url  = cam["rtsp_url"]
    zone_id   = cam.get("zone_id", "")
    zone_name = cam.get("zone_name") or cam.get("name") or zone_id[:12]
    venue_id  = cam.get("venue_id", "")

    analyzer = _load_analyzer(model_path)
    loop     = asyncio.get_running_loop()
    backoff  = 5

    while True:
        # ── Connect ────────────────────────────────────────────
        _print_camera_event(camera_id, f"connecting → {rtsp_url}", "info")

        def _open_cap():
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        cap: cv2.VideoCapture = await loop.run_in_executor(executor, _open_cap)
        is_open = await loop.run_in_executor(executor, cap.isOpened)

        if not is_open:
            await loop.run_in_executor(executor, cap.release)
            _print_camera_event(
                camera_id,
                f"could not open stream — retry in {backoff}s",
                "warn",
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 300)
            continue

        backoff = 5
        _print_camera_event(camera_id, "connected ✓", "ok")

        # ── Read loop ──────────────────────────────────────────
        try:
            while True:
                ret, frame = await loop.run_in_executor(executor, cap.read)
                if not ret or frame is None:
                    _print_camera_event(camera_id, "stream ended — reconnecting", "warn")
                    break

                result: dict = await loop.run_in_executor(
                    executor, analyzer.analyze_frame, frame
                )

                now    = datetime.now(timezone.utc)
                posted = False

                if not no_ingest and venue_id and zone_id:
                    status = await loop.run_in_executor(
                        executor,
                        lambda: api_post(api_url, "/ingest/camera", {
                            "venue_id":     venue_id,
                            "zone_id":      zone_id,
                            "person_count": result["person_count"],
                            "timestamp":    now.isoformat(),
                            "camera_id":    camera_id,
                        }),
                    )
                    posted = status == 201

                _print_result(
                    camera_id=camera_id,
                    zone_name=zone_name,
                    result=result,
                    ts=now,
                    posted=posted,
                    verbose=verbose,
                )

                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            await loop.run_in_executor(executor, cap.release)
            raise
        except Exception as exc:
            logger.warning("camera %s error in read loop: %s", camera_id, exc)
        finally:
            await loop.run_in_executor(executor, cap.release)

        # Back-off before reconnect
        _print_camera_event(camera_id, f"reconnecting in {backoff}s", "warn")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 300)


# ── Main ──────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    api_url  = args.api_url.rstrip("/")
    venue_id = args.venue_id

    # ── Fetch venue info ───────────────────────────────────────
    print(_dim(f"Fetching cameras for venue {venue_id} from {api_url} …"))
    cameras: list[dict] = api_get(api_url, f"/venues/{venue_id}/cameras")

    if not cameras:
        print(_yellow("No cameras registered for this venue."))
        print()
        print("Register a camera first:")
        print(f"  POST {api_url}/venues/{venue_id}/cameras")
        print("  Body: { camera_id, rtsp_url, zone_id, name }")
        print()
        raise SystemExit(0)

    # Attach venue_id to each camera record (API omits it in the list)
    for cam in cameras:
        cam.setdefault("venue_id", venue_id)

    _print_header(cameras)

    executor = ThreadPoolExecutor(
        max_workers=max(len(cameras) * 2, 4),
        thread_name_prefix="rtsp",
    )

    tasks = [
        asyncio.create_task(
            _camera_task(
                cam=cam,
                api_url=api_url,
                interval=args.interval,
                no_ingest=args.no_ingest,
                model_path=args.model,
                verbose=args.verbose,
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
        epilog=__doc__,
    )
    parser.add_argument(
        "--venue-id", required=True, metavar="UUID",
        help="Venue UUID to process cameras for",
    )
    parser.add_argument(
        "--api-url", default="http://localhost:8000", metavar="URL",
        help="F3 API base URL  [default: http://localhost:8000]",
    )
    parser.add_argument(
        "--interval", type=int, default=30, metavar="SEC",
        help="Seconds between frame captures  [default: 30]",
    )
    parser.add_argument(
        "--no-ingest", action="store_true",
        help="Do not POST results to /ingest/camera (print only)",
    )
    parser.add_argument(
        "--model", default="yolov8n.pt", metavar="FILE",
        help="YOLOv8 weights file  [default: yolov8n.pt]",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show bounding-box and coverage details for each frame",
    )

    args = parser.parse_args()

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print()
        print(_dim("Stopped."))
        sys.exit(0)


if __name__ == "__main__":
    main()
