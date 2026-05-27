#!/usr/bin/env python3
"""
register_camera.py — Register a real RTSP camera with Fan Flow & Fusion.

Builds the RTSP URL from camera credentials, tests the live connection
with OpenCV, then registers the camera via the F3 API.

Usage:
    python3 register_camera.py
    python3 register_camera.py --api-url https://web-production-34b8b.up.railway.app
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

API_URL = "https://web-production-34b8b.up.railway.app"


# ── Helpers ───────────────────────────────────────────────────────

def _ask(prompt: str, default: str = "", secret: bool = False) -> str:
    suffix = f" [{default}]" if default and not secret else ""
    try:
        if secret:
            import getpass
            value = getpass.getpass(f"  {prompt}: ").strip()
        else:
            value = input(f"  {prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)
    return value if value else default


def _hr(char: str = "─", width: int = 60) -> None:
    print(char * width)


def _api_get(api_url: str, path: str) -> list | dict:
    req = urllib.request.Request(
        api_url.rstrip("/") + path,
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"\nAPI error {exc.code} for {path}") from exc
    except Exception as exc:
        raise SystemExit(f"\nCannot reach API at {api_url}: {exc}") from exc


def _api_post(api_url: str, path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        api_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise SystemExit(f"\nAPI error {exc.code}: {detail}") from exc
    except Exception as exc:
        raise SystemExit(f"\nCannot reach API at {api_url}: {exc}") from exc


def _test_rtsp(rtsp_url: str) -> bool:
    """
    Try to open the RTSP stream with OpenCV and read one frame.
    Returns True on success, False on failure.
    Masks credentials in any error messages printed to the console.
    """
    try:
        import cv2
    except ImportError:
        print("  opencv-python-headless not installed — skipping connection test")
        return True

    # Mask credentials for safe logging: rtsp://user:***@host/...
    safe_url = rtsp_url
    try:
        from urllib.parse import urlparse, urlunparse
        p = urlparse(rtsp_url)
        if p.password:
            masked = p._replace(netloc=f"{p.username}:***@{p.hostname}:{p.port or 554}")
            safe_url = urlunparse(masked)
    except Exception:
        pass

    print(f"\n  Testing connection to {safe_url} …", flush=True)
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        cap.release()
        print("  ✗  Could not open stream — check IP, port, credentials, and network.")
        return False

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("  ✗  Stream opened but no frame received — camera may be rebooting.")
        return False

    h, w = frame.shape[:2]
    print(f"  ✓  Connection OK — frame size {w}×{h}")
    return True


# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="F3 camera registration")
    parser.add_argument("--api-url", default=API_URL, metavar="URL",
                        help=f"F3 API base URL  [default: {API_URL}]")
    args = parser.parse_args()
    api  = args.api_url.rstrip("/")

    print()
    print("Fan Flow & Fusion — Camera Registration")
    _hr("═")
    print(f"  API: {api}")
    _hr()

    # ── Pick venue ─────────────────────────────────────────────────
    print("\n  Fetching venues…", end=" ", flush=True)
    venues = _api_get(api, "/venues")
    if not venues:
        raise SystemExit("\nNo venues found. Run setup_venue.py first.")
    print(f"{len(venues)} found\n")

    for i, v in enumerate(venues):
        print(f"  [{i+1}]  {v['name']:<30}  {v['id']}")
    print()

    venue_id = _ask("Venue ID (paste from list above)")
    if not venue_id:
        raise SystemExit("Venue ID is required.")
    if venue_id not in {v["id"] for v in venues}:
        raise SystemExit(f"Venue ID not found: {venue_id}")

    # ── Pick zone ──────────────────────────────────────────────────
    print("\n  Fetching zones…", end=" ", flush=True)
    zones = _api_get(api, f"/venues/{venue_id}/zones")
    if not zones:
        raise SystemExit("\nNo zones found for this venue. Run setup_venue.py first.")
    print(f"{len(zones)} found\n")

    for z in zones:
        print(f"  {z['name']:<28}  {z['type']:<12}  {z['id']}")
    print()

    zone_id = _ask("Zone ID (paste from list above)")
    if not zone_id:
        raise SystemExit("Zone ID is required.")
    if zone_id not in {z["id"] for z in zones}:
        raise SystemExit(f"Zone ID not found: {zone_id}")

    # ── Camera credentials ─────────────────────────────────────────
    _hr()
    print("\n  CAMERA CREDENTIALS\n")
    camera_id = _ask("Camera ID (unique name, e.g. cam-entrance-01)", "cam-01")
    camera_name = _ask("Camera display name (e.g. Main Entrance Camera)", camera_id)
    cam_ip    = _ask("Camera IP address (e.g. 192.168.1.100)")
    if not cam_ip:
        raise SystemExit("Camera IP is required.")

    cam_port  = _ask("RTSP port", "554")
    cam_user  = _ask("Username", "admin")
    cam_pass  = _ask("Password", secret=True)
    cam_path  = _ask("RTSP path", "/h264Preview_01_main")

    rtsp_url = f"rtsp://{cam_user}:{cam_pass}@{cam_ip}:{cam_port}{cam_path}"

    # ── Test connection ────────────────────────────────────────────
    ok = _test_rtsp(rtsp_url)
    if not ok:
        print()
        proceed = _ask("Connection failed. Register anyway? (yes/no)", "no").lower()
        if proceed not in ("yes", "y"):
            print("Aborted.")
            sys.exit(0)

    # ── Register via API ───────────────────────────────────────────
    print(f"\n  Registering camera '{camera_id}'…", end=" ", flush=True)
    result = _api_post(api, f"/venues/{venue_id}/cameras", {
        "camera_id":    camera_id,
        "rtsp_url":     rtsp_url,
        "zone_id":      zone_id,
        "name":         camera_name,
        "coverage_pct": 100.0,
    })
    print("done")

    # ── Summary ────────────────────────────────────────────────────
    print()
    _hr("═")
    print("  CAMERA REGISTERED\n")
    print(f"  Camera ID:  {camera_id}")
    print(f"  Name:       {camera_name}")
    print(f"  Venue ID:   {venue_id}")
    print(f"  Zone ID:    {zone_id}")
    print(f"  Status:     {result.get('status', 'connected')}")
    print()
    print("  Start processing frames:")
    print(f"    python3 stream.py --venue-id {venue_id}")
    _hr("═")
    print()


if __name__ == "__main__":
    main()
