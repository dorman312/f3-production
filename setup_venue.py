#!/usr/bin/env python3
"""
setup_venue.py — Interactive venue and zone setup for Fan Flow & Fusion.

Creates a venue and all its zones via the F3 API, then prints all
generated IDs so you can copy them into register_camera.py or the dashboard.

Usage:
    python3 setup_venue.py
    python3 setup_venue.py --api-url https://web-production-34b8b.up.railway.app
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

API_URL = "https://web-production-34b8b.up.railway.app"

ZONE_TYPES = ["gate", "concession", "restroom", "exit", "corridor", "general"]


# ── Helpers ───────────────────────────────────────────────────────

def _post(api_url: str, path: str, body: dict) -> dict:
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
        body_text = exc.read().decode(errors="replace")
        raise SystemExit(f"\nAPI error {exc.code}: {body_text}") from exc
    except Exception as exc:
        raise SystemExit(f"\nCannot reach API at {api_url}: {exc}") from exc


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"  {prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)
    return value if value else default


def _ask_float(prompt: str, default: float) -> float:
    while True:
        raw = _ask(prompt, str(default))
        try:
            return float(raw)
        except ValueError:
            print(f"    Please enter a number.")


def _ask_int(prompt: str, default: int) -> int:
    while True:
        raw = _ask(prompt, str(default))
        try:
            return int(raw)
        except ValueError:
            print(f"    Please enter a whole number.")


def _ask_zone_type() -> str:
    types_str = ", ".join(ZONE_TYPES)
    while True:
        val = _ask(f"Zone type ({types_str})", "general").lower()
        if val in ZONE_TYPES:
            return val
        print(f"    Choose one of: {types_str}")


def _hr(char: str = "─", width: int = 60) -> None:
    print(char * width)


# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="F3 interactive venue setup")
    parser.add_argument("--api-url", default=API_URL, metavar="URL",
                        help=f"F3 API base URL  [default: {API_URL}]")
    args = parser.parse_args()
    api  = args.api_url.rstrip("/")

    print()
    print("Fan Flow & Fusion — Venue Setup")
    _hr("═")
    print(f"  API: {api}")
    _hr()

    # ── Venue ──────────────────────────────────────────────────────
    print("\n  VENUE DETAILS\n")
    venue_name    = _ask("Venue name", "Lions Field Waconia")
    venue_type    = _ask("Venue type (stadium/park/arena/mall/etc.)", "park")
    venue_address = _ask("Address", "Waconia, MN")

    print("\n  Creating venue…", end=" ", flush=True)
    venue = _post(api, "/venues", {
        "name":    venue_name,
        "type":    venue_type,
        "address": venue_address,
    })
    venue_id = venue["id"]
    print(f"done  id={venue_id}")

    # ── Zones ──────────────────────────────────────────────────────
    print()
    _hr()
    num_zones = _ask_int("\n  How many zones does this venue have?", 1)

    created_zones = []
    for i in range(num_zones):
        print()
        _hr("·")
        print(f"  ZONE {i + 1} of {num_zones}\n")

        name = _ask("Zone name", f"Zone {i + 1}")
        ztype = _ask_zone_type()

        print("  Map position (0.0–1.0 as fraction of map width/height):")
        x_pct = _ask_float("    Left edge  (x_pct)", round(0.1 + (i % 3) * 0.3, 2))
        y_pct = _ask_float("    Top edge   (y_pct)", round(0.1 + (i // 3) * 0.3, 2))
        w_pct = _ask_float("    Width      (w_pct)", 0.20)
        h_pct = _ask_float("    Height     (h_pct)", 0.12)

        area_m2      = _ask_float("  Floor area (m²)", 100.0)
        max_capacity = _ask_int("  Max capacity (people)", 200)

        print(f"  Creating zone '{name}'…", end=" ", flush=True)
        zone = _post(api, f"/venues/{venue_id}/zones", {
            "name":         name,
            "type":         ztype,
            "x_pct":        x_pct,
            "y_pct":        y_pct,
            "w_pct":        w_pct,
            "h_pct":        h_pct,
            "area_m2":      area_m2,
            "max_capacity": max_capacity,
        })
        print(f"done  id={zone['id']}")
        created_zones.append(zone)

    # ── Summary ────────────────────────────────────────────────────
    print()
    _hr("═")
    print("  SETUP COMPLETE\n")
    print(f"  Venue:    {venue_name}")
    print(f"  Venue ID: {venue_id}")
    print()
    print(f"  {'Zone Name':<28}  {'Type':<12}  Zone ID")
    _hr()
    for z in created_zones:
        print(f"  {z['name']:<28}  {z['type']:<12}  {z['id']}")
    print()
    print("  Next steps:")
    print("    python3 register_camera.py   ← attach cameras to these zones")
    print("    python3 stream.py --venue-id", venue_id)
    _hr("═")
    print()


if __name__ == "__main__":
    main()
