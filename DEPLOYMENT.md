# F3 + FanFlow Deployment Guide

## Backend (F3 API) — Deploy First

Both frontends depend on this. Current state: runs locally on `localhost:8000`.

1. **Containerize** — add a `Dockerfile` (Python 3.12-slim, `pip install -r requirements.txt`, `uvicorn main:app`)
2. **Host** — cheapest/fastest options:
   - [Railway](https://railway.app) or [Render](https://render.com) — push repo, auto-deploy, free tier available
   - [Fly.io](https://fly.io) — `fly launch`, good for always-on
3. **Persistence** — data is in-memory and resets on restart. For production add SQLite at minimum, or swap to PostgreSQL
4. **Env vars** — set `JWT_SECRET`, `ENVIRONMENT=production`, `DEBUG=false` in host dashboard

---

## Option A: Mobile App (iOS — FanFlow)

**What's done:** All 12 Swift files written, connects to `localhost:8000`.

**Remaining steps:**
1. Open the Xcode project, add all Swift files to the target
2. Change `base` URL in `FanFlow/Networking/APIClient.swift` from `localhost:8000` to your deployed backend URL
3. Remove `NSAllowsLocalNetworking` from Info.plist (not needed once backend is on HTTPS)
4. Add an App Icon and launch screen
5. **TestFlight** → requires Apple Developer Program ($99/yr), then Archive → distribute
6. **App Store** → add metadata, screenshots, submit for review (~1–3 days)

---

## Option B: Web Frontend

### B1. Standalone (React/Next.js)
1. `npx create-next-app fanflow-web`
2. Mirror the same views: Overview, Map, Concessions, Navigate, Alerts
3. Use `fetch` against the deployed F3 API
4. Deploy to Vercel (free, `git push` auto-deploys)

### B2. Embedded in FastAPI (single deploy)
1. Add a `static/` folder and a single HTML/JS file
2. Serve via `app.mount("/", StaticFiles(directory="static"))` in `main.py`
3. Entire app ships as one container

---

## Recommended Order

```
1. Deploy backend (Railway/Render)  →  unblocks everything
2. Update iOS app URL → TestFlight beta
3. Ship web frontend to Vercel       →  fastest path to shareable link
4. Add persistence (SQLite → Postgres) once you have real traffic
```
