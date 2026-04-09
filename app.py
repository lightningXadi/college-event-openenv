from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from environment import Action, CollegeEventEnv, Observation, StepResult
from grader import grade_from_task_id
from tasks import TaskSpec, get_tasks, tasks_by_id
from fastapi.openapi.docs import get_swagger_ui_html


app = FastAPI(
    title="College Event Registration API",
    description="OpenEnv-compatible environment for college event management.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)

env = CollegeEventEnv()


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"

class TaskInfo(BaseModel):
    id: str
    title: str
    difficulty: str
    description: str
    max_steps: int

class ResetRequest(BaseModel):
    task_id: str

class GraderRequest(BaseModel):
    task_id: str
    events: Optional[List[Dict[str, Any]]] = None
    registrations: Optional[List[Dict[str, Any]]] = None

class GraderResponse(BaseModel):
    task_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    checks: List[Dict[str, Any]]

class BaselineRequest(BaseModel):
    task_id: Optional[str] = None

class BaselineResponse(BaseModel):
    results: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def root():
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>EventEnv — College Event Registration</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Mona+Sans:ital,wdth,wght@0,75..125,200..900;1,75..125,200..900&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<style>
/* ═══════════════════════════════════════════════
   GUMROAD-FEEL DESIGN TOKENS
   Light cream background + deep charcoal darks
   + hot pink accent
═══════════════════════════════════════════════ */
:root {
  /* Core palette */
  --cream:      #edd5b9;
  --cream-2:    #F0EAE0;
  --cream-3:    #E6DDD0;
  --dark:       #1A1A1A;
  --dark-2:     #2A2A2A;
  --dark-3:     #3D3D3D;
  --dark-card:  #222222;

  /* Accent — Gumroad-ish hot pink */
  --pink: #FF7A18;
--pink-dark: #E66500;
--pink-glow: rgba(255,122,24,0.35);


  /* Coral CTA (secondary) */
  --coral:      #FF6154;
  --coral-glow: rgba(255,97,84,0.25);

  /* Text */
  --ink:        #1A1A1A;
  --ink-mid:    #555555;
  --ink-muted:  #888888;
  --ink-light:  #AAAAAA;
  --on-dark:    #F8F4EF;
  --on-dark-mid:#C8BFB4;

  /* Borders */
  --border-light: rgba(26,26,26,0.12);
  --border-dark:  rgba(248,244,239,0.12);

  /* Utility */
  --radius:   12px;
  --radius-lg: 20px;
  --ease:      cubic-bezier(0.22, 1, 0.36, 1);
  --ease-back: cubic-bezier(0.34, 1.56, 0.64, 1);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }

body {
  font-family: 'Mona Sans', sans-serif;
  background: var(--cream);
  color: var(--ink);
  min-height: 100vh;
  overflow-x: hidden;
  -webkit-tap-highlight-color: transparent;
  cursor: none;
}

/* No-select on chrome elements */
nav, h1, h2, h3,
.hero-badge, .btn, .stat, .stat-num, .stat-label,
.fc-title, .fc-icon, .section-label, .method,
.endpoint-note, .footer-pill, .api-title,
.loader-wordmark, .loader-status {
  user-select: none;
  -webkit-user-select: none;
}

/* ═══════════════════════════════════════════════
   LOADER — Gumroad-style morphing wordmark
═══════════════════════════════════════════════ */
#loader {
  position: fixed; inset: 0; z-index: 9999;
  background: var(--dark);
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 32px;
  transition: opacity 0.7s var(--ease), visibility 0.7s;
}
#loader.hide { opacity: 0; visibility: hidden; pointer-events: none; }

/* Pink morphing blob */
.loader-blob-wrap {
  position: relative; width: 80px; height: 80px;
  opacity: 0;
  animation: fadeUp 0.5s var(--ease) 0.1s forwards;
}
.loader-blob {
  width: 80px; height: 80px;
  background: linear-gradient(135deg, var(--pink), var(--coral));
  animation: blobMorph 3s ease-in-out infinite;
  box-shadow: 0 0 40px 10px var(--pink-glow), 0 0 80px 20px rgba(255,97,84,0.15);
}
@keyframes blobMorph {
  0%   { border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%; transform: rotate(0deg) scale(1); }
  25%  { border-radius: 30% 60% 70% 40% / 50% 60% 30% 60%; transform: rotate(90deg) scale(1.06); }
  50%  { border-radius: 50% 60% 30% 60% / 40% 50% 60% 50%; transform: rotate(180deg) scale(0.96); }
  75%  { border-radius: 60% 40% 60% 30% / 60% 30% 60% 40%; transform: rotate(270deg) scale(1.04); }
  100% { border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%; transform: rotate(360deg) scale(1); }
}
/* Inner glyph */
.loader-blob-inner {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Instrument Serif', serif;
  font-size: 32px; color: var(--dark); font-style: italic;
  z-index: 1;
}

.loader-wordmark {
  font-family: 'Instrument Serif', serif;
  font-size: 20px; color: var(--on-dark);
  letter-spacing: -0.3px;
  opacity: 0;
  animation: fadeUp 0.5s var(--ease) 0.4s forwards;
}
.loader-wordmark span {
  color: var(--pink);
}

/* Progress bar */
.loader-progress {
  width: 180px; height: 3px;
  background: rgba(255,255,255,0.08);
  border-radius: 99px; overflow: hidden;
  opacity: 0;
  animation: fadeUp 0.4s var(--ease) 0.55s forwards;
}
.loader-fill {
  height: 100%; width: 0;
  background: linear-gradient(90deg, var(--pink), var(--coral));
  border-radius: 99px;
  box-shadow: 0 0 8px var(--pink-glow);
  animation: fillBar 1.8s var(--ease) 0.65s forwards;
}
@keyframes fillBar { 0%{width:0} 60%{width:72%} 100%{width:100%} }

.loader-status {
  font-size: 10px; font-weight: 600;
  letter-spacing: 2.5px; text-transform: uppercase;
  color: rgba(248,244,239,0.28);
  opacity: 0;
  animation: fadeUp 0.4s var(--ease) 0.75s forwards;
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ═══════════════════════════════════════════════
   CUSTOM CURSOR
═══════════════════════════════════════════════ */
#cursor-dot {
  position: fixed; width: 8px; height: 8px;
  background: var(--pink); border-radius: 50%;
  pointer-events: none; z-index: 9998;
  transform: translate(-50%, -50%);
  transition: width 0.15s var(--ease), height 0.15s var(--ease),
              background 0.15s, border-radius 0.15s;
  will-change: transform;
  box-shadow: 0 0 10px 2px var(--pink-glow);
}
#cursor-ring {
  position: fixed; width: 30px; height: 30px;
  border: 1.5px solid var(--pink); border-radius: 50%;
  pointer-events: none; z-index: 9997;
  transform: translate(-50%, -50%);
  transition: width 0.28s var(--ease), height 0.28s var(--ease),
              border-color 0.2s, opacity 0.25s, border-radius 0.2s;
  opacity: 0.5; will-change: transform;
}
body.cursor-hover #cursor-dot  { width: 12px; height: 12px; background: var(--coral); border-radius: 3px; }
body.cursor-hover #cursor-ring { width: 44px; height: 44px; border-color: var(--coral); opacity: 0.25; border-radius: 8px; }
.sq-particle {
  position: fixed; pointer-events: none; z-index: 9996;
  will-change: transform, opacity; border-radius: 2px;
}
@media(hover:none){ #cursor-dot, #cursor-ring, .sq-particle { display:none; } body { cursor: auto; } }

/* ═══════════════════════════════════════════════
   BACKGROUND — textured cream with subtle noise
═══════════════════════════════════════════════ */
.bg-texture {
  position: fixed; inset: 0; z-index: 0;
  pointer-events: none;
  background:
    radial-gradient(ellipse 70% 50% at 15% 0%, rgba(255,144,232,0.07) 0%, transparent 60%),
    radial-gradient(ellipse 50% 60% at 90% 100%, rgba(255,97,84,0.06) 0%, transparent 60%),
    var(--cream);
}
.bg-grain {
  position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: 0.04;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  background-size: 160px 160px;
}
/* Diagonal rule lines — very subtle */
.bg-lines {
  position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: 0.11;
  background-image: repeating-linear-gradient(
    -45deg,
    var(--ink) 0px, var(--ink) 1px,
    transparent 1px, transparent 28px
  );
}

/* ═══════════════════════════════════════════════
   LAYOUT
═══════════════════════════════════════════════ */
.page { position: relative; z-index: 1; max-width: 1080px; margin: 0 auto; padding: 0 24px 80px; }
.page-content { opacity: 0; transition: opacity 0.8s var(--ease); }
.page-content.visible { opacity: 1; }

/* ═══════════════════════════════════════════════
   SCROLL REVEAL
═══════════════════════════════════════════════ */
.reveal { opacity: 0; transform: translateY(24px); transition: opacity 0.65s var(--ease), transform 0.65s var(--ease); }
.reveal.in { opacity: 1; transform: translateY(0); }
.d0{transition-delay:0s} .d1{transition-delay:0.08s} .d2{transition-delay:0.16s}
.d3{transition-delay:0.24s} .d4{transition-delay:0.32s}

/* ═══════════════════════════════════════════════
   NAVBAR — dark pill floating
═══════════════════════════════════════════════ */
nav {
  position: sticky; top: 14px; z-index: 200;
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 18px;
  margin: 14px 0 0;
  background: var(--dark);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 50px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.22), 0 1px 0 rgba(255,255,255,0.05) inset;
  transition: box-shadow 0.3s;
}
nav.scrolled {
  box-shadow: 0 16px 48px rgba(0,0,0,0.32), 0 1px 0 rgba(255,255,255,0.05) inset;
}
.nav-logo {
  font-family: 'Instrument Serif', serif;
  font-size: 19px; letter-spacing: -0.2px;
  color: var(--on-dark);
  display: flex; align-items: center; gap: 9px;
  transition: opacity 0.2s;
}
.nav-logo:hover { opacity: 0.8; }
.nav-logo .dot {
  width: 9px; height: 9px;
  background: var(--pink); border-radius: 50%;
  box-shadow: 0 0 10px 3px var(--pink-glow);
  animation: dotPulse 2.4s ease-in-out infinite;
}
@keyframes dotPulse {
  0%,100% { transform: scale(1); opacity: 1; }
  50%     { transform: scale(0.5); opacity: 0.5; }
}
.nav-links { display: flex; align-items: center; gap: 8px; }
.nav-links a {
  font-size: 13px; font-weight: 600;
  color: rgba(248,244,239,0.72);
  text-decoration: none;
  padding: 8px 18px; border-radius: 50px;
  border: 1px solid rgba(255,255,255,0.1);
  transition: color 0.18s, background 0.18s, border-color 0.18s, transform 0.2s var(--ease-back), box-shadow 0.2s;
}
.nav-links a:hover {
  color: var(--on-dark);
  background: rgba(255,255,255,0.08);
  border-color: rgba(255,255,255,0.18);
  transform: translateY(-1px);
}
.nav-links a:active { transform: translateY(0); }
.nav-links a.filled {
  background: var(--pink); color: var(--dark);
  border-color: var(--pink);
  font-weight: 700;
  box-shadow: 0 2px 16px var(--pink-glow);
}
.nav-links a.filled:hover {
  background: var(--pink-dark); border-color: var(--pink-dark);
  box-shadow: 0 4px 24px rgba(255,144,232,0.45);
  transform: translateY(-2px) scale(1.03);
}

/* ═══════════════════════════════════════════════
   HERO
═══════════════════════════════════════════════ */
.hero {
  padding: 80px 0 60px;
  display: grid; grid-template-columns: 1fr auto;
  align-items: end; gap: 32px;
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 7px;
  font-size: 11px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1.4px;
  color: var(--ink-mid);
  border: 1.5px solid var(--border-light);
  background: rgba(26,26,26,0.05);
  border-radius: 100px; padding: 5px 14px; margin-bottom: 22px;
  transition: transform 0.2s var(--ease-back), box-shadow 0.2s, background 0.2s;
}
.hero-badge:hover { transform: scale(1.04); background: rgba(26,26,26,0.08); }
.hero-badge::before {
  content: '';  width: 6px; height: 6px;
  background: var(--pink); border-radius: 50%;
  box-shadow: 0 0 8px 2px var(--pink-glow);
  animation: dotPulse 2s ease-in-out infinite;
}
h1 {
  font-family: 'Instrument Serif', serif;
  font-size: clamp(46px, 5.8vw, 78px);
  line-height: 1.03; letter-spacing: -2px;
  color: var(--dark); max-width: 700px;
  font-weight: 400;
}
h1 em {
  font-style: italic; color: var(--pink);
  text-shadow: 0 0 60px var(--pink-glow);
}
.hero-sub {
  font-size: 16.5px; color: var(--ink-mid);
  line-height: 1.72; max-width: 520px;
  margin-top: 20px; font-weight: 400;
}
.hero-cta { display: flex; gap: 12px; margin-top: 36px; flex-wrap: wrap; }

/* ═══════════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════════ */
.btn {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 13px 26px; border-radius: 50px;
  font-family: 'Mona Sans', sans-serif;
  font-size: 14px; font-weight: 700;
  text-decoration: none; cursor: none;
  transition: transform 0.18s var(--ease-back),
              box-shadow 0.2s var(--ease),
              background 0.15s, color 0.15s, border-color 0.15s;
  position: relative; overflow: hidden; letter-spacing: -0.1px;
}
.btn::before {
  content: ''; position: absolute; inset: 0;
  background: rgba(255,255,255,0.08); opacity: 0;
  transition: opacity 0.18s;
}
.btn:hover::before { opacity: 1; }
.btn:hover { transform: translate(-2px, -3px); }
.btn:active { transform: translate(1px, 1px); }

.btn-primary {
  background: var(--dark); color: var(--on-dark);
  border: 2px solid var(--dark);
  box-shadow: 4px 4px 0 var(--dark-3), 0 4px 20px rgba(26,26,26,0.18);
}
.btn-primary:hover {
  background: var(--dark-2);
  box-shadow: 6px 7px 0 var(--dark-3), 0 8px 32px rgba(26,26,26,0.25);
}
.btn-secondary {
  background: transparent; color: var(--dark);
  border: 2px solid var(--border-light);
  box-shadow: 4px 4px 0 rgba(26,26,26,0.12);
}
.btn-secondary:hover {
  background: rgba(26,26,26,0.04);
  border-color: rgba(26,26,26,0.3);
  box-shadow: 6px 7px 0 rgba(26,26,26,0.18);
}
.btn-pink {
  background: var(--pink); color: var(--dark);
  border: 2px solid var(--pink);
  box-shadow: 4px 4px 0 var(--pink-dark), 0 4px 20px var(--pink-glow);
}
.btn-pink:hover {
  background: var(--pink-dark); border-color: var(--pink-dark);
  box-shadow: 6px 7px 0 #b040a0, 0 8px 32px rgba(255,144,232,0.45);
}

/* ═══════════════════════════════════════════════
   STATS STRIP — dark background
═══════════════════════════════════════════════ */
.stats {
  display: flex; gap: 1px;
  background: var(--border-light);
  border: 1.5px solid var(--border-light);
  border-radius: var(--radius-lg); overflow: hidden;
  margin: 0 0 72px;
}
.stat {
  flex: 1; background: var(--dark);
  padding: 28px 20px; text-align: center;
  transition: background 0.22s, transform 0.22s var(--ease-back);
  cursor: default;
}
.stat:hover { background: var(--dark-2); transform: translateY(-4px); }
.stat-num {
  font-family: 'Instrument Serif', serif;
  font-size: 32px; color: var(--on-dark);
  line-height: 1; transition: color 0.2s;
  font-style: italic;
}
.stat:hover .stat-num { color: var(--pink); }
.stat-label {
  font-size: 10.5px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1.2px;
  color: var(--on-dark-mid); margin-top: 7px;
}

/* ═══════════════════════════════════════════════
   SECTION LABEL
═══════════════════════════════════════════════ */
.section-label {
  font-size: 10.5px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1.8px;
  color: var(--ink-muted);
  margin-bottom: 28px;
  display: flex; align-items: center; gap: 12px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border-light); }

/* ═══════════════════════════════════════════════
   FEATURE CARDS — mixed light/dark
═══════════════════════════════════════════════ */
.features {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 20px; margin-bottom: 72px;
}
.feature-card {
  padding: 32px; border-radius: var(--radius-lg);
  border: 1.5px solid;
  transition: transform 0.22s var(--ease-back),
              box-shadow 0.22s var(--ease), background 0.2s;
  cursor: default;
}
/* Card 1 — light */
.feature-card:nth-child(1) {
  background: var(--cream-2);
  border-color: var(--border-light);
  box-shadow: 4px 4px 0 var(--cream-3);
}
.feature-card:nth-child(1):hover {
  transform: translate(-3px, -5px);
  box-shadow: 8px 10px 0 var(--cream-3), 0 12px 40px rgba(0,0,0,0.08);
}
/* Card 2 — dark (hero card) */
.feature-card:nth-child(2) {
  background: var(--dark);
  border-color: rgba(255,255,255,0.08);
  box-shadow: 4px 4px 0 rgba(0,0,0,0.2);
}
.feature-card:nth-child(2):hover {
  transform: translate(-3px, -5px);
  box-shadow: 8px 10px 0 rgba(0,0,0,0.5), 0 12px 40px rgba(0,0,0,0.25);
}
.feature-card:nth-child(2) .fc-title { color: var(--on-dark); }
.feature-card:nth-child(2) .fc-body  { color: var(--on-dark-mid); }
/* Card 3 — pink-tinted */

/* Card 3 — same as card 1 */
.feature-card:nth-child(3) {
  background: var(--cream-2);
  border-color: var(--border-light);
  box-shadow: 4px 4px 0 var(--cream-3);
}

.feature-card:nth-child(3):hover {
  transform: translate(-3px, -5px);
  box-shadow: 8px 10px 0 var(--cream-3), 0 12px 40px rgba(0,0,0,0.08);
}



.feature-card:active { transform: translate(0, 0); }
.fc-icon {
  font-size: 28px; margin-bottom: 18px; display: block;
  transition: transform 0.35s var(--ease-back); will-change: transform;
}
.feature-card:hover .fc-icon { transform: translateY(-5px) scale(1.14) rotate(-4deg); }
.fc-title {
  font-family: 'Instrument Serif', serif;
  font-size: 23px; color: var(--dark);
  margin-bottom: 10px; line-height: 1.18;
}
.fc-body { font-size: 14px; color: var(--ink-mid); line-height: 1.7; font-weight: 400; }

/* ═══════════════════════════════════════════════
   API SECTION — dark block
═══════════════════════════════════════════════ */
.api-section {
  background: var(--dark);
  border-radius: var(--radius-lg); padding: 48px;
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 48px; align-items: center; margin-bottom: 72px;
  border: 1.5px solid rgba(255,255,255,0.07);
  box-shadow: 0 16px 60px rgba(0,0,0,0.22), 6px 6px 0 rgba(0,0,0,0.4);
}
.api-title {
  font-family: 'Instrument Serif', serif;
  font-size: 38px; color: var(--on-dark); line-height: 1.08;
  margin-bottom: 14px; font-weight: 400;
}
.api-title em { color: var(--pink); font-style: italic; }
.api-desc {
  font-size: 14.5px; color: var(--on-dark-mid);
  line-height: 1.72; margin-bottom: 26px; font-weight: 400;
}

.endpoint-list { display: flex; flex-direction: column; gap: 8px; }
.endpoint {
  display: flex; align-items: center; gap: 12px;
  padding: 12px 16px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: var(--radius); font-size: 13px;
  transition: background 0.18s, transform 0.18s var(--ease-back),
              border-color 0.18s, box-shadow 0.18s;
  cursor: default;
}
.endpoint:hover {
  background: rgba(255,255,255,0.08);
  transform: translateX(6px);
  border-color: rgba(255,255,255,0.15);
  box-shadow: 0 4px 18px rgba(0,0,0,0.2);
}
.method {
  font-weight: 800; font-size: 10px; letter-spacing: 0.8px;
  padding: 3px 9px; border-radius: 6px; min-width: 46px; text-align: center;
}
.method.get  { background: rgba(255,144,232,0.18); color: var(--pink); }
.method.post { background: rgba(255,97,84,0.18);   color: #FF9080; }
.endpoint-path { font-family: 'Courier New', monospace; color: var(--on-dark); font-size: 13px; }
.endpoint-note { margin-left: auto; font-size: 11px; color: var(--on-dark-mid); font-weight: 500; }

.api-section .btn {
  border-color: rgba(255,255,255,0.15); color: var(--on-dark);
  background: rgba(255,255,255,0.07);
  box-shadow: 3px 3px 0 rgba(255,255,255,0.04);
}
.api-section .btn:hover {
  background: rgba(255,255,255,0.12);
  box-shadow: 5px 6px 0 rgba(255,255,255,0.06);
}

/* ═══════════════════════════════════════════════
   FOOTER
═══════════════════════════════════════════════ */
footer {
  margin-top: 60px; font-size: 14px; color: var(--ink-muted);
  display: flex; justify-content: space-between; align-items: center;
  padding-top: 24px; border-top: 1.5px solid var(--border-light);
}
footer strong { color: var(--ink); }
.footer-pill {
  background: var(--dark); color: var(--on-dark);
  padding: 7px 18px; border-radius: 100px;
  font-size: 11px; font-weight: 700; letter-spacing: 0.3px;
  transition: transform 0.2s var(--ease-back), box-shadow 0.2s;
  box-shadow: 3px 3px 0 var(--dark-3);
}
.footer-pill:hover { transform: scale(1.06) translateY(-2px); box-shadow: 5px 6px 0 var(--dark-3); }

/* ═══════════════════════════════════════════════
   RESPONSIVE
═══════════════════════════════════════════════ */
@media(max-width: 768px) {
  .hero           { grid-template-columns: 1fr; }
  .features       { grid-template-columns: 1fr; }
  .api-section    { grid-template-columns: 1fr; padding: 28px; }
  .stats          { flex-direction: column; gap: 0; }
  nav             { margin: 8px 0 0; padding: 10px 14px; top: 8px; }
  body            { cursor: auto; }
}

@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(18px); }
  to   { opacity: 1; transform: translateY(0); }
}
body { user-select: none; -webkit-user-select: none; }
p    { user-select: text; }
</style>
</head>
<body>

<!-- ══ LOADER ══ -->
<div id="loader">
  <div class="loader-blob-wrap">
    <div class="loader-blob"></div>
    <div class="loader-blob-inner">E</div>
  </div>
  <div class="loader-wordmark">Event<span>Env</span></div>
  <div class="loader-progress"><div class="loader-fill"></div></div>
  <div class="loader-status">Initialising environment</div>
</div>

<!-- ══ CURSOR ══ -->
<div id="cursor-dot"></div>
<div id="cursor-ring"></div>

<!-- ══ BACKGROUND ══ -->
<div class="bg-texture"></div>
<div class="bg-grain"></div>
<div class="bg-lines"></div>

<!-- ══ PAGE ══ -->
<div class="page">
<div class="page-content">

  <nav id="main-nav">
    <div class="nav-logo"><span class="dot"></span>EventEnv</div>
    <div class="nav-links">
      <a href="/docs">API Explorer</a>
      <a href="/tasks" class="filled">View Tasks →</a>
    </div>
  </nav>

  <section class="hero">
    <div>
      <div class="hero-badge">OpenEnv Hackathon</div>
      <h1>College Events,<br><em>Intelligently</em><br>Managed.</h1>
      <p class="hero-sub">
        An OpenEnv-compatible AI environment for event registration,
        capacity enforcement, and cancellation — built for agents
        that reason, plan, and act.
      </p>
      <div class="hero-cta">
        <a href="/docs" class="btn btn-primary">Explore API →</a>
        <a href="/tasks" class="btn btn-secondary">View Tasks</a>
      </div>
    </div>
  </section>

  <div class="stats reveal d0">
    <div class="stat"><div class="stat-num">REST</div><div class="stat-label">API Interface</div></div>
    <div class="stat"><div class="stat-num">FastAPI</div><div class="stat-label">Framework</div></div>
    <div class="stat"><div class="stat-num">Graded</div><div class="stat-label">Task System</div></div>
    <div class="stat"><div class="stat-num">AI</div><div class="stat-label">Agent Ready</div></div>
  </div>

  <div class="section-label reveal d0">What it does</div>
  <div class="features">
    <div class="feature-card reveal d0">
      <span class="fc-icon">🎓</span>
      <div class="fc-title">Event Management</div>
      <p class="fc-body">Create and manage multiple college events with configurable capacity limits and real-time availability tracking.</p>
    </div>
    <div class="feature-card reveal d1">
      <span class="fc-icon">🧠</span>
      <div class="fc-title">AI Environment</div>
      <p class="fc-body">Fully OpenEnv-compatible with reward signals, task specifications, and a grader for evaluating agent performance.</p>
    </div>
    <div class="feature-card reveal d2">
      <span class="fc-icon">📋</span>
      <div class="fc-title">Student Registration</div>
      <p class="fc-body">Register students, prevent duplicates, handle cancellations, and enforce waitlists — all through a clean HTTP API.</p>
    </div>
  </div>

  <div class="api-section reveal d0">
    <div>
      <div class="api-title">Clean API.<br><em>Zero friction.</em></div>
      <p class="api-desc">Every endpoint is designed for agents and humans alike. Reset the environment, take actions, and get scored — all over HTTP.</p>
      <a href="/docs" class="btn">Open API Explorer →</a>
    </div>
    <div class="endpoint-list">
      <div class="endpoint reveal d0"><span class="method get">GET</span><span class="endpoint-path">/tasks</span><span class="endpoint-note">List all tasks</span></div>
      <div class="endpoint reveal d1"><span class="method post">POST</span><span class="endpoint-path">/env/reset</span><span class="endpoint-note">Start episode</span></div>
      <div class="endpoint reveal d2"><span class="method get">GET</span><span class="endpoint-path">/env/state</span><span class="endpoint-note">Current obs.</span></div>
      <div class="endpoint reveal d3"><span class="method post">POST</span><span class="endpoint-path">/env/step</span><span class="endpoint-note">Take action</span></div>
      <div class="endpoint reveal d1"><span class="method post">POST</span><span class="endpoint-path">/grader</span><span class="endpoint-note">Score episode</span></div>
      <div class="endpoint reveal d2"><span class="method post">POST</span><span class="endpoint-path">/baseline</span><span class="endpoint-note">Run baseline</span></div>
    </div>
  </div>

  <footer class="reveal d0">
    <div>Built with <strong>FastAPI</strong> &amp; <strong>OpenEnv</strong></div>
    <div class="footer-pill">Hackathon Project 2025</div>
  </footer>

</div><!-- /page-content -->
</div><!-- /page -->

<script>
/* ── LOADER ── */
(function(){
  var loader  = document.getElementById('loader');
  var content = document.querySelector('.page-content');
  function hide(){
    loader.classList.add('hide');
    setTimeout(function(){
      content.classList.add('visible');
      loader.style.display = 'none';
      var heroEls = document.querySelectorAll('.hero-badge,.hero h1,.hero .hero-sub,.hero .hero-cta');
      heroEls.forEach(function(el, i){
        el.style.animation = 'fadeInUp 0.7s cubic-bezier(0.22,1,0.36,1) '+(i*0.12)+'s both';
      });
    }, 600);
  }
  if(document.readyState === 'complete') setTimeout(hide, 2200);
  else window.addEventListener('load', function(){ setTimeout(hide, 2200); });
})();

/* ── CURSOR PARTICLE ENGINE ── */
(function(){
  var dot  = document.getElementById('cursor-dot');
  var ring = document.getElementById('cursor-ring');
  if(!dot || !ring) return;

  var mx=0, my=0, rx=0, ry=0;
  var COLORS = ['#FF90E8','#FF6154','#E060C8','#FF9080'];
  var MAX_PARTICLES = 16;
  var particles = [];

  document.addEventListener('mousemove', function(e){
    mx = e.clientX; my = e.clientY;
    dot.style.left = mx+'px';
    dot.style.top  = my+'px';
    spawnParticle(mx, my);
  });

  (function loop(){
    rx += (mx-rx)*0.11;
    ry += (my-ry)*0.11;
    ring.style.left = rx+'px';
    ring.style.top  = ry+'px';
    requestAnimationFrame(loop);
  })();

  document.addEventListener('mouseover', function(e){
    var t = e.target.closest('a,button,.btn,.feature-card,.stat,.endpoint,.footer-pill');
    document.body.classList.toggle('cursor-hover', !!t);
  });

  document.addEventListener('mouseleave', function(){ dot.style.opacity='0'; ring.style.opacity='0'; });
  document.addEventListener('mouseenter', function(){ dot.style.opacity='1'; ring.style.opacity='0.5'; });

  var lastSpawn = 0;
  function spawnParticle(x, y){
    var now = Date.now();
    if(now - lastSpawn < 42) return;
    lastSpawn = now;
    if(particles.length >= MAX_PARTICLES) return;

    var size  = 4 + Math.random()*5;
    var color = COLORS[Math.floor(Math.random()*COLORS.length)];
    var vx    = (Math.random()-0.5)*2.2;
    var vy    = (Math.random()-0.5)*2.2 - 0.6;
    var rot   = Math.random()*360;
    var drot  = (Math.random()-0.5)*8;
    var maxLife = 30 + Math.floor(Math.random()*18);

    var el = document.createElement('div');
    el.className = 'sq-particle';
    el.style.cssText = [
      'width:'+size+'px','height:'+size+'px',
      'background:'+color,'border-radius:2px',
      'left:'+x+'px','top:'+y+'px','opacity:0.65'
    ].join(';');
    document.body.appendChild(el);
    particles.push({el:el, x:x, y:y, vx:vx, vy:vy, rot:rot, drot:drot, life:0, maxLife:maxLife});
  }

  (function animParticles(){
    for(var i=particles.length-1; i>=0; i--){
      var p = particles[i];
      p.life++; p.x += p.vx; p.y += p.vy; p.vy += 0.07; p.rot += p.drot;
      var progress = p.life / p.maxLife;
      p.el.style.left      = p.x+'px';
      p.el.style.top       = p.y+'px';
      p.el.style.opacity   = (1-progress)*0.6;
      p.el.style.transform = 'translate(-50%,-50%) rotate('+p.rot+'deg) scale('+(1-progress*0.6)+')';
      if(p.life >= p.maxLife){
        p.el.parentNode && p.el.parentNode.removeChild(p.el);
        particles.splice(i,1);
      }
    }
    requestAnimationFrame(animParticles);
  })();
})();

/* ── SCROLL ANIMATIONS ── */
(function(){
  var obs = new IntersectionObserver(function(entries){
    entries.forEach(function(entry){
      if(entry.isIntersecting){
        entry.target.classList.add('in');
        obs.unobserve(entry.target);
      }
    });
  },{threshold:0.1, rootMargin:'0px 0px -36px 0px'});
  document.querySelectorAll('.reveal').forEach(function(el){ obs.observe(el); });
})();

/* ── NAV SCROLL ── */
(function(){
  var nav = document.getElementById('main-nav');
  window.addEventListener('scroll', function(){
    nav.classList.toggle('scrolled', window.scrollY > 24);
  },{passive:true});
})();
</script>

</body>
</html>"""


@app.get("/tasks", response_model=List[TaskInfo])
def list_tasks() -> List[TaskInfo]:
    out: List[TaskInfo] = []
    for t in get_tasks():
        out.append(TaskInfo(
            id=t.id, title=t.title, difficulty=t.difficulty,
            description=t.description, max_steps=t.max_steps,
        ))
    return out


@app.post("/env/reset", response_model=Observation)
def env_reset(req: ResetRequest) -> Observation:
    try:
        return env.reset(req.task_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Unknown task_id")


@app.get("/env/state", response_model=Observation)
def env_state() -> Observation:
    return env.state()


@app.post("/env/step", response_model=StepResult)
def env_step(action: Action) -> StepResult:
    return env.step(action)


@app.post("/grader", response_model=GraderResponse)
def grader(req: GraderRequest) -> GraderResponse:
    tasks: Dict[str, TaskSpec] = tasks_by_id()
    if req.task_id not in tasks:
        raise HTTPException(status_code=404, detail="Unknown task_id")
    if req.events is None or req.registrations is None:
        from environment import _read_json
        events = _read_json("events.json")
        registrations = _read_json("registrations.json")
    else:
        events = req.events
        registrations = req.registrations
    try:
        result = grade_from_task_id(req.task_id, events=events, registrations=registrations)
    except ValueError:
        raise HTTPException(status_code=404, detail="Unknown task_id")
    return GraderResponse(task_id=req.task_id, score=float(result["score"]), checks=result["checks"])


@app.post("/baseline", response_model=BaselineResponse)
def baseline(req: BaselineRequest) -> BaselineResponse:
    """Runs a simple, deterministic baseline (no external model calls)."""
    from baseline import run_baseline_episode
    tasks = get_tasks()
    if req.task_id is not None:
        by = tasks_by_id()
        if req.task_id not in by:
            raise HTTPException(status_code=404, detail="Unknown task_id")
        tasks = [by[req.task_id]]
    results: List[Dict[str, Any]] = []
    for t in tasks:
        results.append(run_baseline_episode(env, t.id))
    return BaselineResponse(results=results)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


from fastapi.responses import HTMLResponse  # noqa: F811

@app.get("/docs", include_in_schema=False)
async def custom_docs():
    swagger = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="College Event API Explorer"
    ).body.decode()
    return HTMLResponse(f"""<!DOCTYPE html>
<html><head>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Mona+Sans:wdth,wght@75..125,200..900&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<style>
  body{{margin:0;font-family:'Mona Sans',sans-serif;background:#F8F4EF;}}
  .banner{{
    position:sticky;top:0;z-index:999;
    background:rgba(26,26,26,0.97);
    backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
    padding:12px 24px;display:flex;align-items:center;gap:14px;
    border-bottom:2px solid #FF90E8;
    box-shadow:0 4px 24px rgba(0,0,0,0.3);
  }}
  .banner-logo{{
    font-family:'Instrument Serif',serif;font-size:18px;
    color:#F8F4EF;display:flex;align-items:center;gap:8px;
  }}
  .banner-dot{{width:8px;height:8px;background:#FF90E8;border-radius:50%;display:inline-block;box-shadow:0 0 8px rgba(255,144,232,0.6);}}
  .banner a{{
    color:#FF90E8;text-decoration:none;font-size:12.5px;font-weight:600;
    padding:6px 16px;border:1px solid rgba(255,144,232,0.3);border-radius:50px;
    transition:background 0.2s;
  }}
  .banner a:hover{{background:rgba(255,144,232,0.1);}}
  .banner-tag{{
    margin-left:auto;font-size:10px;font-weight:700;text-transform:uppercase;
    letter-spacing:1.2px;color:#888;background:rgba(255,255,255,0.05);
    padding:4px 12px;border-radius:100px;border:1px solid rgba(255,255,255,0.1);
  }}
</style>
</head><body>
  <div class="banner">
    <span class="banner-logo"><span class="banner-dot"></span>EventEnv</span>
    API Explorer
    <a href="/">← Back to Home</a>
    <span class="banner-tag">OpenEnv Hackathon</span>
  </div>
  {swagger}
</body></html>""")
