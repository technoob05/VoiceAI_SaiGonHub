"""
Native desktop wrapper for the VoiceAI SaiGonHub Edge S2ST Gradio demo.

Runs the exact same 100%-on-device speech-to-speech translation UI from
``web_demo_edge.py``, but presents it inside a native OS window (pywebview) with no
browser chrome — a one-click "desktop app" feel for demos, closer to my-translator.

What it does, in order:
  1. Force UTF-8 stdout (Vietnamese transcripts print cleanly on Windows cp1252).
  2. ``warmup()`` — load ASR + MT + TTS BEFORE the window opens (~60-90 s, one time).
  3. Launch the Gradio Blocks NON-blocking (prevent_thread_lock=True) on a free local
     port in 7870..7875 (avoids clashing with the dev web demo on 7862).
  4. Open a native pywebview window pointing at that local URL.
  5. When the window closes, exit the whole process cleanly (os._exit(0)).

Run:
    python desktop_app.py
    # or double-click run_edge_app.bat

Test hook (headless / CI): set env EDGE_APP_NO_WINDOW=1 to SKIP the pywebview window.
The server still launches; the URL is printed; the process sleeps briefly then exits so
you can curl http://127.0.0.1:<port>/ and assert HTTP 200 without a GUI blocking.
"""
from __future__ import annotations

import os
import sys
import socket
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Force UTF-8 stdout/stderr (Windows cp1252) so Vietnamese text prints cleanly.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from web_demo_edge import build_ui, warmup  # noqa: E402

WINDOW_TITLE = "VoiceAI SaiGonHub — Edge S2ST"
PORT_CANDIDATES = (7870, 7871, 7872, 7873, 7874, 7875)


def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """True if we can bind (host, port) right now — i.e. nothing else is listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port() -> int:
    """First free port in PORT_CANDIDATES (7870..7875). Falls back to the last one."""
    for p in PORT_CANDIDATES:
        if _port_is_free(p):
            return p
    print("[desktop] WARNING: all preferred ports busy; forcing last candidate.", flush=True)
    return PORT_CANDIDATES[-1]


def main() -> None:
    no_window = os.environ.get("EDGE_APP_NO_WINDOW") == "1"

    # 1) Load all models before we accept any request / open any window.
    warmup()

    # 2) Pick a free local port (avoid the dev web demo on 7862).
    port = _pick_port()
    url = f"http://127.0.0.1:{port}"

    # 3) Launch Gradio NON-blocking — prevent_thread_lock returns immediately and
    #    serves in a background thread, so we can drive the desktop window afterwards.
    print(f"[desktop] launching Gradio (non-blocking) on {url} …", flush=True)
    demo = build_ui().queue(default_concurrency_limit=1)
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=True,
        inbrowser=False,
        show_error=True,
    )
    print(f"[desktop] server up at {url}", flush=True)

    # --- headless test hook -----------------------------------------------------------
    if no_window:
        print(f"[desktop] EDGE_APP_NO_WINDOW=1 — skipping pywebview window.", flush=True)
        print(f"[desktop] URL ready for probing: {url}", flush=True)
        time.sleep(5)
        print("[desktop] no-window run complete; exiting.", flush=True)
        os._exit(0)

    # 4) Open the native desktop window (blocks until the user closes it).
    import webview  # imported lazily so the headless path never needs a GUI backend

    print(f"[desktop] opening native window → {url}", flush=True)
    webview.create_window(WINDOW_TITLE, url, width=1200, height=850)
    webview.start()

    # 5) Window closed → tear the whole process (and its Gradio thread) down cleanly.
    print("[desktop] window closed — shutting down.", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
