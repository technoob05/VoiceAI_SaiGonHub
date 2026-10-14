# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for the VoiceAI SaiGonHub Edge S2ST DESKTOP app.
#
#   Build:   pyinstaller edge_app.spec
#   Output:  dist/VoiceAI-Edge-S2ST/VoiceAI-Edge-S2ST.exe   (windowed, no console)
#
# ---------------------------------------------------------------------------------------
# IMPORTANT — read before building:
#
#   * This is ONEDIR (a folder), NOT onefile. Onedir is strongly preferred here because
#     gradio/sherpa-onnx/ctranslate2 ship many native .dll/.so + data files that onefile
#     unpacks to a temp dir on every launch (slow, and breaks some native loaders).
#
#   * MODELS ARE NOT BUNDLED. The ASR (PhoWhisper), MT (NLLB-200-distilled), Piper/MMS TTS,
#     GTCRN denoise and Silero VAD weights under ``data/raw/`` (and the HF cache) are large
#     (hundreds of MB to several GB). Bundling them would make the build enormous. The
#     recommended pattern is FIRST-RUN DOWNLOAD: ship the app WITHOUT weights and let the
#     pipeline fetch/cache them on first launch (the pipeline already resolves models from
#     data/raw/ or the HF cache). If you want a fully offline installer instead, add the
#     model folders to ``datas`` below (see the commented example) and accept the size.
#
#   * This spec is authored best-effort and has NOT been built/verified in this environment
#     (running pyinstaller is heavy). Treat hidden imports / collected packages as a strong
#     starting point; a first real build may surface a few more missing submodules or data
#     files (typical for gradio + native-ML wheels).
#
#   * gradio needs its bundled frontend + template metadata. We collect them via
#     collect_data_files('gradio') / collect_data_files('gradio_client') and
#     copy_metadata('gradio') so ``importlib.metadata.version('gradio')`` works at runtime.
# ---------------------------------------------------------------------------------------

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

block_cipher = None

# --- data files ------------------------------------------------------------------------
datas = []
datas += collect_data_files("gradio")
datas += collect_data_files("gradio_client")
datas += collect_data_files("safehttpx")          # gradio runtime dep, ships data
datas += collect_data_files("groovy")             # gradio runtime dep, ships data
datas += collect_data_files("sherpa_onnx")
datas += collect_data_files("faster_whisper")     # tokenizer/assets
datas += collect_data_files("webview")            # pywebview JS/HTML bridge assets

# gradio (and friends) read their own installed version via importlib.metadata.
datas += copy_metadata("gradio")
datas += copy_metadata("gradio_client")
datas += copy_metadata("safehttpx")
datas += copy_metadata("groovy")

# OPTIONAL — bundle model weights for a fully offline installer (large!). Uncomment and
# point at the real folders to embed them next to the exe:
#   datas += [("data/raw", "data/raw")]
#   datas += [("../data/raw", "data/raw")]

# --- native libs -----------------------------------------------------------------------
binaries = []
binaries += collect_dynamic_libs("sherpa_onnx")
binaries += collect_dynamic_libs("ctranslate2")
binaries += collect_dynamic_libs("onnxruntime")
binaries += collect_dynamic_libs("soundfile")     # bundled libsndfile

# --- hidden imports --------------------------------------------------------------------
hiddenimports = []
hiddenimports += collect_submodules("gradio")
hiddenimports += collect_submodules("gradio_client")
hiddenimports += collect_submodules("sherpa_onnx")
hiddenimports += collect_submodules("faster_whisper")
hiddenimports += collect_submodules("ctranslate2")
hiddenimports += collect_submodules("webview")
hiddenimports += [
    "gradio",
    "gradio_client",
    "sherpa_onnx",
    "faster_whisper",
    "ctranslate2",
    "soundfile",
    "numpy",
    "webview",
    # pywebview auto-selects a GUI backend at runtime; on Windows it is EdgeChromium
    # (WebView2) via pythonnet/clr — pull the likely backends so it resolves frozen.
    "webview.platforms.winforms",
    "webview.platforms.edgechromium",
    "clr",
    # web_demo_edge imports these at module load:
    "web_demo_edge",
    "pipeline_fast",
]


a = Analysis(
    ["desktop_app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VoiceAI-Edge-S2ST",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # windowed (no console) — native app feel
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="edge_app.ico",  # optional: drop a .ico here for a branded taskbar icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="VoiceAI-Edge-S2ST",
)
