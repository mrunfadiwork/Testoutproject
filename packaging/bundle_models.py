"""
build/bundle_models.py
======================
Downloads InsightFace buffalo_sc models (if not already present) and copies
them into project/insightface_models/ so PyInstaller can bundle them with the .exe.

Run this ONCE before running PyInstaller:
    python build/bundle_models.py
"""
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Destination inside the project (PyInstaller will pick this up)
DST_ROOT = os.path.join(PROJECT_ROOT, "insightface_models", "models", "buffalo_sc")


def download_and_bundle():
    print("Step 1: Ensuring InsightFace buffalo_sc models are downloaded...")

    # Setting CPUExecutionProvider ensures this works in CI without GPU
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("  Models downloaded/verified.")

    # Locate the downloaded models
    from insightface.utils.storage import BASE_REPO_URL  # noqa
    import insightface
    # InsightFace stores models under INSIGHTFACE_HOME (default: ~/.insightface)
    insightface_home = os.environ.get(
        "INSIGHTFACE_HOME",
        os.path.join(os.path.expanduser("~"), ".insightface")
    )
    src = os.path.join(insightface_home, "models", "buffalo_sc")

    if not os.path.isdir(src):
        print(f"ERROR: Model directory not found at {src}")
        sys.exit(1)

    print(f"\nStep 2: Copying models from:\n  {src}\n  -> {DST_ROOT}")
    os.makedirs(DST_ROOT, exist_ok=True)

    copied = 0
    for fname in os.listdir(src):
        s = os.path.join(src, fname)
        d = os.path.join(DST_ROOT, fname)
        shutil.copy2(s, d)
        size_kb = os.path.getsize(d) // 1024
        print(f"  Copied: {fname}  ({size_kb} KB)")
        copied += 1

    print(f"\nDone. {copied} model file(s) bundled to:\n  {DST_ROOT}")


if __name__ == "__main__":
    download_and_bundle()
