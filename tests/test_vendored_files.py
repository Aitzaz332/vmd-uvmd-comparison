"""The third-party files must stay byte-identical to upstream (hashes from third_party/SOURCES.md)."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HASHES = {
    "third_party/MAGN/uvmd.py": "32f25dc8b9cb9241fc53e6fabd3a06be4fa0e48b524f7b7c013173e435c8d798",
    "third_party/vmdpy/vmdpy.py": "4d1b294bb7e699e2c0b481a358254aa042f51394996048196b5f9f3953fda9ee",
}


def test_hashes():
    for rel, h in HASHES.items():
        data = (ROOT / rel).read_bytes()  # .gitattributes keeps these files byte-exact (-text)
        assert hashlib.sha256(data).hexdigest() == h, f"{rel} differs from upstream"


def test_loader_keeps_only_imports_and_model_classes():
    from decomp.uvmd_magn import MODEL_CLASSES, UPSTREAM
    tree = ast.parse(UPSTREAM.read_text(encoding="utf-8"))
    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    assert classes == list(MODEL_CLASSES)
