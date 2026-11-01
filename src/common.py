"""Общие пути и настройки. Все результаты хранятся внутри artifacts/."""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT / "configs/pretrain.json"
ROOT = Path(os.environ.get("MMOT_ARTIFACT_ROOT", "artifacts"))
if not ROOT.is_absolute():
    ROOT = PROJECT / ROOT
ROOT = ROOT.resolve()
RAW_ROOT = ROOT / "data/raw"
ROOT.mkdir(parents=True, exist_ok=True)

for name, folder in {
    "HF_HOME": "cache/huggingface",
    "TORCH_HOME": "cache/torch",
    "MPLCONFIGDIR": "cache/matplotlib",
}.items():
    path = ROOT / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(name, str(path))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def relative_path(path):
    return Path(os.path.relpath(Path(path).resolve(), PROJECT)).as_posix()

def portable(value):
    if isinstance(value, Path):
        return relative_path(value)
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    if isinstance(value, str):
        for path in [ROOT, PROJECT]:
            for spelling in [str(path), path.as_posix()]:
                value = value.replace(spelling, relative_path(path))
        return value
    return value

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(portable(value), ensure_ascii=False, indent=2), encoding="utf-8")

def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)

def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()

def config():
    return read_json(CONFIG_PATH)

def default_run():
    return config()["run_name"]

def data_fingerprint():
    return file_hash(ROOT / "reports/data_report.json")
