"""Собрать архив best и last для GitHub Release."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import ROOT, PROJECT, default_run, file_hash
import zipfile

name = "genius55m-v1.0.0.zip"
output = PROJECT / "release" / name
output.parent.mkdir(exist_ok=True)
run = ROOT / "checkpoints" / default_run()
files = [run / "run.json"]
for folder in [run / "best", run / "last", ROOT / "tokenizer"]:
    files.extend(sorted(folder.glob("*.json")))
    files.extend(sorted(folder.glob("*.safetensors")))
for folder in [run / "best", run / "last"]:
    if not (folder / "model.safetensors").is_file():
        raise FileNotFoundError("Сначала обучите модель или скачайте её из релиза.")
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
    for path in files:
        archive.write(path, (Path("artifacts") / path.relative_to(ROOT)).as_posix())
checksum = file_hash(output)
output.with_suffix(".zip.sha256").write_text(checksum + "  " + name + "\n", encoding="utf-8")
print("release/" + name)
print("release/" + name + ".sha256")
