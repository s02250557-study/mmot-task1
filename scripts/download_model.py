"""Скачать модель из GitHub Release и проверить SHA256."""
import argparse
import hashlib
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

PROJECT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", help="OWNER/REPOSITORY; по умолчанию определяется из git remote origin.")
parser.add_argument("--tag", default="v1.0.0")
args = parser.parse_args()
repo = args.repo
if not repo:
    try:
        origin = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=PROJECT, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        parser.error("Укажите --repo OWNER/REPOSITORY.")
    match = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$", origin)
    if not match:
        parser.error("Не удалось определить GitHub-репозиторий. Укажите --repo OWNER/REPOSITORY.")
    repo = match.group(1)
if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
    parser.error("Формат --repo: OWNER/REPOSITORY.")
if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
    parser.error("Недопустимый тег релиза.")
name = "genius55m-" + args.tag + ".zip"
url = f"https://github.com/{repo}/releases/download/{args.tag}/{name}"
folder = PROJECT / "release"
folder.mkdir(exist_ok=True)
target = folder / name
partial = target.with_suffix(".partial")
print("Скачивание:", url, flush=True)
request = urllib.request.Request(url + ".sha256", headers={"User-Agent": "genius55m"})
with urllib.request.urlopen(request, timeout=60) as stream:
    expected = stream.read().decode("utf-8").split()[0]
request = urllib.request.Request(url, headers={"User-Agent": "genius55m"})
sha = hashlib.sha256()
with urllib.request.urlopen(request, timeout=120) as stream, partial.open("wb") as out:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        sha.update(chunk)
        out.write(chunk)
if sha.hexdigest() != expected:
    partial.unlink()
    raise RuntimeError("SHA256 архива не совпал. Повторите скачивание.")
partial.replace(target)
with zipfile.ZipFile(target) as archive:
    for member in archive.infolist():
        name_path = PurePosixPath(member.filename)
        if (name_path.is_absolute() or ".." in name_path.parts or "\\" in member.filename
                or not member.filename.startswith(("artifacts/checkpoints/genius55m/", "artifacts/tokenizer/"))):
            raise RuntimeError("Недопустимый путь в архиве.")
    for member in archive.infolist():
        destination = PROJECT.joinpath(*PurePosixPath(member.filename).parts)
        if member.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as out:
                shutil.copyfileobj(source, out)
print("Готово: artifacts/checkpoints/genius55m/{best,last}")
