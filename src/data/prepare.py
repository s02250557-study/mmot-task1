"""1. Скачать один корпус, очистить, убрать дубли и разделить по песням."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, RAW_ROOT, config, read_json, write_json, file_hash

import argparse
import collections
import hashlib
import html
import json
import random
import re
import time
import unicodedata
import urllib.request

WORDS = re.compile(r"[a-zа-яё]+", re.I)
RUSSIAN = re.compile(r"[а-яё]", re.I)
SERVICE_LINE = re.compile(
    r"^(?:\d*\s*(?:contributors?|embed|translations?|you might also like|"
    r"read more|see .* live|get tickets|lyrics|текст песни|перевод песни|"
    r"реклама|куплет\s*\d*|припев|интро|аутро|бридж|verse\s*\d*|chorus))\s*[:.!\d]*$", re.I)
BRACKETS = re.compile(r"\[[^\[\]]{0,500}\]")
URL = re.compile(r"https?://\S+|www\.\S+", re.I)

def normalize(text):
    return " ".join(WORDS.findall(text.lower().replace("ё", "е")))

def clean_text(text):
    text = html.unescape(unicodedata.normalize("NFC", str(text)))
    # В этом корпусе часть переносов дополнительно экранирована.
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", " ")
    text = text.replace("\\\"", "\"").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u00ad", "").replace("\u200b", "").replace("\ufeff", "")
    text = BRACKETS.sub("", text)
    text = URL.sub("", text)
    # Типичный хвост Genius после последней строки.
    text = re.sub(r"\d*\s*Embed\s*$", "", text, flags=re.I)
    result = []
    for line in text.splitlines():
        line = "".join(c for c in line if not unicodedata.category(c).startswith("C"))
        line = re.sub(r"\s+", " ", line).strip()
        if SERVICE_LINE.fullmatch(line):
            continue
        if re.match(r"^(?:\d+\s+Contributors?\b|You might also like\b|See .+ Live\b|Get tickets\b)", line, re.I):
            continue
        # Убираем отдельные неразмеченные заголовки; слова внутри стиха сохраняются.
        if re.fullmatch(r"(?:припев|куплет|интро|аутро|бридж|verse|chorus)(?:\s*\d+)?(?:\s*[:—-].*)?", line, re.I):
            continue
        if line or (result and result[-1]):
            result.append(line)
    return "\n".join(result).strip()


def download(cfg):
    folder = RAW_ROOT
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / cfg["filename"]
    manifest = folder / "source.json"
    url = f'https://huggingface.co/datasets/{cfg["repo"]}/resolve/{cfg["revision"]}/{cfg["filename"]}'
    if target.exists() and manifest.exists():
        meta = read_json(manifest)
        if meta["url"] == url and file_hash(target) == meta["sha256"]:
            print("Исходный файл уже скачан и SHA256 проверен.", flush=True)
            return target
        raise RuntimeError("Исходный файл или его ревизия изменились. Проверьте data/raw.")
    partial = target.with_suffix(".partial")
    for attempt in range(4):
        try:
            # При ошибке повторяем загрузку целиком, чтобы не склеить несовместимые ответы.
            request = urllib.request.Request(url, headers={"User-Agent": "mmot-homework/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as out:
                total = int(response.headers.get("Content-Length") or 0)
                size, mark = 0, 0
                while True:
                    chunk = response.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    size += len(chunk)
                    if size - mark >= 32 * 1024 * 1024:
                        print(f"Скачано {size / 1e6:.1f} MB", flush=True)
                        mark = size
                if total and size != total:
                    raise RuntimeError("Неполный ответ сервера.")
            # JSON должен успешно прочитаться до признания загрузки завершённой.
            with partial.open(encoding="utf-8-sig") as stream:
                records = json.load(stream)
            if not isinstance(records, list) or not records:
                raise ValueError("Ожидался непустой JSON-массив песен.")
            del records
            partial.replace(target)
            write_json(manifest, {"repo": cfg["repo"], "revision": cfg["revision"],
                       "url": url, "sha256": file_hash(target), "bytes": target.stat().st_size,
                       "license": "not specified in dataset card"})
            return target
        except Exception:
            if attempt == 3:
                raise
            print("Ошибка загрузки, повтор через несколько секунд.", flush=True)
            time.sleep(3)

def title_key(row):
    # Ремиксы/live и английские переводы названий объединяются с основной песней.
    title = re.sub(r"\([^)]*\)|\[[^]]*\]", "", row.get("title", ""))
    title = re.sub(r"\b(remix|live|version|ремикс|лайв)\b.*", "", title, flags=re.I)
    artist = re.sub(r"\([^)]*\)", "", row.get("artist", ""))
    return normalize(artist) + "|" + normalize(title)

def prepare(source, cfg, seed):
    raw = read_json(source)
    stats = collections.Counter(raw_records=len(raw))
    before_words = collections.Counter()
    after_words = collections.Counter()
    exact = set()
    songs, examples = [], []
    for index, row in enumerate(raw):
        stats["genre_selected_records"] += 1
        original = row.get("lyrics") or ""
        stats["genre_selected_raw_text_bytes"] += len(original.encode("utf-8"))
        before_words.update(WORDS.findall(original.lower()))
        text = clean_text(original)
        words = WORDS.findall(text.lower())
        letters = [c for c in text if c.isalpha()]
        ratio = len(RUSSIAN.findall(text)) / max(len(letters), 1)
        lines = sum(bool(line.strip()) for line in text.splitlines())
        reason = None
        if len(text) < cfg["min_chars"] or len(words) < cfg["min_words"] or lines < cfg["min_lines"]:
            reason = "too_short"
        elif len(text) > cfg["max_chars"]:
            reason = "too_long"
        elif ratio < cfg["min_russian_ratio"]:
            reason = "language_filter"
        elif len(set(words)) / max(len(words), 1) < 0.12:
            reason = "extreme_repetition"
        elif "[" in text or "]" in text:
            reason = "broken_markup"
        if reason:
            stats[reason] += 1
            continue
        norm = normalize(text)
        digest = hashlib.sha256(norm.encode()).hexdigest()
        if digest in exact:
            stats["exact_duplicates"] += 1
            continue
        exact.add(digest)
        songs.append({"id": str(index), "artist": row.get("artist", ""), "title": row.get("title", ""),
                      "text": text, "key": title_key(row), "norm": norm, "sha256": digest})
        if len(examples) < 8:
            examples.append({"title": row.get("title"), "before": original[:1800], "after": text[:1800]})
    del raw
    print(f"После базовой очистки: {len(songs)} песен. Группируем похожие версии...", flush=True)

    # Union-find: пересекающиеся версии целиком попадают в один split.
    parents = list(range(len(songs)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    titles, chunks, shingle_sets = {}, {}, []
    for i, song in enumerate(songs):
        key = song["key"]
        if key in titles:
            union(i, titles[key])
        else:
            titles[key] = i
        words = song["norm"].split()
        # 32 последовательных слова — достаточно длинный общий фрагмент.
        fingerprints = set()
        for start in range(0, max(0, len(words) - 31), 16):
            chunk = " ".join(words[start:start + 32])
            fingerprints.add(hashlib.blake2b(chunk.encode(), digest_size=8).digest())
        candidates = set()
        for fingerprint in fingerprints:
            if fingerprint in chunks:
                candidates.add(chunks[fingerprint])
            else:
                chunks[fingerprint] = i
        is_duplicate = False
        for other in candidates:
            union(i, other)
            other_set = shingle_sets[other]
            overlap = len(fingerprints & other_set) / max(1, min(len(fingerprints), len(other_set)))
            size_ratio = min(len(fingerprints), len(other_set)) / max(1, max(len(fingerprints), len(other_set)))
            if overlap >= 0.85 and size_ratio >= 0.85:
                is_duplicate = True
        shingle_sets.append(fingerprints)
        song["near_duplicate"] = is_duplicate
    groups = collections.defaultdict(list)
    for i, song in enumerate(songs):
        if song.pop("near_duplicate"):
            stats["near_duplicates"] += 1
            continue
        groups[find(i)].append(song)
    stats["groups"] = len(groups)
    stats["clean_available_records"] = sum(map(len, groups.values()))
    stats["clean_available_bytes"] = sum(len(s["text"].encode("utf-8")) for g in groups.values() for s in g)
    print(f"Доступно чистого текста: {stats['clean_available_bytes'] / 1e6:.2f} MB", flush=True)

    # Случайный, воспроизводимый отбор целых песен. Нельзя обрезать текст посреди песни.
    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)
    selected, used = [], 0
    for group_id, group in group_items:
        for song in group:
            size = len(song["text"].encode("utf-8"))
            if cfg["max_bytes"] is None or used + size <= cfg["max_bytes"]:
                song["group"] = str(group_id)
                selected.append(song)
                used += size
    split_dir = ROOT / "data/splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    counts, sizes, lengths = collections.Counter(), collections.Counter(), []
    streams = {name: (split_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n")
               for name in ["train", "val", "test"]}
    try:
        for song in selected:
            # Назначение раздела одинаково для всех версий и общих фрагментов группы.
            value = int(hashlib.sha256(f"{seed}:{song['group']}".encode()).hexdigest()[:12], 16) / 16**12
            split = "test" if value < cfg["test_fraction"] else (
                "val" if value < cfg["test_fraction"] + cfg["validation_fraction"] else "train")
            song.pop("key")
            song.pop("norm")
            streams[split].write(json.dumps(song, ensure_ascii=False) + "\n")
            counts[split] += 1
            sizes[split] += len(song["text"].encode("utf-8"))
            after_words.update(WORDS.findall(song["text"].lower()))
            lengths.append(len(song["text"]))
    finally:
        for stream in streams.values():
            stream.close()
    if any(counts[name] < 10 for name in streams):
        raise RuntimeError("Недостаточно песен в одном из разделов.")
    lengths.sort()
    report = {"source": read_json(RAW_ROOT / "source.json"), "seed": seed, "filters": cfg,
              "counts": dict(stats), "selected_text_bytes": used,
              "has_100mb_available": stats["clean_available_bytes"] >= 100000000,
              "split_records": dict(counts), "split_text_bytes": dict(sizes),
              "length_chars": {"min": lengths[0], "median": lengths[len(lengths)//2],
                               "p95": lengths[int(len(lengths)*0.95)], "max": lengths[-1]},
              "top_words_before": before_words.most_common(100),
              "top_words_after": after_words.most_common(100),
              "split_sha256": {name: file_hash(split_dir / f"{name}.jsonl") for name in streams},
              "limitations": ["Language filter is a Russian-letter heuristic, not a language classifier.",
                 "Shared 32-word chunks are sampled with stride 16; remaining near-duplicates are possible.",
                 "Artist overlap is allowed. Groups keep detected versions/fragments in one split."]}
    write_json(ROOT / "reports/data_report.json", report)
    write_json(ROOT / "reports/cleaning_examples.json", examples)
    print(json.dumps({k: report[k] for k in ["selected_text_bytes", "has_100mb_available", "split_records"]},
                     ensure_ascii=False, indent=2), flush=True)
    if not report["has_100mb_available"]:
        print("ВНИМАНИЕ: 100 MB не набралось. До полного обучения обсудите объём корпуса.", flush=True)

def prepared_data_matches(cfg, seed):
    report_path = ROOT / "reports/data_report.json"
    if not report_path.exists():
        return False
    report = read_json(report_path)
    if report["filters"] != cfg or report["seed"] != seed:
        return False
    if report["source"] != read_json(RAW_ROOT / "source.json"):
        return False
    for split, digest in report["split_sha256"].items():
        path = ROOT / f"data/splits/{split}.jsonl"
        if not path.exists() or file_hash(path) != digest:
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    settings = config()
    source = download(settings["data"])
    if not args.download_only:
        if any((ROOT / "checkpoints").glob("*/training_state.pt")):
            if prepared_data_matches(settings["data"], settings["seed"]):
                print("Данные уже готовы и проверены. Сохраняем их без изменений.", flush=True)
                return
            raise RuntimeError("В этой папке уже есть обучение. Не пересоздаём изменённые данные. Используйте отдельную папку MMOT_ARTIFACT_ROOT.")
        prepare(source, settings["data"], settings["seed"])

if __name__ == "__main__":
    main()
