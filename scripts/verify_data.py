"""Проверить отсутствие пересечений split и служебных маркеров."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import ROOT, config, read_rows, read_json, write_json, file_hash
import re

# Слово embedded в настоящей строке песни не является меткой Embed.
SERVICE_HINT = re.compile(r"\bcontributors?\b|(?:\b|\d)embed\b|you might also like|https?://|www\.", re.I)

def main():
    report = read_json(ROOT / "reports/data_report.json")
    settings = config()["data"]
    groups, hashes, total, counts, suspicious = {}, {}, 0, {}, []
    for split in ["train", "val", "test"]:
        count = 0
        path = ROOT / f"data/splits/{split}.jsonl"
        assert file_hash(path) == report["split_sha256"][split], f"Изменился файл {split}"
        for row in read_rows(path):
            assert groups.setdefault(row["group"], split) == split, "Группа попала в разные split."
            assert row["sha256"] not in hashes, "Одинаковые тексты в корпусе."
            hashes[row["sha256"]] = split
            assert "[" not in row["text"] and "]" not in row["text"], "Остались квадратные скобки."
            assert "\\n" not in row["text"], "Остались экранированные переносы."
            assert len(row["text"].splitlines()) >= settings["min_lines"]
            for line in row["text"].splitlines():
                if SERVICE_HINT.search(line):
                    suspicious.append({"id": row["id"], "split": split, "line": line[:180]})
            total += len(row["text"].encode("utf-8"))
            count += 1
        counts[split] = count
    limit = settings["max_bytes"]
    if limit is not None:
        assert total <= limit
    assert total == report["selected_text_bytes"]
    assert counts == report["split_records"]
    write_json(ROOT / "reports/data_verified.json",
               {"split_groups_disjoint": True, "exact_duplicates": 0, "text_bytes": total,
                "records": counts, "suspicious_service_lines": suspicious})
    if suspicious:
        raise RuntimeError(f"Есть подозрительные строки: {len(suspicious)}. См. data_verified.json.")
    print(f"Проверено {sum(counts.values())} песен, {total:,} байт. Утечек обнаруженных групп и служебных маркеров нет.")

if __name__ == "__main__":
    main()
