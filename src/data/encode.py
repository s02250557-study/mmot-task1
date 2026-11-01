"""3. Превратить train/val/test в компактные массивы токенов."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, config, read_rows, write_json, read_json, file_hash, data_fingerprint
import os
# Обрабатываем несколько песен за вызов: Rust-токенизатор использует 4 CPU-потока.
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAYON_NUM_THREADS"] = "4"
import numpy as np
from transformers import PreTrainedTokenizerFast

def song_batches(path):
    batch = []
    for row in read_rows(path):
        batch.append(row["text"])
        if len(batch) == 128:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    report = read_json(ROOT / "reports/tokenizer_report.json")
    if report["data_fingerprint"] != data_fingerprint():
        raise RuntimeError("После обучения токенизатора изменились данные.")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(ROOT / "tokenizer", local_files_only=True)
    if len(tokenizer) >= 65536:
        raise ValueError("Для uint16 нужен словарь меньше 65536.")
    output = ROOT / "data/tokenized"
    output.mkdir(parents=True, exist_ok=True)
    block = config()["model"]["max_position_embeddings"]
    manifest = {"dtype": "uint16", "block_size": block, "data_fingerprint": data_fingerprint(),
                "tokenizer_sha256": file_hash(ROOT / "tokenizer/tokenizer.json"), "splits": {}}
    for split in ["train", "val", "test"]:
        count = docs = 0
        path = output / f"{split}.bin"
        with path.open("wb") as stream:
            for texts in song_batches(ROOT / f"data/splits/{split}.jsonl"):
                encoded = tokenizer(texts, add_special_tokens=False, padding=False,
                                    truncation=False, verbose=False,
                                    return_attention_mask=False, return_token_type_ids=False)
                for ids in encoded["input_ids"]:
                    ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
                    np.asarray(ids, dtype=np.uint16).tofile(stream)
                    count += len(ids)
                    docs += 1
        blocks = (count - 1) // block
        if blocks < 1:
            raise ValueError(f"Мало токенов в {split}.")
        manifest["splits"][split] = {"tokens": count, "documents": docs, "blocks": blocks,
                                      "sha256": file_hash(path)}
        print(f"{split}: {docs} песен, {count:,} токенов, {blocks:,} блоков.")
    write_json(output / "manifest.json", manifest)

if __name__ == "__main__":
    main()
