"""2. Обучить один byte-level BPE только на train и сохранить статистики."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, config, read_rows, write_json, file_hash, data_fingerprint
from collections import Counter
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAYON_NUM_THREADS"] = "4"
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

def main():
    settings = config()["tokenizer"]
    source = ROOT / "data/splits/train.jsonl"
    if not source.exists():
        raise FileNotFoundError("Сначала запустите src/data/prepare.py")
    output = ROOT / "tokenizer"
    if (output / "tokenizer.json").exists():
        report_path = ROOT / "reports/tokenizer_report.json"
        if report_path.exists():
            from common import read_json
            old = read_json(report_path)
            if (old["data_fingerprint"] == data_fingerprint()
                    and old["tokenizer_sha256"] == file_hash(output / "tokenizer.json")
                    and old["requested_vocab_size"] == settings["vocab_size"]
                    and old.get("fit_max_bytes") == settings["fit_max_bytes"]):
                print("Токенизатор уже готов и соответствует данным. Повторное обучение не нужно.")
                return
        raise FileExistsError("Токенизатор существует, но его отчёт/данные не совпадают. Не перезаписываем его.")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=settings["vocab_size"], min_frequency=settings["min_frequency"],
        special_tokens=["<pad>", "<bos>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    fit_stats = {"records": 0, "text_bytes": 0}
    def texts():
        # Порядок train уже случайно перемешан на этапе prepare.py с seed=42.
        for row in read_rows(source):
            size = len(row["text"].encode("utf-8"))
            if fit_stats["text_bytes"] + size > settings["fit_max_bytes"]:
                continue
            fit_stats["records"] += 1
            fit_stats["text_bytes"] += size
            if fit_stats["records"] % 2000 == 0:
                print(f"Токенизатор прочитал {fit_stats['records']} песен...", flush=True)
            yield row["text"]
    print("Обучаем один BPE на выборке train до 20 MB; 4 CPU-потока.", flush=True)
    tokenizer.train_from_iterator(texts(), trainer=trainer)
    print("Словарь готов. Считаем частоты и проверяем восстановление текста...", flush=True)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>",
        model_max_length=config()["model"]["max_position_embeddings"], clean_up_tokenization_spaces=False)
    output.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(output)
    counts = Counter()
    tokens = chars = words = records = 0
    for row in read_rows(source):
        text = row["text"]
        ids = fast.encode(text, add_special_tokens=False, verbose=False)
        if fast.decode(ids, clean_up_tokenization_spaces=False) != text:
            raise RuntimeError("Токенизатор не восстанавливает исходный текст.")
        counts.update(ids)
        tokens += len(ids)
        chars += len(text)
        words += len(text.split())
        records += 1
    frequencies = [{"id": i, "token": fast.convert_ids_to_tokens(i), "count": counts[i]}
                   for i in range(len(fast))]
    write_json(ROOT / "reports/tokenizer_report.json", {
        "type": "byte-level BPE trained from scratch on train only",
        "requested_vocab_size": settings["vocab_size"], "actual_vocab_size": len(fast),
        "fit_max_bytes": settings["fit_max_bytes"], "fit_sample": fit_stats,
        "records": records, "train_tokens_without_special": tokens,
        "tokens_per_word": tokens / words, "chars_per_token": chars / tokens,
        "unused_tokens": sum(x["count"] == 0 for x in frequencies),
        "tokens_seen_at_most_10_times": sum(0 < x["count"] <= 10 for x in frequencies),
        "roundtrip_train": True, "data_fingerprint": data_fingerprint(),
        "tokenizer_sha256": file_hash(output / "tokenizer.json"), "frequencies": frequencies})
    print(f"Готово: {len(fast)} токенов; train: {tokens:,} токенов; {tokens / words:.2f} токена/слово.")

if __name__ == "__main__":
    main()
