"""6. Test loss, unigram baseline и фиксированные новые префиксы."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, PROJECT, config, read_json, read_rows, write_json, file_hash, data_fingerprint, default_run
import argparse
from collections import Counter
import math
import re
import numpy as np
import torch
from torch.nn import functional as F
from inference import load_model, generate

def words(text):
    return re.findall(r"[а-яёa-z]+", text.lower().replace("ё", "е"))

def repetition(text):
    ws = words(text)
    grams = [tuple(ws[i:i+3]) for i in range(max(0, len(ws)-2))]
    return 1 - len(set(grams)) / max(1, len(grams)) if grams else 0.0

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=default_run())
    parser.add_argument("--max-blocks", type=int, default=512, help="0 = весь test.")
    parser.add_argument("--no-samples", action="store_true")
    args = parser.parse_args()
    if args.max_blocks < 0:
        raise ValueError("--max-blocks должно быть >=0.")
    model, tokenizer = load_model(ROOT / "checkpoints" / args.run / "best")
    manifest = read_json(ROOT / "data/tokenized/manifest.json")
    run = read_json(ROOT / "checkpoints" / args.run / "run.json")
    if manifest["data_fingerprint"] != data_fingerprint() or run["data_fingerprint"] != data_fingerprint():
        raise RuntimeError("Модель и test относятся к разным версиям корпуса.")
    if run["tokenizer_sha256"] != file_hash(ROOT / "tokenizer/tokenizer.json"):
        raise RuntimeError("Токенизатор изменился после обучения.")
    for split in ["train", "test"]:
        if file_hash(ROOT / f"data/tokenized/{split}.bin") != manifest["splits"][split]["sha256"]:
            raise RuntimeError(f"Изменился {split}.bin.")
    train = np.memmap(ROOT / "data/tokenized/train.bin", dtype=np.uint16, mode="r")
    test = np.memmap(ROOT / "data/tokenized/test.bin", dtype=np.uint16, mode="r")
    counts = np.ones(len(tokenizer), dtype=np.float64)
    for start in range(0, len(train), 1000000):
        counts += np.bincount(train[start:start+1000000], minlength=len(tokenizer))
    log_prob = np.log(counts / counts.sum())
    block = model.config.max_position_embeddings
    available = (len(test) - 1) // block
    take = available if args.max_blocks == 0 else min(available, args.max_blocks)
    indices = np.linspace(0, available-1, take, dtype=int)
    losses, baselines = [], []
    with torch.no_grad():
        for index in indices:
            start = int(index) * block
            x = torch.tensor(test[start:start+block].astype(np.int64)[None], device=model.device)
            targets = test[start+1:start+block+1].astype(np.int64)
            y = torch.tensor(targets[None], device=model.device)
            logits = model(x).logits
            losses.append(F.cross_entropy(logits.float().reshape(-1, len(tokenizer)), y.reshape(-1)).item())
            baselines.append(float(-log_prob[targets].mean()))
    result = {"run": args.run, "test_blocks": take, "available_test_blocks": available,
              "test_targets": take * block, "test_loss": float(np.mean(losses)),
              "test_perplexity": math.exp(float(np.mean(losses))),
              "unigram_loss": float(np.mean(baselines)),
              "unigram_perplexity": math.exp(float(np.mean(baselines))),
              "data_fingerprint": data_fingerprint(), "tokenizer_sha256": run["tokenizer_sha256"],
              "checkpoint_config_sha256": file_hash(ROOT / "checkpoints" / args.run / "best/config.json"),
              "checkpoint_weights_sha256": file_hash(ROOT / "checkpoints" / args.run / "best/model.safetensors"),
              "benchmark_sha256": file_hash(PROJECT / "configs/benchmark.json")}
    samples = []
    if not args.no_samples:
        benchmark = read_json(PROJECT / "configs/benchmark.json")
        for prompt in benchmark["prompts"]:
            for seed in benchmark["seeds"]:
                sample = generate(model, tokenizer, prompt, seed, benchmark["max_new_tokens"])
                sample["repeated_word_trigrams_fraction"] = repetition(sample["continuation"])
                samples.append(sample)
        # Ищем все 12-словные фрагменты генераций в train, не загружая индекс train в RAM.
        needles = set()
        sample_chunks = []
        for sample in samples:
            ws = words(sample["continuation"])
            chunks = {" ".join(ws[i:i+12]) for i in range(max(0, len(ws)-11))}
            sample_chunks.append(chunks)
            needles.update(chunks)
        found = set()
        for row in read_rows(ROOT / "data/splits/train.jsonl"):
            ws = words(row["text"])
            for i in range(max(0, len(ws)-11)):
                phrase = " ".join(ws[i:i+12])
                if phrase in needles:
                    found.add(phrase)
        for sample, chunks in zip(samples, sample_chunks):
            sample["train_match_12word_fraction"] = len(chunks & found) / max(1, len(chunks))
        result["samples"] = samples
        result["mean_repeated_trigrams"] = float(np.mean([s["repeated_word_trigrams_fraction"] for s in samples]))
        result["mean_train_match_12word_fraction"] = float(np.mean([s["train_match_12word_fraction"] for s in samples]))
    result["limitations"] = [
        "Perplexity is comparable only with the same tokenizer and test blocks.",
        "Low loss and rhyme alone do not establish coherence.",
        "Read all fixed samples and assess coherence, grammar and style manually.",
        "No instruction following or factual QA is promised without SFT."]
    write_json(ROOT / "reports" / f"evaluation_{args.run}.json", result)
    print({k: v for k, v in result.items() if k not in ["samples", "limitations"]})

if __name__ == "__main__":
    main()
