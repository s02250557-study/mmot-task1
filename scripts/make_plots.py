"""Графики корпуса, токенизатора и обучения. Все результаты идут."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import ROOT, read_json, read_rows, default_run
import argparse
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=default_run())
    args = parser.parse_args()
    output = ROOT / "reports/figures"
    output.mkdir(parents=True, exist_ok=True)
    lengths = [len(row["text"]) for row in read_rows(ROOT / "data/splits/train.jsonl")]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(lengths, bins=60)
    ax.set(xlabel="Characters per song", ylabel="Songs", title="Clean train song lengths")
    fig.tight_layout()
    fig.savefig(output / "song_lengths.png", dpi=150)
    plt.close(fig)
    data = read_json(ROOT / "reports/data_report.json")
    fig, ax = plt.subplots(figsize=(9,5))
    top = data["top_words_after"][:25]
    ax.barh([w for w, n in top][::-1], [n for w, n in top][::-1])
    ax.set(title="Most frequent words after cleaning", xlabel="Occurrences")
    fig.tight_layout()
    fig.savefig(output / "word_frequencies.png", dpi=150)
    plt.close(fig)
    tokenizer_path = ROOT / "reports/tokenizer_report.json"
    if tokenizer_path.exists():
        tokenizer = read_json(tokenizer_path)
        values = sorted([r["count"] for r in tokenizer["frequencies"] if r["count"]], reverse=True)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.loglog(range(1, len(values)+1), values)
        ax.set(xlabel="Token rank", ylabel="Train frequency", title="One BPE vocabulary: token frequency distribution")
        fig.tight_layout()
        fig.savefig(output / "token_frequencies.png", dpi=150)
        plt.close(fig)
    metrics = ROOT / "logs" / args.run / "metrics.jsonl"
    if metrics.exists():
        rows = list(read_rows(metrics))
        fig, ax = plt.subplots(figsize=(8,4))
        for key in ["train_loss", "val_loss"]:
            series = [(r["step"], r[key]) for r in rows if key in r]
            ax.plot([x for x,y in series], [y for x,y in series], label=key, marker=".")
        ax.set(xlabel="Optimizer steps", ylabel="Cross entropy", title=args.run)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"loss_{args.run}.png", dpi=150)
        plt.close(fig)
    print(output)

if __name__ == "__main__":
    main()
