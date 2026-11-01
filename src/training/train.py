"""4. Претрейн с нуля: AMP, SDPA, логи, best/last и возобновление."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, PROJECT, config, read_json, write_json, file_hash, data_fingerprint, default_run, CONFIG_PATH
import argparse
from contextlib import nullcontext
import json
import math
import random
import signal
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from model.build import build_model
from inference import generate

def load_tokens(split):
    return np.memmap(ROOT / f"data/tokenized/{split}.bin", dtype=np.uint16, mode="r")

def make_batch(tokens, generator, batch_size, block_size, device):
    blocks = (len(tokens) - 1) // block_size
    starts = generator.integers(0, blocks, size=batch_size) * block_size
    x = np.stack([tokens[s:s + block_size].astype(np.int64) for s in starts])
    y = np.stack([tokens[s + 1:s + block_size + 1].astype(np.int64) for s in starts])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

def amp_context(device, dtype):
    return torch.autocast("cuda", dtype=dtype) if device == "cuda" else nullcontext()

@torch.no_grad()
def estimate_loss(model, tokens, batch_size, block_size, batches, device, dtype):
    model.eval()
    # Один и тот же набор val-блоков на каждой оценке, без подглядывания в test.
    generator = np.random.default_rng(12345)
    values = []
    for _ in range(batches):
        x, y = make_batch(tokens, generator, batch_size, block_size, device)
        with amp_context(device, dtype):
            logits = model(x).logits
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
        values.append(loss.item())
    model.train()
    return sum(values) / len(values)

def learning_rate(step, settings):
    if step < settings["warmup_steps"]:
        return settings["learning_rate"] * (step + 1) / max(1, settings["warmup_steps"])
    progress = (step - settings["warmup_steps"]) / max(1, settings["max_steps"] - settings["warmup_steps"])
    factor = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return settings["min_learning_rate"] + factor * (settings["learning_rate"] - settings["min_learning_rate"])

def check_resume_settings(saved, current):
    # Можно увеличить только бюджет. Остальные параметры должны совпадать.
    budget_keys = {"max_steps", "target_token_passes"}
    old = {k: v for k, v in saved.items() if k not in budget_keys}
    new = {k: v for k, v in current.items() if k not in budget_keys}
    if old != new or current["max_steps"] < saved["max_steps"]:
        raise RuntimeError("Для resume нужны те же настройки; разрешено только увеличение бюджета.")
    return current["max_steps"] > saved["max_steps"]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=default_run())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, help="Остановиться после N дополнительных шагов, не меняя расписание LR.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batches", type=int)
    parser.add_argument("--allow-small-corpus", action="store_true")
    args = parser.parse_args()
    if not args.run.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Имя запуска: только буквы, цифры, _ и -.")
    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps должно быть положительным.")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size должно быть положительным.")
    if args.eval_batches is not None and args.eval_batches < 1:
        raise ValueError("--eval-batches должно быть положительным.")
    cfg = config()
    settings = cfg["training"].copy()
    if args.batch_size:
        settings["micro_batch"] = args.batch_size
    if args.eval_batches:
        settings["eval_batches"] = args.eval_batches
    data_report = read_json(ROOT / "reports/data_report.json")
    if not data_report["has_100mb_available"] and not args.allow_small_corpus:
        raise RuntimeError("В корпусе меньше 100 MB. Обсудите результат или явно используйте --allow-small-corpus.")
    manifest = read_json(ROOT / "data/tokenized/manifest.json")
    block = cfg["model"]["max_position_embeddings"]
    tokens_per_step = settings["micro_batch"] * settings["gradient_accumulation"] * block
    if settings["max_steps"] is None:
        settings["max_steps"] = math.ceil(settings["target_token_passes"] * manifest["splits"]["train"]["tokens"] / tokens_per_step)
    fingerprint = data_fingerprint()
    token_hash = file_hash(ROOT / "tokenizer/tokenizer.json")
    if manifest["data_fingerprint"] != fingerprint or manifest["tokenizer_sha256"] != token_hash:
        raise RuntimeError("Данные/токенизатор изменились. Повторите encode.py.")
    for split in ["train", "val"]:
        if file_hash(ROOT / f"data/tokenized/{split}.bin") != manifest["splits"][split]["sha256"]:
            raise RuntimeError(f"Изменился массив {split}.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA недоступна. Полный претрейн рассчитан на GPU.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(ROOT / "tokenizer", local_files_only=True)
    run_dir = ROOT / "checkpoints" / args.run
    log_dir = ROOT / "logs" / args.run
    state_path = run_dir / "training_state.pt"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError("Запуск уже существует. Используйте --resume или другое --run.")
    if args.resume and not state_path.exists():
        raise FileNotFoundError("Нет training_state.pt для продолжения.")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(tokenizer)
    params = sum(p.numel() for p in model.parameters())
    model.to(device)
    if settings["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if settings["gradient_checkpointing"] and not model.is_gradient_checkpointing:
        raise RuntimeError("Gradient checkpointing не включился.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 betas=(0.9, 0.95), weight_decay=settings["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    generator = np.random.default_rng(seed)
    start, best, tokens_seen = 0, float("inf"), 0
    if args.resume:
        # Только свой доверенный checkpoint: в нём есть состояния optimizer и RNG.
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["data_fingerprint"] != fingerprint or state["tokenizer_sha256"] != token_hash:
            raise RuntimeError("Нельзя продолжить на другом корпусе или токенизаторе.")
        if state["model_settings"] != cfg["model"] or state["seed"] != seed:
            raise RuntimeError("Для resume нужны та же модель и seed.")
        if check_resume_settings(state["settings"], settings):
            print(f"Бюджет увеличен: {state['settings']['max_steps']} -> {settings['max_steps']} шагов. "
                  "Для дальнейшего обучения используется обновлённое cosine-расписание.", flush=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        generator.bit_generator.state = state["batch_rng"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        start, best, tokens_seen = state["step"], state["best_val"], state["tokens_seen"]
        del state
    train_tokens, val_tokens = load_tokens("train"), load_tokens("val")
    block = cfg["model"]["max_position_embeddings"]
    stop = min(settings["max_steps"], start + args.steps) if args.steps else settings["max_steps"]
    if start >= stop:
        print("Запланированное обучение уже завершено.")
        return
    run_info = {"parameters": params, "model_settings": cfg["model"], "training": settings,
                "data_fingerprint": fingerprint, "tokenizer_sha256": token_hash,
                "seed": seed, "device": torch.cuda.get_device_name(), "amp_dtype": str(dtype),
                "torch": torch.__version__, "attention": "sdpa (backend chosen by torch)",
                "gradient_checkpointing_active": model.is_gradient_checkpointing,
                "tokens_per_step": settings["micro_batch"] * settings["gradient_accumulation"] * block,
                "initialization": "random, no pretrained weights",
                "steps_per_token_pass": len(train_tokens) / tokens_per_step,
                "planned_token_passes": settings["max_steps"] * tokens_per_step / len(train_tokens),
                "config_file": str(CONFIG_PATH), "artifact_root": str(ROOT)}
    write_json(log_dir / "run.json", run_info)
    write_json(run_dir / "run.json", run_info)
    print(json.dumps(run_info, ensure_ascii=False, indent=2), flush=True)
    metrics_path = log_dir / "metrics.jsonl"
    if args.resume and metrics_path.exists():
        # После аварии могли остаться логи шагов, которые ещё не попали в checkpoint.
        kept = []
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record["step"] <= start:
                kept.append(line)
        metrics_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    writer = SummaryWriter(str(log_dir), purge_step=start + 1 if args.resume else None)
    log_file = metrics_path.open("a", encoding="utf-8")
    def log(record):
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_file.flush()
        for key, value in record.items():
            if key != "step" and isinstance(value, (int, float)):
                writer.add_scalar(key, value, record["step"])
        writer.flush()
        print(json.dumps(record, ensure_ascii=False), flush=True)
    def export(folder):
        model.save_pretrained(folder, safe_serialization=True)
        tokenizer.save_pretrained(folder)
    def save(step):
        export(run_dir / "last")
        state = {"step": step, "best_val": best, "tokens_seen": tokens_seen,
                 "settings": settings, "model_settings": cfg["model"], "seed": seed,
                 "data_fingerprint": fingerprint, "tokenizer_sha256": token_hash,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "batch_rng": generator.bit_generator.state, "python_rng": random.getstate(),
                 "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all()}
        temp = state_path.with_suffix(".tmp")
        torch.save(state, temp)
        temp.replace(state_path)
    def validation_samples(step):
        prompts = read_json(PROJECT / "configs/benchmark.json")["prompts"][:2]
        model.eval()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            samples = [generate(model, tokenizer, prompt, seed=42, max_new_tokens=64) for prompt in prompts]
        write_json(ROOT / "samples" / args.run / f"step_{step:06d}.json", samples)
        for i, sample in enumerate(samples):
            writer.add_text(f"continuations/{i}", sample["text"], step)
        model.train()
    if start == 0:
        initial = estimate_loss(model, val_tokens, settings["micro_batch"], block,
                                settings["eval_batches"], device, dtype)
        log({"step": 0, "val_loss": initial, "perplexity": math.exp(min(initial, 50))})
        best = initial
        export(run_dir / "best")
        validation_samples(0)
        save(0)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    timer, interval_loss, interval_steps, interval_tokens = time.perf_counter(), 0.0, 0, 0
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        if not stop_requested:
            print("Получен Ctrl+C. Завершаю текущий шаг, затем валидация и сохранение...", flush=True)
        stop_requested = True
    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    try:
        for step in range(start, stop):
            lr = learning_rate(step, settings)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(settings["gradient_accumulation"]):
                x, y = make_batch(train_tokens, generator, settings["micro_batch"], block, device)
                with amp_context(device, dtype):
                    logits = model(x).logits
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Loss стал NaN/Inf. Последний сохранённый checkpoint сохранён.")
                step_loss += loss.item()
                scaler.scale(loss / settings["gradient_accumulation"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            tokens = settings["micro_batch"] * settings["gradient_accumulation"] * block
            tokens_seen += tokens
            interval_tokens += tokens
            interval_steps += 1
            interval_loss += step_loss / settings["gradient_accumulation"]
            done = step + 1
            if done % settings["log_every"] == 0 or done == stop:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - timer
                log({"step": done, "train_loss": interval_loss / interval_steps,
                     "learning_rate": lr, "tokens_per_second": interval_tokens / elapsed,
                     "tokens_seen": tokens_seen, "train_token_passes": tokens_seen / len(train_tokens),
                     "grad_norm": float(grad_norm), "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
                     "peak_reserved_vram_gib": torch.cuda.max_memory_reserved() / 1024**3})
                interval_loss, interval_steps, interval_tokens = 0.0, 0, 0
                timer = time.perf_counter()
            if done % settings["eval_every"] == 0 or done == stop:
                value = estimate_loss(model, val_tokens, settings["micro_batch"], block,
                                      settings["eval_batches"], device, dtype)
                log({"step": done, "val_loss": value, "perplexity": math.exp(min(value, 50))})
                if value < best:
                    best = value
                    export(run_dir / "best")
                validation_samples(done)
            if done % settings["save_every"] == 0 or done == stop:
                save(done)
            if stop_requested:
                # Веса, optimizer и RNG сохраняются только после полного шага.
                value = estimate_loss(model, val_tokens, settings["micro_batch"], block,
                                      settings["eval_batches"], device, dtype)
                log({"step": done, "val_loss": value, "perplexity": math.exp(min(value, 50))})
                if value < best:
                    best = value
                    export(run_dir / "best")
                save(done)
                print(f"Остановлено после шага {done}. Состояние сохранено; продолжение через --resume.", flush=True)
                break
    except KeyboardInterrupt:
        print("Остановка. Можно продолжить с последнего training_state.pt через --resume.", flush=True)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        log_file.close()
        writer.close()
    print(f"Checkpoint: {run_dir}. Best validation loss: {best:.4f}", flush=True)

if __name__ == "__main__":
    main()
