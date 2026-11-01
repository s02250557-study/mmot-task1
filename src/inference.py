"""5. Продолжение текста своей обученной моделью. Это pretrain, не чат."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, config, write_json, default_run
import argparse
from datetime import datetime
import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

def load_model(checkpoint):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Нет checkpoint: {checkpoint}. Сначала обучите модель.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(checkpoint, local_files_only=True,
                                           torch_dtype=dtype, attn_implementation="sdpa")
    return model.to(device).eval(), tokenizer

@torch.no_grad()
def generate(model, tokenizer, prompt, seed=42, max_new_tokens=None, temperature=None):
    settings = config()["generation"]
    limit = model.config.max_position_embeddings
    ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) >= limit:
        raise ValueError(f"Префикс занял {len(ids)} токенов, контекст модели {limit}. Укоротите текст.")
    count = min(max_new_tokens or settings["max_new_tokens"], limit - len(ids))
    temperature = settings["temperature"] if temperature is None else temperature
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    inputs = torch.tensor([ids], device=model.device)
    options = {"do_sample": temperature > 0}
    if temperature > 0:
        options.update(temperature=temperature, top_p=settings["top_p"])
    result = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                            max_new_tokens=count, repetition_penalty=settings["repetition_penalty"],
                            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                            use_cache=True, **options)
    continuation = tokenizer.decode(result[0, len(ids):], skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
    return {"prompt": prompt, "continuation": continuation, "text": prompt + continuation,
            "seed": seed, "prompt_tokens": len(ids), "generated_tokens": result.shape[1] - len(ids),
            "max_new_tokens": count, "temperature": temperature}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=default_run())
    parser.add_argument("--checkpoint", help="Путь к своей папке best или last.")
    parser.add_argument("--prompt", default="Ночной район затих, и только свет в окне\n")
    parser.add_argument("--prompt-file", help="UTF-8 файл с началом текста.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.85)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.temperature < 0:
        raise ValueError("Число токенов должно быть >0, температура >=0.")
    checkpoint = Path(args.checkpoint) if args.checkpoint else ROOT / "checkpoints" / args.run / "best"
    prompt = Path(args.prompt_file).read_text(encoding="utf-8-sig") if args.prompt_file else args.prompt
    model, tokenizer = load_model(checkpoint)
    result = generate(model, tokenizer, prompt, args.seed, args.max_new_tokens, args.temperature)
    result["checkpoint"] = str(checkpoint)
    result["note"] = "Continuation by a base pretrained model; no SFT/chat template."
    print(result["text"])
    path = ROOT / "samples" / f"{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    write_json(path, result)
    print(f"\nРезультат сохранён: {path}")

if __name__ == "__main__":
    main()
