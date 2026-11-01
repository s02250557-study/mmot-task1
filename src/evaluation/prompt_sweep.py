"""Прогнать фиксированный список префиксов, загрузив best только один раз."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT, PROJECT, config, default_run, read_json, write_json, file_hash
from inference import load_model, generate
import argparse
from datetime import datetime, timezone, timedelta
import json
import re
import time

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default=default_run())
    parser.add_argument("--prompts",default=str(PROJECT/"configs/inference_examples.json"))
    parser.add_argument("--temperatures",type=float,nargs="+",default=[0.0])
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--max-new-tokens",type=int,default=32)
    args=parser.parse_args()
    if any(t<0 for t in args.temperatures) or args.max_new_tokens<1:
        raise ValueError("Температуры должны быть >=0, длина >0.")
    prompts=read_json(args.prompts)
    checkpoint=ROOT/"checkpoints"/args.run/"best"
    weights_hash=file_hash(checkpoint/"model.safetensors")
    model,tokenizer=load_model(checkpoint)
    if file_hash(checkpoint/"model.safetensors")!=weights_hash:
        raise RuntimeError("Checkpoint меняется во время загрузки. Повторите после сохранения модели.")
    stamp=datetime.now(timezone(timedelta(hours=3))).strftime("%Y%m%d_%H%M%S")
    folder=ROOT/"reports"/f"prompt_sweep_{stamp}"
    folder.mkdir(parents=True,exist_ok=False)
    metadata={"checkpoint":str(checkpoint),"weights_sha256":weights_hash,
              "tokenizer_sha256":file_hash(checkpoint/"tokenizer.json"),
              "prompts_file":str(Path(args.prompts).resolve()),"prompts_sha256":file_hash(args.prompts),
              "temperatures":args.temperatures,"seed":args.seed,"max_new_tokens":args.max_new_tokens,
              "top_p":config()["generation"]["top_p"],
              "repetition_penalty":config()["generation"]["repetition_penalty"],
              "note":"Exploratory prompt search; all outputs retained. Hints are not a correctness score."}
    write_json(folder/"metadata.json",metadata)
    results=[]
    start=time.perf_counter()
    with (folder/"results.jsonl").open("w",encoding="utf-8") as stream:
        for index, prompt in enumerate(prompts, 1):
            for temperature in args.temperatures:
                result=generate(model,tokenizer,prompt["prompt"],args.seed,args.max_new_tokens,temperature)
                result.update(id=prompt["id"],category=prompt["category"],expected_hints=prompt["expected_hints"])
                early=" ".join(re.findall(r"[а-яёa-z0-9]+",result["continuation"].lower())[:12])
                result["expected_mentioned_early"]=any(h.lower() in early for h in prompt["expected_hints"])
                results.append(result)
                stream.write(json.dumps(result,ensure_ascii=False)+"\n")
                stream.flush()
            print(f"{index}/{len(prompts)}: {prompt['category']}",flush=True)
    write_json(folder/"results.json",results)
    markdown=["# Все результаты поиска префиксов\n",
              "Поиск примеров для демонстрации. Это не независимый тест качества.\n",
              f"Весов SHA256: {weights_hash}\n"]
    for index, prompt in enumerate(prompts, 1):
        markdown.append(f"\n## {prompt['id']}. {prompt['category']}\n\nПрефикс:\n\n~~~text\n{prompt['prompt']}\n~~~\n")
        for row in results:
            if row["id"]==prompt["id"]:
                markdown.append(f"\nТемпература {row['temperature']}, seed {row['seed']}. Только продолжение:\n\n~~~text\n{row['continuation']}\n~~~\n")
    (folder/"all_results.md").write_text("".join(markdown),encoding="utf-8")
    write_json(folder/"summary.json",{"prompts":len(prompts),"generations":len(results),
              "seconds":time.perf_counter()-start,"folder":str(folder),
              "checkpoint_unchanged":file_hash(checkpoint/"model.safetensors")==weights_hash})
    print(f"RESULTS: {folder}",flush=True)

if __name__=="__main__":
    main()
