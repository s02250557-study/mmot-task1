"""Проверить совпадения префиксов и 12-словных фрагментов генераций с train."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from common import ROOT,PROJECT,read_json,read_rows,write_json,file_hash
import argparse
import re

def words(text):
    return re.findall(r"[а-яёa-z0-9]+",text.lower().replace("ё","е"))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",required=True)
    args=parser.parse_args()
    folder=Path(args.folder)
    prompts_path=Path(read_json(folder/"metadata.json")["prompts_file"])
    if not prompts_path.is_absolute():
        prompts_path=PROJECT/prompts_path
    prompts=read_json(prompts_path)
    rows=read_json(folder/"results.json")
    prompt_phrases={p["id"]:" "+" ".join(words(p["prompt"]))+" " for p in prompts}
    prompt_counts={p["id"]:0 for p in prompts}
    row_chunks=[]
    wanted=set()
    for row in rows:
        ws=words(row["continuation"])
        chunks={" ".join(ws[i:i+12]) for i in range(max(0,len(ws)-11))}
        row_chunks.append(chunks)
        wanted.update(chunks)
    found=set()
    records=0
    for row in read_rows(ROOT/"data/splits/train.jsonl"):
        ws=words(row["text"])
        text=" "+" ".join(ws)+" "
        for pid,phrase in prompt_phrases.items():
            if phrase in text:
                prompt_counts[pid]+=1
        for i in range(max(0,len(ws)-11)):
            phrase=" ".join(ws[i:i+12])
            if phrase in wanted:
                found.add(phrase)
        records+=1
    results=[]
    for row,chunks in zip(rows,row_chunks):
        results.append({"id":row["id"],"temperature":row["temperature"],
                        "train_match_12word_fraction":len(chunks & found)/max(1,len(chunks))})
    report={"train_records":records,"train_sha256":file_hash(ROOT/"data/splits/train.jsonl"),
            "prompt_occurrences_in_train":prompt_counts,"generation_overlap":results,
            "note":"Normalized exact matches only. No match does not prove novelty or absence of a fact in training."}
    write_json(folder/"train_overlap.json",report)
    print("Overlap audit saved:",folder/"train_overlap.json")
    print(dict(list(prompt_counts.items())[:10]))

if __name__=="__main__":
    main()
