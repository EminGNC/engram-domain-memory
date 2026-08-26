"""Task-level degerlendirme: HumanEval + MBPP pass@1/pass@10.

Neden? Paired ppl dususu "bellek bir sey ogrendi"yi gosterir ama bunun
GERCEK kod yetenegi mi yoksa boilerplate ezberi mi oldugunu ayirt edemez
(Tur-2 riski). Task-level skor bu ayrimin ikinci kanitidir.

Kullanim (iki asama, cunku uretim agir, skorlama hafif):
    # 1) Uretim: her model icin completions uretir (GPU)
    python eval/eval_task.py generate --model base --out runs/humaneval_base.jsonl
    python eval/eval_task.py generate --model engram --ckpt runs/C3_python/engram_step8000.pt --out runs/humaneval_c3.jsonl

    # 2) Skorlama: pass@1/pass@10 hesaplar (CPU, hizli)
    python eval/eval_task.py score --files runs/humaneval_base.jsonl runs/humaneval_c3.jsonl

Not: Kod calistirma (execution) icin `human-eval` paketi gerekir:
    pip install human-eval  (Kaggle'da sorun cikarsa localde skorr)
Bu script uretim + kendi basit execution runner'ini kullanir.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HUMANEVAL_PROMPT_FIELD = "prompt"
COMPLETION_LEN = 384
NUM_SAMPLES = 10  # pass@10 icin


def load_problems(dataset: str):
    """HF datasets'ten problemleri ceker (internet gerekli)."""
    from datasets import load_dataset

    if dataset == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        return [
            {"task_id": ex["task_id"], "prompt": ex["prompt"],
             "canonical_solution": ex["canonical_solution"], "entry_point": ex["entry_point"],
             "test": ex["test"]}
            for ex in ds
        ]
    elif dataset == "mbpp":
        ds = load_dataset("mbpp", "full", split="test")
        out = []
        for i, ex in enumerate(ds):
            prompt = (
                '"""\n' + ex["text"] + "\n"
                + "\n".join(ex["test_list"][:1]) + "\n""\"\"\"\n"
            )
            out.append({"task_id": f"mbpp/{ex['task_id']}", "prompt": prompt,
                        "entry_point": ex.get("code", "") and "",
                        "test": "\n".join(ex["test_list"]),
                        "canonical_solution": ex["code"]})
        return out
    raise ValueError(dataset)


def build_model(variant: str, ckpt: str | None, device: str):
    import torch
    from transformers import AutoModelForCausalLM
    from configs.config import ModelConfig, resolve_dtype

    model = AutoModelForCausalLM.from_pretrained(ModelConfig.name, dtype=resolve_dtype()).to(device)

    if variant == "engram" and ckpt:
        from configs.config import EngramExperimentConfig
        from src.engram import EngramAttach, EngramModuleConfig, HashConfig

        exp = EngramExperimentConfig()
        h = HashConfig(
            tokenizer_name_or_path=ModelConfig.name,
            layer_ids=exp.layer_ids,
            max_ngram_size=exp.max_ngram_size,
            n_head_per_ngram=exp.n_head_per_ngram,
            vocab_multiplier=exp.vocab_multiplier,
            seed=exp.seed,
        )
        m = EngramModuleConfig(n_embed_per_ngram=exp.n_embed_per_ngram,
                               n_head_per_ngram=exp.n_head_per_ngram)
        attach = EngramAttach(model, h, m, layer_ids=exp.layer_ids)
        state = torch.load(ckpt, map_location=device, weights_only=False)["injections"]
        attach.injections.load_state_dict(state)
        attach.enable()
        print(f"[task] Engram takildi: {ckpt} | alpha={attach.alpha_values()}")
    elif variant == "lora" and ckpt:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, ckpt).eval()
    model.eval()
    return model


def generate(model, tok, problems, n_samples: int, out_path: Path, device: str):
    import torch

    records = []
    for idx, prob in enumerate(problems):
        full_prompt = prob["prompt"]
        inputs = tok(full_prompt, return_tensors="pt").to(device)
        for s in range(n_samples):
            torch.manual_seed(1234 + s)
            gen_kwargs = dict(
                max_new_tokens=COMPLETION_LEN,
                pad_token_id=tok.eos_token_id,
            )
            if s > 0:
                gen_kwargs.update(do_sample=True, temperature=0.8, top_p=0.95)
            with torch.no_grad():
                ids = model.generate(**inputs, **gen_kwargs)
            comp = tok.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            records.append({
                "task_id": prob["task_id"],
                "sample_idx": s,
                "completion": comp,
                "prompt": full_prompt,
                "test": prob.get("test", ""),
                "entry_point": prob.get("entry_point", ""),
            })
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(problems)} problem bitti")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"uretim yazildi: {out_path} ({len(records)} completion)")


def check_correctness(rec: dict) -> bool:
    """Subprocess ile execution: prompt+completion+test birlesik calisir.

    Windows'ta multiprocessing.Queue handle sorunlarina yol actigi icin
    gecici dosya + subprocess kullanilir (cross-platform saglam).
    """
    import subprocess
    import tempfile

    if rec.get("entry_point"):
        program = (rec["prompt"] + "\n" + rec["completion"] + "\n"
                   + rec["test"] + f"\ncheck({rec['entry_point']})\n")
    else:
        program = rec["prompt"] + "\n" + rec["completion"] + "\n" + rec["test"]

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=10)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


def score(files: list[str]):
    """pass@1 = ilk sample dogru mu; pass@10 = en az biri dogru mu."""
    import collections

    stats = {}
    for path in files:
        by_task = collections.defaultdict(list)
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            by_task[r["task_id"]].append(r)
        n_tasks = len(by_task)
        p1 = p10 = 0
        for tid, recs in by_task.items():
            recs.sort(key=lambda r: r["sample_idx"])
            results = [check_correctness(r) for r in recs]
            if results[0]:
                p1 += 1
            if any(results):
                p10 += 1
        stats[path] = {"tasks": n_tasks, "pass@1": p1 / max(n_tasks, 1),
                       "pass@10": p10 / max(n_tasks, 1)}
        print(f"{path}: tasks={n_tasks} pass@1={stats[path]['pass@1']:.4f} "
              f"pass@10={stats[path]['pass@10']:.4f}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--model", choices=["base", "engram", "lora"], required=True)
    g.add_argument("--ckpt", default=None)
    g.add_argument("--dataset", default="humaneval", choices=["humaneval", "mbpp"])
    g.add_argument("--limit", type=int, default=None, help="ilk N problem (hizli test)")
    g.add_argument("--n-samples", type=int, default=NUM_SAMPLES)
    g.add_argument("--out", required=True)

    s = sub.add_parser("score")
    s.add_argument("--files", nargs="+", required=True)

    args = ap.parse_args()

    if args.cmd == "generate":
        problems = load_problems(args.dataset)
        if args.limit:
            problems = problems[: args.limit]
        print(f"{len(problems)} problem, {args.n_samples} sample/problem")
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        model = build_model(args.model, args.ckpt, device)
        generate(model, tok, problems, args.n_samples, Path(args.out), device)

    elif args.cmd == "score":
        score(args.files)


if __name__ == "__main__":
    main()
