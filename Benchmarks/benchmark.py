#!/usr/bin/env python3
"""
OpenRouter Multi-LLM Decompilation Benchmark
Benchmarks models from llms.txt on the LLM4Decompile eval set.
Usage: python benchmark.py [--llms llms.txt] [--output results.json] [--dashboard]
"""

import asyncio
import aiohttp
import argparse
import json
import os
import subprocess
import tempfile
import traceback
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import dotenv
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn

dotenv.load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TESTSET_URL = (
    "https://raw.githubusercontent.com/albertan017/LLM4Decompile/main/"
    "legacy-test/decompile-eval-executable-gcc-obj.json"
)
TESTSET_PATH = "./decompile_eval.json"
OPTS = ("O0", "O1", "O2", "O3")


# ── Eval ────────────────────────────────────────────────────────────────────

def evaluate_func(params):
    """Returns (flag_compile, flag_run). Never raises."""
    try:
        c_func           = params.get("c_func") or ""
        c_test           = params.get("c_test") or ""
        c_func_decompile = params.get("c_func_decompile") or ""
        timeout          = 10
        flag_compile = flag_run = 0

        # Empty/None generation → nothing to evaluate
        if not c_func_decompile.strip():
            return flag_compile, flag_run

        c_include = ""
        for line in c_func.split("\n"):
            if "#include" in line:
                c_include += line + "\n"
                c_func = c_func.replace(line, "")
        for line in c_test.split("\n"):
            if "#include" in line:
                c_include += line + "\n"
                c_test = c_test.replace(line, "")

        c_combine  = c_include + "\n" + c_func_decompile + "\n" + c_test
        c_onlyfunc = c_include + "\n" + c_func_decompile

        with tempfile.TemporaryDirectory() as td:
            pid    = os.getpid()
            c_f    = os.path.join(td, f"combine_{pid}.c")
            exe    = os.path.join(td, f"combine_{pid}")
            c_of   = os.path.join(td, f"onlyfunc_{pid}.c")
            exe_of = os.path.join(td, f"onlyfunc_{pid}")

            Path(c_f).write_text(c_combine,  encoding="utf-8", errors="replace")
            Path(c_of).write_text(c_onlyfunc, encoding="utf-8", errors="replace")

            # Check that the decompiled function alone compiles
            try:
                subprocess.run(
                    ["gcc", "-S", c_of, "-o", exe_of, "-lm"],
                    check=True, timeout=timeout, capture_output=True,
                )
                flag_compile = 1
            except Exception:
                return flag_compile, flag_run

            # Compile combined (function + test harness)
            try:
                subprocess.run(
                    ["gcc", c_f, "-o", exe, "-lm"],
                    check=True, timeout=timeout, capture_output=True,
                )
            except Exception:
                return flag_compile, flag_run

            # Run it twice (mirrors original eval methodology)
            for _ in range(2):
                try:
                    subprocess.run(
                        [exe], capture_output=True, text=True,
                        timeout=timeout, check=True,
                    )
                    flag_run = 1
                except Exception:
                    return flag_compile, flag_run

        return flag_compile, flag_run

    except Exception:
        # Last-resort guard: never let a worker crash the pool
        return 0, 0


def compute_stats(testsets, generations):
    """Run compile/exec evaluation in a subprocess pool, return per-opt stats."""
    tasks = [
        {
            "c_func":           t.get("c_func", ""),
            "c_test":           t.get("c_test", ""),
            # Coerce any non-string (None, Exception) to empty string
            "c_func_decompile": gen if isinstance(gen, str) else "",
        }
        for t, gen in zip(testsets, generations)
    ]

    # Use spawn to avoid fork-related issues; fall back item-by-item on pool crash
    try:
        with ProcessPoolExecutor() as pool:
            flags = list(pool.map(evaluate_func, tasks))
    except Exception as exc:
        print(f"\n  [WARN] ProcessPoolExecutor crashed ({exc}), falling back to serial eval...")
        flags = [evaluate_func(t) for t in tasks]

    stats = {opt: {"compile": 0, "run": 0, "total": 0} for opt in OPTS}
    for t, result in zip(testsets, flags):
        opt = t["type"]
        # result could be an exception if pool.map somehow yielded one
        if isinstance(result, tuple) and len(result) == 2:
            fc, fr = result
        else:
            fc, fr = 0, 0
        stats[opt]["total"]   += 1
        stats[opt]["compile"] += fc
        stats[opt]["run"]     += fr

    for opt, d in stats.items():
        tot = d["total"] or 1
        d["compile_rate"] = round(d["compile"] / tot, 4)
        d["run_rate"]     = round(d["run"]     / tot, 4)

    return stats


# ── OpenRouter ───────────────────────────────────────────────────────────────

async def call_openrouter(session: aiohttp.ClientSession, model: str,
                          asm: str, opt: str, semaphore: asyncio.Semaphore) -> str:
    prompt = (
        f"Decompile the following {opt}-optimized assembly to C. "
        "Output only valid C code, no explanations.\n\nAssembly:\n" + asm
    )
    payload = {
        "model":       model,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  512,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        
        "HTTP-Referer":  "https://github.com/crave-benchmark",
    }

    async with semaphore:
        for attempt in range(5):
            try:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as resp:
                    # Rate limited → back off and retry
                    if resp.status == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    # Server errors → retry
                    if resp.status >= 500:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    # Safely extract content; any of these keys may be absent
                    choices = data.get("choices") or []
                    if not choices:
                        return ""
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")
                    # content can be None (e.g. tool-only responses)
                    return content if isinstance(content, str) else ""
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt == 4:
                    return ""
                await asyncio.sleep(2 ** attempt)
            except Exception:
                # Unexpected error — don't retry, just skip
                return ""
    return ""


async def benchmark_model(session: aiohttp.ClientSession, model: str,
                           testsets: list, progress: Progress, task_id,
                           concurrency: int = 8) -> list:
    sem = asyncio.Semaphore(concurrency)

    async def one(t):
        try:
            result = await call_openrouter(
                session, model, t["input_asm_prompt"], t["type"], sem
            )
        except Exception:
            result = ""
        progress.advance(task_id)
        return result

    results = await asyncio.gather(*[one(t) for t in testsets], return_exceptions=True)
    return [r if isinstance(r, str) else "" for r in results]


# ── Dataset ──────────────────────────────────────────────────────────────────

def load_testset() -> list:
    if not Path(TESTSET_PATH).exists():
        print("Downloading eval dataset...")
        urllib.request.urlretrieve(TESTSET_URL, TESTSET_PATH)
    with open(TESTSET_PATH) as f:
        data = json.load(f)
    counts = defaultdict(int)
    for d in data:
        counts[d["type"]] += 1
    print(f"Loaded {len(data)} test cases  "
          f"({', '.join(f'{k}:{v}' for k, v in sorted(counts.items()))})")
    return data


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _atomic_write(path: str, obj):
    """Write JSON to a temp file then rename, so partial writes never corrupt."""
    tmp = path + ".tmp"
    Path(tmp).write_text(json.dumps(obj, indent=2, ensure_ascii=True))
    os.replace(tmp, path)


def save_checkpoint(path: str, testsets: list, all_gens: dict, results: dict):
    """Persist whatever we have so far. Called after each model finishes eval."""
    out = {
        "summary": results,
        "samples": {
            model: [
                {**t, "output": gen if isinstance(gen, str) else ""}
                for t, gen in zip(testsets, gens)
            ]
            for model, gens in all_gens.items()
        },
    }
    try:
        _atomic_write(path, out)
    except Exception as exc:
        # Last resort: dump to a side-car file rather than lose everything
        fallback = path + ".emergency"
        print(f"\n  [WARN] Could not write {path} ({exc}), trying {fallback}...")
        try:
            Path(fallback).write_text(json.dumps(out, indent=2, ensure_ascii=True))
        except Exception as exc2:
            print(f"  [ERR]  Emergency save also failed: {exc2}")


# ── Dashboard ────────────────────────────────────────────────────────────────

def launch_dashboard(results_path: str):
    try:
        import gradio as gr
        import pandas as pd
    except ImportError:
        raise SystemExit("pip install gradio pandas")

    with open(results_path) as f:
        data = json.load(f)
    summary = data.get("summary", {})

    def build_df():
        rows = []
        for model, stats in summary.items():
            row = {"model": model}
            overall_compile = overall_run = total = 0
            for opt in OPTS:
                d = stats.get(opt, {})
                row[f"{opt} compile%"] = f"{d.get('compile_rate', 0)*100:.1f}"
                row[f"{opt} run%"]     = f"{d.get('run_rate',     0)*100:.1f}"
                overall_compile += d.get("compile", 0)
                overall_run     += d.get("run",     0)
                total           += d.get("total",   0)
            row["avg compile%"] = f"{overall_compile/(total or 1)*100:.1f}"
            row["avg run%"]     = f"{overall_run/(total or 1)*100:.1f}"
            rows.append(row)
        return pd.DataFrame(rows)

    def build_bar_data(metric="run_rate"):
        rows = []
        for model, stats in summary.items():
            label = model.split("/")[-1]
            for opt in OPTS:
                d = stats.get(opt, {})
                rows.append({
                    "model": label,
                    "opt":   opt,
                    "rate":  round(d.get(metric, 0) * 100, 1),
                })
        return pd.DataFrame(rows)

    with gr.Blocks(title="LLM Decompilation Benchmark") as demo:
        gr.Markdown("# LLM Decompilation Benchmark\nAssembly → C  |  Compile & Run pass-rates across O0–O3")

        with gr.Row():
            metric_dd = gr.Dropdown(
                choices=["run_rate", "compile_rate"],
                value="run_rate",
                label="Metric",
            )

        gr.Dataframe(value=build_df(), label="Summary", interactive=False)

        bar = gr.BarPlot(
            value=build_bar_data("run_rate"),
            x="model",
            y="rate",
            color="opt",
            title="Run rate % by model & optimisation level",
            y_title="Pass rate (%)",
            x_title="Model",
            tooltip=["model", "opt", "rate"],
            height=400,
        )

        def refresh(metric):
            df    = build_bar_data(metric)
            title = ("Run" if metric == "run_rate" else "Compile") + " rate % by model & opt"
            return gr.BarPlot(value=df, title=title)

        metric_dd.change(refresh, inputs=metric_dd, outputs=bar)

    print(f"\nLaunching dashboard (results from {results_path})...")
    demo.launch(share=False)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark LLMs via OpenRouter on decompile-eval")
    p.add_argument("--llms",           default="llms.txt",     help="Path to llms.txt")
    p.add_argument("--output",         default="results.json", help="Output JSON path")
    p.add_argument("--concurrency",    type=int, default=8,    help="Concurrent requests per model")
    p.add_argument("--dashboard",      action="store_true",    help="Launch Gradio dashboard after run")
    p.add_argument("--dashboard-only", action="store_true",    help="Skip benchmarking, open dashboard from existing results")
    return p.parse_args()


def main():
    args = parse_args()

    if args.dashboard_only:
        launch_dashboard(args.output)
        return

    if not OPENROUTER_API_KEY:
        raise SystemExit("OPENROUTER_API_KEY not set. Add it to .env or export it.")

    llms_path = Path(args.llms)
    if not llms_path.exists():
        raise SystemExit(f"llms.txt not found: {llms_path}")

    models = [
        line.strip()
        for line in llms_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not models:
        raise SystemExit("No models found in llms.txt")

    print(f"Models to benchmark ({len(models)}):")
    for m in models:
        print(f"  • {m}")

    testsets = load_testset()

    # ── Generation phase ─────────────────────────────────────────────────────
    # all_gens is built incrementally so Ctrl+C mid-run still saves finished models
    all_gens: dict = {}
    results:  dict = {}

    # Load any previously saved generations so we can skip models already done
    if Path(args.output).exists():
        try:
            with open(args.output) as f:
                prev = json.load(f)
            for m, samples in prev.get("samples", {}).items():
                all_gens[m] = [s.get("output", "") for s in samples]
            results = prev.get("summary", {})
            skip = set(all_gens.keys())
            if skip:
                print(f"Resuming — skipping already-done models: {', '.join(skip)}")
        except Exception:
            pass

    pending = [m for m in models if m not in all_gens]

    async def _run():
        if not pending:
            return
        print(f"\nRunning {len(pending)} model(s) in parallel "
              f"({args.concurrency} concurrent requests each)...\n")
        progress = Progress(
            TextColumn("[bold cyan]{task.description:<36}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )

        async def run_one(model, task_id, session):
            gens = await benchmark_model(session, model, testsets, progress, task_id, args.concurrency)
            # Save this model's raw generations immediately on completion
            all_gens[model] = gens
            save_checkpoint(args.output, testsets, all_gens, results)

        with progress:
            task_ids = [
                progress.add_task(m.split("/")[-1][:35], total=len(testsets))
                for m in pending
            ]
            async with aiohttp.ClientSession() as session:
                await asyncio.gather(
                    *[run_one(m, tid, session) for m, tid in zip(pending, task_ids)],
                    return_exceptions=True,
                )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run())
    except KeyboardInterrupt:
        print("\n\nInterrupted — saving whatever finished...")
        save_checkpoint(args.output, testsets, all_gens, results)
        print(f"Partial generations saved → {args.output}")
        print("Re-run the same command to resume from where you left off.")
        return
    finally:
        loop.close()

    # ── Evaluation phase ─────────────────────────────────────────────────────
    print("\n\nEvaluating outputs (compile + run)...")
    for model in models:
        if model in results:
            print(f"  Skipping {model} (already evaluated)")
            continue
        gens = all_gens.get(model, [])
        print(f"  Evaluating {model}...")
        try:
            stats = compute_stats(testsets, gens)
        except Exception as exc:
            print(f"  [ERR] Eval crashed for {model}: {exc}")
            traceback.print_exc()
            stats = {opt: {"compile": 0, "run": 0, "total": len(testsets) // 4,
                           "compile_rate": 0.0, "run_rate": 0.0} for opt in OPTS}
        results[model] = stats
        for opt, d in stats.items():
            print(f"    {opt}  compile {d['compile_rate']:.2%}  |  run {d['run_rate']:.2%}")

        save_checkpoint(args.output, testsets, all_gens, results)

    print(f"\nAll results saved → {args.output}")

    if args.dashboard:
        launch_dashboard(args.output)


if __name__ == "__main__":
    main()
