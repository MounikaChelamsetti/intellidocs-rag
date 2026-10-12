"""Run the IntelliDocs evaluation.

Modes
  offline  No API key needed. Uses a LEXICAL hashing embedder and measures RETRIEVAL only
           (Recall@k, relevance statistics, filter behaviour). It does NOT measure answer quality
           and says nothing about Gemini embeddings.
  live     Needs GEMINI_API_KEY. Uses Gemini embeddings + generation and additionally measures
           answer correctness, citation correctness/coverage and unsupported-answer rate.
           Embeddings are cached in .cache/ so reruns do not re-spend quota.

Examples
  python evaluation/run_evaluation.py --mode offline
  python evaluation/run_evaluation.py --mode live --sleep 7
  python evaluation/run_evaluation.py --mode live --experiment --chunk-sizes 50:10,400:60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CACHE_DIR, EVALUATION_DIR, RESULTS_DIR, SAMPLE_DOCS_DIR, load_settings  # noqa: E402
from src.embeddings import CachedEmbedder, GeminiEmbedder  # noqa: E402
from src.errors import IntelliDocsError  # noqa: E402
from src.evaluation import (  # noqa: E402
    evaluate_answer_case, evaluate_retrieval_case, load_cases, results_to_json, retrieval_recall, summarize,
)
from src.llm import GeminiLLM  # noqa: E402
from src.offline import ExtractiveDemoLLM, HashingEmbedder  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

LIMITATIONS = [
    "Tiny corpus: with only 11 pages (11 chunks at the default size) top-5 retrieval covers ~45%% of the corpus, so Recall@5 is easy; Recall@1/@3 are more informative.",
    "Small synthetic corpus (3 fictional PDFs, 11 pages) with %d hand-written questions: results are a regression check, not evidence of general performance.",
    "Answer correctness and citation checks are automated string/page rules, not human judgement.",
    "Pages are short, so chunk-size comparisons are limited (chunks never cross page boundaries).",
]


def build_pipeline(mode, settings, embedder, llm, chunk, ingest=True):
    p = RAGPipeline(settings, embedder, llm, VectorStore())
    if ingest:
        for pdf in sorted(SAMPLE_DOCS_DIR.glob("*.pdf")):
            r = p.ingest_pdf(pdf.name, pdf.read_bytes(), chunk[0], chunk[1])
            if r.status == "error":
                raise SystemExit(f"Failed to ingest {pdf.name}: {r.message}")
    return p


def threshold_sweep(pipeline, cases, top_k, thresholds):
    cached = []
    for c in cases:
        res = pipeline.retriever.retrieve(c.question, top_k, -1.0)
        cached.append((c, res.candidates))
    rows = []
    for t in thresholds:
        recalls, neg_refused, ans_refused = [], [], []
        for c, cands in cached:
            kept = [x for x in cands if x.score >= t]
            if c.answerable:
                recalls.append(retrieval_recall([(x.filename, x.page_number) for x in kept], c.expected_sources))
                ans_refused.append(1.0 if not kept else 0.0)
            else:
                neg_refused.append(1.0 if not kept else 0.0)
        rows.append({
            "min_score": t,
            "recall_at_k_after_filter": round(sum(recalls) / len(recalls), 4),
            "answerable_blocked_by_filter": round(sum(ans_refused) / len(ans_refused), 4),
            "unanswerable_blocked_by_filter": round(sum(neg_refused) / len(neg_refused), 4),
        })
    return rows


def parse_chunks(spec):
    return [tuple(int(x) for x in part.split(":")) for part in spec.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["offline", "live"], default="offline")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=None, help="default: settings value (live) or 0.10 (offline)")
    ap.add_argument("--chunk", default="400:60", help="size:overlap for the main run")
    ap.add_argument("--experiment", action="store_true", help="retrieval-only comparison across --chunk-sizes")
    ap.add_argument("--chunk-sizes", default="50:10,400:60")
    ap.add_argument("--sleep", type=float, default=7.0, help="seconds between live generation calls (rate limits)")
    ap.add_argument("--questions", default=str(EVALUATION_DIR / "questions.json"))
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    settings = load_settings()
    cases = load_cases(Path(args.questions))
    live = args.mode == "live"
    if live:
        if not settings.gemini_api_key:
            raise SystemExit("Live mode needs GEMINI_API_KEY (env var or .env). Use --mode offline without a key.")
        embedder = CachedEmbedder(GeminiEmbedder(settings), CACHE_DIR / "embeddings.json")
        llm = GeminiLLM(settings)
        min_score = settings.min_relevance if args.min_score is None else args.min_score
    else:
        embedder, llm = HashingEmbedder(), ExtractiveDemoLLM()
        min_score = 0.10 if args.min_score is None else args.min_score
    chunk = parse_chunks(args.chunk)[0]
    started = time.time()

    report = {
        "mode": args.mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedder": embedder.name,
        "generation_model": settings.generation_model if live else None,
        "top_k": args.top_k,
        "min_score": min_score,
        "chunk_size_overlap": list(chunk),
        "answer_metrics_measured": live,
        "limitations": [(l % len(cases) if "%d" in l else l.replace("%%", "%")) for l in LIMITATIONS],
    }
    if not live:
        report["note"] = ("OFFLINE LEXICAL RUN: retrieval-only. Answer correctness, citation metrics and "
                          "unsupported-answer rate were NOT measured (they require a live LLM).")
    try:
        pipeline = build_pipeline(args.mode, settings, embedder, llm, chunk)
        results = []
        for i, case in enumerate(cases):
            if live:
                results.append(evaluate_answer_case(pipeline, case, args.top_k, min_score))
                if i < len(cases) - 1:
                    time.sleep(args.sleep)
            else:
                results.append(evaluate_retrieval_case(pipeline, case, args.top_k, min_score))
        report["summary"] = summarize(results, with_answers=live)
        report["per_case"] = results_to_json(results)
        report["threshold_sweep"] = threshold_sweep(pipeline, cases, args.top_k, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        if args.experiment:
            exp = []
            for size, overlap in parse_chunks(args.chunk_sizes):
                p = build_pipeline(args.mode, settings, embedder, llm, (size, overlap))
                rs = [evaluate_retrieval_case(p, c, args.top_k, min_score) for c in cases]
                s = summarize(rs, with_answers=False)
                exp.append({"chunk_size": size, "overlap": overlap, "chunks_indexed": p.store.count(),
                            "recall_at_1": s["recall_at_1"], "recall_at_3": s["recall_at_3"], "recall_at_k": s["recall_at_k"], "hit_rate_at_k": s["hit_rate_at_k"],
                            "top1_score_answerable": s["retrieval_top1_score_answerable"]})
            report["chunk_size_experiment (retrieval only)"] = exp
    except IntelliDocsError as exc:
        raise SystemExit(f"Evaluation aborted: {exc}")
    if live:
        report["embedding_cache"] = {"hits": embedder.hits, "misses": embedder.misses}
    report["runtime_seconds"] = round(time.time() - started, 1)

    out = Path(args.output) if args.output else RESULTS_DIR / f"{args.mode}_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in report if k not in ("per_case",)}, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
