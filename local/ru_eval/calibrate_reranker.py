"""Scores local/ru_eval/calibration_pairs.json with one or more CrossEncoder
models and prints the score distribution per (language direction, label).

Used to pick RELEVANCE_THRESHOLD after switching away from the English-only
ms-marco reranker — the old 3.0 was calibrated on English ms-marco logits and
means nothing for a different model's output scale.

    ./venv/bin/python local/ru_eval/calibrate_reranker.py [model ...]
"""
import json
import statistics
import sys
from pathlib import Path

from sentence_transformers import CrossEncoder

PAIRS_PATH = Path(__file__).with_name("calibration_pairs.json")
DEFAULT_MODELS = ["cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-v2-m3"]


def main(models: list[str]) -> None:
    pairs = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
    for model_name in models:
        model = CrossEncoder(model_name)
        scores = model.predict([(p["query"], p["passage"]) for p in pairs])
        for pair, score in zip(pairs, scores):
            pair.setdefault("scores", {})[model_name] = float(score)

        print(f"\n=== {model_name} ===")
        groups: dict[tuple[str, str], list[float]] = {}
        for pair, score in zip(pairs, scores):
            groups.setdefault((pair["lang"], pair["label"]), []).append(float(score))
        print(f"{'direction':10s} {'label':11s} {'n':>2s} {'min':>8s} {'mean':>8s} {'max':>8s}")
        for (lang, label), values in sorted(groups.items()):
            print(f"{lang:10s} {label:11s} {len(values):2d} {min(values):8.3f} "
                  f"{statistics.mean(values):8.3f} {max(values):8.3f}")

        relevant = [s for p, s in zip(pairs, scores) if p["label"] == "relevant"]
        partial = [s for p, s in zip(pairs, scores) if p["label"] == "partial"]
        irrelevant = [s for p, s in zip(pairs, scores) if p["label"] == "irrelevant"]
        print(f"separation: worst relevant {min(relevant):.3f} | best irrelevant {max(irrelevant):.3f} "
              f"| partial range {min(partial):.3f}..{max(partial):.3f}")

        # Every pair whose score contradicts its label under the best split.
        print("ranking violations (irrelevant scoring above a relevant of the same query):")
        by_query: dict[str, list[tuple[str, float]]] = {}
        for pair, score in zip(pairs, scores):
            by_query.setdefault(pair["query"], []).append((pair["label"], float(score)))
        violations = 0
        for query, labelled in by_query.items():
            best_irrelevant = max((s for lbl, s in labelled if lbl == "irrelevant"), default=None)
            worst_relevant = min((s for lbl, s in labelled if lbl == "relevant"), default=None)
            if best_irrelevant is not None and worst_relevant is not None and best_irrelevant > worst_relevant:
                violations += 1
                print(f"  {query[:50]!r}: irrelevant {best_irrelevant:.3f} > relevant {worst_relevant:.3f}")
        print(f"  total: {violations}")

    out = PAIRS_PATH.with_name("calibration_scores.json")
    out.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nscores written to {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_MODELS)
