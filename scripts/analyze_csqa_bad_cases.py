#!/usr/bin/env python3
"""Analyze CSQA in-house bad cases from QAGNN predictions and KG artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from statistics import fmean
from typing import Any

import pickle


LABELS = ["A", "B", "C", "D", "E"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_csv", required=True, help="Path to test_e*_preds.csv")
    parser.add_argument(
        "--statements_path",
        default="data/csqa/statement/train.statement.jsonl",
        help="CSQA train statement jsonl",
    )
    parser.add_argument(
        "--inhouse_train_qids",
        default="data/csqa/inhouse_split_qids.txt",
        help="In-house train qids file",
    )
    parser.add_argument(
        "--grounded_path",
        default="data/csqa/grounded/train.grounded.jsonl",
        help="Grounded jsonl aligned with train statements choices",
    )
    parser.add_argument(
        "--graph_path",
        default="data/csqa/graph/train.graph.adj.pk",
        help="Graph pickle aligned with train statements choices",
    )
    parser.add_argument(
        "--concept_path",
        default="data/cpnet/concept.txt",
        help="Concept vocabulary",
    )
    parser.add_argument(
        "--output_dir",
        default="analysis/bad_cases_csqa",
        help="Output directory for analysis artifacts",
    )
    parser.add_argument("--top_k", type=int, default=50, help="Number of bad cases to export in detail")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            data.append(json.loads(line))
    return data


def load_qids(path: str) -> set[str]:
    out = set()
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            qid = line.strip()
            if qid:
                out.add(qid)
    return out


def load_preds(path: str) -> dict[str, str]:
    preds: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        for row in reader:
            if len(row) < 2:
                continue
            qid = row[0].strip()
            label = row[1].strip()
            if qid:
                preds[qid] = label
    return preds


def load_concept2id(path: str) -> dict[str, int]:
    c2i = {}
    with open(path, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            c = line.strip()
            if c:
                c2i[c] = idx
    return c2i


def norm_token(t: str) -> str:
    s = t.lower().strip()
    for suf in ("ing", "ed", "es", "s"):
        if len(s) > 4 and s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s


def safe_mean(vals: list[float]) -> float | None:
    return float(fmean(vals)) if vals else None


def to_float_or_none(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (float, int)):
        if math.isnan(float(x)):
            return None
        return float(x)
    return None


def build_case_records(
    statements: list[dict[str, Any]],
    grounded: list[dict[str, Any]],
    graph_data: list[dict[str, Any]],
    test_qids: set[str],
    preds: dict[str, str],
    concept2id: dict[str, int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing_pred = 0
    for q_idx, s_obj in enumerate(statements):
        qid = s_obj["id"]
        if qid not in test_qids:
            continue
        if qid not in preds:
            missing_pred += 1
            continue

        gold = s_obj["answerKey"]
        pred = preds[qid]
        gold_i = LABELS.index(gold)
        pred_i = LABELS.index(pred) if pred in LABELS else -1
        choice_text = {c["label"]: c["text"] for c in s_obj["question"]["choices"]}

        choice_stats = []
        qc_first = grounded[q_idx * 5]["qc"]
        qc_norm = [norm_token(x) for x in qc_first]
        dup_norm = len(qc_norm) - len(set(qc_norm))

        ac_signature_counter: Counter[tuple[str, ...]] = Counter()
        for c in range(5):
            g = grounded[q_idx * 5 + c]
            gg = graph_data[q_idx * 5 + c]
            label = LABELS[c]
            ac = g.get("ac", [])
            ac_signature_counter[tuple(sorted(ac))] += 1

            concept_ids = gg["concepts"]
            qmask = gg["qmask"]
            amask = gg["amask"]
            cid2score = gg["cid2score"]
            node_cnt = int(len(concept_ids))
            qnode_cnt = int(qmask.sum()) if hasattr(qmask, "sum") else int(sum(qmask))
            anode_cnt = int(amask.sum()) if hasattr(amask, "sum") else int(sum(amask))
            ac_scores = []
            for ac_word in ac:
                cid = concept2id.get(ac_word)
                if cid is None:
                    continue
                sc = cid2score.get(cid)
                if sc is not None:
                    ac_scores.append(float(sc))
            ac_score = max(ac_scores) if ac_scores else None

            choice_stats.append(
                {
                    "label": label,
                    "choice_text": choice_text.get(label, ""),
                    "qc_count": len(g.get("qc", [])),
                    "ac_count": len(ac),
                    "ac_list": ac,
                    "node_count": node_cnt,
                    "qnode_count": qnode_cnt,
                    "anode_count": anode_cnt,
                    "ac_score": ac_score,
                }
            )

        dup_ac_groups = sum(1 for _, v in ac_signature_counter.items() if v > 1)
        gold_stat = choice_stats[gold_i]
        pred_stat = choice_stats[pred_i] if 0 <= pred_i < 5 else None
        gold_score = to_float_or_none(gold_stat.get("ac_score"))
        pred_score = to_float_or_none(pred_stat.get("ac_score") if pred_stat else None)
        score_margin = (pred_score - gold_score) if (pred_score is not None and gold_score is not None) else None

        rec = {
            "qid": qid,
            "question": s_obj["question"]["stem"],
            "gold_label": gold,
            "pred_label": pred,
            "correct": int(gold == pred),
            "gold_choice": choice_text.get(gold, ""),
            "pred_choice": choice_text.get(pred, ""),
            "qc_count": len(qc_first),
            "qc_norm_dup_count": dup_norm,
            "dup_ac_groups": dup_ac_groups,
            "gold_node_count": gold_stat["node_count"],
            "pred_node_count": pred_stat["node_count"] if pred_stat else None,
            "gold_ac_score": gold_score,
            "pred_ac_score": pred_score,
            "pred_minus_gold_ac_score": score_margin,
            "choices": choice_stats,
        }
        records.append(rec)

    if missing_pred:
        print(f"[warn] missing predictions for {missing_pred} test qids")
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    wrong = [r for r in records if not r["correct"]]
    correct = [r for r in records if r["correct"]]

    def collect(rows: list[dict[str, Any]], key: str) -> list[float]:
        out = []
        for r in rows:
            v = r.get(key)
            if isinstance(v, (float, int)):
                out.append(float(v))
        return out

    summary = {
        "total_eval_questions": total,
        "num_correct": len(correct),
        "num_wrong": len(wrong),
        "acc": (len(correct) / total) if total else None,
        "means_correct": {
            "qc_count": safe_mean(collect(correct, "qc_count")),
            "qc_norm_dup_count": safe_mean(collect(correct, "qc_norm_dup_count")),
            "dup_ac_groups": safe_mean(collect(correct, "dup_ac_groups")),
            "gold_node_count": safe_mean(collect(correct, "gold_node_count")),
            "gold_ac_score": safe_mean(collect(correct, "gold_ac_score")),
        },
        "means_wrong": {
            "qc_count": safe_mean(collect(wrong, "qc_count")),
            "qc_norm_dup_count": safe_mean(collect(wrong, "qc_norm_dup_count")),
            "dup_ac_groups": safe_mean(collect(wrong, "dup_ac_groups")),
            "gold_node_count": safe_mean(collect(wrong, "gold_node_count")),
            "gold_ac_score": safe_mean(collect(wrong, "gold_ac_score")),
            "pred_minus_gold_ac_score": safe_mean(collect(wrong, "pred_minus_gold_ac_score")),
        },
    }
    return summary


def dump_outputs(
    output_dir: str,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    top_k: int,
    meta: dict[str, Any],
) -> None:
    ensure_dir(output_dir)
    wrong = [r for r in records if not r["correct"]]
    wrong_sorted = sorted(
        wrong,
        key=lambda x: (
            x["pred_minus_gold_ac_score"] if x["pred_minus_gold_ac_score"] is not None else -1e9,
            x["qc_norm_dup_count"],
            x["dup_ac_groups"],
        ),
        reverse=True,
    )
    top = wrong_sorted[:top_k]

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as fout:
        json.dump({"meta": meta, "summary": summary}, fout, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "bad_cases_top.jsonl"), "w", encoding="utf-8") as fout:
        for r in top:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    flat_fields = [
        "qid",
        "gold_label",
        "pred_label",
        "correct",
        "qc_count",
        "qc_norm_dup_count",
        "dup_ac_groups",
        "gold_node_count",
        "pred_node_count",
        "gold_ac_score",
        "pred_ac_score",
        "pred_minus_gold_ac_score",
        "question",
        "gold_choice",
        "pred_choice",
    ]
    with open(os.path.join(output_dir, "all_eval_cases.csv"), "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=flat_fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k) for k in flat_fields})

    with open(os.path.join(output_dir, "bad_cases_top.md"), "w", encoding="utf-8") as fout:
        fout.write("# CSQA Bad Case Analysis (Top Cases)\n\n")
        fout.write(f"- total eval questions: {summary['total_eval_questions']}\n")
        fout.write(f"- accuracy from preds: {summary['acc']:.4f}\n")
        fout.write(f"- wrong cases: {summary['num_wrong']}\n\n")
        fout.write("## Group Stats\n\n")
        fout.write("### Correct\n")
        for k, v in summary["means_correct"].items():
            fout.write(f"- {k}: {v}\n")
        fout.write("\n### Wrong\n")
        for k, v in summary["means_wrong"].items():
            fout.write(f"- {k}: {v}\n")
        fout.write("\n## Top Bad Cases\n")
        for i, r in enumerate(top, start=1):
            fout.write(f"\n### {i}. {r['qid']}\n")
            fout.write(f"- question: {r['question']}\n")
            fout.write(f"- gold/pred: {r['gold_label']} ({r['gold_choice']}) -> {r['pred_label']} ({r['pred_choice']})\n")
            fout.write(f"- qc_count={r['qc_count']}, qc_norm_dup_count={r['qc_norm_dup_count']}, dup_ac_groups={r['dup_ac_groups']}\n")
            fout.write(
                "- ac_score(gold/pred/margin)="
                f"{r['gold_ac_score']} / {r['pred_ac_score']} / {r['pred_minus_gold_ac_score']}\n"
            )
            fout.write("- choice KG stats:\n")
            for ch in r["choices"]:
                fout.write(
                    f"  - {ch['label']}: ac={ch['ac_list']} node={ch['node_count']} "
                    f"qnode={ch['qnode_count']} anode={ch['anode_count']} ac_score={ch['ac_score']}\n"
                )


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)

    statements = load_jsonl(args.statements_path)
    grounded = load_jsonl(args.grounded_path)
    try:
        with open(args.graph_path, "rb") as fin:
            graph_data = pickle.load(fin)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to load graph pickle dependencies. Please run with amem_env Python:\n"
            "/home/tsdhyan/miniconda3/envs/amem_env/bin/python scripts/analyze_csqa_bad_cases.py ..."
        ) from exc
    concept2id = load_concept2id(args.concept_path)
    inhouse_train = load_qids(args.inhouse_train_qids)
    preds = load_preds(args.preds_csv)

    if len(grounded) != len(statements) * 5:
        raise ValueError(f"grounded length mismatch: {len(grounded)} vs {len(statements)}*5")
    if len(graph_data) != len(statements) * 5:
        raise ValueError(f"graph length mismatch: {len(graph_data)} vs {len(statements)}*5")

    test_qids = {s["id"] for s in statements if s["id"] not in inhouse_train}
    records = build_case_records(statements, grounded, graph_data, test_qids, preds, concept2id)
    summary = summarize(records)
    meta = {
        "preds_csv": args.preds_csv,
        "statements_path": args.statements_path,
        "grounded_path": args.grounded_path,
        "graph_path": args.graph_path,
        "test_qids_count": len(test_qids),
        "pred_rows": len(preds),
    }
    dump_outputs(args.output_dir, summary, records, args.top_k, meta)

    print(f"[done] outputs written to: {args.output_dir}")
    print(f"[done] acc={summary['acc']:.4f} wrong={summary['num_wrong']}/{summary['total_eval_questions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
