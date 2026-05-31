#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DEPRECATED (2026-05-31):
# Dynamic/on-the-fly KG generator is archived due to severe performance regression.
# This script is blocked by default. Override only for legacy reproduction:
#   ALLOW_DEPRECATED_DYNAMIC_KG=1 python scripts/generate_dynamic_graph_from_llm.py ...

import argparse
import ast
import hashlib
import json
import os
import pickle
import re
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


MERGED_RELATIONS = [
    "antonym",
    "atlocation",
    "capableof",
    "causes",
    "createdby",
    "desires",
    "hascontext",
    "hasproperty",
    "hassubevent",
    "isa",
    "madeof",
    "notcapableof",
    "notdesires",
    "partof",
    "receivesaction",
    "relatedto",
    "usedfor",
]


def fmt_eta(seconds):
    if seconds < 0 or not np.isfinite(seconds):
        return "unknown"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def load_concepts(concept_path):
    id2concept = []
    with open(concept_path, "r", encoding="utf-8") as fin:
        for line in fin:
            c = line.strip()
            if c:
                id2concept.append(c)
    concept2id = {c: i for i, c in enumerate(id2concept)}
    return concept2id, id2concept


def normalize_concept(text):
    s = text.strip().lower()
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9_ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace(" ", "_")


def map_relation(rel):
    s = rel.strip().lower().replace(" ", "").replace("_", "")
    for i, r in enumerate(MERGED_RELATIONS):
        rr = r.lower().replace("_", "")
        if s == rr:
            return i
    rel_alias = {
        "is_a": "isa",
        "isa": "isa",
        "in": "atlocation",
        "locatedin": "atlocation",
        "locatedat": "atlocation",
        "partof": "partof",
        "hasa": "hasproperty",
        "hasproperty": "hasproperty",
        "usedfor": "usedfor",
        "cause": "causes",
        "causes": "causes",
        "related": "relatedto",
        "relatedto": "relatedto",
    }
    mapped = rel_alias.get(s, "relatedto")
    return MERGED_RELATIONS.index(mapped)


def parse_triples(raw_text):
    text = raw_text.strip()
    triples = []
    if not text:
        return triples

    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                h = item.get("head") or item.get("h") or item.get("subject") or ""
                r = item.get("relation") or item.get("r") or item.get("predicate") or "relatedto"
                t = item.get("tail") or item.get("t") or item.get("object") or ""
                if h and t:
                    triples.append((str(h), str(r), str(t)))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                triples.append((str(item[0]), str(item[1]), str(item[2])))
        if triples:
            return triples

    lines = [ln.strip("-* \t") for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if "|" in ln:
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))
                continue
        if "\t" in ln:
            parts = [p.strip() for p in ln.split("\t")]
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))
                continue
        m = re.match(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)", ln)
        if m:
            triples.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
    return triples


def llm_generate_triples(statement_text, max_triples, cache_dir, mode, timeout_sec=90):
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((mode + "||" + statement_text).encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as fin:
            data = json.load(fin)
        return data.get("triples", [])

    if mode == "heuristic":
        return []

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        if mode == "llm":
            raise RuntimeError("OPENAI_API_KEY is not set, but mode=llm.")
        return []

    base_url = os.environ.get(
        "OPENAI_BASE_URL",
        os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
    ).rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    prompt = (
        "You are helping build a QA reasoning graph.\n"
        "Given one statement, output 5-10 concise commonsense triples in JSON array format.\n"
        "Each item must be: {\"head\": \"...\", \"relation\": \"...\", \"tail\": \"...\"}.\n"
        "Use relations close to ConceptNet style, prefer one-hop useful facts only.\n"
        "Do not output anything except JSON.\n\n"
        f"Statement: {statement_text}\n"
    )

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You produce structured commonsense triples only."},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
        triples = parse_triples(content)[:max_triples]
        with open(cache_file, "w", encoding="utf-8") as fout:
            json.dump({"triples": triples, "raw": content}, fout, ensure_ascii=False)
        return triples
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LLM HTTPError: {e.code} {msg}") from e


def llm_generate_triples_with_retry(
    statement_text,
    max_triples,
    cache_dir,
    mode,
    timeout_sec=90,
    retries=3,
    retry_sleep_sec=3.0,
):
    last_err = None
    for t in range(retries + 1):
        try:
            return llm_generate_triples(
                statement_text,
                max_triples=max_triples,
                cache_dir=cache_dir,
                mode=mode,
                timeout_sec=timeout_sec,
            )
        except Exception as e:
            last_err = e
            if t >= retries:
                break
            sleep_sec = retry_sleep_sec * (2 ** t)
            print(f"[warn] LLM request failed (attempt {t + 1}/{retries + 1}): {e}")
            print(f"[warn] sleeping {sleep_sec:.1f}s then retry...")
            time.sleep(sleep_sec)
    raise last_err


def extract_statement_text(g, s):
    for key in ("sent", "statement", "text"):
        if key in g and isinstance(g[key], str) and g[key].strip():
            return g[key].strip()
        if key in s and isinstance(s[key], str) and s[key].strip():
            return s[key].strip()
    if "question" in s and isinstance(s["question"], dict):
        stem = s["question"].get("stem", "")
        if stem:
            return stem.strip()
    return json.dumps(g, ensure_ascii=False)


def build_graph_entry(
    grounded,
    statement,
    concept2id,
    mode,
    max_nodes,
    max_triples,
    cache_dir,
    timeout_sec=90,
    retries=3,
    retry_sleep_sec=3.0,
):
    qc = grounded.get("qc", []) or []
    ac = grounded.get("ac", []) or []
    q_ids = [concept2id[c] for c in qc if c in concept2id]
    a_ids = [concept2id[c] for c in ac if c in concept2id]
    if len(q_ids) == 0 and len(a_ids) == 0:
        # Keep loader invariant qam[0] == True.
        q_ids = [0]

    statement_text = extract_statement_text(grounded, statement)
    triples = llm_generate_triples_with_retry(
        statement_text,
        max_triples=max_triples,
        cache_dir=cache_dir,
        mode=mode,
        timeout_sec=timeout_sec,
        retries=retries,
        retry_sleep_sec=retry_sleep_sec,
    )

    mapped_edges = []
    qa_front = []
    qa_seen = set()
    for cid in q_ids + a_ids:
        if cid not in qa_seen:
            qa_front.append(cid)
            qa_seen.add(cid)
    node_set = set(qa_front)
    edge_node_order = []
    edge_seen = set()

    for h, r, t in triples:
        hh = normalize_concept(h)
        tt = normalize_concept(t)
        if hh not in concept2id or tt not in concept2id:
            continue
        hid = concept2id[hh]
        tid = concept2id[tt]
        rid = map_relation(r)
        mapped_edges.append((hid, rid, tid))
        node_set.add(hid)
        node_set.add(tid)
        if hid not in edge_seen and hid not in qa_seen:
            edge_node_order.append(hid)
            edge_seen.add(hid)
        if tid not in edge_seen and tid not in qa_seen:
            edge_node_order.append(tid)
            edge_seen.add(tid)

    if len(node_set) == 0:
        if q_ids:
            node_set.add(q_ids[0])
        elif a_ids:
            node_set.add(a_ids[0])
        else:
            node_set.add(0)

    extra_nodes = [cid for cid in edge_node_order if cid not in qa_seen]
    if len(extra_nodes) < (len(node_set) - len(qa_front)):
        # Fill remaining extras deterministically.
        rest = sorted([cid for cid in node_set if cid not in qa_seen and cid not in set(extra_nodes)])
        extra_nodes.extend(rest)
    node_list = qa_front + extra_nodes
    if len(node_list) > max_nodes:
        node_list = node_list[:max_nodes]

    node_pos = {cid: i for i, cid in enumerate(node_list)}
    n_node = len(node_list)
    n_rel = len(MERGED_RELATIONS)

    rows = []
    cols = []
    vals = []
    for hid, rid, tid in mapped_edges:
        if hid in node_pos and tid in node_pos:
            src = node_pos[hid]
            dst = node_pos[tid]
            rows.append(rid * n_node + src)
            cols.append(dst)
            vals.append(1.0)

    if not vals:
        rows.append(0)
        cols.append(0)
        vals.append(1.0)

    adj = sp.coo_matrix((np.array(vals, dtype=np.float32), (np.array(rows), np.array(cols))),
                        shape=(n_rel * n_node, n_node),
                        dtype=np.float32)

    q_set = set(q_ids)
    a_set = set(a_ids)
    qmask = np.array([cid in q_set for cid in node_list], dtype=np.bool_)
    amask = np.array([cid in a_set for cid in node_list], dtype=np.bool_)
    concepts = np.array(node_list, dtype=np.int64)
    cid2score = {int(cid): 1.0 for cid in node_list}

    return {
        "adj": adj,
        "concepts": concepts,
        "qmask": qmask,
        "amask": amask,
        "cid2score": cid2score,
    }


def main():
    if os.environ.get("ALLOW_DEPRECATED_DYNAMIC_KG", "0") != "1":
        print("[DEPRECATED] scripts/generate_dynamic_graph_from_llm.py is archived and disabled by default.")
        print("[DEPRECATED] Reason: dynamic KG graphs frequently degenerated and hurt CSQA accuracy.")
        print("[DEPRECATED] To run legacy code anyway, set ALLOW_DEPRECATED_DYNAMIC_KG=1.")
        return 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded-path", required=True)
    parser.add_argument("--statement-path", default=None)
    parser.add_argument("--concept-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--cache-dir", default=".cache/dynamic_kg")
    parser.add_argument("--mode", choices=["auto", "llm", "heuristic"], default="auto")
    parser.add_argument("--max-triples", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=48)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--sleep-ms", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep-sec", type=float, default=3.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--heartbeat-sec", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    if args.mode == "auto":
        mode = "llm" if os.environ.get("OPENAI_API_KEY", "").strip() else "heuristic"
    else:
        mode = args.mode

    concept2id, _ = load_concepts(args.concept_path)
    cache_dir = Path(args.cache_dir)

    grounded_rows = []
    with open(args.grounded_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                grounded_rows.append(json.loads(line))

    statement_rows = [{} for _ in range(len(grounded_rows))]
    if args.statement_path and os.path.exists(args.statement_path):
        tmp = []
        with open(args.statement_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    tmp.append(json.loads(line))
        if len(tmp) == len(grounded_rows):
            statement_rows = tmp
        else:
            print(
                f"[warn] statement size {len(tmp)} != grounded size {len(grounded_rows)}; "
                "fallback to grounded text only."
            )

    total = len(grounded_rows) if args.limit <= 0 else min(len(grounded_rows), args.limit)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = Path(str(out_path) + ".partial.pkl")

    out = []
    start_i = 0
    if args.resume:
        if out_path.exists():
            with open(out_path, "rb") as fin:
                maybe_done = pickle.load(fin)
            if isinstance(maybe_done, list) and len(maybe_done) == total:
                print(f"[resume] full output already exists: {out_path} ({len(maybe_done)}/{total})")
                print("[resume] nothing to do.")
                return
        if partial_path.exists():
            try:
                with open(partial_path, "rb") as fin:
                    out = pickle.load(fin)
                if not isinstance(out, list):
                    raise RuntimeError(f"Invalid partial checkpoint format: {partial_path}")
                start_i = len(out)
                print(f"[resume] loaded partial checkpoint: {partial_path} ({start_i}/{total})")
                if start_i > total:
                    raise RuntimeError(
                        f"Partial checkpoint is longer than target total ({start_i} > {total}). "
                        "Please remove the partial file and rerun."
                    )
            except (EOFError, pickle.UnpicklingError, RuntimeError) as e:
                broken_path = partial_path.with_suffix(partial_path.suffix + f".broken.{int(time.time())}")
                os.replace(partial_path, broken_path)
                out = []
                start_i = 0
                print(f"[warn] partial checkpoint is corrupted: {e}")
                print(f"[warn] moved broken checkpoint to: {broken_path}")
                print("[resume] restart from index 0 (API cache still reusable).")

    def save_partial():
        tmp_path = partial_path.with_suffix(partial_path.suffix + ".tmp")
        with open(tmp_path, "wb") as fout:
            pickle.dump(out, fout, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, partial_path)

    fallback_count = 0
    t0 = time.time()
    last_heartbeat = t0
    pbar = tqdm(range(start_i, total), desc=f"dynamic-graph({mode})")
    for i in pbar:
        g = grounded_rows[i]
        s = statement_rows[i]
        try:
            entry = build_graph_entry(
                g,
                s,
                concept2id=concept2id,
                mode=mode,
                max_nodes=args.max_nodes,
                max_triples=args.max_triples,
                cache_dir=cache_dir,
                timeout_sec=args.timeout_sec,
                retries=args.retries,
                retry_sleep_sec=args.retry_sleep_sec,
            )
            out.append(entry)
        except Exception as e:
            print(f"[error] failed at index={i}: {e}")
            if args.allow_failures:
                # Fallback graph with no extra triples.
                fallback_entry = build_graph_entry(
                    g,
                    s,
                    concept2id=concept2id,
                    mode="heuristic",
                    max_nodes=args.max_nodes,
                    max_triples=args.max_triples,
                    cache_dir=cache_dir,
                    timeout_sec=args.timeout_sec,
                    retries=0,
                    retry_sleep_sec=0.0,
                )
                out.append(fallback_entry)
                fallback_count += 1
                print(f"[warn] fallback-to-heuristic at index={i}")
            else:
                save_partial()
                print(f"[resume] partial saved: {partial_path} ({len(out)}/{total})")
                traceback.print_exc()
                raise

        if args.checkpoint_every > 0 and (len(out) % args.checkpoint_every == 0):
            save_partial()
            pbar.set_postfix({"saved": len(out)})

        # Heartbeat progress line: useful when tqdm appears frozen.
        if args.heartbeat_sec > 0:
            now = time.time()
            if (now - last_heartbeat) >= args.heartbeat_sec:
                done = len(out)
                processed = i - start_i + 1
                speed = processed / max(1e-9, (now - t0))
                remaining = total - done
                eta = remaining / max(1e-9, speed)
                print(
                    f"[hb] done={done}/{total} processed={processed} "
                    f"fallback={fallback_count} speed={speed:.2f}/s eta={fmt_eta(eta)}",
                    flush=True,
                )
                last_heartbeat = now
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    with open(out_path, "wb") as fout:
        pickle.dump(out, fout, protocol=pickle.HIGHEST_PROTOCOL)
    if partial_path.exists():
        partial_path.unlink()
    t1 = time.time()

    print(f"[done] mode={mode} rows={len(out)} -> {out_path}")
    print(f"[done] elapsed_sec={t1 - t0:.2f}")


if __name__ == "__main__":
    main()
