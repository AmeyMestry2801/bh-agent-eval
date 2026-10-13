"""
Phase 9 (optional): zero-shot LLM triage variant via Groq's free API, evaluated on a
300-case golden subset of the TEST split against M1/M2/M3 using their saved flags.

The prompt and decision rule below are FROZEN before any test-set LLM output exists.
Smoke mode uses validation rows only. Decision rule: flag crisis iff risk_level == "crisis".
No PR-AUC is reported for the LLM (verbalised confidence is not a calibrated score).

ALL DATA IS SYNTHETIC.
    python -m src.llm_agent --smoke     # 10 validation cases, plumbing check
    python -m src.llm_agent --run       # 300-case golden subset (about 10-12 min at free-tier pace)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.config import (COST_FN_CRISIS, COST_FP_CRISIS, DATA_PROCESSED, RESULTS, ROOT, SEED,
                        ensure_dirs, set_seed)
from src.evaluate import mcnemar_exact, wilson_ci
from src.models import M1Rules

def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_SSL_CTX = _ssl_context()
URL = "https://api.groq.com/openai/v1/chat/completions"
MIN_INTERVAL_S = 3.0          # about 26 requests per minute, under the free 30 RPM limit
MAX_ATTEMPTS = 4
N_GOLDEN = 300
CACHE_DIR = RESULTS / "cache" / "llm"

SYSTEM_PROMPT = """You are a triage assistant for a behavioral-health intake line. Read one \
message and assign ONE risk level:
- no_risk: administrative or neutral content, no distress.
- low: mild stress or low mood, no thoughts of self-harm.
- moderate: significant distress, hopelessness or passive thoughts of death, but no stated \
intent or plan to act.
- crisis: stated intent, plan or imminent danger of self-harm or suicide.
Rules: negated statements ("I am not going to hurt myself") lower the risk. Hedged statements \
("maybe", "kind of") still count if the content indicates risk. Voice messages are speech-to-text \
transcripts and may contain recognition errors. Messages may mix English and Spanish. A missed \
crisis is far worse than a false alarm: if you are torn between crisis and moderate, choose crisis.
Respond with JSON only, exactly: {"risk_level": "no_risk|low|moderate|crisis", \
"reason": "<20 words"}"""
PROMPT_SHA = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]


class LLMOut(BaseModel):
    risk_level: Literal["no_risk", "low", "moderate", "crisis"]
    reason: str = Field(default="", max_length=400)

    @field_validator("risk_level", mode="before")
    @classmethod
    def _norm(cls, v):
        return str(v).strip().lower().replace(" ", "_").replace("-", "_")


def load_env() -> None:
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_last_call = [0.0]


def call_groq(channel: str, text: str, key: str, model: str):
    wait = MIN_INTERVAL_S - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    body = {"model": model, "temperature": 0, "seed": SEED, "max_tokens": 800,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": f"Channel: {channel}\nMessage:\n{text}"}]}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": "bh-agent-eval/1.0"})
    t0 = time.time()
    _last_call[0] = t0
    with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"], data.get("usage", {}), time.time() - t0


def classify(row, key: str, model: str, fallback: M1Rules) -> dict:
    ck = hashlib.sha256(f"{model}|{PROMPT_SHA}|{row.channel}|{row.text}".encode()).hexdigest()
    cpath = CACHE_DIR / f"{ck}.json"
    if cpath.exists():
        d = json.loads(cpath.read_text())
        return {**d, "cached": 1, "latency_s": 0.0, "attempts": 0, "fallback": 0}
    attempts, last_err = 0, ""
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        try:
            content, usage, lat = call_groq(row.channel, row.text, key, model)
            out = LLMOut.model_validate_json(content)
            d = {"llm_level": out.risk_level, "reason": out.reason,
                 "prompt_tokens": usage.get("prompt_tokens", 0),
                 "completion_tokens": usage.get("completion_tokens", 0), "model": model}
            cpath.write_text(json.dumps(d))
            return {**d, "cached": 0, "latency_s": lat, "attempts": attempts, "fallback": 0}
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            if e.code in (400, 401, 403, 404):
                detail = e.read().decode()[:300]
                raise SystemExit(f"Groq refused the request ({e.code}): {detail}\n"
                                 "Check GROQ_API_KEY / GROQ_MODEL in .env.")
            wait = float(e.headers.get("retry-after", 15 * attempts)) if e.code == 429 else 3 * attempts
            time.sleep(min(wait, 60))
        except (ValidationError, json.JSONDecodeError, KeyError) as e:
            last_err = f"parse: {type(e).__name__}"
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = f"net: {getattr(e, 'reason', e)}"
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                raise SystemExit(f"SSL certificate problem: {e}\nRun: pip install certifi")
            if attempts == MAX_ATTEMPTS:
                print(f"  network failure after {attempts} tries: {last_err}")
            time.sleep(3 * attempts)
    return {"llm_level": fallback.predict_class(row.text), "reason": f"FALLBACK ({last_err})",
            "prompt_tokens": 0, "completion_tokens": 0, "model": model, "cached": 0,
            "latency_s": 0.0, "attempts": attempts, "fallback": 1}


def setup():
    load_env()
    key, model = os.environ.get("GROQ_API_KEY", ""), os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not key:
        raise SystemExit("GROQ_API_KEY not found in .env (see Step 1).")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PROCESSED / "conversations_split.csv")
    fb = M1Rules().fit(df[df.split == "train"])
    return key, model, df, fb


def smoke() -> None:
    key, model, df, fb = setup()
    val = df[df.split == "val"]
    pos = val[val.risk_level == "crisis"].sample(5, random_state=SEED)
    neg = val[val.risk_level != "crisis"].sample(5, random_state=SEED)
    print(f"model {model} | prompt sha {PROMPT_SHA} | VALIDATION rows only")
    for r in pd.concat([pos, neg]).itertuples():
        o = classify(r, key, model, fb)
        print(f"[{r.risk_level:9s} -> {o['llm_level']:9s}] fb={o['fallback']} cached={o['cached']} "
              f"{o['latency_s']:.2f}s tok={o['prompt_tokens']}+{o['completion_tokens']} | "
              f"{r.text[:70]!r} | {o['reason'][:60]}")


def metrics_for(name: str, g: pd.DataFrame, flag_col: str, n_nc_test: int, n_test: int) -> dict:
    y = g.crisis.to_numpy().astype(bool)
    f = g[flag_col].to_numpy().astype(bool)
    tp, fn = int((y & f).sum()), int((y & ~f).sum())
    fp, n_nc = int((~y & f).sum()), int((~y).sum())
    fp_est = fp / n_nc * n_nc_test
    lo, hi = wilson_ci(tp, tp + fn)
    return {"method": name, "tp": tp, "fn": fn, "fp_in_sample": fp, "n_noncrisis_sample": n_nc,
            "recall": tp / (tp + fn), "recall_lo": lo, "recall_hi": hi,
            "fp_est_full_test": fp_est, "precision_reweighted": tp / (tp + fp_est) if tp + fp_est else float("nan"),
            "cost_per_1000_reweighted": 1000 * (COST_FN_CRISIS * fn + COST_FP_CRISIS * fp_est) / n_test}


def run() -> None:
    key, model, df, fb = setup()
    test = df[df.split == "test"].copy()
    test["crisis"] = (test.risk_level == "crisis").astype(int)
    tp_path = RESULTS / "test_predictions.csv"
    preds = pd.read_csv(tp_path)
    need = {"conversation_id", "flag_M1", "flag_M2", "flag_M3"}
    if not need.issubset(preds.columns):
        raise SystemExit(f"test_predictions.csv is missing {need - set(preds.columns)}; columns: {list(preds.columns)}")
    n_test, n_nc_test = len(test), int((test.crisis == 0).sum())
    rng = np.random.default_rng(SEED)
    pos = test[test.crisis == 1]
    neg_pool = test[test.crisis == 0]
    neg = neg_pool.iloc[np.sort(rng.choice(len(neg_pool), N_GOLDEN - len(pos), replace=False))]
    gold = pd.concat([pos, neg]).sort_values("conversation_id").reset_index(drop=True)
    gold[["conversation_id"]].to_csv(RESULTS / "llm_golden_ids.csv", index=False)
    print(f"golden subset: {len(gold)} cases = {len(pos)} crisis + {len(neg)} non-crisis | "
          f"test n={n_test}, non-crisis={n_nc_test} | model {model} | prompt sha {PROMPT_SHA}")

    rows = []
    for i, r in enumerate(gold.itertuples(), 1):
        o = classify(r, key, model, fb)
        rows.append({"conversation_id": r.conversation_id, "true": r.risk_level, **o})
        if i % 25 == 0:
            print(f"  {i}/{len(gold)} done (live calls so far: {sum(1 - x['cached'] for x in rows)})")
    out = pd.DataFrame(rows)
    out["flag_LLM"] = (out.llm_level == "crisis").astype(int)
    out = out.merge(preds[["conversation_id", "flag_M1", "flag_M2", "flag_M3"]], on="conversation_id", how="left")
    assert out[["flag_M1", "flag_M2", "flag_M3"]].notna().all().all(), "merge with test_predictions failed"
    out["crisis"] = (out.true == "crisis").astype(int)
    out.to_csv(RESULTS / "llm_predictions.csv", index=False)
    out[["conversation_id", "model", "cached", "attempts", "fallback", "latency_s",
         "prompt_tokens", "completion_tokens"]].to_csv(RESULTS / "llm_log.csv", index=False)

    ms = pd.DataFrame([metrics_for(n, out, c, n_nc_test, n_test) for n, c in
                       [("M1_rules", "flag_M1"), ("M2_tfidf_lr", "flag_M2"),
                        ("M3_xgb", "flag_M3"), (f"LLM_{model}", "flag_LLM")]])
    ms.to_csv(RESULTS / "llm_metrics.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n--- crisis decisions on the golden subset (recall is exact: all 39 test crises included; "
          "precision/cost re-weighted to the full test set) ---")
    print(ms.round(3).to_string(index=False))

    crisis_rows = out[out.crisis == 1]
    mc = mcnemar_exact(crisis_rows.flag_LLM, crisis_rows.flag_M2)
    print(f"\nMcNemar exact, LLM vs M2 on the {len(crisis_rows)} true crises: LLM-only detects "
          f"{mc['a_only']}, M2-only detects {mc['b_only']}, p={mc['p_value']:.3f}")
    live = out[out.cached == 0]
    print(f"fallbacks to M1 rules: {int(out.fallback.sum())}/{len(out)} | retried calls: "
          f"{int((out.attempts > 1).sum())} | live calls this run: {len(live)}")
    if len(live):
        print(f"latency median {live.latency_s.median():.2f}s, p95 {live.latency_s.quantile(.95):.2f}s | "
              f"tokens in/out: {int(out.prompt_tokens.sum())}/{int(out.completion_tokens.sum())} | "
              f"cost on the free tier: $0.00")
    print("4-class agreement with recorded label (LLM): "
          f"{(out.llm_level == out.true).mean():.3f} (subset is crisis-enriched, not deployment mix)")
    (RESULTS / "llm_summary.json").write_text(json.dumps({
        "model": model, "prompt_sha": PROMPT_SHA, "n_golden": len(out), "n_crisis": int(out.crisis.sum()),
        "fallbacks": int(out.fallback.sum()), "mcnemar_llm_vs_m2": mc,
        "metrics": ms.to_dict(orient="records")}, indent=2, default=float))
    print("\nwrote results/llm_predictions.csv, llm_metrics.csv, llm_log.csv, llm_summary.json, llm_golden_ids.csv")


def main() -> None:
    set_seed()
    ensure_dirs()
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--run", action="store_true")
    a = ap.parse_args()
    smoke() if a.smoke else run()


if __name__ == "__main__":
    main()
