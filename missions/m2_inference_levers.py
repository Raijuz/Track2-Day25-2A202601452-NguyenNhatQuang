"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # Extension 4: Reasoning budget analysis (local implementation)
    from finops import sustainability
    r_queries = s_queries = 0
    r_cost = s_cost = 0.0
    r_wh = s_wh = 0.0
    for r in rows:
        inp = int(num(r["input_tokens"]))
        out = int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r["is_reasoning"])))
        pin, pout = MODEL_PRICES[r["route_tier"]]
        req_cost = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        tot_tok = inp + out
        req_wh = sustainability.wh_per_query(tot_tok, is_reasoning=is_reasoning)
        if is_reasoning:
            r_queries += 1
            r_cost += req_cost
            r_wh += req_wh
        else:
            s_queries += 1
            s_cost += req_cost
            s_wh += req_wh

    total_wh = r_wh + s_wh
    reasoning_energy_pct = (r_wh / total_wh * 100) if total_wh else 0.0
    reasoning_cost_pct = (r_cost / (r_cost + s_cost) * 100) if (r_cost + s_cost) else 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n-- [Extension 4] Reasoning vs Standard Query Budget --")
        print(f"Reasoning queries: {r_queries} ({r_queries/len(rows)*100:.1f}%) | Cost: ${r_cost:.2f}/day ({reasoning_cost_pct:.1f}%) | Energy: {r_wh:.1f} Wh ({reasoning_energy_pct:.1f}%)")
        print(f"Standard queries : {s_queries} | Cost: ${s_cost:.2f}/day | Energy: {s_wh:.1f} Wh")
        print("Recommendation   : Gate reasoning via confidence threshold routing to avoid 80x energy explosion.")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "reasoning_queries": r_queries,
        "reasoning_cost_usd": round(r_cost, 2),
        "reasoning_energy_wh": round(r_wh, 1),
    }


if __name__ == "__main__":
    run()
