"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # Extension 2: MBU Right-sizing recommendations (local logic in M1)
    mbu_suggestions = []
    for r in summary:
        gtype = r["gpu_type"]
        mbu = float(r.get("mbu", 0))
        cur_cost = num(cat[gtype]["on_demand_hr"])
        cur_bw = num(cat[gtype]["peak_bw_tbs"])
        cur_vram = num(cat[gtype]["hbm_gb"])
        achieved_bw = mbu * cur_bw

        # Compare with other GPU types in catalog
        candidates = []
        for alt_type, alt_specs in cat.items():
            alt_cost = num(alt_specs["on_demand_hr"])
            alt_bw = num(alt_specs["peak_bw_tbs"])
            alt_vram = num(alt_specs["hbm_gb"])
            if alt_cost < cur_cost and alt_bw >= achieved_bw:
                savings_hr = cur_cost - alt_cost
                candidates.append({
                    "suggested_gpu": alt_type,
                    "hourly_savings": round(savings_hr, 2),
                    "savings_pct": round((savings_hr / cur_cost) * 100, 1),
                    "target_mbu": round(achieved_bw / alt_bw, 3) if alt_bw > 0 else 0.0,
                    "vram_gb": alt_vram,
                })
        if candidates:
            best = max(candidates, key=lambda x: x["hourly_savings"])
            mbu_suggestions.append({
                "gpu_id": r["gpu_id"],
                "current_type": gtype,
                "current_mbu": mbu,
                "suggested_type": best["suggested_gpu"],
                "target_mbu": best["target_mbu"],
                "hourly_savings": best["hourly_savings"],
                "savings_pct": best["savings_pct"],
                "monthly_savings": round(best["hourly_savings"] * 24 * 30, 2),
            })

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        if mbu_suggestions:
            print("\n-- [Extension 2] MBU Right-Sizing Recommendations --")
            print(f"{'GPU':14}{'Current':9}{'MBU':>6}{'Suggest':9}{'Tgt MBU':>8}{'Save/hr':>9}{'Save/mo':>10}")
            for sug in mbu_suggestions:
                print(f"{sug['gpu_id']:14}{sug['current_type']:9}{sug['current_mbu']:>6.3f}"
                      f"{sug['suggested_type']:9}{sug['target_mbu']:>8.3f}"
                      f"${sug['hourly_savings']:>7.2f} ${sug['monthly_savings']:>8.2f}")

    return {
        "summary": summary,
        "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "mbu_suggestions": mbu_suggestions,
    }


if __name__ == "__main__":
    run()
