"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # Extension 5: Carbon-aware scheduling for interruptible workloads (local implementation)
    from finops.sustainability import REGION_CARBON, REGION_PRICE_KWH
    base_region, green_region = "us-east-1", "europe-north1"
    base_carbon_total_g = green_carbon_total_g = 0.0
    base_energy_cost_total = green_energy_cost_total = 0.0
    interruptible_jobs = []

    for j in jobs:
        if bool(int(num(j["interruptible"]))):
            gtype = j["gpu_type"]
            ngpu = int(num(j["num_gpus"]))
            hpd = num(j["hours_per_day"])
            days = num(j.get("days", 30))
            watts = num(cat[gtype]["watts"])
            total_kwh = (hpd * days * ngpu * watts) / 1000.0

            base_c = total_kwh * REGION_CARBON.get(base_region, 380)
            green_c = total_kwh * REGION_CARBON.get(green_region, 30)
            base_e = total_kwh * REGION_PRICE_KWH.get(base_region, 0.12)
            green_e = total_kwh * REGION_PRICE_KWH.get(green_region, 0.09)

            base_carbon_total_g += base_c
            green_carbon_total_g += green_c
            base_energy_cost_total += base_e
            green_energy_cost_total += green_e

            interruptible_jobs.append({
                "job_id": j["job_id"], "gpu_type": gtype, "kwh": round(total_kwh, 1),
                "saved_carbon_kg": round((base_c - green_c) / 1000.0, 2),
            })

    carbon_saved_kg = (base_carbon_total_g - green_carbon_total_g) / 1000.0
    carbon_reduction_pct = (carbon_saved_kg / (base_carbon_total_g / 1000.0) * 100) if base_carbon_total_g else 0.0
    energy_cost_saved = base_energy_cost_total - green_energy_cost_total

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n-- [Extension 5] Carbon-Aware Scheduling (us-east-1 -> europe-north1) --")
        print(f"Baseline Carbon: {base_carbon_total_g/1000.0:,.1f} kg CO2e  |  "
              f"Green Carbon: {green_carbon_total_g/1000.0:,.1f} kg CO2e  |  "
              f"Saved: {carbon_saved_kg:,.1f} kg CO2e ({carbon_reduction_pct:.1f}%)")
        print(f"Electricity cost saved: ${energy_cost_saved:,.2f}")

    return {
        "recommendations": recs,
        "on_demand_monthly": round(on_demand_monthly),
        "optimized_monthly": round(optimized_monthly),
        "savings_pct": round(savings_pct, 1),
        "carbon_saved_kg": round(carbon_saved_kg, 2),
        "carbon_reduction_pct": round(carbon_reduction_pct, 1),
        "energy_cost_saved_usd": round(energy_cost_saved, 2),
    }


if __name__ == "__main__":
    run()
