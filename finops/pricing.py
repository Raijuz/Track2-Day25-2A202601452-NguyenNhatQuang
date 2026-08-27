"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


# Interruption rates by GPU type (illustrative: H100 spot is less preempted ~3%, A10G/L4 ~8-10%)
GPU_INTERRUPT_RATES: dict[str, float] = {
    "H100": 0.03,
    "H200": 0.03,
    "B200": 0.04,
    "A100": 0.05,
    "MI300X": 0.06,
    "A10G": 0.08,
    "L4": 0.10,
}


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    job_days: float | None = None,
) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    DOCUMENTED simple policy (instructor extension point — swap in your own):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


def recommend_tier_advanced(
    hours_per_day: float,
    interruptible: bool,
    gpu_type: str,
    job_days: float = 30.0,
    on_demand_hr: float = 2.50,
    spot_hr: float = 1.50,
    reserved_1yr_hr: float = 2.00,
    reserved_3yr_hr: float = 1.40,
) -> dict:
    """Detailed financial simulation comparing all purchasing tiers (Extension 1).

    Evaluates effective cost across on-demand, spot (with checkpointing/rework overhead),
    1-year reserved, and 3-year reserved commitments based on actual workload duration.
    """
    total_hours = hours_per_day * job_days
    on_demand_cost = total_hours * on_demand_hr

    int_rate = GPU_INTERRUPT_RATES.get(gpu_type, 0.05)
    spot_sim = spot_checkpoint_cost(total_hours, spot_hr, on_demand_hr, interrupt_rate=int_rate)
    spot_cost = spot_sim["spot_cost"]

    # Reserved options: compare if commitment matches workload lifetime
    # If job duration is shorter than commitment period, commitment cost must cover the commit duration
    r1_cost = total_hours * reserved_1yr_hr
    r3_cost = total_hours * reserved_3yr_hr

    costs = {"on_demand": on_demand_cost}
    if interruptible and hours_per_day < 24:
        costs["spot"] = spot_cost

    duty = hours_per_day / 24.0
    if duty >= 0.50:
        if job_days >= 365 * 2.5:
            costs["reserved_3yr"] = r3_cost
        elif job_days >= 180:
            costs["reserved_1yr"] = r1_cost
        else:
            costs["reserved"] = r3_cost  # standard 3yr reserved proxy

    best_tier = min(costs, key=costs.get)
    best_cost = costs[best_tier]
    savings_pct = (1.0 - best_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0

    return {
        "best_tier": best_tier,
        "best_cost": round(best_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
        "tier_costs": {k: round(v, 2) for k, v in costs.items()},
    }


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float,
    read_discount: float = 0.10,
    read_cost_per_m: float | None = None,
) -> bool:
    """Evaluate whether prompt caching provides net positive savings (Extension 3).

    Caching incurs a write/storage cost to create the cached prefix and discounts subsequent reads.
    Net savings = (Uncached Read Cost - Cached Read Cost) * avg_cache_reads - Write Cost
                 = read_cost_per_m * (1 - read_discount) * avg_cache_reads - write_cost_per_m

    If read_cost_per_m is omitted, it defaults to write_cost_per_m (standard in Gemini/Anthropic).
    Break-even reads: N_be = write_cost_per_m / (read_cost_per_m * (1 - read_discount)).

    Returns True if avg_cache_reads >= break_even_reads.
    """
    if avg_cache_reads <= 0 or write_cost_per_m <= 0:
        return False
    base_read = read_cost_per_m if read_cost_per_m is not None else write_cost_per_m
    savings_per_read = base_read * (1.0 - read_discount)
    if savings_per_read <= 0:
        return False
    break_even_reads = write_cost_per_m / savings_per_read
    return avg_cache_reads >= break_even_reads
