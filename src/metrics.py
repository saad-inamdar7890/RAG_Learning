"""
Lightweight in-memory metrics store.
Tracks per-request latencies, step-level traces, and token counts.
Exposes p50/p95/p99 percentiles for the /api/metrics endpoint.
"""
import time
import threading
from collections import deque
from typing import Dict, List, Optional
import statistics

# Rolling window — keep last N requests
_WINDOW = 200

_lock = threading.Lock()

# Each entry: {"total_s": float, "steps": {...}, "tokens": int, "ts": float}
_requests: deque = deque(maxlen=_WINDOW)


def record_request(total_s: float, steps: Dict[str, float], tokens: int = 0) -> None:
    """Call this after every /api/ask request completes."""
    with _lock:
        _requests.append({
            "total_s": round(total_s, 3),
            "steps": {k: round(v, 3) for k, v in steps.items()},
            "tokens": tokens,
            "ts": time.time(),
        })


def get_summary() -> dict:
    """Return aggregated stats over the rolling window."""
    with _lock:
        data = list(_requests)

    if not data:
        return {
            "total_requests": 0,
            "latency": {},
            "avg_tokens_per_request": 0,
            "step_averages_s": {},
        }

    totals = [r["total_s"] for r in data]
    totals_sorted = sorted(totals)
    n = len(totals_sorted)

    def percentile(sorted_vals, p):
        idx = max(0, int(len(sorted_vals) * p / 100) - 1)
        return round(sorted_vals[idx], 3)

    # Aggregate step timings
    step_sums: Dict[str, List[float]] = {}
    for r in data:
        for step, t in r.get("steps", {}).items():
            step_sums.setdefault(step, []).append(t)
    step_avgs = {k: round(statistics.mean(v), 3) for k, v in step_sums.items()}

    avg_tokens = round(statistics.mean(r["tokens"] for r in data), 1)

    return {
        "total_requests": len(data),
        "window_size": _WINDOW,
        "latency": {
            "p50_s":  percentile(totals_sorted, 50),
            "p95_s":  percentile(totals_sorted, 95),
            "p99_s":  percentile(totals_sorted, 99),
            "min_s":  round(min(totals), 3),
            "max_s":  round(max(totals), 3),
            "mean_s": round(statistics.mean(totals), 3),
        },
        "avg_tokens_per_request": avg_tokens,
        # Local Ollama has no $ cost, but we can estimate relative compute units
        "estimated_cost_per_request_usd": round(avg_tokens * 0.000002, 6),
        "step_averages_s": step_avgs,
        "recent_requests": [
            {"total_s": r["total_s"], "tokens": r["tokens"]}
            for r in data[-10:]
        ],
    }
