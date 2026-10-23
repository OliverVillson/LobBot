"""Where did the time go? Per-stage wall time from a REAPer / Harness job's events.log.

    python stage_times.py ~/.reaper/jobs/<job>/events.log

Prints each stage's first-to-last event span, its share of the run, and the throughput
messages it logged (score's "tok/s", teacher's "answered n/m"), so optimizations can be
ranked on measured numbers from a real run before any code changes.
"""
import json
import sys
from collections import OrderedDict

def main(path):
    spans, last_msg = OrderedDict(), {}
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        st, ts = e.get("stage"), e.get("ts")
        if not st or ts is None or st == "pipeline":
            continue
        a, b = spans.get(st, (ts, ts))
        spans[st] = (min(a, ts), max(b, ts))
        if e.get("msg"):
            last_msg[st] = e["msg"]
    total = sum(b - a for a, b in spans.values()) or 1
    for st, (a, b) in spans.items():
        d = b - a
        print(f"{st:10s} {d / 60:7.1f} min  {d / total:5.0%}  {last_msg.get(st, '')[:70]}")
    print(f"{'total':10s} {total / 60:7.1f} min")

if __name__ == "__main__":
    main(sys.argv[1])
