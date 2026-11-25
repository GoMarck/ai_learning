"""Report Qwen3 TP8 xllm_run results. Place next to xllm_run.py."""
import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import xllm_run as suite

BATCH_BEGIN = re.compile(r"phase=worker_get_begin request=(\S+) batch_index=(\d+)")
TID = re.compile(r"\d\d:\d\d:\d\d\.\d+\s+(\d+)\s")
TPOT = re.compile(r"avg tpot: ([\d.]+)ms")


def scenario(backend, cached, plans, completions, batches, shape):
    if backend == "none":
        if any(batches) or completions or plans:
            return "unconfirmed"
        return "local_reuse" if cached else "none_no_reuse"
    if len(plans) != 1 or len(completions) != 1:
        return "unconfirmed"
    host = plans[0]
    shared, quiescent = completions[0]
    if not quiescent or not all(batches):
        return "unconfirmed"
    counts = []
    for batch in batches:
        if sorted(batch) != list(range(len(batch))):
            return "unconfirmed"
        counts.append(sum(h for h, calls in batch.values()))
    if min(counts) < shared:
        return "unconfirmed"
    if host == 0 and cached == 0 and shared == 0:
        return "backend_miss" if max(counts) == 0 else "rank_hits_no_reuse"
    if host == 0 and cached == shared * suite.BLOCK and min(counts) == max(counts) == shared:
        if shared == shape.blocks:
            for batch in batches:
                units = [batch[i][0] for i in sorted(batch)]
                if not shape.valid_hits(units):
                    return "unconfirmed"
            return f"backend_hit{shape.blocks}"
        return "other_backend_hit"
    return "local_or_mixed" if cached else "unconfirmed"


def report(folder, backend, phase, shape, output):
    outlen = 1 if phase == "warm" else 1024
    planned = defaultdict(list)
    completed = defaultdict(list)
    ends = {}
    by_rank = [defaultdict(dict) for _ in suite.RANKS]
    samples = defaultdict(list)
    for rank in (range(1) if backend == "none" else suite.RANKS):
        path = folder / f"node_{rank}.log"
        active = {}
        inits = 0
        store_modes = []
        with path.open(errors="replace") as stream:
            for line in stream:
                if "[KVCacheStore] initialize," in line:
                    if (f"backend={backend}," not in line or f"worker_rank={rank}," not in line
                            or f"tp_rank={rank}, tp_size=8," not in line):
                        raise ValueError(f"unexpected backend or TP rank in {path}")
                    inits += 1
                if rank == 0:
                    if "Master init options:" in line:
                        store_modes.append(f"enable_kvcache_store: {int(backend != 'none')}," in line)
                    if m := suite.END.search(line):
                        if int(m[2]) == outlen:
                            if m[1] in ends or not (t := TPOT.search(line)):
                                raise ValueError(f"duplicate/incomplete request record: {m[1]}")
                            ends[m[1]] = (m[3], int(m[4]), int(m[5]), int(m[6]), float(m[7]), float(t[1]))
                    elif m := suite.PLAN.search(line):
                        planned[m[1]].append(int(m[2]))
                    elif m := suite.DONE.search(line):
                        completed[m[1]].append((int(m[2]), int(m[3])))
                if m := suite.PUT_CALL.search(line):
                    r, blocks, ok, failed, us = map(int, m.groups())
                    if r != rank or ok + failed != blocks:
                        raise ValueError(f"bad PUT record: {path}: {line}")
                    samples["stage", "store_kvc", "failed" if failed else "ok", blocks].append(us / 1000)
                elif m := BATCH_BEGIN.search(line):
                    tid = TID.search(line)[1]
                    if tid in active:
                        raise ValueError(f"overlapping GET in one thread: {path}")
                    active[tid] = (m[1], int(m[2]), [])
                elif m := suite.GET_CALL.search(line):
                    r, blocks, hit, us = map(int, m.groups())
                    tid = TID.search(line)[1]
                    if r != rank or tid not in active or hit > blocks or blocks == 0:
                        raise ValueError(f"unmatched GET; check worker_service=2: {path}: {line}")
                    active[tid][2].append((blocks, hit, us))
                elif m := suite.GET.search(line):
                    tid = TID.search(line)[1]
                    rid, index, hit = m[1], int(m[2]), int(m[3])
                    begin = active.pop(tid, None)
                    if not begin or begin[:2] != (rid, index) or index in by_rank[rank][rid]:
                        raise ValueError(f"unmatched GET end: {path}: {line}")
                    if not begin[2]:
                        raise ValueError(f"missing PrefetchGet elapsed_us: {path}: {line}")
                    by_rank[rank][rid][index] = (hit, begin[2])
        if active:
            raise ValueError(f"incomplete worker GET in {path}")
        if inits != int(backend != "none") or (rank == 0 and store_modes != [True]):
            raise ValueError(f"missing or inconsistent xLLM startup records in {path}")
    rows = []
    counts = Counter()
    for rid, (finish, prompt, cached, generated, ttft, tpot) in ends.items():
        if (finish != "length" or generated != outlen
                or abs(prompt - (shape.warm if phase == "warm" else shape.input_tokens)) >= suite.BLOCK):
            raise ValueError(f"unexpected request shape/finish: {rid}: {ends[rid]}")
        batches = [by_rank[r].get(rid, {}) for r in suite.RANKS]
        label = scenario(backend, cached, planned[rid], completed[rid], batches, shape)
        counts[label] += 1
        samples[label, "TTFT", "", ""].append(ttft)
        if generated > 1:
            samples[label, "TPOT", "", ""].append(tpot)
        for batch in batches:
            for units, calls in batch.values():
                for blocks, hit, us in calls:
                    status = "miss" if hit == 0 else "hit" if hit == blocks else "partial"
                    samples[label, "load_kvc", status, blocks].append(us / 1000)
        rows.append((rid, label, prompt, cached, generated, ttft, tpot if generated > 1 else ""))
    if len(ends) != shape.requests:
        raise ValueError(f"{folder}: expected {shape.requests} completed requests, found {len(ends)}")
    output.mkdir(exist_ok=True, parents=True)
    def save(name, header, records):
        with (output / name).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(records)
    save("request_metrics.csv", ["request_id", "scenario", "prompt_tokens", "cached_tokens",
                                 "output_tokens", "ttft_ms", "tpot_ms"], rows)
    data = []
    for (label, metric, status, blocks), xs in sorted(samples.items()):
        data.append((label, metric, status, blocks, len(xs),
                     *(round(n, 3) for n in suite.latency_stats(xs))))
    header = ["scenario", "metric", "batch_result", "blocks_per_call", "n", "avg_ms", "p50_ms", "p99_ms"]
    save("latency_summary.csv", header, data)
    print(f"\n{folder}: {dict(counts)}")
    for row in [header, *data]:
        print("\t".join(map(str, row)))
    if backend == "none":
        print("load_kvc / store_kvc: N/A (backend disabled)")
    if phase == "warm":
        print("TPOT: N/A (one output token)")
    return counts


def cold(root, backend):
    suite.drive(root, backend, "formal", suite.config())
    folder = root / backend / "formal"
    if not (folder / "node_0.log").is_file():
        print("Cold requests sent. Check backend_miss in the xLLM container using stats --log-dir.")
        return
    shape = suite.workload(root)
    counts = report(folder, backend, "formal", shape, folder)
    if counts != {"backend_miss": shape.requests}:
        raise SystemExit(f"Cold control requires {shape.requests} confirmed backend misses: {dict(counts)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stats")
    p.add_argument("root", type=Path, nargs="?")
    p.add_argument("--log-dir", type=Path)
    suite.add_shape(p)
    p.add_argument("--phase", choices=["warm", "formal"], default="formal")
    p.add_argument("--backend", choices=suite.BACKENDS, action="append")
    p = sub.add_parser("cold")
    p.add_argument("backend", choices=["mooncake", "yuanrong"])
    p.add_argument("root", type=Path)
    a = parser.parse_args()
    root = a.root.resolve() if a.root is not None else None
    if a.cmd == "cold":
        cold(root, a.backend)
        return
    if a.log_dir:
        if len(a.backend or []) != 1:
            parser.error("--log-dir requires one --backend")
        b = a.backend[0]
        report(a.log_dir, b, a.phase, suite.workload(root, a.requests, a.input_tokens),
               (root or Path.cwd()) / b / a.phase)
        return
    if root is None:
        parser.error("provide dataset root or --log-dir")
    names = a.backend or [b for b in suite.BACKENDS if (root / b / a.phase).is_dir()]
    if not names:
        parser.error(f"no {a.phase} rank log directories under {root}")
    for b in names:
        folder = root / b / a.phase
        report(folder, b, a.phase, suite.workload(root, a.requests, a.input_tokens), folder)


if __name__ == "__main__":
    main()
