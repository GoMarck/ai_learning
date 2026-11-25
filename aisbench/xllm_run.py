"""Run from the user's aisbench_auto_tools_prefix checkout on the xLLM host."""
import argparse
import csv
import hashlib
import json
import random
import re
import secrets
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PAIRS, BLOCK, FORMAL = 20, 128, 20480
RANKS = range(8)
BACKENDS = ("none", "mooncake", "yuanrong")
WORDS = "red green blue gray black brown cat dog fox lake hill rock sea ship bird tree road field sky star".split()


@dataclass(frozen=True)
class Workload:
    requests: int = PAIRS
    input_tokens: int = FORMAL

    def __post_init__(self):
        if self.requests < 1 or self.input_tokens < 2 * BLOCK or self.input_tokens % (2 * BLOCK):
            raise ValueError("requests must be positive and input-tokens a positive multiple of 256")

    @property
    def prefix(self):
        return self.input_tokens // 2

    @property
    def warm(self):
        return self.prefix + 32

    @property
    def blocks(self):
        return self.prefix // BLOCK

    def valid_hits(self, batches):
        full, tail = divmod(self.blocks, 16)
        hits = [16] * full + ([tail] if tail else [])
        return batches == hits or batches == hits + [0]


def workload(root=None, requests=None, input_tokens=None):
    manifest = {}
    if root is not None and (root / "inputs.json").is_file():
        manifest = json.loads((root / "inputs.json").read_text())
    return Workload(requests if requests is not None else manifest.get("requests", PAIRS),
                    input_tokens if input_tokens is not None else manifest.get("input_tokens", FORMAL))


def add_shape(parser):
    parser.add_argument("--requests", type=int)
    parser.add_argument("--input-tokens", type=int)


def config():
    if not Path("aisbench_test.py").is_file():
        raise SystemExit("Run in the directory containing aisbench_test.py and config.py")
    sys.path.insert(0, str(Path.cwd()))
    import config as cfg
    return cfg


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenizer_files(model_path):
    # Qwen3 fast tokenizer data, special tokens and optional standalone chat template.
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "added_tokens.json", "chat_template.jinja")
    root = Path(model_path).expanduser()
    return {name: digest(root / name) for name in names if (root / name).is_file()}


def generate(root, shape, model_path=None):
    from transformers import AutoTokenizer
    model_name = None
    if model_path is None:
        cfg = config()
        model_path, model_name = cfg.MODEL_PATH, cfg.MODEL_NAME
    model_path = Path(model_path).expanduser().resolve(strict=True)
    tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    def ids(s):
        return tok.encode(s, add_special_tokens=False)
    def framed(s):
        return tok.apply_chat_template([{"role": "user", "content": s}],
                                       tokenize=True, add_generation_prompt=True, return_dict=False)
    words = [" " + w for w in WORDS if len(ids(" " + w)) == 1]
    if len(words) < 5:
        raise ValueError("Expected Qwen3 vocabulary")
    salt = secrets.token_hex(12)
    rng = random.Random(salt)
    def extend(base, target):
        count = target - len(ids(base))
        tail = rng.choices(words, k=count)
        for _ in range(5):
            s = base + "".join(tail)
            delta = target - len(ids(s))
            if delta == 0:
                return s
            if delta < 0:
                del tail[delta:]
            else:
                tail.extend(rng.choices(words, k=delta))
        raise ValueError("Could not construct exact token length")
    rows = {"warm": [], "formal": []}
    first_blocks = set()
    for i in range(shape.requests):
        head = extend(f"Pair {i:02d} run {salt}.", shape.prefix)
        warm = extend(head + " north", shape.warm)
        formal = extend(head + " south", shape.input_tokens)
        for a, b in ((ids(warm), ids(formal)), (framed(warm), framed(formal))):
            common = next((n for n, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            if common // BLOCK != shape.blocks or len(a) // BLOCK != shape.blocks:
                raise ValueError(f"pair {i}: shared={common}, warm_len={len(a)}")
        first = tuple(b[:BLOCK])
        if first in first_blocks:
            raise ValueError("Cross-pair shared first block")
        first_blocks.add(first)
        for phase, s in (("warm", warm), ("formal", formal)):
            rows[phase].append(json.dumps({"question": s, "answer": "none"}))
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"model": model_name, "model_path": str(model_path), "tokenizer_files": tokenizer_files(model_path), "seed": salt, "requests": shape.requests, "input_tokens": shape.input_tokens, "sha256": {}}
    for phase, lines in rows.items():
        path = root / (phase + ".jsonl")
        path.write_text("\n".join(lines) + "\n")
        manifest["sha256"][path.name] = digest(path)
    (root / "inputs.json").write_text(json.dumps(manifest, indent=2))
    print(f"{root}: {shape.requests} pairs, {shape.blocks} common full blocks per pair.\n{manifest['sha256']}")


END = re.compile(r"\brequest_id: ([^,]+), sequence 0, max_tokens: (\d+),.*?"
                 r"finish_reason: ([^,]+), prompt_tokens: (\d+), num_prefix_cache_tokens: (\d+), "
                 r"generated_tokens: (\d+), ttft: ([\d.]+)ms")
PUT = re.compile(r"\[OffloadPut\] worker_rank=(\d+), tp_rank=(\d+), tp_size=8,.*?success=(\d+), failed=(\d+)")
GET = re.compile(r"phase=worker_get_end request=(\S+) batch_index=(\d+) hit_units=(\d+)")
PLAN = re.compile(r"phase=plan_ready request=(\S+) dp_rank=0 host_tokens=(\d+)")
DONE = re.compile(r"phase=completion_available request=(\S+) dp_rank=0 hit_units=(\d+) store_units=\d+ quiescent=(\d+)")


GET_CALL = re.compile(r"\[PrefetchGet\] worker_rank=(\d+),.*?type=0, blocks=(\d+), hits=(\d+),.*?elapsed_us=(\d+)")
PUT_CALL = re.compile(r"\[OffloadPut\] worker_rank=(\d+),.*?store_write_enabled=1, blocks=(\d+), success=(\d+), failed=(\d+), elapsed_us=(\d+)")


def latency_stats(samples):
    ordered = sorted(samples)
    lo, rem = divmod(99 * (len(ordered) - 1), 100)
    p99 = ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * rem / 100
    return statistics.mean(ordered), statistics.median(ordered), p99


def store_latency_summary(folder, backend):
    if backend == "none":
        print("load_kvc / store_kvc: N/A (外部缓存已关闭)")
        return
    groups = {"load_kvc": defaultdict(list), "store_kvc": defaultdict(list)}
    logged = {op: set() for op in groups}
    for rank in RANKS:
        path = folder / f"node_{rank}.log"
        if not path.is_file():
            continue
        with path.open(errors="replace") as stream:
            for line in stream:
                if m := GET_CALL.search(line):
                    r, blocks, hits, us = map(int, m.groups())
                    if r != rank or blocks == 0 or hits > blocks:
                        raise ValueError(f"{path}: 无效的 PrefetchGet 记录")
                    op = "load_kvc"
                    result = "miss" if hits == 0 else "hit" if hits == blocks else "partial"
                elif m := PUT_CALL.search(line):
                    r, blocks, ok, failed, us = map(int, m.groups())
                    if r != rank or blocks != ok + failed:
                        raise ValueError(f"{path}: 无效的 OffloadPut 记录")
                    op = "store_kvc"
                    result = "failed" if failed else "ok"
                else:
                    continue
                logged[op].add(rank)
                groups[op][result, blocks].append(us / 1000)
    print(f"{backend} 后端批调用耗时（每个 rank 的一次调用为一个样本，单位 ms）")
    print("metric     result   blocks/call       N       avg       P50       P99")
    for op, batches in groups.items():
        if not batches:
            print(f"{op:<10} all               -       0       N/A       N/A       N/A")
        else:
            all_times = [v for values in batches.values() for v in values]
            for (result, blocks), values in [(("all", "-"), all_times), *sorted(batches.items())]:
                avg, p50, p99 = latency_stats(values)
                print(f"{op:<10} {result:<8} {str(blocks):>10} {len(values):7d} "
                      f"{avg:9.3f} {p50:9.3f} {p99:9.3f}")
        if len(logged[op]) < len(RANKS):
            print(f"{op} 有计时样本的 rank: {sorted(logged[op])}；TP8 汇总需检查 node_0..7.log 与 --v=1。")


def latency_header():
    print("backend    reuse     N   TTFT avg      P50      P99   TPOT avg      P50      P99  (ms)")


def latency_summary(folder, backend, shape):
    path = folder / "node_0.log"
    if not path.is_file():
        raise ValueError(f"当前容器看不到 {path}；可在日志容器运行："
                         f"python3 xllm_run.py summary --backend {backend} --log-dir ./log "
                         f"--requests {shape.requests} --input-tokens {shape.input_tokens}")
    groups = {"0%": [], "~50%": [], "other": []}
    ids = set()
    with path.open(errors="replace") as log:
        for line in log:
            m = END.search(line)
            if not m or m[2] != "1024":
                continue
            tpot = re.search(r"avg tpot: ([\d.]+)ms", line)
            if (not tpot or m[1] in ids or m[3] != "length" or int(m[6]) != 1024
                    or abs(int(m[4]) - shape.input_tokens) >= BLOCK):
                raise ValueError(f"{path}: 正式请求 {m[1]} 的完成记录、长度或计时异常")
            ids.add(m[1])
            cached = int(m[5])
            label = "0%" if cached == 0 else "~50%" if cached == shape.prefix else "other"
            groups[label].append((float(m[7]), float(tpot[1])))
    if len(ids) != shape.requests:
        raise ValueError(f"{path}: 找到 {len(ids)}/{shape.requests} 条正式请求，请检查本场日志")
    latency_header()
    for label, samples in groups.items():
        if label == "other" and not samples:
            continue
        values = []
        for times in zip(*samples):
            values.extend(f"{n:8.2f}" for n in latency_stats(times))
        print(f"{backend:<10} {label:<6} {len(samples):3d}   "
              + " ".join(values or [f"{'N/A':>8}"] * 6))
    store_latency_summary(folder, backend)


def wait_warm(folder, backend, shape):
    expected_puts = shape.requests * shape.blocks
    deadline = time.monotonic() + 240
    while True:
        log = (folder / "node_0.log").read_text(errors="replace")
        warm = [m for m in END.findall(log) if m[1] == "1"]
        if any(int(m[3]) // BLOCK != shape.blocks or int(m[4]) or int(m[5]) != 1 for m in warm):
            raise RuntimeError("Warm inputs must be cold, finish at 1 output token and have the expected full prefix blocks")
        puts = []
        for rank in RANKS:
            batches = PUT.findall((folder / f"node_{rank}.log").read_text(errors="replace"))
            if any(int(r) != rank or int(t) != rank or int(f) for r, t, s, f in batches):
                raise RuntimeError(f"Warm PUT error on rank {rank}")
            puts.append(sum(int(s) for r, t, s, f in batches))
        if len(warm) == len({m[0] for m in warm}) == shape.requests and (backend == "none" or puts == [expected_puts]*8):
            print("Warm complete; successful PUT blocks per rank:", puts, flush=True)
            return
        if time.monotonic() > deadline or max(puts) > expected_puts:
            raise RuntimeError(f"Warm barrier: {len(warm)} finished; successful PUT blocks per rank={puts}")
        time.sleep(2)


def drive(root, backend, phase, cfg):
    shape = workload(root)
    manifest = json.loads((root / "inputs.json").read_text())
    if manifest["model"] is not None and cfg.MODEL_NAME != manifest["model"]:
        raise SystemExit("MODEL_NAME in config.py differs from the dataset manifest")
    files = manifest.get("tokenizer_files")
    if files and "tokenizer.json" in files and "tokenizer_config.json" in files:
        if tokenizer_files(cfg.MODEL_PATH) != files:
            raise SystemExit("Tokenizer files in config.py MODEL_PATH differ from dataset generation")
    elif Path(cfg.MODEL_PATH).expanduser().resolve() != Path(manifest["model_path"]).resolve():
        raise SystemExit("MODEL_PATH in config.py differs from dataset generation")
    dataset = root / (phase + ".jsonl")
    if digest(dataset) != manifest["sha256"][dataset.name]:
        raise SystemExit("Dataset changed: " + str(dataset))
    folder = root / backend / phase
    folder.mkdir(parents=True, exist_ok=True)
    print(f"AISBench {backend}/{phase}: {folder / 'driver.log'}", flush=True)
    warm = phase == "warm"
    bench = [sys.executable, "-u", "aisbench_test.py", "--dataset_type", "normal", "--dataset",
             str(dataset), "--input_len", str(shape.warm if warm else shape.input_tokens),
             "--output_len", "1" if warm else "1024", "--data_num", str(shape.requests), "--concurrency", "4",
             "--test_type", "stream", "--request_rate", "0", "--repeat", "1", "--dp", "1", "--npu_num", "8"]
    (folder / "aisbench-command.txt").write_text(shlex.join(bench) + "\n")
    with (folder / "driver.log").open("wb") as log:
        with subprocess.Popen(bench, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
            while chunk := proc.stdout.read1(65536):
                log.write(chunk)
                log.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            status = proc.wait()
    snap = folder / "aisbench.log"
    if Path("aisbench.log").exists():
        shutil.copyfile("aisbench.log", snap)
    if status:
        raise subprocess.CalledProcessError(status, bench)
    text = snap.read_text(errors="replace")
    for key, expected in (("Success Requests", shape.requests), ("Failed Requests", 0)):
        values = re.findall(key + r"\s*[│|]\s*total\s*[│|]\s*(\d+)", text, re.I)
        if not values or int(values[-1]) != expected:
            raise RuntimeError(f"Cannot confirm {key}={expected} in {snap}")
    print(f"AISBench complete: {snap}")
    if not warm:
        try:
            latency_summary(folder, backend, shape)
        except ValueError as e:
            print(f"缓存分组结果暂不可用：{e}", file=sys.stderr)


def verify(folder, backend, shape, output):
    reads = [defaultdict(dict) for _ in RANKS]
    plans, done = defaultdict(list), defaultdict(list)
    reqs = {}
    problems = []
    for rank in (range(1) if backend == "none" else RANKS):
        init = False
        mode = False
        with (folder / f"node_{rank}.log").open(errors="replace") as log:
            for line in log:
                if "[KVCacheStore] initialize," in line:
                    init = True
                    if (f"backend={backend}," not in line or f"worker_rank={rank}," not in line
                            or f"tp_rank={rank}, tp_size=8," not in line):
                        problems.append(f"rank {rank}: unexpected backend or topology")
                if rank == 0 and "Master init options" in line:
                    mode = f"enable_kvcache_store: {int(backend != 'none')}," in line
                if m := GET.search(line):
                    rid, index, hits = m.groups()
                    if int(index) in reads[rank][rid]:
                        problems.append(f"{rid}: rank {rank} repeated batch {index}")
                    reads[rank][rid][int(index)] = int(hits)
                if rank != 0:
                    continue
                if m := PLAN.search(line):
                    plans[m[1]].append(int(m[2]))
                if m := DONE.search(line):
                    done[m[1]].append((int(m[2]), int(m[3])))
                if (m := END.search(line)) and m[2] == "1024":
                    if m[1] in reqs:
                        problems.append(f"{m[1]}: duplicate request completion")
                    reqs[m[1]] = m.groups()[2:]
        if init != (backend != "none") or (rank == 0 and not mode):
            problems.append(f"rank {rank}: backend configuration evidence mismatch")
    if len(reqs) != shape.requests:
        problems.append(f"expected {shape.requests} measured requests, got {len(reqs)}")
    passed, ttfts = 0, []
    output.mkdir(parents=True, exist_ok=True)
    path = output / "per_request.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["request_id", "input_tokens", "used_cached_tokens", "host_tokens_before_get",
                         "coordinated_hit_units", *[f"rank{r}_get_blocks" for r in RANKS], "ttft_ms", "verdict"])
        for rid, (reason, prompt, cached, output, ttft) in reqs.items():
            external = backend != "none"
            ok = (reason == "length" and int(output) == 1024 and abs(int(prompt)-shape.input_tokens) < BLOCK
                  and int(cached) == (shape.prefix if external else 0))
            ok &= plans[rid] == ([0] if external else []) and done[rid] == ([(shape.blocks, 1)] if external else [])
            counts = []
            for rank in RANKS:
                batches = reads[rank][rid]
                ordered = [batches[i] for i in sorted(batches)]
                counts.append(sum(ordered))
                if external:
                    ok &= sorted(batches) == list(range(len(batches))) and shape.valid_hits(ordered)
                else:
                    ok &= not batches
            if not ok:
                problems.append(f"{rid}: see {path} and worker_get_end batches")
            passed += int(ok)
            ttfts.append(float(ttft))
            values = [rid, prompt, cached, str(plans[rid]), str(done[rid]), *counts, ttft, "PASS" if ok else "FAIL"]
            writer.writerow(values)
            print(f"{rid}  used={cached}  GET/rank={counts}  TTFT={ttft}ms  {values[-1]}")
    print(f"{backend}: {passed}/{shape.requests} individual checks passed. {path}")
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"xLLM TTFT mean={statistics.mean(ttfts):.1f} ms, median={statistics.median(ttfts):.1f} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("directory", type=Path)
    gen.add_argument("--model-path", type=Path,
                     help="Local Hugging Face tokenizer/model directory (defaults to config.py MODEL_PATH)")
    add_shape(gen)
    summary = sub.add_parser("summary")
    summary.add_argument("directory", type=Path, nargs="?")
    summary.add_argument("--backend", choices=BACKENDS)
    summary.add_argument("--log-dir", type=Path)
    add_shape(summary)
    for action in ("run", "verify", "wait-warm"):
        p = sub.add_parser(action)
        p.add_argument("backend", choices=BACKENDS)
        p.add_argument("directory", type=Path, nargs=None if action == "run" else "?")
        if action != "run":
            p.add_argument("--log-dir", type=Path)
            add_shape(p)
        if action == "run":
            p.add_argument("--phase", choices=("warm", "formal"), required=True)
    a = ap.parse_args()
    root = a.directory.resolve() if a.directory is not None else None
    logs = getattr(a, "log_dir", None)
    if root is None and logs is None:
        ap.error("provide the dataset root or --log-dir pointing directly to node_0.log's directory")
    shape = workload(root, getattr(a, "requests", None), getattr(a, "input_tokens", None))
    if a.action == "generate":
        generate(root, shape, a.model_path)
    elif a.action == "summary":
        if logs is not None and not a.backend:
            ap.error("summary --log-dir requires --backend")
        names = [a.backend] if a.backend else [b for b in BACKENDS if (root / b / "formal").is_dir()]
        if not names:
            ap.error(f"找不到正式测试目录：{root}")
        errors = []
        for b in names:
            try:
                latency_summary(logs or root / b / "formal", b, shape)
            except ValueError as e:
                errors.append(str(e))
        if errors:
            ap.exit(1, "\n".join(errors) + "\n")
    elif a.action == "verify":
        verify(logs or root / a.backend / "formal", a.backend, shape, (root or Path.cwd()) / a.backend)
    elif a.action == "wait-warm":
        wait_warm(logs or root / a.backend / "warm", a.backend, shape)
    else:
        drive(root, a.backend, a.phase, config())


if __name__ == "__main__":
    main()
