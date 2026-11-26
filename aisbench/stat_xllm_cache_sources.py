"""Single-run Qwen3 TP8 benchmark: local/remote observables from xLLM logs."""
import argparse
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("logdir", type=Path, help="directory directly containing node_0.log")
p.add_argument("--requests", type=int, default=100)
args = p.parse_args()
files = [args.logdir / f"node_{i}.log" for i in range(8)]
if not files[0].is_file():
    p.error(f"未找到 {files[0]}")
rows = {}
runs = []
for line in files[0].open(errors="replace"):
    if "Master init options:" in line:
        runs.append(line)
    if "finish_reason:" in line:
        m = re.search(r"\brequest_id:\s*([^,\s]+).*?prompt_tokens:\s*(\d+),\s*num_prefix_cache_tokens:\s*(\d+)", line)
        if m:
            rows[m[1]] = (int(m[2]), int(m[3]))
if len(runs) != 1 or len(rows) != args.requests:
    p.error(f"应为一次完整测试；识别到 {len(runs)} 次启动，{len(rows)} 条完成请求")
flag = re.search(r"enable_kvcache_store:\s*([01])\b", runs[0])
bs = re.search(r"\bblock_size:\s*(\d+)", runs[0])
if not flag or not bs:
    p.error("无法从启动日志确认 Store 开关或 block_size")
total = sum(n for n, h in rows.values())
reuse = sum(h for n, h in rows.values())
if not total:
    p.error("prompt_tokens 总数为 0")
def show(label, n):
    print(f"{label}: {n:,}/{total:,} = {n/total:.2%}")
print(f"完成请求: {len(rows)}，其中有前缀复用: {sum(h > 0 for _, h in rows.values())}")
show("全部实际复用 token", reuse)
if flag[1] == "0":
    show("本地实际复用 token", reuse)
    show("本请求直接从后端恢复并复用 token", 0)
    print("load_kvc / store_kvc: N/A（backend Store 已关闭）")
    raise SystemExit
if any(not f.is_file() for f in files):
    p.error("启用 Store 的 TP8 测试需要完整 node_0.log 到 node_7.log")
hits, tries = [], []
latencies = defaultdict(list)
for rank, f in enumerate(files):
    n = h = init = 0
    for line in f.open(errors="replace"):
        if "[KVCacheStore] initialize," in line:
            init += int(f"worker_rank={rank}," in line and "tp_size=8," in line)
        if "[KVCacheStore][PrefetchGet]" in line:
            m = re.search(r"worker_rank=(\d+),.*?type=(\d+), blocks=(\d+), hits=(\d+)", line)
            if not m or int(m[1]) != rank or m[2] != "0":
                p.error("仅适用于一机 TP8、普通 Qwen3 KV、每个日志一个 rank")
            blocks, found = int(m[3]), int(m[4])
            elapsed = re.search(r"\belapsed_us=(\d+)", line)
            if not elapsed or not 0 <= found <= blocks or not blocks:
                p.error(f"{f}: PrefetchGet 日志缺少计时或块数无效")
            us = int(elapsed[1])
            n += blocks
            h += found
            group = "全 miss" if not found else "全命中" if found == blocks else "部分命中"
            latencies["load_kvc 全部"].append(us)
            latencies["load_kvc " + group].append(us)
        if "[KVCacheStore][OffloadPut]" in line:
            m = re.search(r"worker_rank=(\d+),.*?store_write_enabled=([01]),.*?failed=(\d+), elapsed_us=(\d+)", line)
            if not m or int(m[1]) != rank:
                p.error(f"{f}: OffloadPut 日志格式或 rank 不符")
            if m[2] == "1":
                us = int(m[4])
                latencies["store_kvc 全部"].append(us)
                group = "含失败" if int(m[3]) else "全成功"
                latencies["store_kvc " + group].append(us)
    if init != 1 or not n:
        p.error(f"rank {rank} 初始化或 GET 日志不足；请确认单次完整运行、--v=1")
    hits.append(h); tries.append(n)
print(f"后端 GET 分片块命中率: {sum(hits):,}/{sum(tries):,} = {sum(hits)/sum(tries):.2%}")
print("各 rank 成功读回分片块数:", hits)
upper = min(reuse, min(hits) * int(bs[1]))
show("本请求直接从后端恢复并复用 token【上限】", upper)
show("本地实际复用 token【下限】", reuse - upper)
if upper:
    print("现有日志没有逐请求的最终来源拆分；上限不能当作实际值。")

print("\n后端批调用耗时（TP8 合并，每个 rank 的一次调用算一个样本；单位 ms）")
print("操作\t批次数\tavg\tp50\tp99")
for label in ("load_kvc 全部", "load_kvc 全命中", "load_kvc 部分命中",
              "load_kvc 全 miss", "store_kvc 全部", "store_kvc 全成功",
              "store_kvc 含失败"):
    samples = sorted(latencies[label])
    if not samples:
        print(f"{label}\t0\tN/A\tN/A\tN/A")
        continue
    # Inclusive linear percentile (same as numpy.percentile's default).
    lo, remainder = divmod(99 * (len(samples) - 1), 100)
    hi = min(lo + 1, len(samples) - 1)
    p99 = samples[lo] + (samples[hi] - samples[lo]) * remainder / 100
    print(f"{label}\t{len(samples)}\t{mean(samples)/1000:.3f}\t"
          f"{median(samples)/1000:.3f}\t{p99/1000:.3f}")
