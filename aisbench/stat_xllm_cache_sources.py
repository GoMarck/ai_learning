"""Report prefix reuse and KV store latency for one single-DP Qwen3 benchmark."""
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def show(label, low, high, total):
    if low == high:
        print(f"{label}: {low:,}/{total:,} = {low/total:.2%}")
    else:
        print(f"{label}: {low:,}..{high:,}/{total:,} = {low/total:.2%}..{high/total:.2%}")


def summarize_sources(rows, probes, plans, completions, store, block_size, rank_hits):
    total = sum(prompt for prompt, cached in rows.values())
    reuse = sum(cached for prompt, cached in rows.values())
    print(f"完成请求: {len(rows)}，其中有前缀复用: {sum(cached > 0 for _, cached in rows.values())}")
    show("全部实际复用 token", reuse, reuse, total)

    # Legacy PrefixMatch carries sequence_id=-1. Never zip probes with completion
    # order: multiple requests may be in flight. Only compare aggregate cohorts.
    by_length = defaultdict(list)
    for hbm, host in probes:
        by_length[max(hbm, host)].append(hbm)
    cached_lengths = Counter(cached for _, cached in rows.values())
    matched = Counter({k: len(v) for k, v in by_length.items()}) == cached_lengths
    if matched:
        print("HBM/Host 按每请求一次 PrefixMatch 的采样口径估算；探测条数和长度分布已与完成记录核对。")
    elif reuse:
        print("PrefixMatch 探测缺失或与请求不一致；无法用它拆分 HBM/Host，以下显示可确认的范围。")

    hbm_bounds = {k: (min(v), max(v)) for k, v in by_length.items()} if matched else {}
    hbm_total = sum(hbm for hbm, _ in probes) if matched else 0
    hbm_upper = hbm_total if matched else reuse
    host_low = host_high = backend_low = backend_high = 0
    for rid, (_, cached) in rows.items():
        if not cached:
            continue
        # Probes of equal final length can have different splits. Keep bounds
        # if an anonymous probe cannot be assigned uniquely to a request.
        lo, hi = hbm_bounds.get(cached, (0, cached))
        host_end_lo, host_end_hi = 0, cached
        if not store:
            host_end_lo = cached
        elif len(plans[rid]) == len(completions[rid]) == 1:
            base, units = plans[rid][0]
            hit, done_units, ready = completions[rid][0]
            if ready and units == done_units and hit <= units and base + hit * block_size >= cached:
                host_end_lo = host_end_hi = min(base, cached)
            elif ready and units == done_units and hit <= units and base + hit * block_size < cached:
                # The contiguous host prefix ends before the final cached prefix:
                # only HBM could have supplied that whole prefix for flat KV.
                lo = hi = cached
        host_low += max(0, host_end_lo - hi)
        host_high += max(0, host_end_hi - lo)
        backend_low += cached - max(host_end_hi, hi)
        backend_high += cached - max(host_end_lo, lo)

    # Every restored block requires a hit on each TP rank. Per-rank totals
    # give an upper bound only; ranks may have hit different requests.
    if rank_hits:
        backend_high = min(backend_high, min(rank_hits) * block_size)
    backend_high = min(backend_high, reuse - hbm_total)
    if backend_low > backend_high:
        raise ValueError("GET 汇总与逐请求证据不一致，请检查日志是否缺失")
    host_low = max(host_low, reuse - hbm_upper - backend_high)
    host_high = min(host_high, reuse - hbm_total - backend_low)
    hbm_low = max(hbm_total, reuse - host_high - backend_high)
    hbm_upper = min(hbm_upper, reuse - host_low - backend_low)
    if hbm_low > hbm_upper or host_low > host_high:
        raise ValueError("层级匹配与完成记录不一致，请检查是否存在重试或缺失日志")
    show("本地实际复用 token（HBM + 原驻 Host）", reuse - backend_high, reuse - backend_low, total)
    show("HBM 复用 token", hbm_low, hbm_upper, total)
    show("本地 Host DDR 新增复用 token", host_low, host_high, total)
    show("本请求从后端读回并复用 token", backend_low, backend_high, total)


def report(args):
    rows, sizes, starts, probes = {}, set(), [], []
    plans, completions = defaultdict(list), defaultdict(list)
    inits, hits, tries = Counter(), Counter(), Counter()
    latencies = defaultdict(list)
    ranks_seen = set()
    files = sorted(f for f in args.logdir.glob("node_*.log") if re.fullmatch(r"node_\d+\.log", f.name))
    if not files:
        raise ValueError("目录中未找到 node_<编号>.log")
    for path in files:
        with path.open(errors="replace") as stream:
            for line in stream:
                if "Master init options:" in line:
                    if path.name == "node_0.log":
                        starts.append(line)
                elif "world_size" in line or "[KVCacheStore] initialize," in line:
                    size = re.search(r"\btp_size\s*=\s*(\d+)", line)
                    if size:
                        sizes.add(int(size[1]))
                    dp = re.search(r"\bdp_size\s*=\s*(\d+)", line)
                    if dp and int(dp[1]) != 1:
                        raise ValueError("仅支持一个 DP 组的日志")
                    if "[KVCacheStore] initialize," in line:
                        fields = dict(re.findall(r"(\w+)=(\d+)", line))
                        rank = int(fields["worker_rank"])
                        if int(fields["tp_rank"]) != rank:
                            raise ValueError(f"{path}: worker_rank 与 tp_rank 不一致")
                        inits[rank] += 1
                elif path.name == "node_0.log" and "finish_reason:" in line:
                    m = re.search(r"\brequest_id:\s*([^,\s]+).*?prompt_tokens:\s*(\d+),\s*num_prefix_cache_tokens:\s*(\d+)", line)
                    if m:
                        rid, prompt, cached = m.groups()
                        if rid in rows or int(cached) > int(prompt):
                            raise ValueError(f"重复或异常的完成记录: {rid}")
                        rows[rid] = int(prompt), int(cached)
                elif path.name == "node_0.log" and "[HostCache][PrefixMatch]" in line:
                    m = re.search(r"hbm_tokens=(\d+) host_tokens=(\d+)", line)
                    if m:
                        probes.append(tuple(map(int, m.groups())))
                elif path.name == "node_0.log" and "phase=plan_ready " in line:
                    m = re.search(r"request=(\S+) dp_rank=0 host_tokens=(\d+) store_units=(\d+)", line)
                    if m:
                        plans[m[1]].append((int(m[2]), int(m[3])))
                elif path.name == "node_0.log" and "phase=completion_available " in line:
                    m = re.search(r"request=(\S+) dp_rank=0 hit_units=(\d+) store_units=(\d+) quiescent=([01])", line)
                    if m:
                        completions[m[1]].append(tuple(map(int, m.groups()[1:])))
                elif "[KVCacheStore][PrefetchGet]" in line:
                    f = dict(re.findall(r"(\w+)=(\d+)", line))
                    rank, kind, blocks, found, us = (int(f[k]) for k in ("worker_rank", "type", "blocks", "hits", "elapsed_us"))
                    if kind or not blocks or found > blocks:
                        raise ValueError(f"{path}: 不支持的 GET 记录")
                    tries[rank] += blocks
                    hits[rank] += found
                    ranks_seen.add(rank)
                    group = "全 miss" if not found else "全命中" if found == blocks else "部分命中"
                    latencies["load_kvc 全部"].append(us)
                    latencies["load_kvc " + group].append(us)
                elif "[KVCacheStore][OffloadPut]" in line:
                    f = dict(re.findall(r"(\w+)=(\d+)", line))
                    ranks_seen.add(int(f["worker_rank"]))
                    if f["store_write_enabled"] == "1":
                        group = "含失败" if int(f["failed"]) else "全成功"
                        latencies["store_kvc 全部"].append(int(f["elapsed_us"]))
                        latencies["store_kvc " + group].append(int(f["elapsed_us"]))
    if len(starts) != 1 or len(rows) != args.requests or not sum(prompt for prompt, _ in rows.values()):
        raise ValueError(f"应为一次完整测试；识别到 {len(starts)} 次启动，{len(rows)} 条完成请求")
    flag = re.search(r"\benable_kvcache_store:\s*([01])\b", starts[0])
    bs = re.search(r"\bblock_size:\s*(\d+)", starts[0])
    if not flag or not bs or not int(bs[1]):
        raise ValueError("无法从启动日志确认 Store 开关或 block_size")
    store = flag[1] == "1"
    if len(sizes) > 1 or (store and not sizes):
        raise ValueError(f"无法确定唯一 TP 大小: {sorted(sizes)}")
    tp = next(iter(sizes), None)
    print(f"TP: {tp or '日志未记录'}，backend store: {'开启' if store else '关闭'}")
    if store:
        expected = set(range(tp))
        if inits != Counter(expected) or not ranks_seen <= expected:
            raise ValueError(f"需要每个 TP rank 的一次初始化，预期 {sorted(expected)}，实际 {dict(inits)}")
    have_all_gets = store and len(tries) == tp
    summarize_sources(rows, probes, plans, completions, store, int(bs[1]),
                      list(hits.values()) if have_all_gets else [])
    if not store:
        print("load_kvc / store_kvc: N/A（backend Store 已关闭）")
        return
    if have_all_gets:
        show("后端 GET 分片块命中率", sum(hits.values()), sum(hits.values()), sum(tries.values()))
        print("各 rank 成功读回分片块数:", [hits[r] for r in range(tp)])
    else:
        print(f"完整 GET 命中率不可用；有 GET 记录的 rank: {sorted(tries)}；耗时仅汇总已记录的调用。")
    print(f"后端批调用耗时（TP{tp}，单位 ms）\n操作\t批次数\tavg\tp50\tp99")
    for label in ("load_kvc 全部", "load_kvc 全命中", "load_kvc 部分命中", "load_kvc 全 miss",
                  "store_kvc 全部", "store_kvc 全成功", "store_kvc 含失败"):
        samples = sorted(latencies[label])
        if not samples:
            print(f"{label}\t0\tN/A\tN/A\tN/A")
            continue
        lo, rem = divmod(99 * (len(samples) - 1), 100)
        hi = min(lo + 1, len(samples) - 1)
        p99 = samples[lo] + (samples[hi] - samples[lo]) * rem / 100
        print(f"{label}\t{len(samples)}\t{mean(samples)/1000:.3f}\t{median(samples)/1000:.3f}\t{p99/1000:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logdir", type=Path, help="directory with one run's node_<rank>.log files")
    parser.add_argument("--requests", type=int, default=100)
    try:
        report(parser.parse_args())
    except (ValueError, KeyError) as e:
        parser.error(str(e))
