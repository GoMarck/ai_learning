# 用 AISBench 验证 xLLM 的后端前缀缓存

仓库附带一份 Qwen3-8B 数据集，包含 100 条预热请求和 100 条正式请求。每条正式请求的正文长 20,480 个 token，其中 10,240 个 token 与对应的预热请求共享。按每块 128 个 token 计算，目标是**每条正式请求从后端复用 80 块**。聊天模板会让实际输入略长，按服务端 prompt token 数计算的复用率约为 50%。

| 阶段 | 文件 | 请求数 | 每条请求的正文 token 数 | 请求输出 token 数 |
| --- | --- | ---: | ---: | ---: |
| 预热 | warm.jsonl | 100 | 10,272 | 1 |
| 正式测试 | formal.jsonl | 100 | 20,480 | 1,024 |

## 准备环境和数据

使用已能执行原版 `aisbench_test.py` 的环境。在其 `config.py` 中，将 `MODEL_PATH` 设为测试机上的 Qwen3-8B 模型目录，`MODEL_NAME` 设为实际服务名，并配置服务地址。生成数据还需要安装有 Qwen3 tokenizer 支持的 Transformers。[AISBench 驱动仓库](https://github.com/rayn-zzz/aisbench_auto_tools_prefix)

从本仓库的 `aisbench` 目录执行以下命令。脚本按 TP8、DP1、并发 4 运行，xLLM 及后端服务由你独立启动。

```bash
TOOLS="$PWD"
cd /path/to/aisbench_auto_tools_prefix
RUN="$PWD/runs/qwen3_8b_pairs100"
mkdir -p "$RUN"
cp "$TOOLS/datasets/qwen3_8b_pairs100/"{inputs.json,warm.jsonl,formal.jsonl} "$RUN/"
```

如需自行生成，两步复制命令（`mkdir`、`cp`）可换成下面的命令。生成器会创建新目录。三组实验使用同一个 `RUN`，保留相同的请求和顺序。已附的数据集可跨机器使用，运行时会比对 tokenizer 文件摘要。

```bash
RUN="$PWD/runs/qwen3_8b_pairs100_new"
python3 "$TOOLS/xllm_run.py" generate "$RUN" \
  --model-path /data/models/Qwen3-8B --requests 100 --input-tokens 20480
```

## DataSystem 或 Mooncake

选择 `BACKEND=yuanrong`（DataSystem）或 `BACKEND=mooncake`。用本次实验独占的空缓存或新模型命名空间启动 xLLM，打开对应的 backend store，使用 `block_size=128`、`prefetch_batch_size=16`，保留各 rank 的详细日志（`--v=1`）。warm 和 formal 的 xLLM 日志分别写入空目录。

```bash
BACKEND=yuanrong
python3 "$TOOLS/xllm_run.py" run "$BACKEND" "$RUN" --phase warm
```

在可读取 xLLM 日志的容器确认预热写入完成。若容器独立，在该容器中把 `TOOLS` 设为脚本目录，并设置相同的 `BACKEND`。每个 rank 应成功写入 8,000 个块。

```bash
python3 "$TOOLS/xllm_run.py" wait-warm "$BACKEND" \
  --log-dir /path/to/warm/log --requests 100 --input-tokens 20480
```

重启 xLLM 以清空进程内 HBM/Host 前缀缓存。保持相同的模型、model_id、TP 拓扑和后端地址，并保持后端服务与已写入的数据可用。等模型服务就绪后，在 AISBench 容器执行第二次测试。

```bash
python3 "$TOOLS/xllm_run.py" run "$BACKEND" "$RUN" --phase formal
```

## 无后端对照

用相同的模型和 TP8 配置新启动 xLLM，设置 `--enable_kvcache_store=false`，直接跑同一份 formal。不同请求之间没有共享的完整首块，预期 100 条均为零复用。

```bash
python3 "$TOOLS/xllm_run.py" run none "$RUN" --phase formal
```

## 查看实际 token 复用率

在 xLLM 日志容器执行。日志目录应包含一次启动记录和本轮 100 条正式请求的完成记录，warm 阶段单独存放。

```bash
python3 "$TOOLS/stat_xllm_cache_sources.py" /path/to/formal/log --requests 100
```

无后端时只需 `node_0.log`。输出的“本地实际复用 token”按所有请求的 `sum(num_prefix_cache_tokens) / sum(prompt_tokens)` 计算。“其中有前缀复用”给出至少命中一个块的请求数。

`stat_xllm_cache_sources.py` 从启动日志的 `tp_size` 自动识别单个 DP 组的 TP 大小。启用后端时，日志目录须覆盖该组的每个 worker。例如 TP4 通常放入 `node_0.log` 到 `node_3.log`，脚本按记录中的 rank 检查完整性。每个请求的 prompt 和复用 token 只在主日志统计一次。

输出分别列出 HBM、本地 Host DDR 新增贡献、从后端读回并实际复用的 token 数及其占总输入的比例。HBM 优先；Host 比 HBM 多覆盖的部分才可能贡献额外复用。对于没有请求 ID 的 `[HostCache][PrefixMatch]`，只在探测次数和前缀长度分布与完成请求相符时，按每请求一次探测估算 HBM 贡献。请单独保留本轮 formal 的启动、探测及完成日志。

建议以 `--v=1` 保留层级匹配和批调用。运行版本若提供 KVTrace，可用 `--vmodule=hierarchy_block_manager_pool=2` 保留按 request 关联的 `plan_ready` 和 `completion_available`。它们记录预取前的本地 Host 前缀和后端连续命中块数，可帮助区分本地 DDR 与后端回填。日志不足时输出上下界，区间不可相加。后端 GET 的分片块命中率及 load_kvc/store_kvc 的平均值、P50 和 P99 按实际记录的调用另行统计。

下面的 `xllm_run.py verify` 和 `kv20_metrics.py` 仍按原 TP8 成对实验验收。

## 验收和时延统计

在日志容器执行，`LOG` 直接指向本轮 formal 的 `node_0.log` 所在目录。有后端时，需要 `node_0.log` 至 `node_7.log`，以及含请求 ID 的 `plan_ready`、`worker_get_end`、`completion_available` 打点和请求完成日志。缺少打点的运行无法用此脚本逐条确认后端命中。

```bash
BACKEND=yuanrong  # 或 mooncake、none
LOG=/path/to/formal/log
python3 "$TOOLS/xllm_run.py" verify "$BACKEND" \
  --log-dir "$LOG" --requests 100 --input-tokens 20480
python3 "$TOOLS/kv20_metrics.py" stats --backend "$BACKEND" --phase formal \
  --log-dir "$LOG" --requests 100 --input-tokens 20480
```

- `verify` 逐请求核对已复用 token 数和各 rank 的 GET 块数。有后端时，预期每条复用 10,240 个 token、每个 rank 各读回 80 块，最终打印 `100/100 individual checks passed`。
- `stats` 按命中场景统计 TTFT、TPOT 和有打点的 load_kvc/store_kvc 批调用时延，给出平均值、P50 和 P99，单位为毫秒。以 `--log-dir` 运行时，逐请求验收表保存在当前目录的 `<backend>/per_request.csv`，统计表在 `<backend>/formal/`。
- AISBench 原始输出同步打印，副本位于 `$RUN/<backend>/<phase>/driver.log` 和 `aisbench.log`。如果压测容器看不到 rank 日志，仍可在日志容器中单独执行上述验收。
