# GLM-5.3 W8A8C8 在 A3 单机上使用 Mooncake Store

用于验证 [xLLM PR #2855](https://github.com/xLLM-AI/xllm/pull/2855) 的 KV 缓存异步写回。命令使用 16 个逻辑 NPU，开启 MTP，要求本机已启动 Mooncake Store 且草稿权重已导出到 `--draft_model` 指定的目录。主模型使用 W8A8C8 权重，运行时将 `kv_cache_dtype` 设为 `auto` 以支持 Host offload。

## 导出 GLM-5.3 MTP 权重

在 xLLM 源码仓库根目录执行：

```bash
python3 tools/export_mtp.py \
  --input-dir /models/GLM-5.3-w8a8c8 \
  --output-dir /models/GLM-5.3-w8a8c8-mtp
```

## 启动 xLLM 服务

```bash
xllm serve /models/GLM-5.3-w8a8c8 \
  --model_impl=python \
  --port=18994 \
  --master_node_addr=127.0.0.1:10015 \
  --nnodes=16 \
  --dp_size=2 \
  --ep_size=16 \
  --max_memory_utilization=0.86 \
  --max_tokens_per_batch=4096 \
  --max_seqs_per_batch=16 \
  --block_size=128 \
  --kv_cache_dtype=auto \
  --indexer_cache_dtype=auto \
  --enable_prefix_cache=true \
  --host_blocks_factor=2 \
  --enable_kvcache_store=true \
  --store_protocol=tcp \
  --store_master_server_address=127.0.0.1:50051 \
  --store_metadata_server=P2PHANDSHAKE \
  --store_local_hostname=127.0.0.1:12345 \
  --enable_chunked_prefill=true \
  --enable_graph=true \
  --enable_schedule_overlap=true \
  --communication_backend=hccl \
  --acl_graph_decode_batch_size_limit=2 \
  --draft_model=/models/GLM-5.3-w8a8c8-mtp \
  --num_speculative_tokens=3 \
  --tool_call_parser=auto \
  --log-dir=logs
```
