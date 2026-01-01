# Mooncake

## 1. 概述

Mooncake 提出了一种以 KVCache 为中心的 PD 分离架构，用于提升 LLM 系统整体推理的吞吐量。而 Mooncake 作为 KVCache 的存储系统，利用 CPU/DRAM/SSD 以及节点间 RDMA 的高速传输，为这种分离架构提供了高速的 KVCache 存取。

![](images/architecture.png)

## 2. PD 分离架构的优势

在 LLM 服务中请求的生命周期通常包含两个阶段：Prefill（生成首个 token）和 Decode（逐步生成后续 token）。

当前大部分推理系统（vLLM、TensorRT-LLM）采用的是 PD 合并部署，即将 Prefill 处理与 Decode 处理放到同一个进程中。当由于 Prefill 阶段与 Decode 阶段的处理差异及其带来的对硬件诉求的差异，会导致这种合并部署无法带来足够好的 SLO（服务等级目标）。

### 2.1 SLO 判定标准

在评价大模型平台的性能时，不能只是简单地通过吞吐量（Throughput）来判定。单纯依赖 Throughput 作为指标，并不能反映延迟表现，系统看似处理了大量请求，但其中不少未能满足 SLO，最终呈现给用户的仍是不理想的服务体验。

大模型服务中最常用的 SLO 包括：
- **TTFT (Time To First Token)**：首 token 响应延迟，直接影响用户的等待体验。
- **TPOT (Time Per Output Token)**：衡量两个连续生成的 token 之间的平均延迟，决定交互的流畅程度。

为了更好地评价大模型的表现，后续引入了 Goodput 的概念，即在满足 TTFT 和 TPOT 要求的前提下，每秒完成的有效请求数，它能更好地衡量一个大模型平台的性能。

![](images/goodput.png)

### 2.2 Prefill 与 Decode 的矛盾

#### 2.2.1 Prefill

TTFT 由 Prefill 阶段所决定，在 Prefill 阶段需要并行处理整个输入序列来生成首个 token，属于计算密集型操作，对计算的算力要求较高，而对内存带宽要求较小（毕竟只生成了首 token）。

实际处理过程中，请求的 prompt 是按照 batch 分组处理的，随着 batch size 的增加，Prefill 阶段吞吐量的提升很快趋于平缓。这是因为 prefill 属于 compute-bound，当 batch 中的总 tokens 数超过一定规模后，GPU 的计算能力已经被完全吃满，再增加请求只会延长整体处理时间（**TTFT 增加！！！**），而不会带来明显的吞吐提升。

#### 2.2.2 Decode

TPOT 由 Decode 阶段所决定，在 Decode 阶段则是逐个生成后续 token，需要频繁访问 KVCache，属于内存密集型操作，对算力要求相对较低。

实际处理过程中，同样是按照 batch 分组进行处理，随着 batch size 的增加，吞吐量的增长趋势越来越显著。这是因为 Decode 阶段是 memory-bound，即相比于计算，读写数据的时间要更多。所以在 Decode 阶段中，如果我们能提升 batch size，就能把计算强度提起来，吞吐量就上去了。

#### 2.2.3 矛盾点

从 2.2.1 与 2.2.2 的描述中可以看出，Prefill 阶段与 Decode 阶段的主要矛盾点如下：
1. Prefill 阶段与 Decode 阶段对于硬件的要求差异较大，但是它们却部署在同一进程中使用相同的卡，容易造成相互干扰，导致 SLO 不理想；
2. Batch size 的调节很难做，小的 batch size 照顾了 TTFT，却容易导致 TPOT 下降，反之亦然。

### 2.3 PD 分离

基于以上 PD 合并部署的劣势，后续提出了一种新方案：PD 分离架构。当一个请求到达系统时，它会先被分配到 Prefill 实例完成 prefill 阶段；随后系统将其中间状态（主要是 KV Cache）迁移到 Decode 实例，并执行多步 decode 以生成后续 token；当生成完成后，请求才会离开系统。它具有以下优势：
1. **干扰避免**：prefill 和 decode 各自独立运行，更快地完成计算，也更容易满足各自的 SLO；
1. **资源分配与并行策略解耦**：Prefill 阶段属于计算密集型操作，而 Decode 阶段属于内存密集型操作，分离架构可以使计算和存储可以朝着两个独立的方向做优化；
2. **Batching 策略更灵活**：在 Batching 策略中，可以为 PD 设置合适的 batch size，将 PD 阶段的吞吐量提升至最优；
3. **部署更灵活**：xPyD 的部署方式可以满足更多不同类型应用的需求，同时也能充分提高 GPU 的利用率；
4. **弹性粒度更细**：无需将 PD 一起弹性扩缩，只需要基于实际情况扩缩 P 实例或者 D 实例。

## 3. Mooncake 架构

![](images/mooncake-store-simple-arch.jpg)

Mooncake 主要包含两个组件：Master 与 Client。

### 3.1 Mooncake Master

Mooncake Master 是一个独立的进程，对其他组件暴露 RPC 服务，典型的中心化组件。它的主要职责是：
1. **集群管理**：协调整个集群的逻辑存储空间池，管理节点的加入和离开事件；
2. **资源管理**：负责对象空间分配和元数据维护。

### 3.2 Mooncake Client

Mooncake Client 具有两种独立的职责：
- **SDK提供**：提供 `put`、`get` 等 API 以供 LLM 层调用；
- **数据节点**：托管了一段连续内存，该内存属于集群内存池子的一部分，可供给其他 Client 使用。数据传输实际是在 Client 侧发生的，绕过了 Master。

Client 可以只扮演这两种职责的任意一个：
- 当 `global_segment_size` 设置为 0 时，Client 只扮演 SDK 提供者的角色；
- 当 `local_buffer_size` 设置为 0 时，Client 只扮演数据节点的角色。

因此，Client 的初始化有两种模式：
- **Embedded 模式**：Client 同时扮演 SDK 提供者与数据节点两种角色；
- **Standalone 模式**：Client 仅扮演 SDK 提供者角色。

## 4. Mooncake 核心 API

### 4.1 Init

```cpp
ErrorCode Init(const std::string& local_hostname,
               const std::string& metadata_connstring,
               const std::string& protocol,
               void** protocol_args,
               const std::string& master_server_entry);
```

其中 `protocol` 字段用于指定 TE 的传输协议，如 `TCP` 或 `RDMA`。

### 4.2 Put

```cpp
tl::expected<void, ErrorCode> Put(const ObjectKey& key,
                                  std::vector<Slice>& slices,
                                  const ReplicateConfig& config);

struct ReplicateConfig {
    size_t replica_num{1};                    // Total number of replicas for the object
    bool with_soft_pin{false};               // Whether to enable soft pin mechanism for this object
    std::string preferred_segment{};         // Preferred segment for allocation
};
```

![](images/mooncake-store-simple-put.png)

`Put` 会将与键关联的值存储到分布式内存池中。配置参数允许指定所需的副本数以及存储值的首选段。启用持久化后，Put 操作还会异步触发对 SSD 的持久化操作。

副本保证与尽力而为的可靠性行为：
- 对象的每个切片都保证复制到不同的内存段，从而确保数据分布在不同的存储节点上；
- 不同对象的不同切片可能在同一内存段上；
- 尽力而为的副本策略：如果可用空间不足以容纳所有请求的副本，会尽力为对象提供副本。

### 4.3 Get

```cpp
tl::expected<void, ErrorCode> Get(const std::string& object_key, 
                                  std::vector<Slice>& slices);
```

![](images/mooncake-store-simple-get.png)

### 4.4 Remove

```cpp
tl::expected<void, ErrorCode> Remove(const ObjectKey& key);
```

## 5. 依赖组件

### 5.1 Mooncake Master 设计

[Mooncake Master 设计](mooncake_master.md)

### 5.2 Transfer Engine 设计

[Transfer Engine 设计](transfer_engine.md)

## 6. 典型使用场景

### 6.1 vllm ascend + Mooncake

vllm-ascend v0.11.0 已原生支持 Mooncake，可以与 Mooncake 实现如下功能：
- **PD 分离部署**：P 实例与 D 实例之间的 KVCache 传输；
- **前缀缓存加速**：借助 Mooncake 的多级缓存，提升 Prefill 阶段的 TTFT。

部署命令如下：

- 部署 Mooncake Master

    ```bash

    ```

- 部署 vLLM 实例：
    
    ```bash
    
    ```