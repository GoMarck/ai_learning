# PD合并部署指南

## 0. 环境准备

### 0.1 硬件

vllm ascend 基于华为昇腾卡进行适配，属于华为昇腾团队为 vllm 做的支持 NPU 平台的插件，因此硬件上需要确保有华为昇腾的 NPU 卡，当前验证过的卡有：
- 910B A2
- 910C

### 0.2 软件

运行 vllm ascend 推荐使用官方推荐的镜像环境，因此请确保机器上已安装 Docker 并处于运行状态。

<details>
<summary>Docker安装详细教程</summary>

#### Docker安装教程

- Ubuntu 用户：
    
    卸载旧版本：
    ```bash
    sudo apt-get remove docker docker-engine docker.io containerd runc
    ```
    安装依赖工具：
    ```bash
    sudo apt update
    sudo apt install apt-transport-https ca-certificates curl gnupg lsb-release
    ```
    添加 Docker 官方 GPG 密钥：
    ```bash
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    ```
    配置稳定版仓库：
    ```bash
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    ```
    安装 Docker Engine：
    ```bash
    sudo apt install docker-ce docker-ce-cli containerd.io
    ```
    设置为开机自启动：
    ```bash
    sudo systemctl enable --now docker
    sudo docker run hello-world
    ```

- OpenEuler 用户：
    ```bash
    # 1. 更新系统
    sudo dnf update -y

    # 2. 一键安装官方包
    sudo dnf install -y docker

    # 3. 启动并设为开机自启
    sudo systemctl enable --now docker

    # 4. 验证
    docker --version
    sudo docker run --rm hello-world
    ```

#### Docker安装其他注意事项

1. 配置非root用户免sudo：
    ```bash
    sudo usermod -aG docker $USER
    newgrp docker
    ```


2. Docker 镜像国内源加速配置：
    ```bash
    sudo mkdir -p /etc/docker
    sudo tee /etc/docker/daemon.json <<-'EOF'
    {
    "registry-mirrors": ["https://mirror.aliyuncs.com","https://docker.registry.cyou"]
    }
    EOF
    sudo systemctl daemon-reload && sudo systemctl restart docker
    ```
</details>

## 1. 部署

### 1.1 拉取官方镜像

vllm-ascend 目前发布了多个官方发行版，截止2025年1月1日已发布了 v0.12.0rc1，稳定版推荐 v0.11.0：

| 版本     | 镜像                                     |
|---------|------------------------------------------|
| v0.10.0rc1 | quay.io/ascend/vllm-ascend:v0.10.0rc1 |
| v0.11.0    | quay.io/ascend/vllm-ascend:v0.11.0    |
| v0.12.0rc1 | quay.io/ascend/vllm-ascend:v0.12.0rc1 |

### 1.2 拉起容器

以 vllm-ascend v0.11.0 为例拉起容器

```bash
docker run \
    --name "docker_name" \
    --privileged \
    -itu root \
    -d \
    --net=host \
    --device=/dev/davinci0:/dev/davinci0 \
    --device=/dev/davinci1:/dev/davinci1 \
    --device=/dev/davinci2:/dev/davinci2 \
    --device=/dev/davinci3:/dev/davinci3 \
    --device=/dev/davinci4:/dev/davinci4 \
    --device=/dev/davinci5:/dev/davinci5 \
    --device=/dev/davinci6:/dev/davinci6 \
    --device=/dev/davinci7:/dev/davinci7 \
    --device=/dev/davinci_manager:/dev/davinci_manager \
    --device=/dev/devmm_svm:/dev/devmm_svm \
    --device=/dev/hisi_hdc:/dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it quay.io/ascend/vllm-ascend:v0.11.0 bash
```

### 1.3 下载镜像

以下载 Qwen 2.5 7B 模型为例，执行命令如下：
```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /workspace/models
```

### 1.4 部署 vllm 实例

```bash
ASCEND_RT_VISIBLE_DEVICES=0 vllm serve /workspace/models/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8080 \
    --max-num-batched-tokens 45000 \
    --tensor-parallel-size 1 \
    --max-model-len 30000 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enforce-eager > /tmp/vllm.log 2>&1 &
```

环境变量说明：
- `ASCEND_RT_VISIBLE_DEVICES`：指定使用环境中的 NPU 卡，需要确保卡空闲

参数说明：
- `host & port`：用于指定对外监听的IP和端口号，后续请求都从这个IP和端口号接收；
- `max-num-batched-tokens`：决定 vLLM 在单次 forward 里最多把多少个 token 拼成一个大 batch 送进模型，数值越大 → 单次 forward 能吞下的序列越多，吞吐越高，但延迟也会拉长，显存占用增加；数值越小 → batch 被切得更碎，延迟降低，可吞吐会掉；
- `tensor-parallel-size`：张量并行度，小模型并行度为1就够了，提升并行度会额外占用其他的NPU卡，需要同步调整环境变量 `ASCEND_RT_VISIBLE_DEVICES`；
- `max-model-len`：最大序列长度，如果该值设置过小而需要处理的序列很长请求会报错，但是设置过长会需要更多的NPU显存；
- `gpu-memory-utilization`：NPU内存使用最大阈值，如果设置过小会造成模型加载失败，Qwen 7B 模型权重为14+GB显存。

额外参数说明：
- `seed`：设置推理的随机seed，在性能测试下一般会设置该值确保两次推理结果一致；
- `--kv-transfer-config`：如果需要配置多级缓存，则需要额外设置该参数，大致如下：
    ```bash
    vllm serve ... \
    --kv-transfer-config '{"role":"kv_both","connector":"xxx"}'
    ```
    不同类型的connector可能会有不同的配置项，他们都通过json文件解析配置项读取。

### 1.5 快速验证

通过 curl 请求可以快速验证模型是否部署成功：

```bash
curl -X POST "http://${SERVER_IP}:8080/v1/completions" \
     -H "Content-Type: application/json" \
     -d '{
          "model": "'"/workspace/models/Qwen2.5-7B-Instruct"'",
          "prompt": "What is your name?",
          "max_tokens": 50,
          "temperature": 0
        }'
```

> **注意事项**：
>
> model字段请确保与 vLLM 实例的模型路径**完全一致**，否则会报 404 Not Found错误。

### 1.6 Benchmark 验证

安装vllm bench：

```bash
pip install vllm[bench]
```

验证：
- 随机数据验证
    ```bash
    vllm bench serve \
        --base-url=http://${SERVER_IP}:8080 \
        --dataset-name=random \
        --random-input-len=512 \
        --random-output-len=256 \
        --max-concurrency=8 \
        --num-prompts=3000 \
        --model=/workspace/models/Qwen2.5-7B-Instruct
    ```

- 已有数据集验证

    <details>
    <summary>数据集生成</summary>
    
    gen_random_dataset.py：

    ```python
    import json
    import logging
    import os
    import random
    import string
    from tqdm import tqdm
    from transformers import AutoTokenizer

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    def gen_random_string(length=10):
        """Generate a random string of specified length"""
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))


    def gen_random_tokens(tokenizer, num_tokens):
        """
        Generate a piece of random text that, after tokenization, has exactly the specified number of tokens.

        Args:
            tokenizer: The transformers tokenizer instance to use.
            num_tokens: Target number of tokens.

        Returns:
            A randomly generated text string.
        """
        token_ids = []
        while len(token_ids) < num_tokens:
            random_word = gen_random_string(random.randint(3, 8))
            token_ids.extend(tokenizer.encode(" " + random_word, add_special_tokens=False))

        final_token_ids = token_ids[:num_tokens]
        return tokenizer.decode(final_token_ids)


    def gen_random_prompts(tokenizer, num_groups, num_prompts_per_group, prefix_tokens, suffix_tokens):
        """
        Generate a random prompt dataset with length based on token count.

        Args:
            tokenizer: The transformers tokenizer instance to use.
            num_groups: Number of groups.
            num_prompts_per_group: Number of prompts per group.
            prefix_tokens: Number of prefix tokens.
            suffix_tokens: Number of suffix tokens.

        Returns:
            A list of randomly generated prompts.
        """
        prompts = []
        logging.info(
            f"Starting to generate dataset (Number of groups: {num_groups}, Prompts per group: {num_prompts_per_group})...")

        for _ in tqdm(range(num_groups), desc="Generating groups"):
            prefix = gen_random_tokens(tokenizer, prefix_tokens)
            for _ in range(num_prompts_per_group):
                suffix = gen_random_tokens(tokenizer, suffix_tokens)
                prompt = prefix + " " + suffix
                prompts.append(prompt)

        random.shuffle(prompts)
        return prompts


    def save_to_file(prompts, output_file):
        """Save generated prompts to a JSONL file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for prompt in tqdm(prompts, desc="Writing to file"):
                data = {"prompt": prompt}
                json_line = json.dumps(data, ensure_ascii=False)
                f.write(json_line + '\n')

        logging.info(f"Successfully saved {len(prompts)} entries to {output_file}")


    def main():
        # ==============================================================================
        # Configuration - Parts you need to modify
        # ==============================================================================
        config = {
            # Fill in your local model folder path or model name on Hugging Face here
            # For example: 'gpt2', './my_local_llama_model', 'Qwen/Qwen1.5-7B-Chat'
            'tokenizer_path': '/data/models/qwen2.5_7B_Instruct',
            'num_groups': 30,
            'num_prompts_per_group': 100,
            'prefix_length': 12 * 1024,  # Prefix token count
            'suffix_length': 12 * 1024,  # Suffix token count
            'output_dir': '.',
            'output_file': 'dataset_24k_tokens_50p.jsonl',
            'seed': 42
        }

        # Set random seed
        random.seed(config['seed'])

        # ==============================================================================
        # Load generic tokenizer from specified path
        # ==============================================================================
        tokenizer_path = config['tokenizer_path']
        logging.info(f"Loading tokenizer from '{tokenizer_path}'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logging.info("Tokenizer pad_token not set, has been set to eos_token.")

            logging.info("Tokenizer loaded successfully.")
        except Exception as e:
            logging.error(f"Error: Unable to load tokenizer from path '{tokenizer_path}'.")
            logging.error(
                f"Please confirm if the path is correct and if the folder contains files like " +
                "'tokenizer.json' or 'tokenizer_config.json'."
            )
            logging.error(f"Detailed error message: {e}")
            return

        os.makedirs(config['output_dir'], exist_ok=True)
        output_path = os.path.join(config['output_dir'], config['output_file'])
        total_prompts = config['num_groups'] * config['num_prompts_per_group']
        total_tokens = config['prefix_length'] + config['suffix_length']
        logging.info(f"Will generate a total of {total_prompts} entries, each with approximately {total_tokens} tokens.")
        logging.info(f"(Number of groups: {config['num_groups']}, Prompts per group: {config['num_prompts_per_group']})")
        prompts = gen_random_prompts(tokenizer, config['num_groups'], config['num_prompts_per_group'],
                                    config['prefix_length'], config['suffix_length'])
        save_to_file(prompts, output_path)


    if __name__ == "__main__":
        main()    
    ```

    config参数说明：
    - tokenizer_path：模型路径
    - num_groups：数据组数
    - num_prompts_per_group：每组提示词数量
    - prefix_length：序列前缀长度，用于测试前缀缓存使用
    - suffix_length：序列后缀长度
    - output_file：文件名

    </details>

    运行命令：

    ```bash
    vllm bench serve \
        --base-url=http://${SERVER_IP}:8080 \
        --dataset-name=custom \
        --dataset-path=${DATASET_PATH} \
        --max-concurrency=8 \
        --custom-output-len=256 \
        --num-prompts=3000 \
        --model=/workspace/models/Qwen2.5-7B-Instruct
    ```

命令运行完成后，vllm bench会输出benchmark总结。