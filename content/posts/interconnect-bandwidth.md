---
title: 分布式训练的真瓶颈是互联带宽
date: 2026-09-20T16:02:57+08:00
author: "丁志强"
draft: false
tags: ["分布式训练", "互联带宽", "NCCL", "InfiniBand", "Ascend"]
---

加卡不一定更快。当你把一张卡换成八张、八张换成一千张，总有那么一刻，训练 step 时间不再下降——不是 GPU 没在算，而是它们在等。等的不是算力，是带宽。这篇文章要说的是：分布式训练真正的瓶颈，往往不是 FLOPS，而是互联带宽。

## 一、瓶颈的本质：计算-通信比与那条越拉越大的「剪刀差」

判断「是不是被带宽卡住」，第一个指标是计算-通信比（compute/communication ratio）：单位算力需要配套搬运多少字节数据。这个比例一旦接近 1，算力就在等网络，而不是在算。DeepSeek-V3 论文写得很直白：跨节点专家并行带来的通信开销，使计算-通信比约为 1:1，必须靠 DualPipe 做计算-通信重叠才不空转 [18]。

更麻烦的是「剪刀差」：算力增速长期快于互联带宽增速。官方口径下，GB200 NVL72 的 LLM 训练性能是 H100 的 4 倍 [7]，而单卡 NVLink 带宽只从 900 GB/s（NVLink 4）涨到 1,800 GB/s（NVLink 5），也就是 2 倍 [6]。连接翻倍，算力翻两番——扩集群的边际收益必然递减（本文量级比较）。算力侧的官方口径可参照 H100 SXM 的 FP16 Tensor Core 规格 1,979 TFLOPS，官方页脚注明「With sparsity」，稠密约折半为 989.5 TFLOPS（本文折算）[8]。

再看机内与机外的鸿沟：DGX H100 单机 NVSwitch 提供 900 GB/s 的 GPU-GPU 带宽，对外每个 ConnectX-7 口最高 400 Gb/s（约 50 GB/s）[5]。单卡 NVLink 对比单口 InfiniBand，约 18 倍（本文推算，标称值换算）。模型还在变大：GPT-3 已到 175B 参数 [14]，ZeRO 面向万亿参数设计 [15]，而互联速率一代才翻一倍（NDR 400 Gb/s → XDR 800 Gb/s）[10]。

## 二、通信原语与通信量下界：all-reduce 到底要搬多少字节

并行训练的每一步，本质是四种集合通信原语的组合：all-reduce、all-gather、reduce-scatter、all-to-all。NCCL 文档明确，ReduceScatter + AllGather 等价于 AllReduce [1]。看懂原语，才知道瓶颈压在哪一段。

ring all-reduce 的通信量下界可以精确写出：每个元素需要 2(n-1) 次传输，n 是 rank 数。NCCL Tests 给出的理想时间 t = S·2(n-1)/(n·B)，S 是字节数，B 是单 rank 对外带宽 [3]。

```text
ring all-reduce（reduce-scatter + all-gather 两阶段）
每个 rank 在链路上的总发送量 = 2(n-1)/n × S
阶段一：reduce-scatter，每 rank 聚合出 S/n 的规约结果
阶段二：all-gather，把各 rank 的部分结果广播给所有人
```

这里有个关键区分：algbw（algorithm bandwidth = S/t）不能直接和硬件峰值比。它随 rank 数下降，直接比峰值会误判「带宽没跑满」。要和峰值同口径比较的是 busbw（bus bandwidth），修正因子 AllReduce 为 2(n-1)/n [3]。8 卡 all-reduce 的因子是 2×7/8 = 1.75，意味着 algbw 只体现链路利用率约 57%（本文推算）。举个具体算例：8 个 rank、参数梯度总量 S = 1 GB，按 ring 走一圈，每个 rank 在链路上实际发送（并接收）1.75 GB 而非 1 GB——多出来的 0.75 GB 正是 reduce-scatter 与 all-gather 两阶段叠加的代价。

集合通信还是全局同步点：任何一个 rank 掉队，全集群一起等。NCCL 要求所有 rank 以相同 count 与 datatype 调用同一集合操作，否则是未定义行为，包括 hang、崩溃、数据损坏 [1]。所以带宽瓶颈常以「长尾抖动」而非「平均带宽低」的形式出现。

MoE/EP 场景下，all-to-all 是主瓶颈：稀疏化让计算变省，却把通信变成主角。MegaBlocks 用 block-sparse 重写 MoE 计算、不丢 token，端到端最多比 Tutel 快 40%、比稠密网络快 2.4 倍 [17]；DeepSeek-V3 则为跨节点 all-to-all 专门写通信 kernel，吃满 IB 与 NVLink 带宽 [18]。

## 三、DP、TP、PP、EP：谁在通信，谁在等待

四种并行策略的通信强度天差地别。DP（数据并行）的通信量正比于参数量——每步梯度 all-reduce，与 microbatch 大小无关，这是「参数量越大、同步越贵」的根源。ZeRO 靠消除显存冗余，让 100B+ 参数模型在 400 张 GPU 上跑到 15 Petaflops [15]。

TP（张量并行）通信最密集，每层前后各一次 all-reduce，因此通常被关在机内 NVLink 域——机内带宽是「必需品」。Megatron-LM 的层内并行只靠插入少量通信算子实现 [11]，每层引入两个互为共轭的通信操作（前向/反向各一次 all-reduce）[13]。

PP（流水并行）不含 all-reduce，但有 bubble 与激活存储代价。naive 组合在千卡级会撞墙：昂贵的跨节点通信、设备大量时间在等别人。Megatron-LM 的 interleaved pipeline 调度把吞吐提升 10% 以上，但 1 万亿参数模型在 3072 GPU 上仍只达到理论峰值的 52% [12]。

EP（专家并行/MoE）的 all-to-all 是当前大模型训练的主瓶颈。MoE 越稀疏，每 token 要跨卡搬的专家数据相对算力越多。DeepSeek-V3 跨节点 EP 的计算-通信比约 1:1，靠 DualPipe 把 all-to-all 与 PP 通信完全隐藏后才实现近满重叠 [18]。

工程上已被验证的路线是：把高通信量算子关进机内，用序列并行/选择性重计算换空间。序列并行 + 选择性激活重计算把激活内存降 5 倍、重计算开销降 90% 以上，在 2240 张 A100 上训练 530B 模型达到 54.2% MFU [13]。

## 四、硬件与拓扑：NVLink、InfiniBand 与昇腾 HCCS/HCCL

机内 scale-up 由 NVLink/NVSwitch 提供非阻塞 all-to-all，它决定 TP/EP 能否放机内。NVLink 每卡带宽第四代 900 GB/s、第五代 1,800 GB/s、第六代 3,000 GB/s，NVLink Switch 内含 SHARP 引擎做 in-network reduction 与 multicast 加速 [6]。GB200 NVL72 用 72 卡 NVLink domain 提供 130 TB/s 聚合 GPU 通信带宽 [7]。

机外 scale-out 由 InfiniBand 承担，速率一代翻倍：NDR 每 lane 100 Gb/s × 4X = 400 Gb/s，XDR 为 800 Gb/s [10]。DGX H100 每台配 8 张单口 ConnectX-7，默认最高 400 Gbps [5]。拓扑用 rail-optimized 胖树：DGX SuperPOD（H100）满配 127 节点，每 32 节点一组 rail-aligned，同 rail 内始终一跳可达，跨 rail 才走 spine 层 [9]。收敛比（oversubscription）是设计参数而非缺陷，DGX 侧连接按接近 4:3 轻度收敛 [9]。

昇腾侧，集合通信由 HCCL 承担，硬件是 HCCS 互联。HCCL 支持 AllReduce、Broadcast、AllGather、ReduceScatter、AlltoAll 等原语，链路/协议支持 HCCS、RoCE、PCIe，算法提供 Mesh、Ring、RHD、NHR 等；其中归约类操作通过「随路」实现、不占用计算资源，计算与通信并发流水执行 [19]。昇腾用「超节点」把 scale-up 域做大：Atlas 900 A3 SuperPoD 支持最多 384×NPU 高速互联、D2D 双向带宽 784 GB/s，官方描述为「384 张 NPU 像一台计算机一样工作」[20]。官方规格页另给出单机灵衢互联总带宽 8×784 GB/s（双向）、RoCE 每 NPU 400 Gbps（单向）[21]。

## 五、有效带宽不等于标称带宽：ECMP、PFC 与算法选择

标称峰值基本拿不到，衡量标准应是 busbw 的「达标率」——busbw 的定义本身就是「换算到可与硬件峰值比较的口径」[3]。换硬件之前，先定位是硬件不够还是没跑满。

```python
# 从 nccl-tests 的 algbw 换算 busbw 与达标率
# peak 用同口径标称值：NVLink 每卡 900 GB/s，或 IB 400 Gb/s ÷ 8 = 50 GB/s
n = 8                       # rank 数
algbw = 270.0               # 示例值：nccl-tests 输出的 algorithm bandwidth (GB/s)
busbw = algbw * 2 * (n - 1) / n   # AllReduce 修正因子
peak_per_gpu = 900.0        # NVLink 标称值，按每卡口径
ratio = busbw / peak_per_gpu
print(f"busbw={busbw:.1f} GB/s, 达标率={ratio:.1%}")
```

跑不满的软原因不少。ECMP 哈希不均会让部分链路先饱和——MegaScale 把「降低 ECMP hashing conflicts」列为网络设计目标，让同组通信尽量落在同一 ToR 下 [16]。RoCEv2 环境下，默认 DCQCN 下 all-to-all 可能引发拥塞与 Priority Flow Control（PFC）水位上升，需自定义拥塞控制并调整 retransmit timer 与 retry count 以便快速恢复 [16]；DGX SuperPOD 参考架构也把 congestion control 与 adaptive routing 列为必需能力 [9]。

collective 算法选择本身就是带宽优化。NCCL_ALGO 可选 Ring、Tree、NVLS、NVLSTree、PAT 等，其中 NVLS 与 NVLSTree 启用 NVLink SHARP offload，可把 all-reduce 下推到 NVSwitch 域执行 [2][6]。全局同步的尾延迟会被放大成全集群空等——所以评价集群要看 P99 而非只看均值（工程经验）。

## 六、工程 checklist：如何实测，如何判定「是不是带宽瓶颈」

第一步先算计算-通信比，再决定「加带宽」还是「做重叠」。ratio 接近 1 时加带宽收益有限，优先做计算-通信重叠（如 DualPipe 路线）[18]，或以调度换吞吐（interleaved PP +10%）[12]。

第二步用 nccl-tests 做可复现的带宽实测 [4]：

```text
# 单机 8 卡
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
# 多机 64 卡（每节点 8 卡）
mpirun -np 64 -N 8 ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```

第三步读输出只认 busbw 列，算达标率 = busbw / 硬件标称带宽，换算后与 NVLink/IB 标称值同口径比较 [3][6][10]。

第四步用扩展效率与 MFU 交叉验证是不是带宽受限：只测通信会漏掉 bubble 与重计算带来的等待。Megatron-LM 在 512 GPU 上训练 8.3B 模型保持 76% scaling efficiency [11]，而 1T 模型在 3072 GPU 上仅 52% 理论峰值 [12]；MFU 明显低于同代基线时，先查通信与 bubble，再查算力。

昇腾侧 checklist 先看 HCCL 选到的是哪条链路（HCCS / RoCE / PCIe）与哪个算法，再确认超节点逻辑规模是否把 EP 关进机内 [19][20]。

一句话收尾：带宽是分布式训练里增长最慢、却最容易被人忽略的那一环。加卡之前，先问一句——你的 busbw 达标了吗？

## 参考

[1] NCCL User Guide — Collective Operations：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
[2] NCCL User Guide — Environment Variables：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
[3] nccl-tests — PERFORMANCE.md：https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
[4] nccl-tests — README：https://raw.githubusercontent.com/NVIDIA/nccl-tests/master/README.md
[5] NVIDIA DGX H100 User Guide — Introduction：https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html
[6] NVIDIA NVLink 与 NVLink Switch 规格页：https://www.nvidia.com/en-us/data-center/nvlink/
[7] NVIDIA GB200 NVL72 产品页：https://www.nvidia.com/en-us/data-center/gb200-nvl72/
[8] NVIDIA H100 规格页：https://www.nvidia.com/en-us/data-center/h100/
[9] DGX SuperPOD（H100）参考架构 — Network Fabrics：https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html
[10] NVIDIA XDR InfiniBand 交换机手册 — Interfaces：https://networking-docs.nvidia.com/xdrswitcheshw/interfaces
[11] Megatron-LM（1909.08053）：https://arxiv.org/abs/1909.08053
[12] Efficient Large-Scale Language Model Training Using Megatron-LM（2104.04473）：https://arxiv.org/abs/2104.04473
[13] Reducing Activation Recomputation in Large Transformer Models（2205.05198）：https://arxiv.org/abs/2205.05198
[14] Language Models are Few-Shot Learners（2005.14165）：https://arxiv.org/abs/2005.14165
[15] ZeRO（1910.02054）：https://arxiv.org/abs/1910.02054
[16] MegaScale（2402.15627）：https://arxiv.org/abs/2402.15627
[17] MegaBlocks（2211.15841）：https://arxiv.org/abs/2211.15841
[18] DeepSeek-V3 Technical Report（2412.19437）：https://arxiv.org/abs/2412.19437
[19] HCCL——昇腾高性能集合通信库：https://www.hiascend.com/developer/techArticles/20240809-1
[20] 华为 Atlas 900 A3 SuperPoD：https://e.huawei.com/cn/products/computing/ascend/atlas-900-a3-superpod
[21] 昇腾 Atlas 650E 服务器规格页：https://www.hiascend.com/hardware/ai-server
