---
title: "国产超节点的互联底牌:UB 灵衢 2.0 的内存语义 vs NVLink 的 load/store 语义"
date: 2026-09-21T09:56:19+08:00
tags: ["超节点", "UnifiedBus", "灵衢2.0", "NVLink", "内存语义"]
author: "丁志强"
draft: false
---

如果把数千张 NPU 装进同一个机柜群,让它对外表现得像「一台」计算机,最难的部分不是把带宽堆到多少,而是让程序能用访存指令(load/store)直接摸到另一张卡上的内存。带宽决定「搬得快不快」,语义决定「能不能像访问本地内存一样访问远端内存」——后者才是超节点能否在逻辑上成为一台计算机的前提。本文对照两条路线:NVIDIA 的 NVLink/NVSwitch load/store 语义,与华为灵衢(UnifiedBus,简称 UB)2.0 的内存语义。

## 一、为什么互联语义比互联带宽更底层

互联的「语义」决定上层程序用哪类原语访问远端资源。load/store 是同步、可直接寻址的内存语义,message/transaction 是异步、需双方约定的消息语义——两者不是快慢之分,而是编程抽象与硬件耦合度之分。UB 直接点破这层关系:单边内存语义适合传大块数据,双边消息语义适合发通知,并以 Write/Send with Immediate 把「写数据 + 发通知」融合成单个硬件原语 [23]。

超节点存在两条技术路线:NVLink/NVSwitch 的域内互联,与 UB 灵衢的 bus 级对等互联。前者以 NVLink GPU Domain 逐代扩大规模,单 GPU 双向带宽从 900 GB/s(NVLink 4)提到 1,800 GB/s(NVLink 5)、3,000 GB/s(NVLink 6),NVL72 聚合 216 TB/s [6];后者在 Atlas 950 超节点上一次给到 8192 卡、互联带宽 16.3 PB/s [19]。两条路线对「规模上限 × 语义强度」的取舍不同,决定了生态与编程模型的走向。

更深一层是总线与网络的范式之别。总线面向节点内紧耦合系统,靠硬件仲裁与 credit 流控,延迟极低但扩展性差;网络面向节点间松耦合系统,报文存储转发,扩展性好但需要端到端拥塞控制。传统做法在超节点内用总线、超节点间用网络,形成两套技术栈与两种编程抽象,跨边界必须协议转换。UB 的核心主张是向上只暴露一套统一内存语义:物理上承认超节点内更像总线、超节点间更像网络,但编程抽象上把两者统一 [23]。

华为给出的「万卡级超节点架构」六大特征——总线级互联、平等协同、全量池化、协议归一、大规模组网、高可用性——是产业侧对「语义比带宽更底层」的官方口径。灵衢正是这一主张的载体,于 2025-09-18 在华为全联接大会正式发布 [19]。代价同样藏在语义里:把过去由操作系统与软件承担的内存管理、地址翻译、一致性维护下沉到硬件。这正是 UB 需要 UMMU/UBMMU、UBFM 等新硬件部件的原因——UB 把复杂度下沉到硬件,同时向应用暴露更多选择(事务序、负载均衡策略),新范式的胜利在于解决「性能与规模」的核心矛盾,而非消灭全部问题 [23]。

## 二、NVLink 的 load/store 语义:把 peer memory 当成显存

CUDA 官方把多 GPU 通信能力分为两类:peer-to-peer 批量内存传输,以及「细粒度 peer-to-peer GPU load/store 内存访问」。官方原文列出 "Fine-grained peer-to-peer GPU load/store memory access",并明确通过 NVLink 实现「更高带宽传输与更细粒度的设备间 load/store 操作」[1]。这是 load/store 语义在 NVIDIA 官方文档里最直接的表述。

P2P 直访的使能机制是一套显式权限模型:`cudaDeviceCanAccessPeer()` 查询可达性,`cudaDeviceEnablePeerAccess()` 按方向开启授权,授权默认关闭且单向。配合 UVA(统一虚拟寻址),同一指针可同时寻址两台设备的显存,使能后 kernel 可直接对 peer 指针做 LD/ST,无需 `cudaMemcpy` [1]。

```cuda
// 1) 查询并开启单向 peer access(默认关闭)
int canAccessPeer = 0;
cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);   // device 0 -> device 1 是否可达
if (canAccessPeer) {
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);            // 允许 device 0 访问 device 1 的显存
}

// 2) kernel 内直接对 peer 指针做 load/store,无需 cudaMemcpy
__global__ void remote_add(float *peer_buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) peer_buf[i] += 1.0f;   // 一次 peer load + 一次 peer store
}
```

「像本地内存一样读写」听起来简单,工程门槛却在内存序与可见性。CUDA 用 thread scope 分级定义传播范围:`thread_scope_block` 的一致性点在 L1,`.cluster`/`thread_scope_device` 在 L2,`thread_scope_system` 在 L2 及相连缓存,覆盖 CPU 与其他 GPU。跨 peer device 的 load/store 传播范围取决于 scope 选择,选错就会产生 data race——官方给出的反例是:block scope 的 store 与另一 block 的 device scope load 不构成原子关系 [3]。fence 与 acquire/release 是这套语义的正确性基础设施,`cuda::atomic_thread_fence(seq_cst, thread_scope_system)` 保证本线程此前的写对设备内、host、以及所有 peer device 的线程可见 [4]。

```cuda
// producer:写完数据后以 release 发布,scope 必须覆盖到 peer device
__global__ void producer(float *buf, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        buf[i] = compute(i);
    cuda::atomic_thread_fence(cuda::memory_order_release,
                              cuda::thread_scope_system);   // 跨设备必须用 system scope
}

// consumer:以 acquire 读取,保证看到 producer 此前的写
__global__ void consumer(const float *buf, int n, float *out) {
    cuda::atomic_thread_fence(cuda::memory_order_acquire,
                              cuda::thread_scope_system);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        out[i] = buf[i];
}
```

load/store 语义能覆盖整个机柜,前提是交换层把 remote 访问做得跟本地 L2 事务足够像。NVSwitch 在域内承担高基数交换与多播:技术白皮书描述其内部为全连接 non-blocking crossbar,数据经 CRC 保护并可 replay,内部数据通路/路由/状态用 ECC 保护 [12];Hot Chips 34 的演讲材料直言「NVLink-Port Interfaces match data-exchange semantics of L2 as closely as possible」[13]。归约与广播则由 NVSwitch 内的 SHARP 引擎与 multicast 承担,把 all-reduce/reduce/broadcast 从计算引擎卸载到交换机,fabric 内归约让传输数据量减半 [15];NCCL 的 NVLS 算法即依赖这一能力 [10]。

NVSHMEM 进一步把 one-sided 操作按底层链路分流:在 NVLink 上「直接翻译为 load/store 操作」,在网络上翻译为 RDMA 原语;经 NVLink 直连的 peer GPU 内存可用库返回的指针直接访问 [8]。这证明内存语义与消息语义不是二选一,而是同一 API 面在不同物理链路上的两种落地。

## 三、UB 灵衢 2.0 的内存语义:统一地址空间 + 双访问语义

灵衢的时间线锚点是理解「国产超节点底牌」的事实基础:2019 年起研究,灵衢 1.0 已随 Atlas 900 于 2025-03 起交付,灵衢 2.0 形成后于 2025-09 开放技术规范。官方口径是:基于灵衢 1.0 的 Atlas 900 超节点累计商用部署超过 300 套、服务 20 多个客户(2025-09-18 时间点);Atlas 950 超节点则基于灵衢 2.0 [19]。协议开放是生态底牌的关键一环:两份规范文档(灵衢®基础规范 2.0、灵衢®固件规范 2.0)由灵衢社区随《灵衢规范许可协议 V1.0》发布,许可给出全球范围、非排他、免费的版权许可与必要权利要求专利不诉承诺,同时限制「不得摘录或引用灵衢规范内容用于开发其他标准」[20]。

内存语义在 UB 上的物理支点,是统一全局物理地址空间。UB 把各 NPU 的 HBM 物理地址范围纳入统一的 UB 地址空间,本地 VA 可直接映射到远端 NPU 的 HBM 物理页,之后像本地内存一样访问。这是它比 RDMA 注册更轻量的根本原因:StrataCL 论文实测 UB 的内存注册延迟比 RDMA 平均快约 9×,注册由「远端导出物理内存句柄 → 本地导入 → 预留本地 VA → 映射」四步构成 [25]。

与 NVLink 最大的结构差异在于,UB 把「直接 load/store」与「异步读写原语」都写进同一协议栈,而不是把后者交给独立网卡。CANN 官方技术解读明确 Ascend 950「同时支持 Load/Store 的同步通信语义,和 URMA 异步消息通信语义」[24];URMA(Unified Remote Memory Access)是 UB 面向应用的统一编程抽象集合,据参与早期 UB 项目的作者视角,由分布式与并行软件实验室主任谭焜博士提出 [23],支持单边、双边、原子操作,是 UMDK 的应用通信基础 [28]。

地址翻译与鉴权被硬件化。Entity 是全局通信基本单元,EID 类似 IP 地址角色;UMMU(UB Memory Management Unit)把全局 UBA 翻译为本地物理地址并做鉴权,UBFM(UB Fabric Manager)只负责 EID 分配、路由配置与资源管理,不做数据转发 [24]。数据路径完全去中心化,这是企业级扩展的前提 [23]。落地层面,UB 同时支持 UBoE 组网,把 UB 协议承载在以太网上:华为口径称集群同时支持 UBoE 与 RoCE,相比传统 RoCE,UBoE 静态时延更低、可靠性更高、交换机与光模块数量更省,因此推荐 UBoE [19]。

内存语义要在 8000+ 卡规模上成立,可靠性必须分层解决。UB 的分层机制是:链路层 LLR、物理层 lane 降级与光模块 2+2 备份、传输层端到端重传,且传输可靠性(Reliability)与事务顺序性(Ordering)在设计上正交解耦 [23];官方口径进一步给出光互联可靠性提升 100 倍、互联距离超过 200 米、百纳秒级故障检测与保护切换 [19]。作者视角还给出了一个 8192 卡规模下光互联平均无故障时间(MTBF)超过 6000 小时的数字,属个人估算口径 [23]。

## 四、语义模型逐项对照:NVLink vs UB

只有逐维度对照才能看出:两条路线并非一条强一条弱,而是在不同维度各有结构性取舍。下表每一行同时给出两侧机制名与出处编号。

| 对照维度 | NVLink 侧(NVIDIA) | UB 灵衢 2.0 侧(华为/昇腾) |
| --- | --- | --- |
| 寻址与地址空间 | UVA 统一寻址 + P2P peer access + VMM mapping:远端显存映射进本地地址空间,同一指针可寻址 peer [1];另有 resource-centric 的 logical endpoint(id + offset)作为替代寻址模型 [2] | 统一全局物理地址空间:各 NPU HBM 物理地址范围纳入 UB 地址空间,本地 VA 经 UBMMU/UMMU 映射到远端 HBM 物理页,由 EID 做全局标识 [23][24][25] |
| 一致性模型与内存序 | 弱序;thread scope 分级 block/cluster/device/system,一致性点 L1/L2/L2/L2+相连缓存 [3];acquire/release + fence 保序 [4];NVSHMEM 在官方规范基础上放宽阻塞取数操作的保序以贴合 GPU 内存模型 [9] | 弱事务序,并把顺序保证拆成执行序与完成序两个正交维度,提供 NO/RO/SO 分级原语(个人视角来源,非官方规范译文)[23];URMA 默认允许乱序执行与乱序完成 [23] |
| 原子操作 | 多播对象 + `multimem` 指令:`multimem.ld_reduce` / `multimem.st` 由 NVSwitch 内 SHARP 引擎承担归约与多播 [5][13][15];NCCL 以 NVLS 算法使用该能力 [10] | URMA 提供远端原子操作(单边/双边/原子三类远端内存操作之一),由 UMDK 暴露给应用 [28][33] |
| 访问粒度与延迟 | 细粒度 peer load/store,官方称 NVLink 端口接口尽量贴近 L2 的数据交换语义 [1][13];NVLink 4 每 GPU 双向 900 GB/s,NVLink 5 1,800 GB/s,NVLink 6 3,000 GB/s [6] | 片内 die-to-die 0.2 µs / 210 GB/s,节点内 0.7 µs / 170 GB/s,跨节点 2.1 µs / 150 GB/s(以上为单方向带宽,与 NVLink 侧双向口径不同);跨节点相比片内延迟约 10×、带宽低 29% [25];华为官方口径 950 系列互联带宽 2 TB/s [19] |
| 可靠性与重传 | 链路级:数据经 CRC 保护并可 replay,NVSwitch 数据通路/路由/状态用 ECC 保护 [12];NVLink 6 Switch 增加控制面韧性、支持半配机架运行与交换托盘热插拔 [6] | 分层:链路层 LLR、物理层 lane 降级与光模块 2+2 备份、传输层端到端重传;传输可靠性(Reliability)与事务顺序性(Ordering)正交解耦 [23];官方口径:光互联可靠性提升 100 倍、互联距离超 200 米、百纳秒级故障检测与保护切换 [19] |
| 可扩展性 / 规模上限 | NVLink GPU Domain 8(NVLink 4)→ 72(NVL72,NVLink 5/6);聚合带宽 7.2 TB/s(NVLink 4)→ 130 TB/s(Blackwell NVL72)→ 216 TB/s(Vera Rubin NVL72)[6] | Atlas 950 超节点 8192 卡、互联带宽 16.3 PB/s、内存 1152 TB;Atlas 960 超节点最大 15488 卡;Atlas 950 SuperCluster 52 万余卡 [19];8192 卡下光互联 MTBF > 6000 小时(作者估算口径)[23] |
| 远端访问的失败语义 | 显式给出取舍:地址映射式访问没有错误传播通道,load/store 只能返回数据或 memory fault;为此引入带显式完成状态的 fabric put/get/atomics [2] | 本次调研来源未给出等价的 per-operation 完成状态口径,待核实 |

表里最容易被误读的一行是「访问粒度与延迟」:统一地址空间不等于均匀访问性能。同一超节点内,片内 die-to-die 为 0.2 µs / 210 GB/s,跨节点为 2.1 µs / 150 GB/s,延迟约 10×、带宽低 29% [25]。这是选型时最容易踩的坑,也是全文「语义比带宽更底层」的边界。出处口径也须分清:表中 [23] 的弱事务序分级属个人视角,已显式降级表述;[25] 的延迟/带宽矩阵属论文实测口径。

## 五、编程与生态后果:NCCL/NVSHMEM 与 UMDK/HCCL 的分野

「语义分野」在主流通信库 API 层面体现得最直接。NCCL 自 2.28 起提供 Device API,按语义把通信拆成三个模块:LSA(Load/Store Accessible,用内存 load/store 直接互访的设备)、Multimem(用硬件多播)、GIN(GPU-Initiated Networking,自 NCCL 2.28.7 起用于跨网络通信)。官方明确 LSA 路径「数据通过内存 load/store 指令传输」;multimem 依赖 NVLink SHARP(`NCCL_NVLS_ENABLE`),device API 依赖 symmetric memory 与 VMM(`NCCL_CUMEM_ENABLE`)[10]。

```cpp
// 注册 symmetric window:远端显存映射进本地地址空间
char *buffer;
size_t size = 256 * 1048576;                      // 256 MiB
ncclMemAlloc((void **)&buffer, size);
ncclWindow_t win;
ncclCommWindowRegister(comm, buffer, size, &win, NCCL_WIN_COLL_SYMMETRIC);

// device kernel:用 memory load/store 完成 in-place AllReduce(LSA 路径)
__global__ void inPlaceAllReduceKernel(ncclDevComm devComm, ncclWindow_t win,
                                       size_t offset, size_t count) {
    const int rank = devComm.lsaRank, nRanks = devComm.lsaSize;
    const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
    const int globalN = blockDim.x * gridDim.x * nRanks;
    for (size_t o = globalTid; o < count; o += globalN) {
        float v = 0.f;
        for (int peer = 0; peer < nRanks; peer++)
            v += ((float *)ncclGetLsaPointer(win, offset, peer))[o];  // peer load
        for (int peer = 0; peer < nRanks; peer++)
            ((float *)ncclGetLsaPointer(win, offset, peer))[o] = v;   // peer store
    }
}
```

NVSHMEM 用 PGAS + symmetric heap 把多卡内存聚合成一个分区全局地址空间,提供 kernel 内可发起的 one-sided get/put/AMO,把通信从 CPU 手里拿走,是 TP/EP 等紧耦合并行形态的编程基础 [8][9]。

```cpp
// symmetric heap 上分配对称缓冲区,所有 PE 用同一地址看到同一块全局空间
float *buf = (float *)nvshmem_malloc(n * sizeof(float));

// device kernel 内发起 one-sided put:把本 PE 数据写入 peer,无需对方参与
__global__ void do_put(float *buf, size_t n, int pe) {
    nvshmem_float_put(buf, buf, n, pe);   // 弱序:需要 fence/quiet 保序与完成
}

nvshmem_fence();   // 保证后续操作观察到的顺序
nvshmem_quiet();   // 等待本 PE 发起的非阻塞操作全部完成
```

NVSHMEM 的可见性语义随链路组合变化,是「同一套 API、不同语义强度」的典型陷阱:在只有 NVLink 的系统上,barrier/quiet/wait 等同步操作保证对 symmetric object 更新的全局可见性;在 NVLink + InfiniBand 混合系统上,只保证本 PE symmetric object 的可见性 [9]。官方文档分两种系统配置分别给出保证范围。

国产侧的对应栈是 UMDK(灵衢内存语义开发包),组件包括 URMA(统一内存语义)、CAM(超节点通信加速库,北向对接 vLLM/SGLang/VeRL)、URPC(统一远程过程调用)、ULOCK、USOCK。UMDK 定位为「以内存语义为核心」的分布式通信软件库,为数据中心网络、超节点内、服务器内卡间提供高性能通信接口,代码托管于 openEuler 镜像仓 [28]。

内存语义红利最终由谁吃掉,StrataCL 给出一条完整证据链:registration-on-allocation(分配即后台注册)、shadow virtual addressing(所有 NPU 用同一 VA 看到同一 buffer)、full-mesh 编程抽象(用 UB 远端 load/store 单步完成通信)、NPU 驱动的 SDMA offload(核只提交 DMA 描述符)。在 CM384 上,集合通信 bus 带宽最高提升 1.6×、MoE dispatch/combine 最高 1.4×;三个生产负载上,LLM 推理吞吐提升 1.9×、P99 TTFT 降低 2.2×、LLM 与 Recsys 训练迭代时间分别缩短 1.4×/1.3×,均为论文实测口径 [25]。对并行形态的影响更直接:小中消息下,依赖多步同步的 ring 类算法被单步 full-mesh 反超——同步开销占 ring 端到端时间在小消息下超过 50%、64 KiB 时约 77%,full-mesh 在 1 MiB 以下平均快 2× 以上、64 KiB 时最高 4.5×,8 MiB 前保持领先,超过 16 MiB 后 ring 因跨节点流量与 fanout 更优而重新占优 [25]。

## 六、工程后果与选型:什么时候用内存语义,什么时候用消息语义

### 6.1 两条路线的赢面场景

load/store 内存语义的赢面是紧耦合、细粒度、小消息、低延迟。但「统一地址空间」不等于「均匀访问性能」:CM384 上访问延迟与带宽矩阵为片内 die-to-die 0.2 µs / 210 GB/s、节点内 0.7 µs / 170 GB/s、跨节点 2.1 µs / 150 GB/s,跨节点相比片内延迟约 10×、带宽低 29% [25]。消息语义的赢面则是跨机柜、可扩展、容错、异构组件接入。内存语义的硬约束在于「访问要么成功要么 fault」,缺少错误传播通道:CUDA 官方在 Compute Fabric Transport 一节明确指出,虚拟地址模型下内核对 peer 地址发起 load/store 时硬件只返回数据或 memory fault,没有中间结果可供程序检查处理;为此才引入 resource-centric 的 logical endpoint(id + offset)与带显式完成状态的 put/get/atomics/reductions,失败可重试或改路 [2]。

### 6.2 规模上限与代价

规模上限与故障域的约束来自拓扑与交换层设计。UB-Mesh 采用分级本地化 nD-FullMesh 拓扑(1D 板内 → 2D 机架内 → 3D 以上跨范围),理由之一是 DP 类集合通信虽只占总流量不到 2% 却需要长距传输;相比传统 Clos 架构,成本效率 2.04×、网络可用性高 7.2%、LLM 训练线性度在 1×–32× 规模内超过 100%,且「一跳覆盖范围 = 交换机端口数 × 芯片端口数」,芯片基数翻倍即一跳超节点规模翻倍(论文口径)[27][26]。两侧的官方数字对比:NVLink 侧 GPU Domain 从 8(NVLink 4)扩到 72(NVL72,NVLink 5/6),Vera Rubin NVL72 聚合 216 TB/s [6];UB 侧 Atlas 950 超节点 8192 卡、16.3 PB/s、内存 1152 TB,Atlas 960 超节点最大 15488 卡,Atlas 950 SuperCluster 由 64 个超节点互联成 52 万余卡集群 [19]。

与计算争资源是 engineer 视角最容易被忽略的代价:UB 带宽并非免费。论文实测要跑到 UB 峰值带宽的 95%,需要 24 个 NPU core 参与,占每 die 48 个核的一半,直接影响 overlap(重叠)策略 [25]。

### 6.3 结论与工程 checklist

判断一段并行程序该走内存语义还是消息语义,可依三条判据 [23]:对端是否需在场——单边即可完成的读/写走内存语义;是否存在多写者与覆盖冲突——需要多发送方在不确定时刻通知同一接收方时,消息语义更合适;数据 vs 通知——大块数据走单边内存语义,通知走双边消息语义。语义融合原语 Write with Immediate / Send with Immediate 在硬件层把「传数据」与「发通知」合并为一个原子操作,消除了「先写数据、再发通知」两步的额外延迟,也避免了单边写与双边通知之间的乱序 [23]。

可实测手段清单:NVIDIA 侧可用 `nvbandwidth` 官方工具做带宽/延迟/超节点内 P2P 实测 [16];CUDA 侧 P2P 直访可通过使能 peer access 后由 kernel 直接对 peer 指针做 LD/ST 来验证 [1]。内存语义正确性 checklist(用户可直接照做):(a) CUDA 侧核对 thread scope 是否覆盖到目标 peer device,跨设备必须用 system scope 的 fence/atomic [3][4];(b) NVSHMEM 侧核对 `nvshmem_fence`/`nvshmem_quiet` 使用是否正确,并注意 NVLink-only 与 NVLink+IB 混合系统可见性语义不同 [9];(c) UB 侧核对 UMMU 映射权限与 EID/Token 鉴权模型(本项依据当前来源尚待核实,建议以 UMDK/HCCL 官方文档补全)。组网选型 checklist:华为推荐 UBoE 而非 RoCE(静态时延更低、可靠性更高、交换机与光模块更省)[19];NVLink 侧需确认目标域规模(NVLink 4 为 8 卡域,NVLink 5/6 经 NVL72 可到 72 卡域)与是否使用 SHARP/NVLS 归约 [6][10]。

## 参考

[1] CUDA Programming Guide — 3.4 Programming Systems with Multiple GPUs: https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html
[2] CUDA Programming Guide — 4.18 Compute Fabric Transport: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/compute-fabric-transport.html
[3] CUDA Programming Guide — 3.2 Advanced Kernel Programming(thread scopes 表): https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html
[4] CUDA Programming Guide — 5.4 C/C++ Language Extensions(内存 fence): https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html
[5] CUDA Driver API — Multicast Groups: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MULTICAST.html
[6] NVIDIA NVLink & NVLink Switch 官方页: https://www.nvidia.com/en-us/data-center/nvlink/
[7] NVIDIA OpenSHMEM Library (NVSHMEM) 文档首页: https://docs.nvidia.com/nvshmem/api/index.html
[8] NVSHMEM — Introduction(含 Communication Transports): https://docs.nvidia.com/nvshmem/api/introduction.html
[9] NVSHMEM — Memory Model: https://docs.nvidia.com/nvshmem/api/gen/mem-model.html
[10] NCCL User Guide — Device API(LSA / Multimem / GIN): https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html
[11] NCCL User Guide 首页: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
[12] NVIDIA NVSwitch: The World's Highest-Bandwidth On-Node Switch(技术白皮书 PDF): https://images.nvidia.com/content/pdf/nvswitch-technical-overview.pdf
[13] Hot Chips 34 — The NVLink-Network Switch 演讲 PDF: https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf
[14] NVIDIA 技术博客 — Upgrading Multi-GPU Interconnectivity with the Third-Generation NVIDIA NVSwitch: https://developer.nvidia.com/blog/upgrading-multi-gpu-interconnectivity-with-the-third-generation-nvidia-nvswitch/
[15] NVIDIA 技术博客 — Advancing Performance with NVIDIA SHARP In-Network Computing: https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/
[16] NVIDIA nvbandwidth(GitHub): https://github.com/NVIDIA/nvbandwidth
[17] NVIDIA GB200 NVL72 官方页: https://www.nvidia.com/en-us/data-center/gb200-nvl72/
[18] DeepEP(GitHub / DeepSeek): https://github.com/deepseek-ai/DeepEP
[19] 华为 — 徐直军 HC2025 主题演讲《以开创的超节点互联技术,引领AI基础设施新范式》: https://www.huawei.com/cn/news/2025/9/hc-xu-keynote-speech
[20] 灵衢社区 — 灵衢规范许可协议 V1.0(载明基础规范 2.0 / 固件规范 2.0): https://www.unifiedbus.com/zh/specification-license-agreement-v1
[21] 灵衢社区 — 资讯:以开创的超节点互联技术,引领AI基础设施新范式: https://www.unifiedbus.com/zh/news/hc-xu-keynote-speech
[22] 灵衢社区 — 首页: https://www.unifiedbus.com/zh
[23] Bojie Li —《Unified Bus 背后的思考》: https://01.me/2025/09/a-story-of-unified-bus
[24] CANN 开发者社区 —《面向Ascend 950,CANN技术架构的变与不变》: https://cann.csdn.net/69d8a96e54b52172bc684f2e.html
[25] arXiv 2607.26444 — StrataCL: Fabric-Native Communication Library for Production Supernodes: https://arxiv.org/abs/2607.26444
[26] arXiv 2609.16787 — Nested Parallel von Neumann Architecture and Nested BSP: https://arxiv.org/abs/2609.16787
[27] arXiv 2503.20377 — UB-Mesh: a Hierarchically Localized nD-FullMesh Datacenter Network Architecture: https://arxiv.org/abs/2503.20377
[28] openEuler 镜像 — UMDK README_zh(灵衢内存语义开发包): https://github.com/openeuler-mirror/umdk/blob/master/README_zh.md
[29] Gitee — openeuler/umdk: https://gitee.com/openeuler/umdk
[30] 华为企业支持 — HPC Cluster Computing Solution 24.0.0 术语&缩略语(灵衢计算网络): https://support.huawei.com/enterprise/zh/doc/EDOC1100502782/4bd6aba5
[31] 昇腾社区 — CANN 商用版 9.0.0 通信算子开发:简介(支持 PCIe/HCCS/RoCE/UB): https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/commopdev/hcclopdev_000001.html
[32] 昇腾社区 — HCCL 集合通信库:术语与相关概念: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/hccl/hcclug/hcclug_000002.html
[33] GitCode — UMDK URMA User Guide(中文): https://gitcode.com/openeuler/umdk/blob/master/doc/ch/urma/URMA+User+Guide.ch.md
[34] 昇腾社区 — CANN 通信库(HCCL)产品页: https://www.hiascend.com/cann/hccl
