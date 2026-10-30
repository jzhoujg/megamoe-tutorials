# 第六章｜NPU 和 GPU：两套硬件，两种掩盖哲学

> **入门主线 · 提高篇 3/3** ｜ [返回目录](../README.md)

**一句话：本教程中的 NPU 实现用 cube/vector 指令流组织并发，GPU 上可以用 program/warp 分工组织并发；两者都要回答同三个问题——谁来做、何时能读、何时能复用缓冲。优化的目标是缩短关键路径。**

本章以当前工作区的 Ascend 单 kernel 前向为主，GPU 对照取自 Triton-distributed-ascend 的 NVIDIA 实现。代码按函数名定位，带 `...` 的片段省略了参数或分支，用来读结构。历史性能数据另标测试条件，不能视为当前版本的新测量。

## 6.1 共同的问题：有空闲资源，还要有能提前做的工作

第三章讲了“边收边算”，这里再追问一句：**是谁在收，谁在算？**

常见做法有两类：让不同 worker 分工，或让不同执行引擎同时推进。两类可以组合：一个 GPU SM 内也有矩阵计算与异步搬运，NPU 的 vector 子核之间也可以分工。因此，“NPU 只能核内融合、GPU 只能切 SM”会把设计空间讲窄。

要让通信与 GEMM 重叠，至少要满足三件事：

1. **依赖允许**：当前 tile 到齐就能算，不必等整批；要先发的未来 tile 也不能被无关等待挡住。
2. **资源允许**：计算与搬运有可并行使用的执行资源，且没有因争用内存带宽、片上空间而得不偿失。
3. **交接正确**：信号发布后数据对消费者可见；复用缓冲前，上一轮消费者已经读完。

下面的 cv、vv 和 GPU worker 分工，都是这三个条件的具体实现。

## 6.2 NPU：cube/vector 如何交接（cv 融合）

在本实现采用的编程模型里，一个混合 program 协调 **cube 矩阵计算侧与两个 vector 子核**，`sub_vec_id()` 区分 vector lane 0/1。MTE 等流水线负责相应的数据搬运。这里用的是执行分工的抽象；具体 AIC/AIV 配比与物理组织要看芯片型号。

片上存储也有分工：UB 主要承载 vector 侧的数据，L0A/L0B 是矩阵输入缓冲，L0C 保存矩阵累加结果。**它们不是所有引擎都能任意读写的一块共享内存。** 当前 FC1 用 `_fixpipe_f322bf16` 把 L0C 的 FP32 累加结果转换并送入 BF16 UB 缓冲，再交给 vector 做激活。

`al.scope(core_mode=...)` 声明代码属于哪一侧，后端据此组织指令流。它提供组织并发的方式；实际重叠仍由数据依赖和资源决定。看 `_partition_pipeline_fc1_activation_group_ub` 中的一组交接：

```python
# src/mega_moe/kernels/fused_forward.py:808-819（摘，buffer 0 分支）
if pipeline_step >= 2:
    if pipeline_step % 2 == 0:
        _wait_fc1_vector_ack(0)
    else:
        _wait_fc1_vector_ack(1)
if pipeline_step % 2 == 0:
    _fixpipe_f322bf16(gate_acc, gate_buffer_0,
                    dual_dst_mode=al.FixpipeDualDstMode.ROW_SPLIT)
    _fixpipe_f322bf16(up_acc, up_buffer_0,
                    dual_dst_mode=al.FixpipeDualDstMode.ROW_SPLIT)
    al.sync_block_set('cube', 'vector', 8, al.PIPE.PIPE_FIX,
                      al.PIPE.PIPE_V)
```

这不是只发一个“算完了”的通知，而是一套**双缓冲的生产—消费—释放协议**：

| 交接 | 源码事件 | 允许后续做什么 |
|---|---|---|
| Cube → Vector：结果已送入 UB | buffer 0/1 分别用事件 8/9，`PIPE_FIX → PIPE_V` | 两个 vector 子核各读自己的行片段，做激活与写回 |
| Vector → Cube：本轮消费结束 | buffer 0/1 分别用事件 10/11，`PIPE_MTE3 → PIPE_FIX` | Cube 可以覆盖对应 UB 缓冲 |

```python
# 同文件:592-595，等待两条 vector lane 应答
@triton.jit
def _wait_fc1_vector_ack(SLOT: tl.constexpr):
    al.sync_block_wait('vector', 'cube', 10 + SLOT,
                       al.PIPE.PIPE_MTE3, al.PIPE.PIPE_FIX)
```

**真正值得学的是 wait 放在哪里。** 源码先完成下一 tile 的 load 和 `tl.dot`，到即将覆盖 UB 时才等上一轮的 ack。这样，下一块矩阵计算就有机会与上一块激活重叠；若把 wait 提到 GEMM 前，会多引入一段本可避免的等待。两组 UB 缓冲分别服务相邻 tile，每组包含 gate/up 数据，代价则是额外片上空间。

还要分清两层同步：上面的 `sync_block_set/wait` 管的是这个混合 program 的 cube/vector 交接；dispatch、FC2 完成和回传到齐，则由 GM 中的信号槽、`signal_op`、`dl.wait` 与 `dl.consume_token` 连接不同 program 或 rank。**局部 UB 已释放，不代表远端 token 已到齐。** `disable_auto_sync=True` 也要求作者显式维护相应依赖，不能理解成“同步可以省略”。

多 stream 的教训同样要放回具体路径。`ops/backward.py:180-194` 记录：2026-08-13，Kimi-K3 w8 t4k、npu-grouped wgrad 后端启用 `MOE_WGRAD_STREAM=1` 后，记录耗时从 112 增至 1402 ms，约慢 12.5×。它说明那条旁路 wgrad 与混合 kernel 并发的方案失败了。该文件 `use_dual` 分支又把纯 cube 的 wgrad 与纯 vector 的激活反向放进两个 stream，并用事件交接。**软件 stream 不自动对应独立硬件资源；能否重叠、重叠是否更快，都要看具体 kernel 组合。**

## 6.3 vv 融合：两个 vector 子核怎样分工

`sub_vec_id()` 可以把两个 vector 子核当成两个 worker。本文把这种协作称为 vv 融合；它与 GPU 的角色划分相似，但 worker 的资源与调度方式并不相同。

先纠正一个容易形成的印象：**lane 1 并非整个 kernel 都只通信，lane 0 也并非独自完成激活。** 当前代码按阶段换角色：

| 阶段 | lane 0 | lane 1 | 源码入口 |
|---|---|---|---|
| 普通路由 scatter | 负责一部分 expert bin | 负责另一部分 expert bin | `_scatter_stable_routes`；MoonEP scatter 另走 lane 0 分支 |
| 波流水 dispatch | 不执行此处 dispatch | 预发前两波，随后发未来波 | `_run_dynamic_wave_pipeline` |
| FC1 后激活 | 处理 tile 的一半行 | 处理另一半行 | `_partition_pipeline_fc1_activation_group_ub` |
| 回传与最终 top-k 归约 | 按 worker 编号分摊任务 | 按 worker 编号分摊任务 | `_return_dynamic_wave` / `_reduce_topk_rows` |

```python
# src/mega_moe/kernels/fused_forward.py:1530-1536（摘，预发阶段）
with al.scope(core_mode='vector', disable_auto_sync=True):
    if sub_vec_id() == 1:
        for prepared_wave in range(2):
            if prepared_wave < global_waves:
                ...  # 可选打点
                _dispatch_dynamic_wave(...)

# 同文件:852，激活阶段两 lane 各取半块行
vector_row_base = sub_vec_id() * vector_block_m

# 同文件:1677-1678，回传阶段两个子核都参加
_fc2_end = _return_dynamic_wave(pid * 2 + sub_vec_id(), step - 2, ...)
```

最后一行的 `pid * 2 + sub_vec_id()` 是 **vector worker 编号**，不是波槽号；波号是下一个参数 `step - 2`。跨阶段复用 worker，能减少固定角色带来的闲置，但也会让 lane 1 的 dispatch、激活、回传互相影响。

vv 的收益要按阶段判断。元数据阶段通常没有可直接并行的 GEMM，启用另一个闲置 vector 子核，就可能缩短整层的前置等待；稳态波流水中，如果 vector 工作已经藏在 cube 计算后面，再缩短它也未必改善 E2E。小任务还可能不值得分工：当前普通 scatter 在 `NUM_EXPERTS <= 32` 时保留单 bin-block 路径，其循环可能只有 lane 0 有工作。

开源项目实践中的前向打点记录保留了两组有用的历史对照：

- **元数据优化有收益。** 2026-09-18 的 8 卡 Kimi-K3 t4k、fp16 保存基线，warmup 5 / samples 20，routing 的 p50 约 10.0 ms，其中 scatter 5.66 ms、counts publish 2.68 ms。次日记录的“发布向量化 + 双 lane 散射 + pad bin 裁剪”组合优化，把 routing 从约 10.0 降至 3.4 ms，**E2E 最小值**从约 28.0 降至 23.0 ms，两轮复现。这里有多项改动，不能把全部收益归给 vv，也不能把 E2E min 与基线表中的 p50 28.85 ms 混为同一统计量。
- **激活括号变长，不足以证明激活算得慢。** 同一历史基线里，两 lane 激活括号为 0.46 / 5.76 ms。把未来波 dispatch 挪到激活 signal 之后，lane 1 的括号从约 5.7M 变为 6.0M ticks，E2E 中性。这次重排没有支持“移动 dispatch 就能消除访存竞争、加速激活”的预期，但也不能排除所有资源竞争。括号可能包含等待；下一步应检查哪次等待真正推迟了 FC2。

当前 scatter 已继续演进，6.5 会看它的代码。上述数字只说明当时的优化结果；后续路由优化记录明确区分了 host 验证与待完成的设备编译、正确性和性能验收。

## 6.4 总纲：从算力平衡走向关键路径平衡

“先 cv 后 vv”可以作为检查稳态流水的起点：先找能与 GEMM 并行的 vector 工作，再检查两个 vector 子核是否分工失衡。**它不是固定优化顺序。** 如果路由元数据阻塞整个流水启动，就应先优化元数据。

一个简化的耗时模型是：

```text
整层耗时 ≈ 前置元数据 + 流水填充 + 稳态推进 + 尾部排空
稳态每波间隔 ≳ max(Cube 服务时间, 较忙的 Vector lane 服务时间,
                    共享带宽约束下的数据搬运时间)
```

这只是理解下限的模型：同步、调度和到达不均匀还会把间隔拉长。`服务时间` 指实际工作占用，不能直接拿包含等待的计时括号代入；单独跑通信和 GEMM 的时间，在并发时也可能因带宽竞争而改变。

代码还揭示了两个约束：**FC1 和 FC2 都使用 cube，需要在同一侧排队；dispatch、激活、回传也共享 vector 侧的执行与访存资源。** 多个阶段处于流水中，不等于它们各自有一套独占引擎。图上画出四个工位，并不会增加硬件吞吐。

因此可以按这个顺序分析一次优化：先找延迟了后续工作的阶段，再找其中的实际计算、搬运或等待；针对原因改一项，对照阶段时间与 E2E；最后检查是否把成本转移到了别处。尤其不要只凭“两条 lane 的累计时间小于整波时间”，就断言剩下的时间全部属于 cube 瓶颈。

## 6.5 NPU 的亲和设计：布局、批量处理与 UB 预算

NPU 上通常应尽量把数据组织成大块、连续、对齐的访问。这也有利于 GPU；在本实现里，额外要关注 Triton Ascend 后端如何把向量临时量映射到 UB。下面三个例子分别回答“数据怎么排”“索引怎么处理”“临时量怎么存”。

**第一，按消费者需要的布局写，消掉后续重排。** `dispatch_fc2_bwd.py:4-8` 的模块注释写得很直接：Vector 以 expert-major 布局写 `peer_mem`，Cube 连续读取，因此不需要消费端的 `local_sort` gather。源 token 原来不连续，生产端仍要按路由取行；收益是把这次取行与发送结合，免去接收端再 gather 一遍。不能把它解释为所有离散访问都消失了。

**第二，把逐标量串行工作改成有界的向量块。** `_publish_count_row` 仍由 `pid == 0` 的 lane 0 调用，但归并方法已经改变：

```python
# src/mega_moe/kernels/fused_forward.py:237-245
bin_offsets = tl.arange(0, NUM_BINS_PAD)
counts = tl.zeros((NUM_BINS_PAD,), dtype=tl.int32)
for core_id in range(0, NUM_PROGRAM_CORES):
    counts += tl.load(
        core_bucket_cursor_ptr
        + core_id * cursor_stride
        + bin_offsets
    )
```

旧版逐 bucket、逐 core 累计；现在每次读一整行，沿 bucket 维并行相加。**优化不一定要多开 worker，也可以减少一个 worker 内的串行依赖。** 历史记录的约 28,672 次 load 对应 `896 experts × 32 cores`；2.68 ms 除以次数约为 93 ns，只是整段平均成本，包含循环和地址计算，不能当作硬件单次 load 延迟。

scatter 则进一步改变算法。当前 `_scatter_stable_routes` 在 `NUM_EXPERTS > 32` 时调用 `_scatter_routes_by_ordinal`：每个 lane 拥有互斥的 expert 半区，每次读 32 条 route，求“同一 expert 在本块之前出现了几次”，再加到该 expert 的 cursor 上：

```python
# 同文件:432-441（摘，valid 仅包含本 lane 负责的有效 expert）
matches = (experts[:, None] == experts[None, :]) & earlier & valid[None, :]
ordinal = tl.sum(matches.to(tl.int32), 1)
rows = tl.gather(cursors, local_experts, 0) + ordinal
```

两个 lane 都扫描 route 列表，但只写自己 expert 区间的结果。这样减少了按 expert block 反复扫描 route 的工作，同时保留稳定顺序；代价是块内两两比较、gather 与 histogram，仍需实测。稳定写入位置满足：

```text
send_row = 更小 expert 的总行数
         + 更小 core 编号对本 expert 的贡献
         + 当前 core 内更早 route 对本 expert 的贡献
```

这是并行化后必须保持的正确性条件。不能为了均摊工作，让两个 lane 无保护地修改同一个 expert cursor。也要看实际路由分布：对半切 expert 编号，并不保证热门 expert 的工作量恰好对半。

类似地，`moonep_planning.py` 的 `_kernel_moonep_b0_b1` 用一个 `(R, BLOCK_E)` 块读取多 source 的计数，再归约，替代按 source 编译期逐个展开。块化减少展开和调度成本；块越大，临时数据也越多，仍要受 UB 预算约束。

**第三，减少重复临时量，控制存活范围。** `fused_forward.py:1500-1507` 记录，2026-09-17 的打点实现曾在每个现场分别构造小临时量，后端自动分配过多 UB buffer 导致溢出。当前代码复用 `dummy_ts`、时间戳行基址和累计数组。

这里的 *scattered per-site* 是指打点代码分散在多个位置，**不是 scatter/gather 内存访问的同义词**。这个案例证明的是临时量与后端缓冲分配会影响 UB 占用，不能推出“每个离散地址必然单独占一块 UB”。改块大小、双缓冲或打点时，都应检查编译后的资源占用，并分别验证 `TIMING=0/1`；关闭打点能消除计时代码，也不意味着开启时毫无扰动。

## 6.6 GPU：用真实代码看 worker 分工

GPU 的 SM 内有 CUDA 执行单元和 Tensor Core；支持 TMA 的架构还可异步搬运 tensor tile。**TMA 的 GM/shared-memory 搬运与设备 copy engine 的跨设备复制是不同机制**，不宜合称为“SM 里的通信引擎”。通信可以由 SM 执行，也可以采用 copy engine 等路径。

工作区中有一个直接体现“班组制”的例子：Triton-distributed-ascend 的 NVIDIA `gemm_allreduce.py`。它实现 GEMM + all-reduce，可用于理解角色分工；它不是 UniEP 的 MoE 实现。

```python
# python/triton_dist/kernels/nvidia/gemm_allreduce.py:583-598
# kernel_fused_gemm_allreduce（摘，省略参数）
global_pid = tl.program_id(axis=0)
if global_pid < NUM_COMM_SMS:
    if USE_LD_REDUCE:
        consumer_all_reduce_kernel(...)
    else:
        consumer_all_reduce_load_store_kernel(...)
else:
    kernel_persistent_gemm_notify(..., FUSE_START_OFFSET=NUM_COMM_SMS)
```

前三个关键点可以直接沿调用链核对：

1. **按 program 分角色。** 前 `NUM_COMM_SMS` 个 program 负责通信/归约，其余做 persistent GEMM；后者减去 `FUSE_START_OFFSET` 得到本组编号，再以 `NUM_GEMM_SMS` 为步长取 tile。`program_id` 是逻辑工作编号，代码没有把它硬绑定到某个物理 SM 编号。
2. **按结果就绪交接。** GEMM 写出结果后发布 release 信号；load/store 版本的消费者轮询所需 peer 的 acquire 信号，再读对称内存中的数据并归约。算出一块就能推进对应通信，双方的 tile 映射必须一致。
3. **为相互等待提供调度条件。** `low_latency_gemm_allreduce_op` 用通信 program 数加实际 GEMM program 数构造 grid，传入 `use_cooperative=True` 并请求 cooperative launch；kernel 尾部有 grid barrier。部署时须满足后端支持和驻留资源要求，否则等待中的消费者可能阻塞尚未获调度的生产者。cooperative launch 支持 grid 协作，数据可见性仍靠具体的内存与信号协议。

这种分工有真实成本：给通信更多驻留资源，通常会挤占 GEMM 可用资源；给得太少，通信又可能拖住流水。具体比例应按矩阵形状、通信量与网络测量。warp specialization 则把角色划分下沉到一个 thread block 内的不同 warp，粒度和资源代价又不同。

GPU 也不必总划出一组 SM 搬数据。相邻的 `all_to_all_single_gemm.py` 提供 `all_to_all_copy_engine`：分块复制后设置信号，GEMM 消费端用 `dl.wait` 和 `dl.consume_token` 连接就绪与读取。这条路径说明，**通信生产者可以不同，而数据到齐才消费的原则相同。**

UniEP（HPDC'26）是把上述问题扩展到整层 MoE 的参考工作。本章对 UniEP 只作论文层面的指引；这里能够逐行核对的 GPU 代码是上述 Triton-distributed 实现，不据此推断 UniEP 的具体 SM 配比或内部同步细节。

## 6.7 对照表：比较实现选择，避免当成硬件定律

| 比较项 | 本教程的 Ascend 实现 | 本章 NVIDIA 代码示例 |
|---|---|---|
| 工作分配 | `al.scope` 组织 cube/vector；vector 子核按阶段分工 | fused GEMM + all-reduce 按 program 划分通信与计算角色 |
| 矩阵结果交给消费者 | Fixpipe → UB → vector 激活 | GEMM store → release 信号 → 消费者 acquire |
| 局部协作 | `sync_block_set/wait` 管 cube/vector 缓冲交接 | block 内同步；示例尾部用 cooperative grid barrier |
| 跨 program / rank 就绪 | GM 信号槽、`signal_op`、`dl.wait`、`consume_token` | 对称内存信号与 acquire/release；A2A + GEMM 也用 `dl.wait` |
| 并发资源代价 | UB 双缓冲；vector 任务竞争；FC1/FC2 共用 cube | 通信与计算争用驻留资源、寄存器/shared memory 与带宽 |
| 调优重点 | 元数据关键路径、两 lane 负载、缓冲生命周期 | worker 比例、tile 就绪粒度、驻留与前进条件 |

## 6.8 小结

读这类 kernel，可以沿着三条线走：**谁执行这段工作，哪次等待让它开始，哪次应答允许缓冲被覆盖。** cv 融合利用不同引擎的并行机会，vv 分工改善 vector 内部的任务分配，GPU 的 program/warp 分工则提供另一种资源组织方式。布局与批量处理能减少必须做的工作；阶段打点和 E2E 对照决定这些变化有没有缩短关键路径。

本章的前向历史数据来自 `MOE_FWD_TIMING` 记录，工具入口见第九章 9.3/9.4；旁路 stream 的数字来自反向源码注释，两者不是同一套测量。第四章解释信号与可见性，第九章展开波流水。**下一章进入进阶篇：看 MoonEP 怎样把负载均衡接进大算子。**

源码参考：NPU 前向、反向 stream 分支、反向 dispatch 布局、MoonEP 块化规划、[GPU 分工示例](../../Triton-distributed-ascend/python/triton_dist/kernels/nvidia/gemm_allreduce.py)、[GPU copy-engine 示例](../../Triton-distributed-ascend/python/triton_dist/kernels/nvidia/all_to_all_single_gemm.py)。

---

← [上一章](05-ep-and-framework.md) ｜ [返回目录](../README.md) ｜ [下一章：MoonEP](07-moonep.md) →
