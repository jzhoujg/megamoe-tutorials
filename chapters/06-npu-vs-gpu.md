# 第六章｜NPU 和 GPU：两套硬件，两种掩盖哲学

> **入门主线 · 提高篇 3/3** ｜ [返回目录](../README.md)
>
> 这个部分属于通识性对比，建议阅读

**一句话：NPU 的掩盖单位是"核内指令流"——一个 AI Core 拆出 cube/vector 双流，通信原语和 GEMM 在同一个核里跳舞；GPU 的掩盖单位是"SM 级 worker"——一部分 SM 跑通信，一部分跑计算，kernel 自己当调度器。而无论哪家，本质是同一道题：把两类算力的队列都排满。**

## 6.1 共同的问题

无论什么硬件，"通信和计算同时发生"只有两种实现方式：**要么有两拨人**（一部分硬件发数据、一部分算），**要么一个人有两只手**（同一硬件交错发射两类指令）。NPU 和 GPU 分别把这两种思路走到了极致。

## 6.2 NPU：一个核里的"双人舞"（cv 融合）

昇腾的 AI Core 内部大致是：**cube 单元**（矩阵乘）+ **vector 单元**（向量/标量）+ **MTE 搬运引擎**，共享 UB/L0A/L0B/L0C 这套片上存储。所谓 **cv 融合**，就是在一个 Triton kernel 里用 `al.scope(core_mode=...)` 显式声明"接下来这段代码走 vector 指令流 / 走 cube 指令流"——两条流在同一个物理核内并发。真实代码长这样：

```python
# src/mega_moe/kernels/fused_forward.py:1306-1312（摘）
with al.scope(core_mode='vector', disable_auto_sync=True):
    if sub_vec_id() == 1:            # 双 vector 子核里的 1 号车道
        for prepared_wave in range(2):
            if prepared_wave < global_waves:
                _dispatch_dynamic_wave(...)   # 这条车道专职发数据
```

注意 `sub_vec_id()`：vector 侧还细分成**两个子核（车道）**——0 号车道干重活，1 号车道专职 dispatch（第九章的波流水里它是主角）。两条指令流之间的握手用块内事件：

```python
# src/mega_moe/kernels/fused_forward.py:506-510（摘）
@triton.jit
def _wait_fc1_vector_ack(SLOT: tl.constexpr):
    # Dual Vector subcores acknowledge the same event before Cube reuses UB.
    al.sync_block_wait('vector', 'cube', 10 + SLOT,
                       al.PIPE.PIPE_MTE3, al.PIPE.PIPE_FIX)
```

语义：vector 双子核在指定事件上应答，cube 才能复用这块 UB——**搬运完成通知计算，靠的是流水线事件（PIPE_MTE3/PIPE_FIX），不是全局同步。**

这套玩法的直接推论就是第三章那个 12.5× 教训：NPU 上真正的引擎级重叠**只能**发生在 kernel 内部的 scope 里，host 侧排多个 streams 是排不出引擎重叠的。

## 6.3 vv 融合：把 SM 的"角色划分"移植到 NPU

GPU 那套"一部分 worker 专职发数据、一部分算"能不能搬到 NPU 上？能。昇腾的 vector 侧本身就细分成**两个子核（subcore）**，`sub_vec_id()` 返回 0 或 1——正好当两个"worker"使。仓库里到处是这样的分工：

```python
# src/mega_moe/kernels/fused_forward.py:753,1488（摘）
vector_row_base = (sub_vec_id() * vector_block_m)   # 激活：两 lane 各管半块
...
pid * 2 + sub_vec_id(), ready_combine, route_to_send_ptr, output_ptr,
                                                   # 回传：波槽号 = pid*2 + lane
```

1 号车道专职 dispatch、0 号车道干重活、回传阶段两 lane 对称分摊——这就是 NPU 版的"SM 角色划分"，行话叫 **vv 融合**（两个 vector 子核协同分工）。

**但实测下来，这种融合的效率上限不高。** 三层原因，每层都有带日期的实测数据：

1. **角色划分只是重新切饼，不烙新饼。** 两个子核分的是同一份 vector 生产力，收益上限是延迟隐藏，不是新增算力。真正的新算力在 cube——那个在纯通信/元数据阶段整核闲着的矩阵引擎。
2. **分工不均比不分工更伤。** 相位打点（第九章 9.2）抓到过一次典型案例：`route_scatter` 5.66 ms 全部压在单个 vector lane 上，**另一个子核整段空转**，pad bin 扫描还有 12.5% 纯浪费。针对性改成"散射双 lane 分摊 + pad bin 裁剪 + 发布向量化"后，routing 元数据段 **10.0 → 3.4 ms、e2e 28.0 → 23.0 ms**（2026-09-19，两轮复现）。
3. **vector 侧本来就有富余。** 波流水内的累计打点显示：两 lane 的 busy 合计（lane1 ≈ 8.6M、lane0 ≈ 3.1M ticks）**远低于波周期 18.1M ticks**——vector 是与 cube 工作重叠的 slack，真正卡脖子的是 cube 的 GEMM。

顺带一个漂亮的消除实验：激活两 lane 忙闲差 12×（lane0 0.46 ms vs lane1 5.76 ms），两 lane 之间唯一结构差异是 lane1 独担 dispatch。直觉假设是"dispatch 抢了激活的访存带宽"，于是把 dispatch 挪到激活 signal 之后重排——结果 vact_v1 不降反升（5.7M → 6.0M ticks，两轮复现）、e2e 中性。**假设被实验否决**，这条重排路线就此关闭。猜想的瓶颈要用消除实验锤死，不能靠直觉供着。

## 6.4 总纲：这是一道 vector/cube 算力平衡题

把 6.2 和 6.3 合起来看：MoE 一层的活天然分两味——**vector 味**（搬运、激活、通信原语、reduce、元数据）和 **cube 味**（GEMM）。掩盖设计的全部要义是让两条队列都排满：

- **cv 融合是主战场**：它动用的是另一个引擎的空闲算力，两边都是真金白银；
- **vv 融合是精修**：只在 vector 已成瓶颈、且两 lane 分工不均时才值得做（比如上面 route_scatter 那个案例——单 lane 空转同僚是纯浪费）。

**先 cv 后 vv：先填空闲引擎，再优化忙碌引擎。**

## 6.5 NPU 的亲和设计：为什么不喜欢离散访存

NPU 的向量引擎喜欢**大块、连续、对齐**的访存，讨厌逐元素的离散访问（scatter/gather）。这不是玄学，是会被 UB 硬容量当场教做人的物理约束。仓库里有两条带日期的实测注释：

```python
# src/mega_moe/kernels/fused_forward.py:1276-1283（摘）
# ... The scattered per-site form cost one auto-buffer UB allocation per tiny
# value and overflowed UB (2026-09-17).  Everything except the three hoisted
# names below sits inside `if TIMING:` so the default binary carries none of it.

# src/mega_moe/kernels/fused_forward.py:1538-1540（摘）
    # One shared dummy + one precomputed ts row for all ten stamps (the
    # scattered per-site aranges/multiplies are what overflowed UB).
```

同一个功能，"每个打点现场各自 arange/multiply"的离散写法直接把 UB 撑爆；改成"一次预计算、处处复用"才活下来。离散访存不仅慢——`counts_publish` 那次：单核串行 ~28 672 次依赖链 load，~93 ns/次，2.68 ms 严丝合缝——还会**放大寄存器/UB 占用**：每一条离散地址都要自己的临时缓冲。

绕法三条，一条比一条根本：

1. **布局先行（最根本）**：让数据生来就是连续的。反向 dispatch 的模块头注释一句话说透：

```python
# src/mega_moe/kernels/dispatch_fc2_bwd.py:6-8（摘）
#  ... GEMM. No barrier_all — push and GEMM overlap on every AI core. peer_mem
#  is written expert-major and read contiguously, so no local_sort gather is
#  needed.
```

   排列阶段多花一分功夫（写成 expert-major），消费阶段的 gather **整个消失**——不访存是最快的访存。

2. **块化 gather**：躲不掉的 gather 也按块来，别按元素来：

```python
# src/mega_moe/kernels/moonep_planning.py:48-50（摘）
    # One (R, BLOCK_E) gather per expert block replaces the per-source
    # compile-time unroll, which explodes at wide worlds.
```

3. **数据捎带**：把小数据 pack 进大数据的行里一起搬（下一节 gate 通道就是例子）。

## 6.6 让 reduce 搭便车

NPU 上离散的原子归约很贵，仓库的解法是让归约**上别人的车**。看反向 combine 的第三相——top-k 加权求和与 gate 梯度，两个 reduce 挤在同一趟 kernel 里：

```python
# src/mega_moe/kernels/combine_fc1_bwd.py:358-362（摘）
def _kernel_combine_fc1_bwd_reduce(
    # Phase 3 (Vector): topk-sum reduce peer_mem -> grad_hidden, and gather the
    # packed gate channel -> grad_routing_weights [B*topk] (one value per
    # (token,slot), no sum).
    inv_sort_idxs_ptr,       # int64 [total_send]
    peer_mem_ptr, ...
```

两个便车：其一，topk-sum **不设独立 kernel**，fuse 进回传搬运本来就要碰 peer_mem 的那一相——数据反正要读一遍，读完顺手加完；其二，gate 梯度也不单独算，它被 pack 在 push 行尾的 gate 通道里（行偏移 `H_push` 处），**同一趟扫描顺手取出**。README 反向阶段表里 reduce 一栏只有 0.74 ms——不是它变快了，是它上车了。

## 6.7 GPU：以 SM 为单位的"班组制"

NVIDIA GPU 的 SM 是独立的小 worker：Tensor Core 算矩阵，CUDA core 做杂活，TMA/copy engine 搬数。要做通信计算重叠，GPU 上的思路是**把 worker 分组**：

- 一部分 SM / warp 绑定给通信（NVSHMEM 的 kernel-initiated 通信，`put/signal` 直接从 SM 发出去）；
- 其余 SM 跑 GEMM（cuBLAS / CUTLASS / Triton）；
- 再往上是各种 warp specialization、producer-consumer 队列——本质都是"楼里不同班组各干各的，靠对讲机（信号量/事件）协调"。

UniEP 在 Hopper 上走的正是这条路：把 dispatch + grouped GEMM + combine 整个装进**一个 persistent MegaKernel**，kernel 内部自己把 SM 划成通信组和计算组，自己当操作系统。NPU 的 cv 融合是"一个核里两个人"，GPU 的 megakernel 是"一栋楼里若干班组"——**粒度不同，哲学相同：把调度权从 host 收进 kernel。**

## 6.8 对照表

| | 昇腾 NPU（Mega-MoE-TD） | NVIDIA GPU（UniEP 等） |
|---|---|---|
| 计算单元 | AI Core = cube + vector + MTE | SM = Tensor Core + CUDA core + TMA |
| 通信执行者 | 同核 vector 指令流跑 libshmem 原语 | 专门分出的 SM 组跑 NVSHMEM |
| 融合粒度 | 核内指令流（`al.scope`，cv 融合） | SM/warp 级（persistent kernel 当调度器） |
| 同一引擎再分工 | 双 vector 子核（vv 融合，`sub_vec_id`） | warp specialization |
| 核间同步 | `sync_block_*` 流水线事件 + `dl.wait` | cooperative launch / 信号量 |
| 代表工作 | Mega-MoE-TD | UniEP（HPDC'26） |

## 6.9 小结

两套硬件、两种粒度、一个哲学：**调度进 kernel，同步细到事件。** NPU 内部再分两层：cv 融合吃 cube 的空闲（主战场），vv 融合重切 vector 的饼（精修，且要防单 lane 空转）；对硬件的脾气要顺毛捋——离散访存能免则免（布局先行、块化 gather、数据捎带），reduce 让它搭便车。（本章实测数据全部来自 `MOE_FWD_TIMING` 相位打点，工具用法见第九章 9.2。）本章用到的 `signal_op`/`dl.wait` 原语，第四章有整章拆解。**下一章起是进阶篇：先看 MoonEP 怎么把负载均衡焊进大算子。**

---

← [上一章](05-ep-and-framework.md) ｜ [返回目录](../README.md) ｜ [下一章：MoonEP](07-moonep.md) →
