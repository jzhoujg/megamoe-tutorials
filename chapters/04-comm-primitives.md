# 第四章｜通信原语：signal / wait / epoch 的实现

> **入门主线 · 提高篇 1/3** ｜ [返回目录](../README.md)

**一句话：跨 rank 通信就一套"快递协议"——数据先到（put + fence），通知后到（signal SET 一个单调届号），收件人只等自己的件（dl.wait acquire）；所有死锁几乎都死在违反这三步的顺序上。**

## 4.1 生产者：put → fence → signal

dispatch 搬运一个 tile 的完整收尾（producer 侧）：

```python
# src/mega_moe/kernels/dispatch_fc1.py:410-477（摘）
if DIRECT_REMOTE_STORE:
    remote_input_ptr = dl.symm_at(peer_mem_ptr, dst_rank)
    # ① 把本 tile 的数据写进对端的对称接收区（按行列掩码 store）
    row_offsets = tl.arange(0, _DIRECT_STORE_BLOCK_M)
    ...
    values = tl.load(input_ptr + source_rows[:, None] * stride_input_m
                     + cols[None, :], mask=..., other=0.0)
    tl.store(remote_input_ptr + destination_rows[:, None] * stride_input_m
             + cols[None, :], values, mask=...)

    # ② fence：让上面的 store 真正落地、对端可见
    libshmem_device.fence()
    # ③ 通知：往"这个 tile 专属"的信号槽写届号
    libshmem_device.signal_op(
        signal_mem_ptr + signal_slot * 16,
        signal_epoch,
        libshmem_device.ACLSHMEM_SIGNAL_SET,
        dst_rank,
    )
else:  # putmem 版本：按 token 逐行 put，最后 quiet + signal
    libshmem_device.putmem(dst_base, src_base, hidden * 2, dst_rank)
    ...
    libshmem_device.quiet()
    libshmem_device.signal_op(...)
```

三步的顺序是**铁律**：先数据、后 fence、再 signal。fence 缺了会出现"通知到了货还没到"——消费者等到信号去读，读到的是上一轮的旧数据，这种 bug 不报错，只让你数值对不上，查一晚上。

信号槽的编址也值得看一眼：

```python
signal_slot = (LOCAL_RANK * EXPERTS_PER_RANK + expert_id) * MAX_SOURCE_TILES
              + source_tile
```

**（源 rank，专家，tile 序号）三元组唯一确定一个槽**，槽宽 16 个 int32（`signal_slot * 16`；UDMA 引擎写 64-bit 通知落在前两个 word，cube 侧等低 32 位）。一格一物，永不串味。

## 4.2 消费者：dl.wait + acquire，外加一次二分搜索

消费侧的完整实现是单 kernel 前向里的 `_wait_dispatch_row_range`——"我要算专家 e 的 [row_start, row_end) 行，请把所有和它重叠的源 tile 都等到"：

```python
# src/mega_moe/kernels/fused_forward.py:452-503（摘）
@triton.jit
def _wait_dispatch_row_range(
        signal_mem_ptr, recv_seg_starts_ptr, expert_id, row_start, row_end,
        signal_epoch, WORLD_SIZE: tl.constexpr, ...):
    """Wait for every source tile overlapping [row_start, row_end) of an expert.

    Two W.bit_length()-step binary searches over the segment-start table
    bracket the overlapping sources; the walk then visits exactly those.
    The old O(W) linear scan sat on the FC1 Cube critical path.
    """
    seg_row = recv_seg_starts_ptr + expert_id * (WORLD_SIZE + 1)
    lo_a, hi_a, lo_b, hi_b = 0, WORLD_SIZE, 0, WORLD_SIZE
    for _ in range(SEARCH_STEPS):          # 两组交错二分，并行推进
        ...                                 # 找出 [first_source, last_source)
    ready_token = 0
    for source_id in range(first_source, last_source):
        ...
        signal_slot = ((source_id * EXPERTS_PER_RANK + expert_id)
                       * MAX_SOURCE_TILES + first_tile)
        ready_token += dl.wait(
            signal_mem_ptr + signal_slot * 16, last_tile - first_tile + 1,
            'gpu', 'acquire', waitValue=signal_epoch)
    return ready_token
```

两个细节品一下：

1. **为什么先二分**：W 个源 rank 里哪些和我这段行有重叠？线性扫是 O(W)，坐在 FC1 cube 关键路径上；二分只要 `W.bit_length()` 步（W=128 时 7 步）。**同步代码也要讲算法复杂度**——它和 GEMM 一样占墙钟；
2. **`acquire` 语义**：等到的不仅是"信号到了"，还保证**信号之前写入的数据对我也可见**（内存序）。这正是 producer 那边 fence 的对偶。`dl.wait` 还能一次等连续多个 tile（`count = last_tile - first_tile + 1`），省掉逐槽轮询。

## 4.3 为什么是 SET 一个递增的 epoch，而不是 SET 1

新手最容易想当然的写法：发完就 `signal_op(槽, 1)`，消费者 `waitValue=1`。**错。** 对称内存的信号槽是持久的——上一轮迭代的 1 还躺在槽里。你分不清"这一轮到了"还是"上一轮的残影"。

所以协议是**每轮迭代递增届号（epoch）**：第 n 次前向用 `epoch = n`（跨步持久化在 state 里，重启会归 1——所以 state 不能丢，见第五章），消费者只认 `waitValue=当前届号`。旧的残影自然被忽略，**单调性就是正确性**。（NVSHMEM 的 signal 槽有一模一样的残影问题，业界解法同样是单调届号——这不是昇腾特产。）

## 4.4 barrier 家族

除了 per-tile 信号，还有一组粗粒度原语，按"贵"的程度排：

- `fence` / `quiet`：只管排空自己的写，最便宜；
- `barrier_all_vec`：全部 vector 核到场；
- `barrier_all`：全 rank 全引擎到场，最贵——mega 反向用它分隔 P1–P5 相位（第三章的图），单 kernel 前向只在头部元数据阶段用几次，进了波流水就再也不要它。

**能用细粒度信号就不用 barrier，能用 barrier 就不用 host 同步**——这是贯穿全库的口诀。

## 4.5 这些原语从哪来：triton-distributed-ascend 的接线

`libshmem_device`、`dl.symm_at`、`dl.wait` 不是 Triton 自带的，是 **triton-distributed-ascend** 这层 overlay 接进来的。看 kernel 文件的 import 就明白接线方式：

```python
# src/mega_moe/kernels/dispatch_fc1.py:26-31（原文）
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
import triton.language.extra.cann.extension as al
from triton.language.extra.cann.extension import sub_vec_id
```

整个栈分三层：

1. **底层：ACLSHMEM C 库**（cann-shmem）——对称堆的建立、put/get/signal/fence/barrier 原语、跨机时的 UDMA 通路；
2. **中层：triton_dist overlay**——把这些原语**注册成 Triton 的 device builtin**。于是 kernel 里调 `libshmem_device.signal_op(...)` 和调 `tl.load(...)` 一样自然，编译器直接把它 lower 成对应的通信指令；同时提供 `dl.symm_at`（本地指针 + 目的 rank → 对端地址的翻译）和 `dl.wait`（带 acquire 语义的等待）这两个 device 函数。**通信由此变成 kernel 里的一条普通指令**——这是它能耗进流水线缝隙的前提（呼应第一章）；
3. **host 侧：`import shmem as ash`**——分配对称内存、查询 PE：

```python
# src/mega_moe/runtime/workspace.py:105-121（摘）
def create_moe_forward_context(...):
    """Allocate BF16 token buffers and an FP32 routing-weight buffer."""
    import shmem as ash

    ash_rank = ash.my_pe()
    ash_world_size = ash.pe_count()
    if ash_rank != rank or ash_world_size != world_size:
        raise ValueError(
            "Ascend Mega-MoE requires the EP process group to match "
            "the complete ACLSHMEM world")
    ...
    context.peer_mem = ash.aclshmem_create_tensor(
        [max_received_routes * hidden_size],
        dtype=torch.bfloat16, device_id=rank)
```

注意开头那个检查：EP 进程组必须和 ACLSHMEM 世界**完全重合**——"发数据的地址空间"和"算数据的进程组"必须是同一批人，否则 symm_at 翻译出来的地址属于一个根本不在你进程组里的 PE。

## 4.6 shmem 对称内存的思想模式

最后把思想模型立起来，两种通信世界观对照：

- **集合通信（HCCL/NCCL 的 all-to-all）**：约定时间全员开会。发起一个 collective，所有 rank 都得到场，语义是"集体行动"，你只能等最慢的人——自由度低，但省心；
- **对称内存（shmem 系）**：我直接把东西放你家桌上。每个 rank 在**同一个对称堆**里按**相同顺序**分配**同尺寸**的 buffer，于是"我这边的 offset X"在所有 rank 那里都是同一格局下的合法地址——`dl.symm_at(ptr, dst_rank)` 做完翻译，put/signal 单方面完成，**接收方不需要出场**。

这就是 one-sided RMA（单边通信）的自由度所在：发送时机、粒度、节奏全在自己手里，才能做出 per-tile 的细粒度握手。本章那套 epoch 协议、第五章的分配纪律，全都建立在这个"同构镜像"的世界观上。

自由是有代价的，对称内存的三条纪律（分配顺序全员一致、尺寸取全局最大、谁也不能擅自 free）放到第五章和框架接入一起讲。

## 4.7 小结

快递协议：put → fence → signal（SET 单调届号，槽按三元组编址）；收件 `dl.wait(..., acquire, waitValue=epoch)`，等之前先用二分圈出自己要的件。这套原语由 triton-distributed-ascend 注册成 Triton builtin，底下是 ACLSHMEM 对称堆——**one-sided 的自由度，是大算子能"边发边算"的地基。**

---

← [上一章](03-mega-moe-td-walkthrough.md) ｜ [返回目录](../README.md) ｜ [下一章：EP 与框架接入](05-ep-and-framework.md) →
