# 第五章｜和 EP/DP/PP/TP 的关系，以及怎么接进框架

> **入门主线 · 提高篇 2/3** ｜ [返回目录](../README.md)

**一句话：EP 是 MoE 特有的并行维度，和 DP/TP/PP 正交组合；mega_moe 就是 EP 这一个维度的"通信+计算执行体"，对外只暴露一层 PyTorch 接口——而接口背后，它还得自己管好三本内存账。**

## 5.1 EP：第五个并行维度

经典四维（DP/TP/PP/SP）之外，MoE 多出一个天然维度：**专家并行（EP）**——896 个专家切到 8 张卡，每卡供养 112 个。组合空间里最常见的形态：

```text
               ┌─ attention：TP 或 SP 或 DP
一个 transformer ─┤
               └─ FFN(MoE)：EP（mega_moe 在这里）
PP 按层切，与上面正交
```

通信特征对比（为什么 EP 大规模下划算）：

- TP：每层两次 all-reduce，字节量 ∝ hidden × token；
- EP：一层两次 all-to-all，字节量 ∝ hidden × token × topk——但**专家权重不用每卡复制**，总显存随卡数线性下降。

`FusedMoEForward.__init__` 的真实签名（注意 `num_experts` 必须被 world size 整除——896/8=112）：

```python
# src/mega_moe/ops/forward.py:78-114（摘）
def __init__(
    self,
    ep_group: Optional[torch.distributed.ProcessGroup],
    *,
    max_tokens_per_rank: int,
    hidden_size: int,
    top_k: int,
    num_experts: int,
    config: Optional[MoEForwardConfig] = None,
):
    """Create one BF16-only post-routing MoE forward instance.
    ``ep_group`` must match the complete ACLSHMEM world. ..."""
    ...
    if num_experts % self.world_size:
        raise ValueError("num_experts must be divisible by the EP world size")
    self.experts_per_rank = num_experts // self.world_size
```

## 5.2 接入 PyTorch：三层接口

**第一层：`torch.nn.Module`。** 上面的 `FusedMoEForward` 本身就是 Module，前向编排全在里面。

**第二层：`torch.autograd.Function`。** 训练要反向，于是有 `MegaMoEFunction`——前向调 `op.forward(..., return_saved=True)` 把反向要用的中间量顺手存下，backward 调 `moe_backward_triton`。它的 apply 契约是使用者必须背下来的八股：

```python
# src/mega_moe/ops/function.py:48-68（摘）
"""Args (``apply``, in order):

    op                ``FusedMoEForward`` that will run the pass
    hidden_states     ``[B, H]``                                (grad)
    routing_weights   ``[B, topk]`` contiguous FP32             (grad)
    selected_experts  ``[B, topk]`` int expert ids              (no grad)
    gate_up_weight    packed ``[E_p, H, 2F]`` gate/up table     (grad)
    down_weight       ``[E_p, H, F]``                           (grad)
    peer_mem          shared symmetric buffer at heap offset 0 — must be
                      the session's FIRST symmetric allocation.
    state             caller-owned persistent namespace carrying the
                      backward's cross-step runtime state (``signal_mem`` /
                      ``epoch``)...
"""
```

典型用法：

```python
op = FusedMoEForward(ep_group, max_tokens_per_rank=..., hidden_size=...,
                     top_k=..., num_experts=...,
                     config=MoEForwardConfig(enable_single_kernel_forward=True))
output = MegaMoEFunction.apply(op, hidden_states, routing_weights,
                               selected_experts, gate_up_weight, down_weight,
                               peer_mem, state)
output.backward(dy)   # 自动进入 moe_backward_triton
```

**第三层：宿主训练框架。** 通过 MindSpeed-MM 的 `ep_plan.dispatcher: megamoe` 配置项，把整网训练的 MoE 层指到本库——框架管 DP/PP/优化器，mega_moe 管 EP 这一段。

## 5.3 使用者必须知道的三条纪律

接口虽薄，纪律很硬，全部来自对称内存的物理约束：

1. **single-in-flight**：算子实例的 workspace 是复用的，同一时刻只能有一个前向在飞。代码里有 owner/generation 双重门，反向开始前先验证 saved 没被下一轮前向覆写；
2. **`peer_mem` 必须是会话的第一个对称分配**（heap offset 0）——`dl.symm_at` 只在 offset 0 才能正确解析（第四章的原语依赖这一点）；
3. **`state` 必须跨迭代复用**。docstring 写得很清楚：不带着 state 走，每步都重新分配对称信号槽（**对称堆泄漏**），而且届号归 1（旧信号残影会被读成新的——第四章讲过的正确性问题）。

```python
# src/mega_moe/ops/function.py:61-68（摘）
    state             caller-owned persistent namespace ... Without it every
                      step would re-allocate symmetric tile-signal slots
                      (symmetric heap leak) and restart the SET epoch at 1
                      (stale signal values would read as fresh).
```

**接口的每一句"废话"，都是某个深夜事故的墓志铭。**

## 5.4 算子怎么管自己的内存（怎么存）

框架接入者最容易忽略的一件事：这个算子不是"无状态"的，它自己养着一整套内存。看 workspace 的定义：

```python
# src/mega_moe/runtime/workspace.py:10-18（摘）
@dataclass
class MoEForwardContext:
    """Single-in-flight symmetric buffers and reusable routing workspaces.

    A routing plan must be consumed before the next plan is built because the
    device metadata tensors are intentionally reused in place.
    """
```

一本"三层内存账"：

1. **对称堆（通信面）**：`peer_mem`（token 接收区）、`routing_weight_mem`、`signal_mem` 全部用 `ash.aclshmem_create_tensor` 分配在对称堆上。顺序即纪律——`peer_mem` 必须第一个（第四章）；`signal_mem` 是一张大表同时装 dispatch 就绪槽和 replica 权重就绪槽（`dispatch_signal_slots + 3 * replica_budget`，槽 ABI 16 个 int32）：

```python
# src/mega_moe/runtime/workspace.py:166-175（摘）
    # Dispatch readiness and replica-weight readiness share one eagerly
    # initialized symmetric allocation.  A slot occupies 16 int32 values to
    # match the ACLSHMEM signal ABI used by the existing dispatch pipeline.
    ...
    # UDMA writes a 64-bit notify into the first two words of each aligned
    # slot; Cube dl.wait acquires its low int32 word. Down has two N panels.
```

2. **常驻 workspace（元数据面）**：路由规划的全部中间表（计数、偏移、波偏移……）挂在 context 上**原地复用**——docstring 那句 "intentionally reused in place" 就是 single-in-flight 在内存面的表述：不重新分配、不清零、每轮覆写。省下的是分配器和清零的墙钟，付掉的代价是"上一轮的 plan 没消费完就不能建下一轮"。用完整个 context 走 `finalize()` 释放对称分配。
3. **大激活（save 面）**：反向要吃的 fc1_output 等，量级最大（Kimi-K3 下几百 MB/层），策略也最多——压缩存 / 搬主机 / 重算三档。这层的思想下面说，细节第八章展开。

## 5.5 大激活怎么办（怎么卸载——代码思想）

框架自带的 activation offload（`saved_tensors_hooks` / SwapTensor 那一套）为什么管不到它？因为这个 saved dict 是算子自己攒的，**不在 autograd 的标准保存路径上，框架根本看不见**。所以卸载必须自己长器官：

- **pinned 主机缓冲池**（`MEGAMOE_FC1_OFFLOAD=1`）：前向结束把 fc1_output 挪到锁页主机内存，反向入口再搬回来；
- **旁路 stream 搬运**：搬运和宿主的下一层计算并行，不占关键路径；
- **为什么不能借框架的手**：一句话点透——

```python
# src/mega_moe/ops/function.py:111-115（摘）
# Optional host swap of the ONE big saved activation: with
# MEGAMOE_FC1_OFFLOAD=1 fc1_output moves to a pooled pinned host
# buffer on a side stream (framework async_offload.py SwapTensor
# idiom — the native saved dict is invisible to saved_tensors_hooks,
# so the framework mechanism can't do this) ...
```

**框架够不着的地方，算子自己长手。** 三档省显存菜单（压缩存 → 搬主机 → 全重算）和重算契约的完整设计，见第八章。

## 5.6 开发者注记（宿主接入入口）

（给第三类读者）

- 整网训练入口：Mega-MoE-TD `examples/kimi_k3/finetune_kimik3.sh`（经 MindSpeed-MM 的 `dispatcher: megamoe`）；
- 融合反向的三项配置约束（`recompute` / `enable_activation_offload` / `megamoe_shared_op`）及原因见宿主仓 `examples/kimi_k3/README.md`——违反会在首个 backward 处失败。

## 5.7 小结

EP 与四维正交，mega_moe 是 EP 执行体；接入走 nn.Module → autograd.Function → 宿主 dispatcher 三层；使用者背三条纪律；算子自己管三本内存账（对称堆 / 常驻元数据 workspace / 大激活），卸载思想是"框架看不见的就自己长手"。**下一章抬头看硬件：NPU 和 GPU 的两套掩盖哲学；再往后是三章硬菜：负载均衡、重算、全局流水。**

---

← [上一章](04-comm-primitives.md) ｜ [返回目录](../README.md) ｜ [下一章：NPU 与 GPU](06-npu-vs-gpu.md) →
