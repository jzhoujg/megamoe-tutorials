# 四个 token 的 MoE：CPU 小实验

配合第二、三、四、七、八章使用。全局 4 个 token、4 个专家、top-2、2 个模拟 rank，hidden 和 FFN 中间维都为 2。rank 0 持有 token 0、1 和专家 0、1；rank 1 持有 token 2、3 和专家 2、3。

## 运行

使用已有的 **Python 3.9+、PyTorch 2.x** 环境，在教程仓库根目录运行：

```bash
python examples/tiny_moe.py
```

脚本只有 PyTorch 依赖，所有张量在 CPU 上。不需要 NPU、GPU、Triton、Mega-MoE-TD 或分布式进程组。两个 rank 用数据分桶模拟；没有真实网络通信、信号等待或并行执行，因此不能用脚本耗时推断大算子的性能。

## 按这个顺序读

1. **第二章：路由与 combine。** `naive_moe` 逐 token 调用专家，在专家输出上乘权重；`bucketed_moe` 先按目的 rank / 专家分桶，在激活处乘权重，再按 route_id 还原并求和。两个实现使用同一套无 bias SwiGLU 权重。
2. **第三、四章：布局如何支持 tile 与信号。** `make_plan` 和打印的 offset 对应正文接收表。rank 1 的 e2 两行是 tile A，e3 两行是 tile B。信号槽与 epoch 的推演在正文中，脚本不模拟设备内存序。
3. **第七章：副本改变执行地点。** 固定偏斜路由，把 token 2、3 的 e0 路由移到 rank 1 上的 e0 副本，负载由 `[6,2]` 变为 `[4,4]`。脚本中的副本权重用可微 `clone()` 构造，autograd 自动把梯度加回同一逻辑专家；真实系统需要显式做副本梯度归约。这不是 MoonEP B.0–B.3 规划器的实现。
4. **第八章：选择性重算。** `check_selective_recompute` 保留原始输入、路由映射、权重和 FC1 输出，重建接收 token 和加权激活，恢复阶段没有 FC1 GEMM。脚本为了核对仍保留原始中间量，不能用它测省了多少内存。

## 应看到什么

均衡路由的输出布局应为：

```text
order=[0, 4, 2, 6, 1, 3, 5, 7]
send_counts=[[2, 2], [2, 2]]
rank_loads=[4, 4]
```

随后会打印两组偏斜用例的布局、负载和数值输出。断言全部通过时，出现以下四条结果：

```text
PASS balanced: output, dx, d_route_weights, dW1, dW2
PASS skewed / owner only: output, dx, d_route_weights, dW1, dW2
PASS skewed / e0 replica on rank 1: output, dx, d_route_weights, dW1, dW2
PASS selective recompute: recv_x [8,2], act [8,2]; FC1 was not rerun
```

这里是预期输出，运行时由断言确认。前向与梯度比较使用 FP64，`rtol=1e-10`、`atol=1e-12`，旨在验证代数与映射一致性；这些阈值不能套到真实 BF16/FP8 kernel。梯度以固定、非均匀的 dy 比较，覆盖输入、路由权重、W1 和 W2。专家选择是固定整数，不对 top-k 选择求导；路由权重作为叶子张量，验证的是传回 router 接口的梯度。

## 可以自己改两处

- 把均衡例子里 token 0 的权重从 `[0.75,0.25]` 改为 `[0.25,0.75]`，观察布局不变、输出改变，并确认两条实现仍一致。
- 把偏斜例子的 `destinations[2:,0]=1` 改成只迁移 token 2 的 e0 路由，负载应为 `[5,3]`；输出和逻辑专家梯度仍应匹配参考实现。

返回：[教程目录](../README.md) · [第二章](../chapters/02-dispatch-to-combine.md) · [第七章](../chapters/07-moonep.md) · [第八章](../chapters/08-recompute.md)
