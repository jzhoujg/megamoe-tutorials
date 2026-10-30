# MegaMoE-tutorials

本教程总结了我在开源项目中从头开发 Mega-MoE 训练算子的实践经验——开发过程中真正体会到了"万丈高楼平地起"，我的初衷是把这个过程中积累的经验记录下来，分享给所有的学习者和从业者，一起讨论。

适合的读者有三类，都需要一定的 AI 基础——纯小白看到不懂的词汇，建议随手问问 AI 再接着往下读。第一类：AI infra 入门、想学习领域知识——不必追求会写 kernel、不必抠语法，当伪代码看也行，目的主要是建立感性认识，你更需要关注前六章，尤其是前三章。第二类：自己在写算子、尤其想写 Mega-MoE 算子的人——即使你用的不是 Triton 和 NPU，一些共性的东西、通算掩盖的思想和 trick 仍然值得你参考。第三类：triton-distributed-ascend 的直接开发使用者或算法验证工作者——可以结合文中的代码示例理解实现思路，再按自己的环境验证和调整。

不适合的读者：想要非常细致、准确的 kernel 写法，或者是希望最好能作为黑盒直接植入到自己的项目里，而你的项目用的又是 CUDA/Ascend C/TileLang 等——这种情况就不太适合了。

教程中的经验来自真实的开源项目开发，涵盖基于 Triton 的昇腾 NPU MoE 前向与反向算子实践。这里重点分享设计思路、实现取舍和验证方法。

## 怎么读

- **前六章是入门主线**，前三章尤其适合第一类读者，建议顺序读完。我会按照"Mega-MoE 为什么被需要、由什么组成，以及最基本的'压箱技'——通算掩盖——是怎么设计的"来展开；第四章到第六章稍微做了一些扩展，依次讲底层接入（通信原语）、框架层接入、硬件特性对比（NPU 和 GPU），可以作为知识提高。
- **第七章起各自独立**，是进阶内容，按需取用：MoonEP 负载均衡、重算算子、全局流水等话题都在这里讨论；
- 教程主体介绍基础实现；第七、八章进一步讨论后续规划与 FC1 重算方案，新旧行为按各节说明区分。
- 正文的源码摘录标注 `文件:行号`（行号随仓库演进会漂移，**以函数名为准**），可以当伪代码看——看懂结构和数据流比抠语法重要一百倍。教学小例子则单独放在 `examples/`。
- **想动手验证，先跑 [四个 token 的 CPU 小实验](examples/README.md)**：全局 4 个 token、4 个专家、top-2、2 个模拟 rank。第二、三、四、七、八章沿用这组数据，串起路由布局、tile 就绪、信号槽、副本分担和选择性重算。示例使用 PyTorch，不需要 NPU。

## 目录

| 章 | 文件 | 一句话 | 定位 |
|---|---|---|---|
| 一 | [MoE 是什么，什么又是 Mega-MoE](chapters/01-what-is-moe.md) | MoE 把容量和算力解耦；大算子把这一层焊成一次流水，让通信/访存/计算互相掩盖 | 入门主线 |
| 二 | [经典结构：从 dispatch 到 combine，从前向到反向](chapters/02-dispatch-to-combine.md) | 前向是一条链，反向是一张网——每个 GEMM 裂成 dgrad 和 wgrad，反向计算量 ≈ 前向 × 2 | 入门主线 |
| 三 | [通信与计算掩盖的艺术](chapters/03-comm-compute-overlap.md) | 串行分析法、四种重叠手法与反向阶段打点 | 入门主线 |
| 四 | [通信基础：从对称内存到 Triton custom op](chapters/04-comm-primitives.md) | 对称寻址与数据搬运 → 信号交接 → custom op 编译链；以 wait 的 IR 转换和设备轮询为例 | 入门 · 提高 |
| 五 | [和 EP/DP/PP/TP 的关系，以及怎么接进框架](chapters/05-ep-and-framework.md) | 框架配置 DP/FSDP，MegaMoE 执行 EP；普通参数与专家参数的分组、三层接口及内存归属 | 入门 · 提高 |
| 六 | [NPU 和 GPU：两套硬件，两种掩盖哲学](chapters/06-npu-vs-gpu.md) | 从真实代码看 cv 双缓冲与 GPU worker 分工；按关键路径分析 vv、路由布局和 UB 预算 | 入门 · 提高 |
| 七 | [MoonEP：负载均衡为什么值得焊进大算子](chapters/07-moonep.md) | 木桶效应 → 副本式均衡（B.0–B.3）→ 规划/预取/归约全部 device 侧闭环 | 进阶 |
| 八 | [省显存的艺术：从选择性重算到 FC1 重算](chapters/08-recompute.md) | 从保留或卸载 FC1 输出到反向重算 FC1；比较激活保存、通信、计算与峰值显存 | 进阶 |
| 九 | [前向的全局流水，以及反向的相位制选择](chapters/09-global-pipeline.md) | 前向三级波流水；反向依赖网也可流水化，当前以相位制为骨架并探索局部流水 | 进阶 · 压轴 |

## 参考资料

- UniEP: Unified Expert-Parallel MegaKernel MoE for LLM Training, HPDC 2026 —— [ACM DL](https://dl.acm.org/doi/10.1145/3806645.3807818)
- MegaKernel 方向汇总：[Awesome-MegaKernel](https://github.com/qhy991/Awesome-MegaKernel)
- 生态：vLLM `fused_moe`（vllm-project）、DeepEP / DeepGEMM（deepseek-ai）
- 论文（按章）：Shazeer et al. 2017《Outrageously Large Neural Networks》；GShard；Switch Transformers；ST-MoE；Mixtral of Experts；DeepSeek-V3 Technical Report（负载均衡与 FP8）

## 更新状态

持续更新中。
