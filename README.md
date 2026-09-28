# MegaMoE-tutorials

本教程是我从头开发 megamoe 训练算子这一过程的总结——开发过程中真正体会到了"万丈高楼平地起"，我的初衷是把这个过程中积累的经验记录下来，分享给所有的学习者和从业者，一起讨论。

适合的读者有三类，都需要一定的 AI 基础——纯小白看到不懂的词汇，建议随手问问 AI 再接着往下读。第一类：AI infra 入门、想学习领域知识，或者急需补课应付面试的人——不必追求会写 kernel、不必抠语法，当伪代码看也行，目的主要是建立感性认识，你更需要关注前六章，尤其是前三章。第二类：自己在写算子、尤其想写 megamoe 算子的人——即使你用的不是 Triton 和 NPU，一些共性的东西、通算掩盖的思想和 trick 仍然值得你参考。第三类：triton-distributed-ascend 的直接开发使用者或算法验证工作者——本教程所依据的仓库完全能跑通，可以参考并按需修改。

不适合的读者：想要非常细致、准确的 kernel 写法，最好能直接植入到自己的项目里，而你的项目用的又是 CUDA/Ascend C/TileLang 等——这种情况就不太适合了。

本书的一大优势就是确实是从头搭建了一个仓库，代码是实实在在的，经验也是货真价实的，需要更进一步，可以前往开源项目：[Mega-MoE-TD](https://gitcode.com/jzhoujg/Mega-MoE-TD.git)（基于 Triton-distributed-ascend 的昇腾 NPU MoE 前向/反向大算子库）。

## 怎么读

- **前六章是入门主线**，前三章尤其适合第一类读者，建议顺序读完。我会按照"megamoe 为什么被需要、由什么组成，以及最基本的'压箱技'——通算掩盖——是怎么设计的"来展开；第四章到第六章稍微做了一些扩展，依次讲底层接入（通信原语）、框架层接入、硬件特性对比（NPU 和 GPU），可以作为知识提高。
- **第七章起各自独立**，是进阶内容，按需取用：MoonEP 负载均衡、重算算子、全局流水等话题都在这里讨论；
- 全文代码摘自真实源码，标注 `文件:行号`（行号随仓库演进会漂移，**以函数名为准**），可以当伪代码看——看懂结构和数据流比抠语法重要一百倍。

## 目录

| 章 | 文件 | 一句话 | 定位 |
|---|---|---|---|
| 一 | [MoE 是什么，什么又是 Mega-MoE](chapters/01-what-is-moe.md) | MoE 把容量和算力解耦；大算子把这一层焊成一次流水，让通信/访存/计算互相掩盖 | 入门主线 |
| 二 | [经典结构：从 dispatch 到 combine，从前向到反向](chapters/02-dispatch-to-combine.md) | 前向是一条链，反向是一张网——每个 GEMM 裂成 dgrad 和 wgrad，反向计算量 ≈ 前向 × 2 | 入门主线 |
| 三 | [通信与计算掩盖的艺术](chapters/03-mega-moe-td-walkthrough.md) | 掩盖五招：引擎分工、细粒度信号、双 buffer、影子流水、尾部填充 | 入门主线 |
| 四 | [通信原语：signal / wait / epoch 的实现](chapters/04-comm-primitives.md) | put→fence→signal（单调届号）+ dl.wait(acquire)；triton-distributed-ascend 的接线与 shmem 对称内存思想 | 入门 · 提高 |
| 五 | [和 EP/DP/PP/TP 的关系，以及怎么接进框架](chapters/05-ep-and-framework.md) | EP 是第五个并行维度；nn.Module→autograd→宿主三层接入；算子自己的三本内存账与卸载思想 | 入门 · 提高 |
| 六 | [NPU 和 GPU：两套硬件，两种掩盖哲学](chapters/06-npu-vs-gpu.md) | cv 融合 vs SM 班组制；vv 融合为何上限不高——本质是 vector/cube 算力平衡；NPU 亲和设计与"让 reduce 搭便车" | 入门 · 提高 |
| 七 | [MoonEP：负载均衡为什么值得焊进大算子](chapters/07-moonep.md) | 木桶效应 → 副本式均衡（B.0–B.3）→ 规划/预取/归约全部 device 侧闭环 | 进阶 |
| 八 | [省显存的艺术：重算怎么设计](chapters/08-recompute.md) | 激活可重算、排列元数据绝不能重算；三档菜单（压缩存/搬主机/全重算）本质是行情兑换 | 进阶 |
| 九 | [前向的全局流水，以及为什么后向没有](chapters/09-global-pipeline.md) | 链可流、网只可切：前向三级波流水一次 launch；后向相位制 + 段内塞满 | 进阶 · 压轴 |

## 参考资料

- 仓库：[Mega-MoE-TD](https://gitcode.com/jzhoujg/Mega-MoE-TD.git)（全部代码出处）
- UniEP: Unified Expert-Parallel MegaKernel MoE for LLM Training, HPDC 2026 —— [ACM DL](https://dl.acm.org/doi/10.1145/3806645.3807818)
- MegaKernel 方向汇总：[Awesome-MegaKernel](https://github.com/qhy991/Awesome-MegaKernel)
- 整网集成：[MindSpeed-MM_MoonEP](https://gitcode.com/jzhoujg/MindSpeed-MM_MoonEP.git)（`moonep` 分支）
- 生态：vLLM `fused_moe`（vllm-project）、DeepEP / DeepGEMM（deepseek-ai）
- 论文（按章）：Shazeer et al. 2017《Outrageously Large Neural Networks》；GShard；Switch Transformers；ST-MoE；Mixtral of Experts；DeepSeek-V3 Technical Report（负载均衡与 FP8）

## 待补清单（v0.1 → v0.2）

- [x] 各章插图（第一批已补，见 `pic/`）：transformer block 图（一章 1.1，SVG 源件 `pic/transformer-block.svg`）、通算重叠图（三章 3.1）、前/反向算子链全景（三章 3.2 / 二章 2.3）、前向三车道甘特图（九章 9.1）、规划与预取车道图（七章 7.6）、重算路径与内存账（八章 8.5）
- [ ] 插图待补（第二批）：第二章形状账图、第三章后向相位图（3.4 现为 ASCII）、真正的后向流水甘特图
- [ ] 第七章 MoonEP 实测数字：跑 `moonep-skewed` 配对用例（trimmed-skewed vs skewed，开/关 MoonEP 对照）后填入
- [ ] 第六章 GPU 侧贴真实代码（UniEP 开源代码发布后补；现为文字描述）
- [ ] 第九章补一次真实相位打点数据表的完整示例
- [ ] 术语表与"进一步阅读"分章链接（视读者反馈决定是否加附录）
