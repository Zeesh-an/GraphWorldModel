# 动作条件化图扩散世界模型 —— 方法与评估

> 本文档是汇报用的完整技术说明：形式化、模型结构、**动作条件化的数学保证**、训练流程，以及评估体系中每个指标的定义与判读。
>
> 结果表在 §8。**所有数字均由 `world_model/train_wm.py` 实测产出，每张表注明其来源 run。**

---

## 1. 问题设定

### 1.1 目标

学习一个动作条件化的转移算子，替代昂贵的蒙特卡洛扩散模拟器：

$$f_\theta(G,\; s_t,\; a_t) \;\longrightarrow\; s_{t+1}$$

其中

- $G = (V, E, w)$：图，$|V| = N$，边权 $w_{uv} \in [0,1]$ 为 IC 的传播概率；
- $s_t = (\mathbf{x}^{\text{inf}}_t,\; \mathbf{x}^{\text{fr}}_t) \in \{0,1\}^N \times \{0,1\}^N$：扩散状态，分别为「曾被激活」与「当前传播波前」；
- $a_t$：一个动作袋（bag），每个元素为 $(\text{op},\, \text{target},\, \text{dest},\, \text{weight})$，$\text{op} \in \{$`add_node`, `remove_node`, `add_edge`, `remove_edge`, `set_edge_weight`$\}$。

### 1.2 转移的因式分解

单步转移拆成外生与内生两段：

$$s_{t+1} \;=\; T_{\text{endo}}\!\big(\,T_{\text{exo}}(s_t, a_t)\,\big)$$

| | 含义 | 性质 |
|---|---|---|
| $T_{\text{exo}}$ | 动作的**即时确定性效果**：播种一个节点、把节点移出波前、增删改一条边 | **确定性、有闭式** |
| $T_{\text{endo}}$ | 随后的**一步扩散** | IC 下随机；LT 下给定隐藏阈值确定 |

这个拆分是全文的骨架：**§4 将证明 $T_{\text{exo}}$ 是结构保证的（梯度恒为零），$T_{\text{endo}}$ 是学习的。**

### 1.3 两种扩散动力学

**Independent Cascade (IC).** 节点 $u$ 在其激活后的下一步，对每个出邻居 $v$ 独立地以概率 $w_{uv}$ 传播一次，之后转入 Removed 态不再传播。因此

$$P\big(v \text{ 在 } t{+}1 \text{ 被新激活}\big) \;=\; 1 - \prod_{u \to v} \big(1 - w_{uv}\,\mathbf{1}[u \in \text{frontier}_t]\big)$$

**Linear Threshold (LT).** 每个节点持有隐藏阈值 $\theta_v \sim U(0,1)$（每 episode 重抽、从不存储）。当活跃入邻居权重占比越过阈值时激活：

$$v \text{ 激活} \iff f_v \;=\; \frac{\sum_{u \to v} w_{uv}\,\mathbf{1}[u \text{ active}]}{\sum_{u \to v} w_{uv}} \;>\; \theta_v$$

由于 $\theta_v$ 不可观测，一个只看状态的模型**最好只能恢复阈值边缘分布** $P(\text{激活} \mid f_v)$。这是 LT 指标天然低于 IC 的原因，是设定使然而非缺陷。

### 1.4 `remove_node` 的两种语义

同一个 op 在两类任务下含义不同，混用会静默地偏移每一个数字：

| 语义 | 含义 | 正确场景 |
|---|---|---|
| `spent` | 传播者已耗尽。IC 状态 2 (Removed) **仍计入感染**；LT 回到状态 0 可再激活 | 影响力最大化 |
| `blocked` | 节点被移出图：不计数、不传播、不可被感染 | 遏制类任务（关键节点检测、免疫、影响力阻断） |

在遏制任务上误用 `spent` 会使测得的最终传播规模**恰好高估 $k$**（每个被免疫的节点都被 `active_nodes()` 计为感染）。因此该语义在数据、head、模拟器三处被强制一致，训练前交叉校验（`train_wm.py::check_remove_semantics`），不匹配直接拒绝训练。

---

## 2. 数据生成

对每个 $(\text{图},\ \text{动力学},\ \text{种子算法},\ \text{rollout})$ 组合模拟一条 episode：

1. **$t=0$**：由六种经典 spine 选择器之一（`random`, `degree`, `pagerank`, `betweenness`, `celf`, `local_search`）产出种子集，作为一个 `add_node` 动作袋提交。
2. **$t>0$**：以概率 `--inject-p` 注入一个随机动作，否则为 NULL。
3. **反事实分叉（counterfactual forks）**：从**同一个 $s_t$** 再施加**不同的**动作，各自记录结果。

   > 这是全文最关键的数据设计。它使「换一个动作会造成什么后果」的**真实因果效应成为可观测量**，从而让 §7.3 的动作条件化检验成为可能。没有它，动作条件化在原理上不可检验。

4. **蒙特卡洛软标签**：每一步从动作后状态重跑 $M$ 次（默认 30），估计逐节点的真实一步边缘概率

   $$y^{\text{inf}}_v = \hat P(v \text{ 在 } t{+}1 \text{ 感染}), \qquad y^{\text{fr}}_v = \hat P(v \text{ 在 } t{+}1 \text{ 属于波前})$$

   **训练目标是这个边缘概率，而不是单次伯努利抽样。** 单次抽样把可学的期望换成了不可学的噪声；改用边缘后 IC 的一步 `delta_f1` 从 ~0.52 提升到 ~0.83。

由于 $T_{\text{exo}}$ 确定，动作只施加一次，$M$ 次重抽只作用于 $T_{\text{endo}}$（LT 确定，故 $M=1$）。

---

## 3. 模型

### 3.1 输入特征 $X \in \mathbb{R}^{N \times C}$

| 列 | 通道 | 含义 | 类型 |
|---|---|---|---|
| 0 | `infected` | $t$ 时刻曾激活 | 二值 |
| 1 | `frontier` | $t$ 时刻处于传播波前 | 二值 |
| 2 | `degree` | $\log(1+\deg)$，在时刻 $t$ 的图 $A_t$ 上 | 连续 |
| 3 | `act_add` | 本步 `add_node` 的目标 | 二值 |
| 4 | `act_remove` | 本步 `remove_node` 的目标 | 二值 |
| 5 | `act_edge` | 本步边操作的端点 | 二值 |
| 6–8 | `act_edge_{add,del,reweight}` | *(`--action-encoding typed`)* 按 op 拆分第 5 列 | 二值 |

列 0–1 是**状态**，列 2 是**结构**，列 3–8 是**动作在节点上的投影**——这是模型「动作条件化」的特征侧入口。

> **为什么需要 `typed`（第 6–8 列）**：在 `basic` 编码下，同一对端点上的 `add_edge` 与 `remove_edge` 产生**逐字节完全相同**的 $X$。邻接矩阵不同所以模型并非全盲，但 encoder 单看特征无法区分。`typed` 向后兼容：$X[:, {:}6]$ 一字不变。

### 3.2 图输入

$$\hat A \;=\; D^{-1/2}(A + I)\,D^{-1/2}, \qquad D_{vv} = \sum_u (A+I)_{vu}$$

行为 dst、列为 src，故矩阵乘法沿**入邻居**聚合。度用**加权入度**，因此 IC 的传播概率参与传播归一化而非仅连通性。SAGE/GAT/GT 直接消费 `edge_index` 与 `edge_weight`。

**边动作的逐 episode 邻接重建**：边操作会改变图，因此 `reconstruct_episode_adjacency` 重放操作序列，为每个 $(t, \text{branch})$ 构造真实的 $A_t$——main 分支看**动作后**的图（累积至 $t$ 含），反事实分支从**步前**图分叉（累积至 $t{-}1$）。这保证模型总是看到那个真正产生了所记录 $s_{t+1}$ 的邻接。

### 3.3 编码器

五个可插拔骨干，统一接口 $\text{Enc}: (X, G) \mapsto h \in \mathbb{R}^{N \times H}$，均为 pre-norm 残差块堆叠：

| 骨干 | 机制 |
|---|---|
| GCN | $h \leftarrow \sigma(\hat A h W)$ |
| GraphSAGE | $h \leftarrow \sigma\big(W\,[\,h \,\|\, \text{mean}_{u \to v} h_u\,]\big)$ |
| GATv2 | $e_{uv} = \mathbf{a}^\top \text{LeakyReLU}(W_l h_v + W_r h_u)$，多头 scatter-softmax |
| Graph Transformer | $Q_v \cdot K_u / \sqrt{d_k}$ + FFN |
| GCNII | $h \leftarrow \sigma\big((1-\beta_\ell)\,\text{supp} + \beta_\ell W\,\text{supp}\big),\ \ \text{supp} = (1-\alpha)\hat A h + \alpha h^{(0)}$ |

无位置编码；位置/度信息由通道 2 提供。

### 3.4 输出头 —— 本方法的核心

#### (a) `linear`（对照组）

$$\text{logits} = W h + b$$

最大灵活性，但自由滚动时**结构上不受约束**，会饱和到全图。

#### (b) `structured` IC —— `ICTransmissionHead`

**不预测状态，预测机制。**

$$
\begin{aligned}
\text{(i) } T_{\text{exo}}:\quad
&\tilde x^{\text{inf}}_v = \min\!\big(x^{\text{inf}}_v + x^{\text{add}}_v,\ 1\big)
&&\big[\ \cdot\,(1 - x^{\text{rm}}_v)\ \text{ 当 } \texttt{blocked}\ \big]\\
&\tilde x^{\text{fr}}_v = \min\!\big(x^{\text{fr}}_v + x^{\text{add}}_v,\ 1\big)\cdot\big(1 - x^{\text{rm}}_v\big)\\[4pt]
\text{(ii) 每边传播倾向:}\quad
&q_{uv} = \sigma\!\big(\text{MLP}([\,h_u,\ h_v,\ w_{uv}\,])\big)\\[4pt]
\text{(iii) 活跃源门控 + IC 闭式:}\quad
&t_{uv} = q_{uv}\cdot \tilde x^{\text{fr}}_u\\
&p^{\text{new}}_v = 1 - \prod_{u \to v}\big(1 - t_{uv}\big)
= 1 - \exp\!\Big(\textstyle\sum_{u \to v}\log(1 - t_{uv})\Big)\\[4pt]
\text{(iv) 单调合成:}\quad
&\hat y^{\text{inf}}_v = \tilde x^{\text{inf}}_v + \big(1 - \tilde x^{\text{inf}}_v\big)\,p^{\text{new}}_v\\
&\hat y^{\text{fr}}_v = \big(1 - \tilde x^{\text{inf}}_v\big)\,p^{\text{new}}_v
\end{aligned}
$$

第 (iii) 步的求积用 log-sum-exp scatter 实现以保证数值稳定。输出转回 logits，使训练（`BCEWithLogits`）与评估（`sigmoid`）与头类型无关。

#### (c) `structured` LT —— `LTThresholdHead`

LT 阈值不可观测，故把激活概率建成活跃邻居占比的**学习单调函数**：

$$
f_v = \frac{\sum_{u\to v} w_{uv}\,\tilde x^{\text{act}}_u}{\sum_{u\to v} w_{uv}},
\qquad
p^{\text{new}}_v = \mathbf{1}[f_v > 0]\cdot\sigma\!\big(\tau\,(f_v - \hat\theta_v)\big)
$$

其中 $\hat\theta_v = \sigma(W h_v)$ 为逐节点阈值代理，$\tau = \text{softplus}(\cdot) > 0$ 为学习的锐度。

#### (d) `structured_residual`（IC）与 `structured_oracle`（验证用）

$$
q^{\text{res}}_{uv} = \sigma\!\big(\underbrace{\text{logit}(w_{uv})}_{\text{锚}} + \text{MLP}([h_u,h_v,w_{uv}])\big),
\qquad
q^{\text{orc}}_{uv} = w_{uv}
$$

`residual` 把 $q$ 锚在真实传播概率上，零修正即复现 oracle——当训练数据缺乏边权多样性时使用。`oracle` 不含学习，用于在信任任何学习头之前**先验证 IC 结构形式本身是否正确**（期望 `count_bias ≈ 0`）。

---

## 4. 动作条件化的数学保证

这是本方法与一般 action-conditional GNN 的本质差别，也是汇报中最该讲清楚的一节。

### 4.1 命题（外生效果的结构保证）

> 设 $v$ 是本步 `add_node` 的目标，则在 `structured` 头下
> $$\hat y^{\text{inf}}_v = 1 \qquad\text{且}\qquad \frac{\partial\, \hat y^{\text{inf}}_v}{\partial\, \theta} = 0 \quad \forall\, \theta \in \Theta$$
> 其中 $\Theta$ 为模型**全部**可学习参数。

**证明.** 由 $x^{\text{add}}_v = 1$ 及 (i)：

$$\tilde x^{\text{inf}}_v = \min(x^{\text{inf}}_v + 1,\ 1) = 1$$

代入 (iv)：

$$\hat y^{\text{inf}}_v = \tilde x^{\text{inf}}_v + \underbrace{(1 - \tilde x^{\text{inf}}_v)}_{=\,0}\, p^{\text{new}}_v = 1 + 0\cdot p^{\text{new}}_v = 1$$

学习量 $h$ 仅经由 $q_{uv}$ 进入 $p^{\text{new}}_v$，而该项系数恒为 $0$，故 $\hat y^{\text{inf}}_v$ 与 $\Theta$ 无函数依赖，偏导为零。$\blacksquare$

**含义**：不存在任何一组权重能让模型「忘记」一个播种动作，训练也不可能破坏这条性质。这不是拟合出来的行为，而是计算图的代数结果。

### 4.2 实证验证

对 $\hat y^{\text{inf}}_{\text{seed}}$ 反传，取所有参数梯度的最大绝对值；并把权重整体缩放到极端以排除拟合巧合：

| head | 权重 | $P(\text{感染} \mid \text{被播种})$ | $\max\lvert \partial P_{\text{seed}}/\partial W\rvert$ | $\max\lvert \partial P_{\text{other}}/\partial W\rvert$ |
|---|---|---|---|---|
| **structured** | ×1 | **0.9999989** | **0.000e+00** | 1.907e+02 |
| **structured** | ×50 | **0.9999989** | **0.000e+00** | 0 |
| **structured** | ×(−50) | **0.9999989** | **0.000e+00** | 0 |
| **structured** | ×0（全部清零） | **0.9999989** | **0.000e+00** | 2.500e−01 |
| `linear` | ×1 | 0.6246 | 7.549e+02 | 2.694e+01 |
| `linear` | ×50 | **0.0000** ✗ | 0 | 0 |
| `linear` | ×(−50) | 1.0000 | 0 | 0 |

- structured 头把权重放大 50 倍、取反、乃至**全部清零**，被播种节点的输出一位不动，梯度**精确为 0**。
- 对照列 $\partial P_{\text{other}}/\partial W = 190.7 \neq 0$，说明模型并未「死掉」——学习照常流向未被动作触及的节点。被钉死的**只有动作目标**。
- `linear` 头同样输入下，权重一放大就断言「被播种的节点未被感染」（$P = 0$）。没有任何机制拦住它。

> $0.9999989$ 而非 $1.0$ 源自 $\text{clamp}(\varepsilon,\, 1-\varepsilon),\ \varepsilon = 10^{-6}$——logits 必须有限，否则 `BCEWithLogits` 数值溢出。这是数值下限而非模型误差。

### 4.3 结构保证覆盖的完整清单

同样的代数给出以下**全部**为结构保证的性质：

| 性质 | 机制 |
|---|---|
| `add_node` $\Rightarrow \hat y^{\text{inf}} = 1$ | $\tilde x^{\text{inf}} = \min(x^{\text{inf}} + x^{\text{add}}, 1)$ |
| `remove_node` $\Rightarrow$ 移出波前 | $\times\,(1 - x^{\text{rm}})$ |
| `blocked` 移除 $\Rightarrow \hat y^{\text{inf}} = 0$ | $\tilde x^{\text{inf}} \times (1 - x^{\text{rm}})$ |
| **无活跃入邻居 $\Rightarrow p^{\text{new}} = 0$（自终止，不饱和）** | 门控 $t_{uv} = q_{uv}\cdot \tilde x^{\text{fr}}_u$ |
| **单调性：已感染者恒保持感染** | $\hat y^{\text{inf}} = \tilde x^{\text{inf}} + (1-\tilde x^{\text{inf}})\,p^{\text{new}}$ |
| 边动作生效 | 动作后邻接 $A_t$ 直接进入 $p^{\text{new}}$ 的求积 |

**自终止是不饱和的根本原因**：一个无活跃入邻居的易感节点 $p^{\text{new}} = 0$ 在结构上成立，级联必然自行熄灭。`linear` 头无此约束，自由滚动的 `count_bias` 达 **+49**（100 节点图上感染 99 个）。

### 4.4 边动作为何无需额外特征

对 IC/LT，**动作后的邻接与边权就是充分统计量**：加边使该边真实出现在 `edge_index` 中，$p^{\text{new}}$ 的求积必然经过它；改权重直接改变 $q$ 的输入 $w_{uv}$；删边使该边从求积中消失。动作的全部动力学后果已编码在图里，无需再告诉模型「你刚做了一次加边」。

（本项目据此**未**引入逐边动作张量——它在因果上是冗余的。`typed` 编码补的是 encoder 的**特征可分性**，而非动力学正确性。）

### 4.5 保证的边界 —— 必须明说的部分

$T_{\text{exo}}$ 有结构保证；$T_{\text{endo}}$ **完全是学习的**：

| | 保证 | 来源 |
|---|---|---|
| $T_{\text{exo}}$（即时效果） | ✅ 结构，梯度 = 0 | 闭式 |
| **传播强度 $q_{uv}$** | ❌ 学习 | $\sigma(\text{MLP}([h_u,h_v,w]))$ |
| **动作的下游 / 二阶后果** | ❌ 学习 | 经由 $q$ |

**这一点对 agent 内环至关重要**：内环比较的是「播种 A 相对播种 B 在时程末多扩散多少节点」，该量**完全由学习出的 $q$ 决定，无任何结构兜底**。一个模型可以在 `delta_f1` 与 `count_bias` 上都很好看，同时对所有动作预测**相同**的下游效应——那样它作为 simulator 勉强及格，作为 planner 彻底无用。

§7.3 的检验就是为覆盖这个缺口而建。**结构保证与检验不重叠，互补。**

---

## 5. 训练

**教师强制单步**。批次为真实 $(s_t, a_t)$，预测下一步边缘并对 MC 软标签计分：

$$\mathcal{L} \;=\; \text{BCEWithLogits}\big(\text{logits}[:,0],\ y^{\text{inf}}\big) \;+\; \text{BCEWithLogits}\big(\text{logits}[:,1],\ y^{\text{fr}}\big)$$

| 设置 | 值 | 说明 |
|---|---|---|
| 优化器 | Adam, lr $10^{-3}$, weight decay $5\times10^{-4}$ | |
| 批 | 32 个 transition，拼成**分块对角**大图 | 块间不连通，等价于逐图前向但一次完成 |
| 类不平衡 | `--pos-weight auto` 上采样正类（比值截断至 $[1,50]$） | **结构化头必须用 `off`**：`pos_weight` 会全局抬高 $q$ 并摧毁一步精度；结构形式已排除全零退化 |
| 模型选择 | 每 epoch 在验证集上算 `delta_f1`，取最佳存档 | |
| 早停 | `--patience` 个 epoch 无提升即停 | |

**分块对角批处理**：$B$ 个样本的节点编号偏移后拼成一张不连通大图，消息传递不会跨样本边界，因此在数学上严格等价于逐样本独立前向。

---

## 6. 评估体系总览

四族指标，各回答一个不同的问题：

| 族 | 回答的问题 | 结果块 |
|---|---|---|
| 一步（教师强制） | 给定真实状态，下一步预测准不准？ | `test` |
| 自由滚动 | 当作模拟器连续跑，轨迹保真吗？会不会饱和？ | `rollout` / `rollout_ood` |
| **动作条件化** | **模型真的在读动作，而且读对了吗？** | `action_conditioning` |
| 规划 | 用它来**做决策**，损失多少？ | `planning` / `planning_budget` |

---

## 7. 指标定义与判读

### 7.1 一步指标（`test`）

阈值 0.5；软标签在 0.5 处二值化用于 F1，保留原值用于 Brier。

| 指标 | 定义 | 判读 |
|---|---|---|
| `infected_acc` / `frontier_acc` | 逐节点准确率 | **高是容易的**——多数节点状态不变。必须对照 `persistence` 基线读 |
| `new_infection_f1` | 限定在 $t$ 时易感的节点上，对「新感染」求 F1 | 实质性的一步数字 |
| `delta_f1` | 限定在状态**发生改变**的节点上求 F1 | 早停与头条指标 |
| `brier_infected` | $\mathbb{E}\big[(\hat p - y)^2\big]$，对**软**边缘 | 校准度，越低越好 |
| `add_seed_success` | `add_node` 目标被预测为感染的比例 | 结构化头下**恒为 1.0**（§4.1），故它验证实现而非模型能力 |
| `action_sensitivity` | 同一状态下不同动作产生的相异输出数 | **弱指标**：只回答「输出动了吗」，乱动与动对得分相同。已由 §7.3 取代 |
| `persistence` | 「下一步 = 当前」基线 | 其 `delta_f1` 与 `new_infection_f1` 恒为 0 |

> ⚠️ **IC 下 `delta_f1` $\equiv$ `new_infection_f1`**。IC 单调（节点不会脱感染），故「状态改变」与「新感染」是同一集合，两列在代数上是**同一个数**，不是两份证据。结果 JSON 自动标记 `test.delta_f1_is_new_infection_f1`。

### 7.2 自由滚动（`rollout`）

把模型当作**随机模拟器**：每步从预测边缘**采样**（而非阈值化），滚 $n$ 条轨迹，与真实模拟器在**同一动作序列**下的 MC 轨迹比较。

采样采用**耦合抽样**：从波前边缘一次抽出新感染集合，再由动作语义导出两个通道。逐通道独立抽样会产生不一致状态（幽灵传播者：$\text{frontier}=1$ 而 $\text{infected}=0$），系统性地夸大自由滚动。

| 指标 | 定义 | 判读 |
|---|---|---|
| `ens_marg_mae` | $\text{mean}\lvert \hat p_{\text{model}} - \hat p_{\text{true}}\rvert$，跨节点与步 | 主保真度指标 |
| `ens_count_w1` | 感染计数**分布**的逐步 Wasserstein-1 | 比对均值更强：分布形状是否吻合 |
| **`ens_count_bias`** | 逐步 $\mathbb{E}[\text{model}] - \mathbb{E}[\text{true}]$ | **$\approx 0$ 无偏；$\gg 0$ 饱和。这是结构化头的核心成果指标** |
| `ens_final_count_model/true` | 最终感染计数均值 | 端到端保真度 |

**为何 IC 更有意义**：LT 的真值重跑会重新抽阈值，比较只具指示性。

### 7.3 动作条件化（`action_conditioning`）—— 本项目新增

三个检验，每个都有**明确的零假设**，因此失败可读：

#### (a) `counterfactual_effect` —— 直接度量因果效应

数据中同一状态记录了多个动作，各带自己的 MC 真值，故**真实因果效应已知**。对每一对 $(a, a')$：

$$
d^{\text{true}} = y_{\text{MC}}(s,a) - y_{\text{MC}}(s,a'), \qquad
d^{\text{pred}} = f_\theta(s,a) - f_\theta(s,a')
$$

二者均为 $(N,)$ 向量，因此度量的是动作的**全部下游效应**，不只目标节点。

| 指标 | 判读 |
|---|---|
| `effect_pearson` | 1 = 效应被精确预测；0 = 被忽略 |
| **`effect_mae_norm`** | $\text{mean}\lvert d^{\text{pred}} - d^{\text{true}}\rvert \big/ \text{mean}\lvert d^{\text{true}}\rvert$。**恒等于 1.0 正是「预测无任何效应」的得分**（$d^{\text{pred}} \equiv 0$）。$< 1$ 才说明动作携带真信号 |
| `effect_sign_agree` | 有真实效应的节点上方向正确的比例（随机 = 0.5） |
| `effect_magnitude_ratio` | $\text{mean}\lvert d^{\text{pred}}\rvert / \text{mean}\lvert d^{\text{true}}\rvert$；$<1$ 反应不足，$>1$ 反应过度 |
| `n_pairs` | 反事实对数量。**为 0 表示该数据集无法支撑此检验** |

#### (b) `action_ablation` —— 破坏它，看疼不疼

同一批状态，(i) 清空动作、(ii) 在记录间**打乱**动作，重新计分。

打乱比清空更强：动作的**总体分布不变**（同样的动作、同样的数量），只破坏「哪个动作配哪个状态」的配对，模型无法靠动作的先验分布蒙混。

> **`shuffle_delta_f1_drop` $\approx 0$ 即检验失败**：说明所报的一步精度不读动作也能拿到。

#### (c) `exogenous_fidelity` —— 闭式部分是否精确复现

$T_{\text{exo}}$ 有标准答案，检查是否精确命中，并报**最差节点**而非均值——一个节点违规即可见，不被平均掩盖。

#### 判定的保守性

输出单一 `action_conditioned` 布尔值与 `verdict` 字符串。**「无法检验」不计为通过**：split 中没有反事实对，或带动作记录少于 2 条（打乱退化为恒等），一律输出 `UNTESTABLE` 而非一个看起来像结论的数字。

#### 指标本身的有效性验证

一个指标只有在**该失败时真的失败**才值得报告。故用两个**保证不理解动作**的模型对打：

| 探针模型 | 行为 | 应判 |
|---|---|---|
| `ConstantModel` | 无视一切输入，恒定输出 | FAIL |
| `StateOnlyModel` | 持久性预测：用状态、不用动作 | FAIL |

第二个尤为关键——它**并不笨**，在稀疏扩散上分数不难看，正是「看起来可用、作为 planner 无用」的那类模型。实测两者均**精确**得到 `effect_mae_norm = 1.0` 并被判 FAIL，structured 头判 PASS。若三者同分，该指标即为装饰。

### 7.4 离策略滚动（`rollout_ood`）

§7.2 的滚动重放的是**记录的**动作序列，而数据生成时动作是**均匀随机注入**的。因此它是一个**在策略**数字：它说明模型在训练动作分布下保真，对 agent 实际提出的分布**不置一词**。

`--ood-policies` 在**模型与真实模拟器两侧施加同一条策略生成的序列**，重做同一比较：

| 策略 | 每步动作 | 隔离出什么 |
|---|---|---|
| `null` | 无 | 纯扩散——此处若崩，说明动作通道在代偿一个学坏的扩散项 |
| `degree_seed` | 播种第 $t$ 高度节点 | agent 式极端，离均匀注入最远 |
| `random_seed` | 播种随机节点 | `degree_seed` 的对照：同频率、无结构定向 |
| `block_hubs` | 移除第 $t$ 高度节点 | 遏制类对应物 |

策略按契约**与状态无关**（两侧必须重放同一序列，状态相关策略会在两侧状态分岔时失同步）且**仅含节点操作**（邻接图由记录的边操作重建）。

### 7.5 规划

#### (a) 单步（`planning`）

在采样状态上按预测一步传播为候选 `add_node` 打分，取 argmax，度量 $\text{regret} = \text{oracle} - \text{true}(\text{chosen})$。

> ⚠️ **该块评的是单个动作、单步、对单步 oracle**，比外环真正提出的问题**容易得多**，无法区分「一步排序好」与「多步模拟器可用」。应作为健全性检查引用，而非世界模型可作规划器的证据。

#### (b) 预算内全时程（`planning_budget`）—— 本项目新增

IM 问题的真实形式：选 $k$ 个种子，度量**全时程**传播。模型用**自己的多步滚动**作为目标函数贪心构造种子集（即 agent 的用法），所有 arm 最终由真实模拟器在时程末打分。

参照系为**同候选池上的贪心蒙特卡洛**而非真最优：精确 $k$-子集 IM 是 NP-hard，greedy-MC 是文献通用的 $(1-1/e)$ 基准。

| 指标 | 判读 |
|---|---|
| `budget_spread_{model,greedy_mc,degree,random}` | 各 arm 的 $k$-集真实全时程传播 |
| **`budget_regret_norm`** | $(\text{greedy\_mc} - \text{model})/\text{greedy\_mc}$，**放弃掉的可达传播比例**。无量纲，跨图跨 $k$ 可比 |
| `budget_seed_overlap` | 与 greedy-MC 种子集的重合比例 |

### 7.6 多 seed 与显著性

单 seed 无法把真实的小幅改进与 run 间噪声区分开。`aggregate_seeds.py` 对每个标量给出 mean / std / n，并对任意两列报告

$$\text{gap\_over\_se} = \frac{\bar\mu_1 - \bar\mu_2}{\sqrt{\sigma_1^2/n_1 + \sigma_2^2/n_2}}$$

判定 `separated` 当且仅当 $\lvert \text{gap}\rvert > 2\,\text{SE}_{\text{pooled}}$。**这不是 p 值**——3–5 个 seed 上做 t 检验是表演。它给出原始比值，让读者自行判断差距是否越出噪声。

---

## 8. 帕累托前沿

### 8.1 为何不能用单一排行榜

评估有两条相互制约的轴，且没有可辩护的加权方式把它们压成一个数：

- **保真度** —— `ens_marg_mae`、$\lvert$`count_bias`$\rvert$、`budget_regret_norm`、`delta_f1`、`brier`
- **成本** —— `real_env_episodes`、`evaluator_seconds`、`train_seconds`、`forward_passes`

oracle 评估器赢下每一个保真度列、输掉每一个成本列；单次真实 episode 的 native 评估器反之。**按保真度排序等于默认成本免费——而这恰恰是世界模型要攻击的假设。**

### 8.2 定义

点 $p$ **支配** $p'$，当且仅当

$$\forall\, i:\ p_i \succeq p'_i \quad\wedge\quad \exists\, j:\ p_j \succ p'_j$$

（$\succeq$ 按各目标自己的 sense：误差/成本取 min，F1/传播取 max。）前沿即未被支配的点集。

**缺失值永不构成支配**：某目标上没有数字，不等于有个好数字。这使部分完成的结果行可以安全参与。

`abs_count_bias` 折成绝对值：$-3$ 与 $+3$ 同样不保真，带符号轴会把过预测与欠预测放到同一条轴的两端并让其中一个「最优」。

### 8.3 超体积

仅提供二维（`hypervolume_2d`）。更高维下参考点无法辩护，且单一标量重新引入了前沿本要避免的加权。一条保真度轴对一条成本轴，是它作为公平摘要成立的唯一情形。

---

## 9. 结果

> ### ⚠️ 本节当前为 **预期值（PROJECTED）**，非实测
>
> 下列带 `~` 标记的数字是**基于已有实测锚点的外推估计**，用于汇报前的方案讨论与版面预留。
> **每个估计都在「估计依据」列注明推理来源。**
> 实测运行（13 组配置，BA-100 × 20 图，IC）正在进行，完成后本节整体替换，届时移除本提示框。
>
> 未标 `~` 的数字为**已实测**，来源在表下注明。

### 9.1 已实测锚点

这些数字是真实测量的，是下方所有估计的基准。

**（a）IC / LT 主结果** — BA-100 × 20 图，structured SAGE，seed 42，来源 `world_model/checkpoints/RESULTS.md`

| | IC | LT |
|---|---|---|
| 一步 `delta_f1` | **0.8294** | **0.539** |
| `brier_infected` | **0.0012** | **0.0320** |
| rollout `ens_marg_mae` | **0.0946** | — |
| rollout `ens_count_w1` | **2.606** | — |
| rollout `ens_count_bias` | **−0.577** | **−1.43** |
| 最终计数 模型 / 真值 | **36.62 / 36.70** | **47.3 / 46.2** |
| `plan_regret` 模型 / degree / random | **0.244 / 0.269 / 3.11** | **0.196 / 0.271 / 3.37** |

**（b）头类型对照** — 同配置

| head | `ens_count_bias` | 判读 |
|---|---|---|
| `linear` | **+49** | 100 节点图上感染 99 个，完全饱和 |
| `structured` | **−0.577** | 不饱和 |
| `structured_oracle`（$q = w$，零学习） | `ens_marg_mae` **≈ 0.091** | 结构形式本身正确 |

**（c）动作条件化保证** — §4.2 梯度检验，本次实测

| head | $P(\text{感染}\mid\text{被播种})$ | $\max\lvert\partial P_{\text{seed}}/\partial W\rvert$ |
|---|---|---|
| `structured`（权重 ×1 / ×50 / ×−50 / ×0） | **0.9999989**（四种均同） | **0.000e+00** |
| `linear`（权重 ×1 / ×50 / ×−50） | 0.6246 / **0.0000** / 1.0000 | 7.549e+02 / 0 / 0 |

**（d）指标有效性探针** — 本次实测，`tests/test_action_conditioning.py`

| 探针模型 | `effect_mae_norm` | `effect_pearson` | 判定 |
|---|---|---|---|
| `ConstantModel`（无视输入） | **1.0000** | **0.0000** | **FAIL** |
| `StateOnlyModel`（只用状态） | **1.0000** | — | **FAIL** |
| `structured` head | **< 1.0** | > 0 | **PASS** |

**（e）计算成本** — 本机 CPU 实测，BA-100 × 20 图

| 阶段 | 实测 |
|---|---|
| 数据生成（`--mc-marginals 30`） | **9 s** |
| 训练 20 epoch（sage/structured，hidden 64，batch 32） | **82 s**（≈ 4.1 s/epoch） |
| 一步 + rollout + `action_conditioning` 评估 | **≈ 19 s** |
| `+ --ood-policies`（3 策略） | **+27 s** |
| `+ --plan-budget-k 5`（3 图） | **+33 s** |
| 一次完整 run（150 epoch，patience 30，含早停） | **≈ 103 s** |

---

### 9.2 预期值：动作条件化（`action_conditioning`）

| 指标 | 预期 | 估计依据 |
|---|---|---|
| `effect_pearson` | **~0.75 – 0.88** | 效应向量由 $T_{\text{exo}}$（精确）与 $p^{\text{new}}$（学习）两部分构成。$T_{\text{exo}}$ 贡献目标节点上的精确 ±1，$p^{\text{new}}$ 的精度由已实测 `ens_marg_mae` 0.0946 ≈ oracle 0.091 界定。故相关性应高但不达 1 |
| `effect_mae_norm` | **~0.35 – 0.55** | 零假设恰为 1.0（§7.3a）。目标节点分量精确 ⇒ 该分量误差为 0；误差全部来自下游 $p^{\text{new}}$。下游节点占多数但效应幅度小，故显著低于 1 而非接近 0 |
| `effect_sign_agree` | **~0.80 – 0.92** | 方向比幅度容易；随机基线 0.5 |
| `effect_magnitude_ratio` | **~0.85 – 1.05** | 已实测 `count_bias` −0.577 显示模型**略保守**，故预期略低于 1 |
| `shuffle_delta_f1_drop` | **~0.15 – 0.35** | 阈值 0.5 下，动作错配主要影响被播种节点本身（必然翻转）。已实测 `delta_f1` 0.829，`persistence` 基线 0；drop 应显著为正但不至于打回基线 |
| `seeded_p_infected_worst` | **0.9999989** | **非估计**：§4.1 命题 + §9.1(c) 实测，为 clamp 上界 |
| 判定 | **PASS** | §9.1(d) 已实测 structured 头判 PASS |

### 9.3 预期值：离策略滚动（`rollout_ood`）

| 策略 | `ens_marg_mae` | `ens_count_bias` | 估计依据 |
|---|---|---|---|
| 记录序列（在策略） | **0.0946** | **−0.577** | **已实测**，§9.1(a) |
| `null`（纯扩散） | **~0.06 – 0.09** | **~−0.3 – −0.8** | 无动作 ⇒ 只考扩散项，任务更简单；但也失去 $T_{\text{exo}}$ 的免费正确性 |
| `random_seed` | **~0.09 – 0.12** | **~−0.6 – −1.2** | 与训练注入分布最接近，退化应最小 |
| `degree_seed` | **~0.12 – 0.16** | **~−1.2 – −2.5** | 离分布最远。小规模端到端实测显示该策略下 mae 由 0.076 → 0.101（**×1.33**），按同比例外推 |

> **这是预期中最应被关注的一行**：若 `degree_seed` 的退化倍数显著大于 ×1.5，则说明世界模型作为 agent 内环的保真度**不能由在策略数字外推**，需要在训练数据中加入结构定向的动作注入。

### 9.4 预期值：预算内全时程规划（`planning_budget`，$k=5$）

| 指标 | 预期 | 估计依据 |
|---|---|---|
| `budget_spread_greedy_mc` | **~46 – 52** | 已实测单种子最终传播 36.7；$k=5$ 全时程应高于此 |
| `budget_spread_model` | **~43 – 50** | 由 `budget_regret_norm` 反推 |
| `budget_spread_degree` | **~44 – 51** | BA 图 degree 近最优（RESULTS.md 已声明该图族 degree-trivial） |
| `budget_spread_random` | **~15 – 25** | 已实测单步 `plan_regret_random` 3.11 vs 模型 0.244（**×13**），随机基线极弱 |
| **`budget_regret_norm`** | **~0.03 – 0.10** | 即放弃 3–10% 可达传播。依据：已实测单步 regret 0.244 相对最终传播 36.7 约为 0.7%，但多步复合会放大 |
| `budget_seed_overlap` | **~0.4 – 0.7** | 传播是次模的，不同种子集常给出相近传播，重合度中等即可 |

> **预期 `budget_regret_norm` 与 `degree` 不可分**（差距落在误差棒内）。原因已在 RESULTS.md 声明：BA 的 hub 结构使 degree 近最优。**该对比要有意义必须换到 WS / SBM / 真实图**——这是预期结论中最重要的一条，而非模型的缺陷。

### 9.5 预期值：$w$ 隐藏消融（`--hide-edge-weights`）

> **这是整套实验里唯一能回答「模型是否真的学到了动力学」的一组。**

| 指标 | $w$ 可见（已实测） | $w$ 隐藏（预期） | 估计依据 |
|---|---|---|---|
| `delta_f1` | **0.8294** | **~0.68 – 0.78** | $q$ 须从结构推断而非读取。IC 单步的可预测性主要来自 $w$，故有实质下降 |
| `brier_infected` | **0.0012** | **~0.004 – 0.012** | 校准度对缺失 $w$ 更敏感 |
| `ens_marg_mae` | **0.0946** | **~0.12 – 0.17** | oracle 下限 0.091 不再可达 |
| `ens_count_bias` | **−0.577** | **~−1.5 – −3.5** | 结构门控仍保证不饱和，但幅度校准变差，预期更保守（更负） |
| `effect_mae_norm` | ~0.35 – 0.55 | **~0.5 – 0.75** | 下游效应精度随 $q$ 精度同步退化 |

**判读规则（无论实测落在何处，结论如下）：**

| 若 `ens_marg_mae`($w$ 隐藏) | 结论 |
|---|---|
| **< 0.13** | 强结果。编码器从**纯结构**推断出了接近 oracle 的传播概率，「从数据恢复未知动力学」在 IC 上成立 |
| **0.13 – 0.20** | 中性。学到了部分结构信号，但 $w$ 仍是主要信息来源。表述应为「在已知传播概率的设定下的高保真模拟器」 |
| **> 0.20** | 该 claim 在 IC 上不成立，只能由 LT 支撑（其阈值从不存储，无答案可抄） |

### 9.6 预期值：帕累托前沿（保真度 × 成本）

五骨干 × structured 头，横轴 `train_seconds`（实测），纵轴 `ens_marg_mae`（预期）：

| 配置 | `ens_marg_mae` | `train_seconds` | 前沿？ | 估计依据 |
|---|---|---|---|---|
| `gcn` / structured | **~0.10 – 0.13** | **~80 – 110** | 可能 | 最便宜；无边级注意力 |
| **`sage` / structured** | **0.0946**（已实测） | **~100 – 140** | **是** | 已实测为唯一在 IC 与 LT **双双**保真的骨干 |
| `gat` / structured | **~0.09 – 0.12** | **~180 – 300** | 可能被支配 | 多头注意力显著更贵，保真度提升未必抵得过 |
| `gt` / structured | **~0.09 – 0.12** | **~200 – 350** | 可能被支配 | 同上，且更贵 |
| `gcnii` / structured | **~0.10 – 0.14** | **~150 – 250** | 可能被支配 | 深层设计在 100 节点图上无用武之地 |
| `linear` head | **~0.35 – 0.55** | ~最便宜 | **是**（成本端） | `count_bias` **+49** 已实测；便宜但完全不保真，作为前沿的成本端锚点 |

![保真度-成本帕累托前沿（预期值）](docs/figures/pareto_fidelity_cost_projected.png)

*图 1：保真度 × 成本帕累托前沿。**数值为预期值**（`sage` 的保真度 0.0946 与 linear 头的饱和为实测）。实心 = 前沿，空心 = 被支配。橙色虚线为 IC oracle（$q=w$，零学习）的保真度下限 0.091。左图全量程，右图放大到结构化头。*

前沿由 `world_model/pareto.py` 的支配逻辑计算（图与前沿由构造一致）：

| | 结果 |
|---|---|
| **前沿** | `linear head`、`gcn/structured`、`sage/structured` |
| **被支配** | `gat/structured` ← 被 `sage` 支配；`gt/structured` ← 被 `sage`、`gat` 支配；`gcnii/structured` ← 被 `gcn`、`sage` 支配 |

**预期结论**：前沿由 `linear`（极便宜、极不保真，纯成本端锚点）、`gcn`（便宜、保真度尚可）与 `sage/structured`（最保真、成本适中）构成；注意力类骨干（`gat`/`gt`）**被 `sage` 支配**——更贵且保真度无优势。这与 RESULTS.md 中「SAGE 是所选骨干，因为它是唯一在两种动力学上都保真且最便宜」的既有实测判断一致。

> **前沿的读法**：`linear` 在前沿上**不代表它可用**——它以 `count_bias` **+49**（完全饱和）换来最低成本。这正是帕累托前沿的价值：它呈现取舍而不替你做选择。可用性的门槛（不饱和）是一个**约束**，不是一根轴；施加该约束后前沿只剩 `gcn` 与 `sage`。

图由 `world_model/plot_pareto.py` 生成，可直接在真实结果上重跑：

```bash
python -m world_model.plot_pareto <dir>/*.json --out docs/figures/pareto.png \
    --fidelity ens_marg_mae --cost train_seconds
```

### 9.7 预期值：多 seed 显著性

| 对比 | 预期 `gap_over_se` | 预期 `separated` | 估计依据 |
|---|---|---|---|
| `plan_regret_model` vs `plan_regret_degree` | **~0.4 – 1.2** | **False** | 已实测 0.244±0.055 vs 0.269±0.088，单 seed 下差距已落在误差棒内 |
| `plan_regret_model` vs `plan_regret_random` | **> 10** | **True** | 已实测 0.244 vs 3.109（**×13**） |
| `structured` vs `linear`（`count_bias`） | **> 20** | **True** | 已实测 −0.577 vs +49 |

> **这是必须写进汇报的一条**：「模型优于 degree」在 BA-100 上**不成立**（不可分），而「模型远优于 random」与「结构化头远优于 linear 头」**成立且巨大**。把不可分的对比写成胜出，是这份结果最容易被审稿人抓住的点。

---

## 10. 复现

```bash
# 1. 数据（BA-100 × 20 图，IC，MC 边缘标签）
python -m data.generate_wm_data --dataset ba --num-graphs 20 --syn-nodes 100 \
    --models IC --algorithms random degree pagerank betweenness \
    --mc-marginals 30 --action-ops add_node remove_node \
    --out-dir results/influence_maximization/ba/default/data

# 2. 训练 + 全套评估
python -m world_model.train_wm \
    --data-dir results/influence_maximization/ba/default/data \
    --diffusion-model IC --model sage --head structured --pos-weight off \
    --hidden-dim 64 --n-layers 3 --epochs 400 --patience 50 --batch-size 32 \
    --seed 42 --plan-demo --plan-graphs 5 \
    --ood-policies degree_seed random_seed null \
    --plan-budget-k 5 --plan-budget-graphs 3

# 3. w 隐藏消融（bandit 信息态）
python -m world_model.train_wm ... --hide-edge-weights

# 4. 多 seed 聚合与显著性
python -m world_model.aggregate_seeds <dir>/*.json --out agg.json

# 5. 帕累托前沿
python -m world_model.aggregate_seeds <dir>/*.json --pareto \
    --fidelity ens_marg_mae --cost train_seconds

# 6. 单元测试
python -m pytest tests/ -q
```

---

## 11. 现有结果尚未确立的部分

| 缺口 | 为何重要 | 如何填补 |
|---|---|---|
| **IC 下 $q$ 的输入含真实 $w$** | `structured` 头取 $\text{MLP}([h_u,h_v,w])$，而 oracle 即 $q = w$——模型被告知了答案。达到 oracle 是**下限而非上限**。「从数据恢复未知动力学」目前只由 LT 支撑（其阈值每 episode 重抽且从不存储，无答案可抄） | `--hide-edge-weights --head structured` |
| 单 seed | 0.025 量级的差距无法与 run 间噪声区分 | ≥5 seed + `aggregate_seeds` |
| 仅在策略滚动 | 训练动作为均匀随机注入，agent 不是 | `--ood-policies` |
| 仅单步规划 | 与真实 IM 问题差距大 | `--plan-budget-k` |
| 无跨图族迁移证据 | 训练与测试同分布同规模，而「替代模拟器」的卖点正需泛化性 | 在 BA 上训练，在 WS/SBM 及更大 $N$ 上评估 |

**诚实的归因**：在结构化头下，`add_seed_success = 1.0`、不饱和、单调性、自终止**全部来自闭式**，随机初始化的模型同样成立，因此它们验证的是实现正确性而非学习能力。模型实际学到的全部内容是**每条边一个 $q_{uv}$**。这是一个自觉的、可辩护的取舍——它换来自由头无法拥有的结构保证（`linear` 头 `count_bias` **+49** vs 结构化头 $\approx 0$），代价是换动力学须换 head——但必须作为**取舍**陈述，而非作为「世界模型学到了动力学」。
