# GCOMG-FSS：反事实评分与随机评分再分配

程序文件：`gcomg_fss_redistribution.py`。

本版本在 GCMG 的策略选择和 Grand-Canonical 入场规则上加入三类机制：

1. 实际订单通过随机优先撮合决定 Majority-Game／Minority-Game 评分符号；
2. 未采用动作通过自己的反事实订单簿评分，而不是沿用代理实际所选策略的撮合身份；
3. 普通评分之后，再向全部 \(Ns\) 个“代理—持有策略”槽随机分配非负整数单位分。

最终评分更新写成

\[
\boxed{
S_{iR}(t+1)=\alpha S_{iR}(t)+\widetilde g_{iR}(t)+R_{iR}(t)
},
\qquad
\alpha=1-\frac1T.
\]

其中 \(\widetilde g\) 是普通实际／反事实评分；空市场时它被整体豁免为零。\(R_{iR}\) 是随机再分配得分。再分配不会从亏损策略的评分中扣除相应分数，而是按照本轮预算额外加分，因此不能把整个更新理解为严格的全市场评分守恒。

## 1. FSS、个人评分与入场

### 1.1 FSS 策略空间

记忆长度为 `m` 时，公共信息状态数为

\[
P=2^m,\qquad \mu_t\in\{0,\ldots,P-1\}.
\]

一个策略对每个公共信息状态给出固定动作：

\[
A_{R,\mu}\in\{-1,+1\},
\]

其中 `-1` 表示卖出，`+1` 表示买入。完整策略空间包含

\[
|\mathrm{FSS}|=2^P=2^{2^m}
\]

个确定性策略。每个代理初始化时抽取 `s` 个不同策略，形成固定持有集合 \(\mathcal R_i\)；同一代理内不重复，不同代理可以持有相同策略，模拟过程中不更换持有集合。

### 1.2 个人最优策略

评分按“代理—持有策略”存储：

\[
S_{iR}(t),\qquad R\in\mathcal R_i,
\qquad \texttt{scores.shape=(N,s)}.
\]

每轮代理选择个人评分最高的策略：

\[
R_i^*(t)\in\arg\max_{R\in\mathcal R_i}S_{iR}(t),
\qquad
B_i(t)=\max_{R\in\mathcal R_i}S_{iR}(t).
\]

最高分并列时，程序在该代理的并列策略中无偏向随机选择。这里使用精确浮点相等判断，与市场信号的 `tie_tolerance` 无关。

### 1.3 Grand-Canonical 入场

入场条件保持严格大于阈值：

\[
I_i(t)=\mathbf 1\{B_i(t)>r\}.
\]

实际订单为

\[
o_i(t)=I_i(t)A_{R_i^*(t),\mu_t}\in\{-1,0,+1\}.
\]

未入场代理订单为零，但其全部持有策略仍会通过假想入场获得反事实评分，也可能获得随机再分配分数。实际活跃人数

\[
V(t)=\sum_i I_i(t)
\]

不必等于 `N`；当 \(V(t)=0\) 时进入后文的空市场分支。

## 2. 实际订单簿、信号与随机撮合

### 2.1 净需求和市场信号

令实际买、卖订单数为 \(B_t,S_t\)，定义

\[
D_t=B_t-S_t,\qquad
M_t=\max(B_t,S_t),\qquad
m_t=\min(B_t,S_t).
\]

为避免与记忆长度 `m` 混淆，主循环把少数侧人数记为 `m_small`。实际参与人数满足 \(V(t)=M_t+m_t\)。

实际评分信号由需求冲击相对参照阈值的偏离决定：

\[
q_t=\frac{D_t}{\lambda},\qquad z_t=q_t-L.
\]

设 `tie_tolerance = eps`：

| 条件 | `score_signal` \(\chi_t\) | `new_bit` \(b_t\) |
|---|---:|---:|
| \(z_t>\mathrm{eps}\) | `+1` | `1` |
| \(z_t<-\mathrm{eps}\) | `-1` | `0` |
| \(|z_t|\leq\mathrm{eps}\) | 由 `zero_score_tie` 决定 | 随机 `0` 或 `1` |

零偏离时：

- `zero_score_tie="zero"`：\(\chi_t=0\)；
- `zero_score_tie="random"`：\(\chi_t\) 独立随机取 `+1` 或 `-1`；
- 无论采用哪种评分规则，实际公共信息位都随机生成。

本轮末尾只根据真实订单簿更新公共信息：

\[
\mu_{t+1}=(2\mu_t+b_t)\bmod P.
\]

反事实订单簿不产生公共信息位，也不改变真实 \(\mu_{t+1}\)。

### 2.2 随机优先撮合

若实际买卖双方人数不等，程序从多数侧的 \(M_t\) 个代理中均匀随机、无放回地抽取恰好 \(m_t\) 个。少数侧全部订单与这些被抽中的多数侧订单构成内部撮合双方，使用 Majority-Game 评分符号；其余

\[
M_t-m_t=|D_t|
\]

个残余多数侧订单使用 Minority-Game 评分符号。

若 \(B_t=S_t\)，包括空市场，不存在严格多数侧，不抽取优先组。非空且买卖平衡时，全部实际订单视为可以内部撮合。

程序只保存多数侧被抽中的 `priority_mask`，不显式保存一一配对名单，因为普通评分只需要代理所属的评分分组。

### 2.3 实际采用动作的评分

实际撮合身份记为

\[
\rho_i^{\rm actual}(t)=
\begin{cases}
+1,&\text{少数侧、被抽中的多数侧，或买卖平衡时的实际订单},\\
-1,&\text{未被抽中的残余多数侧订单}.
\end{cases}
\]

活跃代理的实际采用策略获得

\[
\boxed{
g_{iR_i^*}(t)=\rho_i^{\rm actual}(t)o_i(t)\chi_t.
}
\]

在 \(L=0\) 且 \(\chi_t\neq0\) 时，残余多数侧订单的普通实际评分确实为 `-1`。若 `L` 非零，\(\chi_t\) 由 \(D/\lambda-L\) 而不是单独由 `D` 决定，因此不能再把每个残余订单的实际增量一概称为负分；但再分配预算仍按第 4 节的代码条件计算。

## 3. 未采用动作的反事实评分

### 3.1 活跃代理替换订单，未入场代理添加订单

每个候选动作 \(a\in\{-1,+1\}\) 都对应一个独立假想订单簿。

对活跃代理，先移除自己的实际订单，再加入候选订单，假想参与人数仍为 \(V(t)\)：

\[
B'_i(a)=B_t-\mathbf 1\{o_i=+1\}+\mathbf 1\{a=+1\},
\]

\[
S'_i(a)=S_t-\mathbf 1\{o_i=-1\}+\mathbf 1\{a=-1\}.
\]

对未入场代理，没有实际订单可移除，候选订单直接添加到真实订单簿，因此假想参与人数为

\[
V(t)+1=M_t+m_t+1.
\]

两种情况的后续计算相同：

\[
D'_i(a)=B'_i(a)-S'_i(a),
\]

\[
M'_i(a)=\max(B'_i(a),S'_i(a)),\qquad
m'_i(a)=\min(B'_i(a),S'_i(a)).
\]

### 3.2 反事实信号和撮合身份

未采用动作使用自己的市场信号：

\[
\chi'_{i,a}=\operatorname{sign}_{\rm eps}
\left(\frac{D'_i(a)}{\lambda}-L\right).
\]

零偏离处理同样由 `zero_score_tie` 和 `tie_tolerance` 控制，但随机数来自独立的反事实评分随机流。

反事实撮合身份为：

- 候选订单处于少数侧，或反事实订单簿买卖平衡：\(\rho'_i(a)=+1\)；
- 候选订单处于多数侧：以
  \[
  \frac{m'_i(a)}{M'_i(a)}
  \]
  的概率取 \(\rho'_i(a)=+1\)，否则取 \(-1\)。

该概率是随机优先撮合下，一个多数侧候选订单被抽中的精确边际概率。程序只抽这个角色结果，不生成完整的反事实配对名单。

未采用动作的普通增量为

\[
\boxed{g_i(a)=\rho'_i(a)a\chi'_{i,a}}.
\]

若候选动作等于活跃代理的实际动作，则不使用虚拟结果，而是覆盖为真实 \(\chi_t\) 和真实撮合身份对应的实际评分。未入场代理的实际订单为零，所以买、卖两个候选动作都按假想入场处理。

### 3.3 两动作计算再映射到全部策略

二元动作空间使每个代理每轮只需计算两个数：

```text
change_if_sell[i] = g_i(-1)
change_if_buy[i]  = g_i(+1)
```

随后根据持有策略在当前 \(\mu_t\) 下的动作进行映射：

\[
g_{iR}(t)=g_i(A_{R,\mu_t}).
\]

代码对应：

```python
score_change = np.where(
    held_actions_now == 1,
    change_if_buy[:, None],
    change_if_sell[:, None],
)
```

因此，同一代理持有的多个策略若在当前公共信息下给出相同动作，本轮普通评分增量相同。反事实订单簿的计算量是每代理两次，而不是每策略两次；最终 `(N, s)` 评分矩阵的更新仍为 \(O(Ns)\)。

普通反事实评分只改变策略分数，不产生虚拟价格、虚拟公共信息，也不改变其他代理的真实订单或评分。

## 4. 随机评分再分配

### 4.1 非空市场的预算

当实际订单簿非空时，本轮预算为

\[
\boxed{
K_t=
\begin{cases}
(M_t-m_t)s=|D_t|s,&\chi_t\neq0,\\
0,&\chi_t=0.
\end{cases}}
\]

预算只检查实际市场是否非空和实际 \(\chi_t\) 是否为零，不逐一检查残余订单本轮普通评分的正负，也不按所有实际／虚拟负分的绝对值求和。

以下情况预算为零：

- 实际 \(\chi_t=0\)，即使部分反事实策略仍有非零评分增量；
- 买卖平衡，因为 \(M_t=m_t\)；
- 其他使 \(M_t-m_t=0\) 的情形。

普通评分增量先保留，随后再分配。也就是说，残余多数侧原有的评分增量不会被删除或抵消，再分配是在其之外额外加入 \(K_t\) 个单位分。

### 4.2 空市场：先豁免，再形成预算

空市场的定义为

\[
B_t=S_t=0.
\]

此时所有代理都未入场，每个候选动作都是在空订单簿上添加一笔假想订单，因而

\[
M'_i(a)=1,\qquad m'_i(a)=0,\qquad \rho'_i(a)=-1.
\]

程序仍然先计算完整的普通 `score_change`，然后执行两步：

\[
K_t=\operatorname{round}\left(\sum_{i,R}|g_{iR}(t)|\right),
\qquad
\widetilde g_{iR}(t)=0.
\]

这里的 `round` 对应代码中的整数预算转换；普通增量本身只可能为 \(-1,0,+1\)。

在默认 \(L=0\) 时，空订单簿中的任一假想动作 \(a\) 都满足

\[
\chi'_{i,a}=a,\qquad g_i(a)=-a^2=-1.
\]

所以每个持有策略的原始普通增量均为 `-1`，并严格有

\[
\boxed{K_t=Ns}.
\]

这些普通减分随后全部豁免，本轮评分只剩旧分衰减和随机再分配：

\[
\boxed{
S_{iR}(t+1)=\alpha S_{iR}(t)+R_{iR}(t)
\qquad\text{（空市场）}.
}
\]

若 `L` 非零，仍应直接从代码中的

\[
K_t=\sum_{i,R}|g_{iR}(t)|
\]

理解预算，并区分两种零偏离设置：`zero_score_tie="random"` 时，所有虚拟 \(\chi'_{i,a}\) 都为 `-1` 或 `+1`，所以任意 `L` 下仍有 \(K_t=Ns\)；`zero_score_tie="zero"` 时，零偏离虚拟信号为零，对应普通增量也被豁免，预算可能小于 `Ns`，极端情况下可以为零。

空市场机制提供了重新积累评分和再次入场的通道，但不保证下一轮一定重启。是否入场仍取决于更新后的个人最高评分是否严格大于 `r`。

### 4.3 单位分如何分配

无论预算来自非空市场还是空市场，分配函数都相同。每个单位分独立、有放回地投向全部 \(Ns\) 个策略槽：

\[
\Pr\{(i,j)\text{ 获得某单位分}\}=\frac1{Ns}.
\]

因此：

- 活跃和未入场代理的持有策略都可能得分；
- 实际采用和未采用策略都可能得分；
- 不区分买方、卖方、内部撮合或残余订单身份；
- 同一策略可以获得多个单位分；
- 所有预算都会分配完毕，不存在只抽取部分预算的步骤。

令 \(R_{iR}(t)\) 为策略槽获得的单位数，则

\[
R_{iR}(t)\in\{0,1,\ldots,K_t\},
\qquad
\sum_{i,R}R_{iR}(t)=K_t.
\]

整个 `(N, s)` 分配矩阵服从总预算为 \(K_t\)、各槽等概率的多项分布。代码用 `rng.integers()` 生成每个单位分的目标槽，再用 `np.bincount()` 汇总；不是逐策略进行一次 Bernoulli 抽签。

### 4.4 守恒的准确范围

随机分配过程本身严格满足

\[
\sum_{i,R}R_{iR}(t)=K_t,
\]

但完整评分系统不守恒：

1. 旧评分先乘以 \(\alpha=1-1/T\)，存在记忆衰减；
2. 非空市场的 \(K_t\) 是约化预算，且普通负分仍保留，加分没有对应的评分扣减；
3. 空市场先删除普通增量，再额外分配其绝对值预算，进一步改变总评分。

因此，`redistribution_units` 表示本轮新增随机分数的总量，不是从其他策略账户实际转移过来的分数总量。

## 5. 单轮执行顺序

`simulate_gcomg()` 每轮按以下顺序推进：

1. 读取当前 \(\mu_t\)，从 `(N, s)` 个人评分中选择最优策略，随机处理最高分并列；
2. 用 `best_score > r` 判断入场，生成包含 `-1`、`0`、`+1` 的 `agent_orders`；
3. 统计真实 \(B_t,S_t,D_t,M_t,m_t\)，生成实际 \(\chi_t\) 和新公共信息位；
4. 调用 `random_priority_matching()`，抽取真实多数侧优先组；
5. 对每个代理分别计算 `change_if_sell` 和 `change_if_buy`：实际动作复用真实评分，其余候选使用自己的反事实订单簿；
6. 按持有策略当前动作映射为完整 `score_change`；
7. 判断实际市场是否为空：非空市场保留普通增量并计算 \((M-m)s\) 或零预算；空市场先按普通增量绝对值求预算，再把普通增量清零；
8. 执行 `scores = alpha * scores + score_change`；
9. 用独立再分配随机流生成 `(N, s)` 非负整数矩阵并加到评分上，同时更新累计再分配；
10. 用真实 \(D_t\) 清算价格；
11. 更新真实公共信息，保存本轮计数、预算、价格和实际 `score_signal`。

再分配发生在普通评分之后、下一轮策略选择之前。它不改变本轮订单、需求、价格或公共信息，只能通过改变评分排名影响后续市场轨迹。

## 6. FSS 存储与计算量

### 6.1 显式与隐式策略空间

| 范围 | 完整策略空间 | 持有策略初始化 |
|---|---|---|
| `m <= 4` | 显式生成完整 `(2**P, P)` 动作矩阵 | 每个代理用 `rng.choice(..., replace=False)` 无放回抽取 |
| `m >= 5` | 不生成完整矩阵，只保存实际持有策略 | 直接生成长度为 `P` 的独立公平比特串，代理内拒绝重复 |

隐式路径中，每条比特串以概率 \(2^{-P}\) 出现；代理内拒绝重复等价于从完整 FSS 均匀无放回抽取 `s` 个策略。不同代理独立抽样，允许持有相同策略。

所有实际持有策略会全局去重，以共享动作映射：

- `candidate_actions`：去重后的 `(K, P)` 动作矩阵，其中 \(K\leq Ns\)；
- `holdings_positions`：每个代理的持有策略在候选矩阵中的行号，形状为 `(N, s)`；
- `final_holdings`：每个代理持有策略的完整 FSS ID，形状为 `(N, s)`。

全局去重只共享动作映射，不共享个人评分或再分配结果。最终结果满足：

```text
final_scores.shape   = (N, s)
final_holdings.shape = (N, s)
total_redistribution_received.shape = (N, s)
sampled_strategy_actions.shape = (K, P)
```

`final_scores[i, j]`、`final_holdings[i, j]` 和 `total_redistribution_received[i, j]` 按同一策略槽一一对应。本程序不重建完整 FSS 的一维全局评分表。

### 6.2 策略 ID 编码

策略 ID 使用小端位序。令 \(b_{R,\mu}\in\{0,1\}\)，则

\[
R=\sum_{\mu=0}^{P-1}b_{R,\mu}2^\mu,
\qquad A_{R,\mu}=2b_{R,\mu}-1.
\]

隐式路径在 `P <= 64` 时使用 `uint64` 保存策略 ID，更宽时使用 Python 任意精度整数。显式路径的持有 ID 使用 `int32`，对当前支持的 `m <= 4` 足够。

### 6.3 每轮开销

每轮主要计算包括：

- 最优策略选择和动作读取：\(O(Ns)\)；
- 两种候选动作的反事实订单簿与角色抽样：\(O(N)\)；
- 普通评分映射和个人评分更新：\(O(Ns)\)；
- 生成 \(K_t\) 个随机目标并汇总：\(O(K_t+Ns)\)；
- 真实撮合、需求、价格和信息更新：\(O(N)\)。

动作映射存储为 \(O(KP)\)，个人评分和累计再分配均为 \(O(Ns)\)。显式 FSS 路径还需保存完整策略矩阵。

## 7. 初始评分与随机数分流

### 7.1 初始评分

所有个人策略评分从同一个值开始：

\[
S_{iR}(0)=
\begin{cases}
\texttt{initial\_score},&\text{若明确指定},\\
r+\texttt{initial\_score\_offset},&\text{否则}.
\end{cases}
\]

直接使用 `GCOMGConfig()` 时，`initial_score=None`、`r=4`、`initial_score_offset=10`，所以初始评分为 `14`。命令行入口则默认 `initial_score=10`，覆盖偏移规则；虽然命令行默认 `r=-100`，初始评分仍为 `10`，不是 `-90`。

所有评分初值相同，因此第一轮最优策略通过并列随机选择产生。若未指定绝对初值，非负偏移保证初始评分不低于 `r`；偏移为零时，第一轮因严格入场条件 `best_score > r` 而全部不入场。

### 7.2 三个独立随机流

程序从一个 `SeedSequence(config.seed)` 派生三个随机流：

| 随机流 | 用途 |
|---|---|
| `rng` | 持有策略初始化、初始公共信息、最优策略并列选择、真实零偏离信息和评分信号、真实随机优先撮合 |
| `virtual_score_rng` | 反事实零偏离评分信号、反事实多数侧优先角色抽样 |
| `redistribution_rng` | 将每个单位分随机投向 `N*s` 个策略槽 |

分流使反事实评分和再分配消耗的随机数不改变真实策略初始化、真实并列选择或真实撮合序列。预算为零时，分配函数直接返回零矩阵，不消耗再分配随机数。

相同代码、配置和种子可复现结果；但不同评分版本的随机流数量和调用序列不同，不能用相同种子期待它们产生相同轨迹。

直接使用 `GCOMGConfig()` 时，`zero_score_tie` 默认为 `zero`；命令行入口默认为 `random`。实验记录中应显式保存该字段。

## 8. 价格形成

定义

\[
h_t=\frac{D_t}{\lambda}.
\]

`price_mode="linear"` 时，正常步骤为

\[
x_{t+1}=x_t+h_t.
\]

若候选价格非正，仅当前步骤回退为

\[
x_{t+1}=x_t\exp(h_t),
\]

下一步自动恢复线性规则。`price_mode="log"` 时，全程使用对数规则。

| 模式与步骤 | `price_change` | `log_return` | `simple_return` |
|---|---|---|---|
| 普通线性步骤 | \(h_t\) | \(\log(x_{t+1}/x_t)\) | \(h_t/x_t\) |
| 线性回退步骤 | 实际增量 \(x_{t+1}-x_t\) | 通常为 \(h_t\) | 通常为 \(\exp(h_t)-1\) |
| 显式 `log` 模式 | 实际增量 \(x_{t+1}-x_t\) | \(h_t\) | \(\exp(h_t)-1\) |

极端下溢时，线性回退会把价格设为最小正浮点数，并按实际新旧价格重算收益率。`result.price_fallback[t]` 记录该步是否回退；此数组不写入 CSV，命令行结束时报告总回退步数。

价格只读取真实净需求，不读取真实撮合身份、反事实订单簿或随机再分配结果。空市场时 \(D_t=0\)，价格保持不变。

## 9. 代码结构

| 类或函数 | 作用 |
|---|---|
| `FSSStrategySpace`、`build_fss()` | 描述或显式构造完整 FSS |
| `_sample_fss_holdings()` | 隐式均匀抽样、代理内去重和全局动作映射去重 |
| `GCOMGConfig` | 保存模型参数、解析市场深度并校验配置 |
| `_pick_best_strategies()` | 从个人评分中选择最优持有策略并随机处理并列 |
| `random_priority_matching()` | 识别真实多数侧并随机抽取优先组 |
| `counterfactual_action_score_changes()` | 为每个代理计算买、卖两种候选动作的普通评分增量 |
| `randomly_redistribute_units()` | 把整数预算独立、有放回地分配到全部策略槽 |
| `simulate_gcomg()` | 执行入场、实际／反事实评分、预算与豁免、再分配、价格和信息更新 |
| `MarketMaker.clear()` | 清算真实净需求并处理线性价格边界 |
| `GCOMGResult` | 保存市场时间序列、撮合计数、再分配记录和个人最终评分 |
| `autocorrelation_function()` | 用 FFT 计算归一化自相关 |
| `summarize_result()` | 计算 burn-in 后的市场、撮合和再分配统计 |
| `save_csv()` | 保存逐轮时间序列 |
| `save_diagnostics_plot()` | 绘制价格、活跃度、优先组、再分配、收益分布和自相关 |
| `save_price_change_distribution_plot()` | 绘制实际价格增量分布与正态 Q–Q 图 |
| `_parse_args()`、`main()` | 命令行入口和文件输出 |

## 10. 安装与运行

依赖 NumPy；生成图形还需要 Matplotlib：

```bash
python -m pip install numpy matplotlib
```

隐式 FSS 示例：

```bash
python gcomg_fss_redistribution.py \
  --N 101 --m 5 --s 2 --r -100 --T 10000 --L 0 \
  --steps 10000 --burn-in 1000 --seed 123 \
  --initial-score 10 --zero-score-tie random \
  --outdir gcomg_redistribution_output
```

显式 FSS 示例：

```bash
python gcomg_fss_redistribution.py \
  --N 101 --m 4 --s 2 --r 4 --T 100 --L 0 \
  --steps 50000 --burn-in 5000 --seed 123 \
  --zero-score-tie random --outdir run_redistribution_m4
```

查看全部选项：

```bash
python gcomg_fss_redistribution.py --help
```

**不能直接依赖当前命令行的全部默认值完成一次运行。** 默认 `steps=10000`，同时默认 `burn-in=10000`，会使诊断阶段没有 burn-in 后观测。运行时至少应减小 `--burn-in`，或增大 `--steps`；完整诊断还需要 burn-in 后至少 101 个观测。

### 参数表

| 参数 | Python 配置默认值 | 命令行默认值 | 含义 |
|---|---:|---:|---|
| `N` / `--N` | `501` | `101` | 代理人数 |
| `m` / `--m` | `3` | `5` | 公共信息记忆长度，支持 1–10 |
| `s` / `--s` | `2` | `2` | 每个代理持有的不同策略数 |
| `r` / `--r` | `4.0` | `-100.0` | 个人最佳策略评分的入场阈值 |
| `T` / `--T` | `100.0` | `10000.0` | 评分遗忘时间尺度 |
| `L` / `--L` | `0.0` | `0.0` | 市场参照阈值 |
| `lam` / `--lambda` | `None` | `None` | 市场深度；`None` 解析为 `10*N` |
| `x0` / `--x0` | `100.0` | `100.0` | 初始价格 |
| `price_mode` / `--price-mode` | `linear` | `linear` | `linear` 或 `log` |
| `steps` / `--steps` | `100000` | `10000` | 模拟轮数 |
| `seed` / `--seed` | `123` | `123` | 三个派生随机流的共同种子 |
| `mu0` / `--mu0` | `None` | `None` | 初始公共信息；未指定时随机生成 |
| `initial_score_offset` / `--initial-score-offset` | `10.0` | `10.0` | 未指定绝对初值时，使用 `r + offset` |
| `initial_score` / `--initial-score` | `None` | `10.0` | 所有个人策略评分的绝对初值，覆盖偏移规则 |
| `zero_score_tie` / `--zero-score-tie` | `zero` | `random` | 实际和反事实零偏离评分规则 |
| `tie_tolerance` | `1e-12` | 无独立选项 | 判断 \(D/\lambda-L\) 是否为零偏离 |
| — / `--burn-in` | — | `10000` | 诊断舍弃的初始轮数 |
| — / `--outdir` | — | `gcomg_output` | 输出目录 |

主要约束包括：`N >= 1`、`1 <= m <= 10`、`s >= 1` 且不超过完整策略数、`T >= 1`、`steps >= 1`、`x0 > 0`、`lambda > 0`。`summarize_result()` 要求 burn-in 后至少保留 4 个观测；命令行完整运行还会计算滞后 100 的自相关，因此至少应保留 101 个 burn-in 后观测。

## 11. 输出文件

### 11.1 `gcomg_timeseries.csv`

每行对应一轮，包含全部模拟轮次，不应用 burn-in。

| 字段 | 含义 |
|---|---|
| `t` | 从 0 开始的轮次 |
| `mu_t` | 本轮决策使用的真实公共信息状态 |
| `new_bit` | 本轮末尾追加的真实公共信息位 |
| `demand` | 真实净超额需求 \(D_t\) |
| `buy_orders`、`sell_orders` | 实际买入和卖出人数 |
| `active_agents` | 实际活跃代理数 \(V(t)\) |
| `majority_count`、`minority_count` | 实际多数侧和少数侧人数 \(M_t,m_t\) |
| `priority_count` | 实际多数侧被随机抽中的优先代理数 |
| `majority_game_agents` | 实际使用 Majority-Game 符号的订单数 |
| `minority_game_agents` | 实际使用 Minority-Game 符号的残余订单数 |
| `majority_direction` | 买方多数 `+1`、卖方多数 `-1`、平衡或空市场 `0` |
| `redistribution_units` | 本轮随机分配的预算 \(K_t\) |
| `max_redistribution_to_one_strategy` | 本轮任一策略槽获得的最高随机单位数 |
| `empty_market` | 实际空市场标记，CSV 中写为 `0` 或 `1` |
| `price` | 本轮真实清算后的 \(x_{t+1}\) |
| `price_change` | 普通线性步为 \(D_t/\lambda\)，其余为实际价格增量 |
| `simple_return`、`log_return` | 实际简单和对数收益率 |
| `score_signal` | 实际订单簿的 \(\chi_t\)，不是任何 \(\chi'_{i,a}\) |

非空且非平衡时：

\[
\texttt{priority\_count}=m_t,
\qquad
\texttt{majority\_game\_agents}=2m_t,
\qquad
\texttt{minority\_game\_agents}=M_t-m_t=|D_t|.
\]

非空且买卖平衡时：

\[
\texttt{priority\_count}=0,
\qquad
\texttt{majority\_game\_agents}=V(t),
\qquad
\texttt{minority\_game\_agents}=0.
\]

空市场时上述人数字段均为零。`minority_game_agents` 不包括未入场但接受反事实评分的代理。

CSV 不保存个人评分、持有策略或累计再分配矩阵；这些可从 Python 结果对象获取。逐代理实际撮合身份、反事实信号和每轮分配矩阵没有存入 `GCOMGResult`，如需检查，只能在主循环中额外记录。`price_fallback` 只保存在结果对象中。

### 11.2 `gcomg_summary.json`

文件包含：

- `parameters`：完整 `GCOMGConfig`；
- `lambda_used`：实际市场深度；
- `diagnostics`：burn-in 后统计量，以及最终个人评分摘要。

主要诊断字段包括：

| 字段或字段组 | 含义 |
|---|---|
| `observations_after_burn_in` | burn-in 后观测数 |
| `mean_demand`、`std_demand`、`excess_kurtosis_demand` | 真实需求均值、总体标准差和超额峰度 |
| `mean_active_agents`、`std_active_agents` | 实际活跃人数统计 |
| `mean_majority_side_agents`、`mean_minority_side_agents` | 实际两侧规模的时间均值 |
| `mean_priority_matched_agents` | 多数侧被抽中人数的时间均值，不是内部撮合双方总人数 |
| `mean_majority_game_share_of_active` | 活跃轮次计算 `majority_game_agents/active_agents`，空市场记为 `0`，再取时间均值 |
| `mean_redistribution_units` | burn-in 后每轮预算 \(K_t\) 的均值 |
| `max_redistribution_to_one_strategy` | burn-in 后单轮单策略最高随机得分的最大值 |
| `empty_market_share` | burn-in 后空市场轮次占比 |
| `market_restart_count` | burn-in 后由空市场转为非空市场的相邻轮次数量 |
| `mean_final_strategy_score`、`max_final_strategy_score` | 模拟结束时 `(N, s)` 个人评分的均值和最大值 |
| `mean_log_return`、`std_log_return`、`excess_kurtosis_log_return` | 对数收益率统计 |
| `acf_return_lag1` | 对数收益率滞后 1 自相关 |
| `acf_abs_return_lag*` | 去均值收益率绝对值的自相关 |
| `acf_volume_lag*` | 实际活跃人数 \(V(t)\) 的自相关 |

`market_restart_count` 只统计 burn-in 后切片内部的 `empty[:-1] & ~empty[1:]` 转移，不包含从 burn-in 前最后一轮到 burn-in 后第一轮之间的跨界转移。

收益率绝对值自相关使用 \(|r_t-\bar r|\)，不是原始 \(|r_t|\)。自相关函数先对输入去均值，再用 FFT 计算协方差；各滞后协方差统一除以样本数，最后除以零滞后协方差。零方差序列的非零滞后可能为 `NaN`，例如长期全员参与或长期空市场时的活跃度自相关。

### 11.3 图形输出

`gcomg_diagnostics.png` 包含：

1. 从模拟起点开始的前至多 3000 步价格；
2. 同时段的活跃代理数、多数侧优先代理数和随机分配预算；
3. burn-in 后标准化对数收益率分布，纵轴为对数刻度；
4. burn-in 后滞后 1 至 100 的收益率、去均值收益率绝对值和活跃人数自相关。

价格和活跃度子图不舍弃 burn-in；分布和自相关子图使用 burn-in 后数据。第二幅子图把人数与评分单位画在同一纵轴上，主要用于观察时间上的共同变化，不表示二者具有相同量纲。

`gcomg_price_change_distribution.png` 使用 burn-in 后实际价格增量

\[
\Delta x_t=x_{t+1}-x_t,
\]

绘制直方图和正态 Q–Q 图。它直接从 `result.price` 计算增量，不依赖含义随价格模式变化的 `price_change` 字段。

## 12. 自检重点与模型边界

运行小规模模拟时，可分别检查两类预算。

非空市场：

\[
\texttt{redistribution\_units}[t]
=(M_t-m_t)s\,\mathbf 1\{\chi_t\neq0\}.
\]

空市场且 \(L=0\)：

\[
\texttt{redistribution\_units}[t]=Ns,
\qquad
\texttt{empty\_market}[t]=\mathrm{True}.
\]

每轮实际分配矩阵的元素应是非负整数，总和严格等于 `redistribution_units[t]`；`total_redistribution_received` 应等于各轮分配矩阵的累计值。结果对象还满足

\[
\texttt{buy\_orders}+\texttt{sell\_orders}
=\texttt{active\_agents}
=\texttt{majority\_count}+\texttt{minority\_count}.
\]

模型中的“流动性收益归属”是评分层面的约化假设，不是真实财富转移。程序没有显式记录现金、持仓、订单年龄、逐笔成交价格、多档限价订单簿、撤单或历史限价单提供者身份。随机分配可能改变策略排名和再入场概率，但不能据此推断真实订单层面的收益流向；空市场补偿提供了重启通道，也不保证市场会持续活跃。
