# GCOMG-FSS：强制全员参与、无评分衰减、关闭评分再分配

程序文件：`gcomg_fss_no_redistribution.py`。

本版本固定两个极限参数：

\[
\boxed{r=-\infty},\qquad \boxed{T=+\infty}.
\]

因此，所有代理每轮都提交订单，策略评分不发生遗忘。评分只由实际采用动作的真实评分和未采用动作的反事实评分构成，不再加入残余订单补偿、空市场补偿或随机再分配分数：

\[
\boxed{S_{iR}(t+1)=S_{iR}(t)+g_{iR}(t)}.
\]

程序仍保留 FSS 策略空间、个人最优策略选择、随机优先撮合、净需求定价和公共信息更新。需要注意，这不是在通用 GCOMG 上只关闭一个输出开关：`r`、`T`、反事实评分和随机数分流都属于本版本的核心设定。

## 1. 状态变量与策略选择

### 1.1 FSS 策略空间

记忆长度为 `m` 时，公共信息状态数为

\[
P=2^m,\qquad \mu_t\in\{0,\ldots,P-1\}.
\]

一个完整策略对每个公共信息状态给出固定动作：

\[
A_{R,\mu}\in\{-1,+1\},
\]

其中 `-1` 表示卖出，`+1` 表示买入。完整策略空间包含

\[
|\mathrm{FSS}|=2^P=2^{2^m}
\]

个确定性策略。每个代理初始化时抽取 `s` 个不同策略，形成固定持有集合 \(\mathcal R_i\)；同一代理内不重复，不同代理可以持有相同策略，模拟过程中不更换持有集合。

### 1.2 个体评分与最优策略

评分按“代理—持有策略”存储：

\[
S_{iR}(t),\qquad R\in\mathcal R_i.
\]

每轮代理选择个人评分最高的策略：

\[
R_i^*(t)\in\arg\max_{R\in\mathcal R_i}S_{iR}(t).
\]

最高分并列时，程序在该代理的并列策略中无偏向随机选择。这里使用精确浮点相等判断，与市场信号的 `tie_tolerance` 无关。

即使不同代理持有同一个 FSS 策略，其评分也不共享，因为两者可能面对不同的实际或反事实撮合结果。代码中的评分数组始终为：

```text
scores.shape = (N, s)
```

### 1.3 强制全员参与

配置校验要求 `r=-inf`，主循环直接构造全为真的 `active` 数组。因此每轮都有

\[
\boxed{V(t)=N},\qquad \boxed{n_{\rm buy}(t)+n_{\rm sell}(t)=N}.
\]

代理实际订单为

\[
o_i(t)=A_{R_i^*(t),\mu_t}\in\{-1,+1\},
\]

不存在订单为零的代理。由于 `N >= 1`，空市场在本版本中不可达。

## 2. 实际订单簿与采用策略评分

### 2.1 净需求和实际市场信号

令本轮实际买、卖订单数为

\[
B_t=n_{\rm buy}(t),\qquad S_t=n_{\rm sell}(t),
\]

则

\[
D_t=B_t-S_t,\qquad q_t=\frac{D_t}{\lambda},\qquad z_t=q_t-L.
\]

设 `tie_tolerance = eps`。实际评分信号 \(\chi_t\) 和公共信息位 \(b_t\) 按以下规则生成：

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

反事实订单簿不会产生公共信息位，也不会改变 \(\mu_{t+1}\)。

### 2.2 随机优先撮合

定义实际订单簿两侧规模

\[
M_t=\max(B_t,S_t),\qquad m_t=\min(B_t,S_t).
\]

为避免与记忆长度 `m` 混淆，主循环把少数侧人数记为 `m_small`。

若买卖双方人数不等，程序从多数侧的 \(M_t\) 个代理中均匀随机、无放回地抽取恰好 \(m_t\) 个。少数侧全部订单与这些被抽中的多数侧订单构成内部撮合双方，使用 Majority-Game 评分符号；其余

\[
M_t-m_t=|D_t|
\]

个残余多数侧订单使用 Minority-Game 评分符号。

若 \(B_t=S_t\)，不存在严格多数侧，不抽取优先组，全部实际订单视为可以内部撮合。

程序只保存多数侧被抽中的 `priority_mask`，不显式保存一一配对名单，因为评分只需要代理所属的评分分组。

### 2.3 实际采用动作的评分

令 \(\rho_i^{\rm actual}(t)\) 表示代理的实际撮合身份：

\[
\rho_i^{\rm actual}(t)=
\begin{cases}
+1,&\text{少数侧、被抽中的多数侧，或买卖平衡时的实际订单},\\
-1,&\text{未被抽中的残余多数侧订单}.
\end{cases}
\]

实际采用策略的评分增量为

\[
\boxed{
g_{iR_i^*}(t)=\rho_i^{\rm actual}(t)o_i(t)\chi_t.
}
\]

关闭再分配并不会取消残余多数侧的负分；该负分仍直接进入个人策略评分。

## 3. 未采用动作的反事实评分

### 3.1 为什么需要反事实订单簿

代理未采用的策略本轮没有提交真实订单。如果只按真实订单簿和实际撮合身份给这些策略评分，就无法区分“同一代理当时改买或改卖”可能面对的不同市场结果。

本程序因此为两种候选动作分别构造反事实订单簿。由于本版本全员参与，假想动作只会**替换代理自己的实际订单**，不存在向订单簿额外添加一个未入场代理的情形。

对代理 \(i\) 和候选动作 \(a\in\{-1,+1\}\)，先移除自己的实际订单，再加入候选订单：

\[
B'_i(a)=B_t-\mathbf 1\{o_i=+1\}+\mathbf 1\{a=+1\},
\]

\[
S'_i(a)=S_t-\mathbf 1\{o_i=-1\}+\mathbf 1\{a=-1\}.
\]

由此得到

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
\left(\frac{D'_i(a)}{\lambda}-L\right),
\]

其中零偏离处理与实际市场相同，也由 `zero_score_tie` 和 `tie_tolerance` 控制，但随机数来自独立的反事实评分随机流。

反事实撮合身份为：

- 候选订单处于少数侧，或反事实订单簿买卖平衡：\(\rho'_i(a)=+1\)；
- 候选订单处于多数侧：以
  \[
  \frac{m'_i(a)}{M'_i(a)}
  \]
  的概率取 \(\rho'_i(a)=+1\)，否则取 \(-1\)。

该概率是随机优先撮合下，一个多数侧候选订单被抽中的精确边际概率。程序只抽这个角色结果，不生成完整的反事实配对名单。

因此，未采用动作的增量为

\[
\boxed{g_i(a)=\rho'_i(a)a\chi'_{i,a}}.
\]

若候选动作恰好等于实际动作，则不使用上述虚拟结果，而是覆盖为第 2.3 节的真实信号和真实撮合结果。

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
scores = scores + score_change
```

所以，同一代理持有的多个策略若在当前公共信息下给出相同动作，本轮获得相同评分增量。反事实订单簿的计算量是每代理两次，而不是每策略两次；最终 `(N, s)` 评分矩阵的更新仍然不可省略。

反事实评分只改变策略分数。它不产生虚拟价格、虚拟公共信息，也不改变其他代理的真实订单。

## 4. 关闭再分配后的评分含义

评分更新没有旧分衰减，也没有额外分配项：

\[
S_{iR}(t+1)=S_{iR}(t)+g_{iR}(t).
\]

因此，本版本的评分变化可以完整归结为：

1. 采用策略：真实订单簿、真实 \(\chi_t\)、真实撮合身份；
2. 未采用策略：各自的反事实订单簿、\(\chi'_{i,a}\) 和反事实撮合身份；
3. 其他加分或补偿：无。

这并不意味着全策略总评分守恒。一般而言，

\[
\sum_{i,R}\Delta S_{iR}(t)=\sum_{i,R}g_{iR}(t)
\]

不必等于零，因为实际内部撮合与残余订单并存，未采用动作还会产生独立的反事实评分。

为兼容前一版输出结构，代码保留了 `randomly_redistribute_units()` 及相关字段，但模拟循环不调用该函数，也不创建再分配随机流。以下结果恒为零：

- `redistribution_units[t]`；
- `max_redistribution_to_one_strategy[t]`；
- `total_redistribution_received[i, j]`。

`empty_market[t]` 恒为 `False`。这不是一次“空市场后免除评分”的结果，而是全员参与使空市场状态根本无法出现。

## 5. 单轮执行顺序

`simulate_gcomg()` 每轮严格按以下顺序推进：

1. 读取当前 \(\mu_t\)，从 `(N, s)` 个人评分中选择最优持有策略，随机处理最高分并列；
2. 令所有代理入场，由所选策略生成 `agent_orders`；
3. 统计真实 \(B_t,S_t,D_t\)，生成实际 \(\chi_t\) 和新的公共信息位；
4. 调用 `random_priority_matching()`，抽取真实多数侧优先组；
5. 对每个代理分别计算 `change_if_sell` 和 `change_if_buy`：实际动作复用真实评分，另一动作使用自己的反事实订单簿；
6. 按持有策略当前动作映射评分增量，并执行 `scores = scores + score_change`；
7. 将再分配和空市场兼容字段写为零或 `False`；
8. 用真实 \(D_t\) 清算价格；
9. 更新真实公共信息，保存本轮计数、价格、收益率和实际 `score_signal`。

真实随机撮合只影响评分及后续策略选择。本轮需求和价格在撮合前已经由实际订单确定；反事实订单簿同样不回写真实市场。

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

全局去重只共享动作映射，不共享个人评分。最终结果满足：

```text
final_scores.shape   = (N, s)
final_holdings.shape = (N, s)
sampled_strategy_actions.shape = (K, P)
```

`final_scores[i, j]` 与 `final_holdings[i, j]` 一一对应。本程序不重建完整 FSS 的一维全局评分表。

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
- 评分映射和个人评分更新：\(O(Ns)\)；
- 真实撮合、需求、价格和信息更新：\(O(N)\)。

动作映射存储为 \(O(KP)\)，个人评分为 \(O(Ns)\)。显式 FSS 路径还需保存完整策略矩阵。

## 7. 初始评分与随机数分流

### 7.1 初始评分

所有 `(agent, strategy)` 评分从同一个有限值开始：

\[
S_{iR}(0)=
\begin{cases}
\texttt{initial\_score},&\text{若其不为 `None`},\\
\texttt{initial\_score\_offset},&\text{否则}.
\end{cases}
\]

默认两者都是 `10`，所以默认初始评分为 `10`。这里不再使用通用 GCOMG 的 \(r+\text{offset}\)，否则 \(r=-\infty\) 会产生无意义的负无穷初值。

`initial_score_offset` 只是保留的绝对值回退接口；只有 `initial_score=None` 时才生效。所有评分初值相同，因此第一轮最优策略通过并列随机选择产生。

### 7.2 两个独立随机流

程序从一个 `SeedSequence(config.seed)` 派生两个随机流：

| 随机流 | 用途 |
|---|---|
| `rng` | 持有策略初始化、初始公共信息、最优策略并列选择、真实零偏离信息和评分信号、真实随机优先撮合 |
| `virtual_score_rng` | 反事实零偏离评分信号、反事实多数侧优先角色抽样 |

分流的目的，是让反事实评分消耗的随机数不改变真实策略初始化、真实并列选择或真实撮合序列。本版本没有第三个再分配随机流。

相同代码、配置和种子可复现结果；但本版本的随机数调用序列与通用 GCOMG 不同，不能用相同种子期待两者轨迹一致。

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

价格只读取真实净需求，不读取真实撮合身份或任何反事实订单簿。

## 9. 代码结构

| 类或函数 | 作用 |
|---|---|
| `FSSStrategySpace`、`build_fss()` | 描述或显式构造完整 FSS |
| `_sample_fss_holdings()` | 隐式均匀抽样、代理内去重和全局动作映射去重 |
| `GCOMGConfig` | 保存参数，解析市场深度，并强制校验 `r=-inf`、`T=+inf` |
| `_pick_best_strategies()` | 从个人评分中选择最优持有策略并随机处理并列 |
| `random_priority_matching()` | 识别真实多数侧并随机抽取优先组 |
| `counterfactual_action_score_changes()` | 为每个代理计算买、卖两种候选动作的评分增量 |
| `randomly_redistribute_units()` | 为兼容保留的随机分配函数，本版本模拟循环不调用 |
| `simulate_gcomg()` | 执行策略选择、实际／反事实评分、价格和公共信息更新 |
| `MarketMaker.clear()` | 清算真实净需求并处理线性价格边界 |
| `GCOMGResult` | 保存市场时间序列、撮合计数、兼容字段和个人最终评分 |
| `autocorrelation_function()` | 用 FFT 计算归一化自相关 |
| `summarize_result()` | 计算 burn-in 后的市场和撮合统计 |
| `save_csv()` | 保存逐轮时间序列 |
| `save_diagnostics_plot()` | 绘制价格、撮合人数、收益分布和自相关 |
| `save_price_change_distribution_plot()` | 绘制实际价格增量分布与正态 Q–Q 图 |
| `_parse_args()`、`main()` | 命令行入口和文件输出 |

## 10. 安装与运行

依赖 NumPy；生成图形还需要 Matplotlib：

```bash
python -m pip install numpy matplotlib
```

隐式 FSS 示例：

```bash
python gcomg_fss_no_redistribution.py \
  --N 101 --m 5 --s 2 --L 0 \
  --steps 10000 --burn-in 1000 --seed 123 \
  --initial-score 10 --zero-score-tie random \
  --outdir gcomg_no_redistribution_output
```

显式 FSS 示例：

```bash
python gcomg_fss_no_redistribution.py \
  --N 101 --m 4 --s 2 --L 0 \
  --steps 50000 --burn-in 5000 --seed 123 \
  --zero-score-tie random --outdir run_no_redistribution_m4
```

查看全部选项：

```bash
python gcomg_fss_no_redistribution.py --help
```

**不能直接依赖当前命令行的全部默认值完成一次运行。** 默认 `steps=10000`，同时默认 `burn-in=10000`，会使诊断阶段没有 burn-in 后观测。运行时至少应减小 `--burn-in`，或增大 `--steps`；完整诊断还需要 burn-in 后至少 101 个观测。

### 参数表

| 参数 | Python 配置默认值 | 命令行默认值 | 含义 |
|---|---:|---:|---|
| `N` / `--N` | `501` | `101` | 代理人数 |
| `m` / `--m` | `3` | `5` | 公共信息记忆长度，支持 1–10 |
| `s` / `--s` | `2` | `2` | 每个代理持有的不同策略数 |
| `r` / `--r` | `-inf` | `-inf` | 固定入场极限，强制全员参与 |
| `T` / `--T` | `+inf` | `+inf` | 固定记忆极限，评分不衰减 |
| `L` / `--L` | `0.0` | `0.0` | 市场参照阈值 |
| `lam` / `--lambda` | `None` | `None` | 市场深度；`None` 解析为 `10*N` |
| `x0` / `--x0` | `100.0` | `100.0` | 初始价格 |
| `price_mode` / `--price-mode` | `linear` | `linear` | `linear` 或 `log` |
| `steps` / `--steps` | `100000` | `10000` | 模拟轮数 |
| `seed` / `--seed` | `123` | `123` | 两个派生随机流的共同种子 |
| `mu0` / `--mu0` | `None` | `None` | 初始公共信息；未指定时随机生成 |
| `initial_score_offset` / `--initial-score-offset` | `10.0` | `10.0` | `initial_score=None` 时的绝对初值回退 |
| `initial_score` / `--initial-score` | `10.0` | `10.0` | 所有个人策略评分的绝对初值 |
| `zero_score_tie` / `--zero-score-tie` | `zero` | `random` | 实际和反事实零偏离评分规则 |
| `tie_tolerance` | `1e-12` | 无独立选项 | 判断 \(D/\lambda-L\) 是否为零偏离 |
| — / `--burn-in` | — | `10000` | 诊断舍弃的初始轮数 |
| — / `--outdir` | — | `gcomg_output` | 输出目录 |

虽然命令行保留了 `--r` 和 `--T`，但传入值仍必须分别为负无穷和正无穷；有限值会被配置校验拒绝。

主要约束包括：`N >= 1`、`1 <= m <= 10`、`s >= 1` 且不超过完整策略数、`steps >= 1`、`x0 > 0`、`lambda > 0`、初始评分有限。`summarize_result()` 要求 burn-in 后至少保留 4 个观测；命令行完整运行还会计算滞后 100 的自相关，因此至少应保留 101 个 burn-in 后观测。

## 11. 输出文件

### 11.1 `gcomg_timeseries.csv`

每行对应一轮，包含全部模拟轮次，不应用 burn-in。

| 字段 | 含义 |
|---|---|
| `t` | 从 0 开始的轮次 |
| `mu_t` | 本轮决策使用的真实公共信息状态 |
| `new_bit` | 本轮末尾追加的真实公共信息位 |
| `demand` | 真实净超额需求 \(D_t\) |
| `buy_orders`、`sell_orders` | 实际买入和卖出人数，二者之和恒为 `N` |
| `active_agents` | 活跃代理数，恒为 `N` |
| `majority_count`、`minority_count` | 实际多数侧和少数侧人数 \(M_t,m_t\) |
| `priority_count` | 实际多数侧被随机抽中的优先代理数 |
| `majority_game_agents` | 实际使用 Majority-Game 符号的订单数 |
| `minority_game_agents` | 实际使用 Minority-Game 符号的残余订单数 |
| `majority_direction` | 买方多数 `+1`、卖方多数 `-1`、平衡 `0` |
| `redistribution_units` | 兼容字段，恒为 `0` |
| `max_redistribution_to_one_strategy` | 兼容字段，恒为 `0` |
| `empty_market` | 兼容字段，CSV 中恒为 `0`（结果数组中为 `False`） |
| `price` | 本轮真实清算后的 \(x_{t+1}\) |
| `price_change` | 普通线性步为 \(D_t/\lambda\)，其余为实际价格增量 |
| `simple_return`、`log_return` | 实际简单和对数收益率 |
| `score_signal` | 实际订单簿的 \(\chi_t\)，不是任何 \(\chi'_{i,a}\) |

非平衡订单流满足：

\[
\texttt{priority\_count}=m_t,
\qquad
\texttt{majority\_game\_agents}=2m_t,
\qquad
\texttt{minority\_game\_agents}=|D_t|.
\]

买卖平衡时：

\[
\texttt{priority\_count}=0,
\qquad
\texttt{majority\_game\_agents}=N,
\qquad
\texttt{minority\_game\_agents}=0.
\]

CSV 不保存个人评分、持有策略或价格回退布尔数组；这些可从 Python 结果对象获取。逐代理实际撮合身份、反事实信号和反事实逐轮增量没有存入 `GCOMGResult`，如需检查，只能在主循环中额外记录。

### 11.2 `gcomg_summary.json`

文件包含：

- `parameters`：完整 `GCOMGConfig`；其中 `r` 和 `T` 分别写成字符串 `"-inf"` 和 `"+inf"`，避免输出非标准 JSON 无穷数值；
- `lambda_used`：实际市场深度；
- `diagnostics`：burn-in 后统计量，以及最终个人评分摘要。

主要诊断字段包括：

| 字段或字段组 | 含义 |
|---|---|
| `observations_after_burn_in` | burn-in 后观测数 |
| `mean_demand`、`std_demand`、`excess_kurtosis_demand` | 真实需求均值、总体标准差和超额峰度 |
| `mean_active_agents`、`std_active_agents` | 分别恒为 `N` 和 `0` |
| `mean_majority_side_agents`、`mean_minority_side_agents` | 实际两侧规模的时间均值 |
| `mean_priority_matched_agents` | 多数侧被抽中人数的时间均值，不是内部撮合双方总人数 |
| `mean_majority_game_share_of_active` | 每轮 `majority_game_agents/active_agents` 的时间均值 |
| `mean_redistribution_units`、`max_redistribution_to_one_strategy` | 兼容字段，恒为零 |
| `empty_market_share`、`market_restart_count` | 兼容字段，恒为零 |
| `mean_final_strategy_score`、`max_final_strategy_score` | 模拟结束时 `(N, s)` 个人评分的均值和最大值 |
| `mean_log_return`、`std_log_return`、`excess_kurtosis_log_return` | 对数收益率统计 |
| `acf_return_lag1` | 对数收益率滞后 1 自相关 |
| `acf_abs_return_lag*` | 去均值收益率绝对值的自相关 |
| `acf_minority_side_lag*` | 实际少数侧人数 \(m_t\) 的自相关 |

收益率绝对值自相关使用 \(|r_t-\bar r|\)，不是原始 \(|r_t|\)。由于 \(V(t)=N\) 是常数，本版本不再计算活跃度自相关，而使用少数侧人数作为订单活动诊断。

自相关函数先对输入去均值，再用 FFT 计算协方差；各滞后协方差统一除以样本数，最后除以零滞后协方差。零方差序列的非零滞后可能为 `NaN`。

### 11.3 图形输出

`gcomg_diagnostics.png` 包含：

1. 从模拟起点开始的前至多 3000 步价格；
2. 同时段的实际少数侧人数和内部撮合双方总人数；
3. burn-in 后标准化对数收益率分布，纵轴为对数刻度；
4. burn-in 后滞后 1 至 100 的收益率、去均值收益率绝对值和少数侧人数自相关。

价格和撮合人数子图不舍弃 burn-in；分布和自相关子图使用 burn-in 后数据。

`gcomg_price_change_distribution.png` 使用 burn-in 后实际价格增量

\[
\Delta x_t=x_{t+1}-x_t,
\]

绘制直方图和正态 Q–Q 图。它直接从 `result.price` 计算增量，不依赖含义随价格模式变化的 `price_change` 字段。

## 12. 自检重点与模型边界

运行小规模模拟时，可用以下恒等式检查主循环：

\[
\texttt{active\_agents}[t]=N,
\qquad
\texttt{buy\_orders}[t]+\texttt{sell\_orders}[t]=N,
\]

\[
\texttt{majority\_count}[t]+\texttt{minority\_count}[t]=N,
\qquad
\texttt{empty\_market}[t]=\mathrm{False}.
\]

所有再分配数组应严格为零。初始评分取整数时，无衰减且普通评分增量属于 \(\{-1,0,+1\}\)，所以任意时刻评分都应为“该整数初值加整数增量”；若出现非整数补偿或旧分衰减，说明运行的不是这份代码或配置已被改动。

本程序仍是策略学习与市场统计的约化模型。反事实订单簿用于策略评分，不是一套并行演化的真实市场；程序也不显式记录现金、持仓、订单年龄、逐笔成交价格、多档限价订单簿或做市商库存。关闭再分配后，残余多数侧负分不再被补偿，个人策略评分完全由上述实际／反事实规则驱动，不能直接等同于标准 Minority Game 的单一全局评分规则。
