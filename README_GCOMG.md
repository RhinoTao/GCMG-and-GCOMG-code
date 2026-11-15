# GCOMG-FSS：随机优先撮合与个体策略评分

`gcomg_fss.py` 在 GCMG-FSS 的基础上修改策略评分规则。程序名采用：

**GCOMG = Great-Canonical Order Matching decided Game**

模型仍使用 Grand-Canonical 入场条件、FSS 策略、净需求价格方程和公共信息历史。新增部分只有一个：每轮根据随机优先撮合结果，决定代理使用 Majority-Game 还是 Minority-Game 的评分符号。

这一改动使评分必须按代理存储。同一个 FSS 策略被不同代理持有时，动作映射相同，但持有者可能因撮合身份不同而获得不同的评分更新。

## 1. 从 GCMG 保留下来的部分

### 1.1 公共信息和 FSS 策略

记忆长度为 `m` 时，公共信息状态数为

\[
P=2^m,\qquad \mu_t\in\{0,\ldots,P-1\}.
\]

一个策略对每个公共信息状态给出固定动作：

\[
A_{R,\mu}\in\{-1,+1\},
\]

其中 `-1` 表示卖出，`+1` 表示买入。完整策略空间包含所有长度为 `P` 的动作向量：

\[
|\mathrm{FSS}|=2^P=2^{2^m}.
\]

每个代理初始化时抽取 `s` 个不同策略，形成固定持有集合 \(\mathcal R_i\)。同一代理内无重复，不同代理可以持有相同策略；模拟过程中不更换持有集合。

### 1.2 个体策略选择

与原 GCMG 的全局共享评分不同，GCOMG 使用

\[
S_{iR}(t),
\]

表示代理 \(i\) 对自己所持策略 \(R\) 的评分。每轮选择其个人评分最高的持有策略：

\[
R_i^*(t)\in\arg\max_{R\in\mathcal R_i}S_{iR}(t),
\qquad
B_i(t)=\max_{R\in\mathcal R_i}S_{iR}(t).
\]

最高评分并列时，程序在该代理的并列策略中无偏向随机选择。这里使用精确浮点相等判断，与需求信号的 `tie_tolerance` 无关。

Grand-Canonical 入场条件保持为严格大于阈值：

\[
I_i(t)=\mathbf 1\{B_i(t)>r\}.
\]

活跃代理执行所选策略在当前公共信息下的动作，未入场代理订单为零：

\[
a_i(t)=I_i(t)A_{R_i^*(t),\mu_t}.
\]

### 1.3 需求、市场信号与公共信息

本轮清算前净超额需求为

\[
D_t=\sum_i a_i(t)=n_{\rm buy}(t)-n_{\rm sell}(t).
\]

市场信号先由需求冲击相对参照阈值的偏离决定：

\[
q_t=\frac{D_t}{\lambda},\qquad z_t=q_t-L.
\]

设 `tie_tolerance = eps`。在非零偏离区域：

| 条件 | `score_signal` \(\chi_t\) | `new_bit` \(b_t\) |
|---|---:|---:|
| \(z_t>\mathrm{eps}\) | `+1` | `1` |
| \(z_t<-\mathrm{eps}\) | `-1` | `0` |

当 \(|z_t|\leq\mathrm{eps}\) 时：

- 新公共信息位始终随机取 `0` 或 `1`；
- `zero_score_tie="zero"` 时，\(\chi_t=0\)；
- `zero_score_tie="random"` 时，\(\chi_t\) 独立随机取 `+1` 或 `-1`。

因此，`zero_score_tie` 只控制评分信号，不决定公共信息位，也不参与最高评分并列的处理。

本轮末尾公共信息按

\[
\mu_{t+1}=(2\mu_t+b_t)\bmod P
\]

更新，即丢弃最旧位，并把最新位追加到 `m` 位历史的最低二进制位。

## 2. 随机优先撮合

### 2.1 买卖两侧人数

`random_priority_matching()` 只读取本轮实际订单数组 `agent_orders`，其中元素为 `-1`、`0` 或 `+1`。定义

\[
M_t=\max(n_{\rm buy},n_{\rm sell}),\qquad
m_t=\min(n_{\rm buy},n_{\rm sell}).
\]

为避免与公共信息记忆长度 `m` 混淆，代码主循环将后者记为 `m_small`。

多数方向 `majority_direction` 为：

- `+1`：买方人数更多；
- `-1`：卖方人数更多；
- `0`：买卖人数相等，包括本轮无人交易。

### 2.2 非平衡订单流

若买卖双方人数不等，程序从多数侧的 \(M_t\) 个活跃代理中，均匀随机、无放回地抽取恰好 \(m_t\) 个，形成多数侧优先撮合一组。

评分分组为：

- 少数侧全部 \(m_t\) 个代理；
- 多数侧被抽中的 \(m_t\) 个代理。

这两部分合计 \(2m_t\) 个代理，代表可以内部交叉撮合的双方，使用 **Majority-Game** 评分符号。

多数侧剩余

\[
M_t-m_t=|D_t|
\]

个代理代表由做市商吸收的残余订单，使用 **Minority-Game** 评分符号。

代码只随机抽取多数侧代理，不显式保存少数侧与多数侧之间的一一配对名单；评分更新只需要知道各代理所属分组。

### 2.3 买卖平衡和无人交易

若 \(n_{\rm buy}=n_{\rm sell}\)，本轮没有严格多数侧，也不抽取优先组：

```text
priority_count = 0
majority_direction = 0
```

所有活跃代理都视为可以内部撮合，并使用 Majority-Game 评分符号。若本轮无人交易，则没有代理进入该活跃分组。

每轮的优先组独立随机抽取。程序不保留上一轮撮合关系，也不模拟历史持仓或往复交易。

## 3. 个体评分更新

### 3.1 评分符号

令 \(\rho_i(t)\) 表示代理本轮的评分身份：

\[
\rho_i(t)=
\begin{cases}
+1,&\text{内部撮合双方，使用 Majority-Game 符号},\\
-1,&\text{残余多数侧或未入场代理，使用 Minority-Game 符号}.
\end{cases}
\]

对代理 \(i\) 持有的每一个策略 \(R\)，更新为

\[
\boxed{
S_{iR}(t+1)=\alpha S_{iR}(t)
+\rho_i(t)A_{R,\mu_t}\chi_t,
}
\qquad
\alpha=1-\frac1T.
\]

代码对应：

```python
scores = (
    alpha * scores
    + role_sign[:, None] * held_actions_now * score_signal
)
```

其中：

- `scores.shape = (N, s)`；
- `held_actions_now[i, j]` 是代理 `i` 的第 `j` 个持有策略在当前 \(\mu_t\) 下的动作；
- `role_sign[i]` 就是 \(\rho_i(t)\)。

例如 \(\chi_t=+1\) 时，内部撮合代理的买入动作加分、卖出动作减分；残余多数侧和未入场代理恰好相反。

### 3.2 未选策略和未入场代理

程序不为每个未选策略重新构造反事实订单簿。代理本轮由实际所选策略决定的撮合身份，会用于更新其全部持有策略的虚拟评分。

未入场代理没有实际订单，不能进入优先撮合组，也不计入残余多数侧人数。但其评分仍按 Minority-Game 符号更新，使未入场策略可以继续积累虚拟评分，代理以后仍能重新入场。

当 \(\chi_t=0\) 时，所有评分只做衰减；撮合分组仍会计算，但不改变本轮评分增量项。

### 3.3 输出人数的定义

结果中有两类容易混淆的计数：

- `priority_count`：只统计多数侧被随机抽中的代理数；
- `majority_game_agents`：统计使用 Majority-Game 评分符号的全部活跃代理，包括内部撮合双方。

非平衡时：

\[
\texttt{priority\_count}=m_t,
\qquad
\texttt{majority\_game\_agents}=2m_t,
\]

\[
\texttt{minority\_game\_agents}=M_t-m_t=|D_t|.
\]

买卖平衡时：

\[
\texttt{priority\_count}=0,
\qquad
\texttt{majority\_game\_agents}=\texttt{active\_agents},
\qquad
\texttt{minority\_game\_agents}=0.
\]

`minority_game_agents` 只统计活跃残余多数侧，不包括虽未入场但接受 Minority-Game 虚拟评分的代理。

## 4. 单轮代码执行顺序

`simulate_gcomg()` 的主循环按以下顺序推进：

1. 从 `(N, s)` 个体评分中选择每个代理的最优持有策略；
2. 用 `best_score > r` 判断入场，生成 `agent_orders`；
3. 计算 \(D_t\)、买卖人数和 \(\chi_t\)，生成新的公共信息位；
4. 调用 `random_priority_matching()` 随机抽取多数侧优先组；
5. 构造 Majority-Game 分组和每个代理的 `role_sign`；
6. 更新每个代理的全部持有策略评分；
7. 按原 GCMG 规则清算价格；
8. 更新公共信息并保存本轮计数、价格、收益率和信号。

随机优先撮合发生在评分更新之前。撮合不改变本轮订单、需求、价格或公共信息位，只改变评分增量符号；后续轮次再通过评分影响和策略选择影响市场轨迹。

## 5. FSS 实现与评分存储

### 5.1 显式和隐式策略空间

| 范围 | 策略空间 | 代理持有策略抽样 |
|---|---|---|
| `m <= 4` | 显式生成完整 `(2**P, P)` 动作矩阵 | 每个代理调用 `rng.choice(..., replace=False)` |
| `m >= 5` | 不生成完整矩阵，只保存实际持有策略 | 直接生成独立公平比特串，代理内拒绝重复 |

隐式路径中，每条长度为 `P` 的比特串以相同概率 \(2^{-P}\) 出现，等价于从完整 FSS 均匀抽取。不同代理独立抽样，允许持有相同策略。

程序对所有实际持有策略全局去重，共享动作映射：

- `candidate_actions`：去重后的策略动作矩阵；
- `holdings_positions`：代理持有策略在候选矩阵中的行号；
- `final_holdings`：每个代理持有策略的完整 FSS ID。

全局去重只共享策略动作，不共享评分。GCOMG 的评分始终是 `(N, s)` 的个体数组。

### 5.2 策略 ID

策略 ID 与动作向量按小端位序对应。令 \(b_{R,\mu}\in\{0,1\}\)，则

\[
R=\sum_{\mu=0}^{P-1}b_{R,\mu}2^\mu,
\qquad
A_{R,\mu}=2b_{R,\mu}-1.
\]

隐式路径在 `P <= 64` 时使用 `uint64` 保存 ID，更宽时使用 Python 任意精度整数。动作存储规模为 \(O(NsP)\)，个体评分存储为 \(O(Ns)\)，不需要为完整 FSS 分配评分向量。

### 5.3 最终结果

`GCOMGResult` 中：

```text
final_scores.shape   = (N, s)
final_holdings.shape = (N, s)
```

`final_scores[i, j]` 与 `final_holdings[i, j]` 一一对应，分别表示代理 `i` 的第 `j` 个持有策略的最终评分和完整 FSS ID。

`sampled_strategy_actions` 保存去重后的实际持有策略动作矩阵。由于评分是个体化的，本程序不再像原 GCMG-FSS 那样重建或返回完整 FSS 的一维全局最终评分表。

## 6. 价格形成

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

下一步恢复线性规则。`price_mode="log"` 时，全程采用对数规则。

| 模式与步骤 | `price_change` | `log_return` | `simple_return` |
|---|---|---|---|
| 普通线性步骤 | \(h_t\) | \(\log(x_{t+1}/x_t)\) | \(h_t/x_t\) |
| 线性回退步骤 | 实际增量 \(x_{t+1}-x_t\) | 通常为 \(h_t\) | 通常为 \(\exp(h_t)-1\) |
| 显式 `log` 模式 | \(h_t\) | \(h_t\) | \(\exp(h_t)-1\) |

极端下溢时，线性回退会把价格设为最小正浮点数，并按实际新旧价格重算收益率。`result.price_fallback[t]` 记录回退是否发生；该数组不写入 CSV，命令行结束时报告总回退步数。

价格形成不读取随机撮合分组。价格仍只由净需求和市场深度决定。

## 7. 初始化与随机性

所有 `(agent, strategy)` 评分从同一个值开始：

\[
S_{iR}(0)=
\begin{cases}
\text{initial\_score},&\text{若明确指定},\\
r+\text{initial\_score\_offset},&\text{否则}.
\end{cases}
\]

默认 `r=4`、`initial_score_offset=10`，初始评分为 `14`，所有代理第一轮均可入场。

随机数生成器依次用于：

- 初始化代理持有策略；
- 未指定 `mu0` 时生成初始公共信息；
- 最优策略评分并列时随机选择；
- 零偏离时生成公共信息位，以及可选的随机评分信号；
- 每轮从多数侧随机无放回抽取优先组。

相同代码、参数和种子可复现相应轨迹。由于每轮撮合也消耗随机数，GCOMG 的随机数调用序列与原 GCMG 不同，不能用同一种子期待两者轨迹相同。

直接使用 `GCOMGConfig()` 时，`zero_score_tie` 默认为 `zero`；命令行入口的实际默认值为 `random`。实验记录中应显式保存该字段。

## 8. 代码结构

| 类或函数 | 作用 |
|---|---|
| `FSSStrategySpace`、`build_fss()` | 描述或显式构造完整 FSS |
| `_sample_fss_holdings()` | 隐式均匀抽样、代理内去重和全局动作映射去重 |
| `GCOMGConfig` | 保存模型参数、解析市场深度并校验配置 |
| `_pick_best_strategies()` | 从个体评分中选择最优持有策略并随机处理并列 |
| `random_priority_matching()` | 识别多数侧并随机抽取优先撮合组 |
| `simulate_gcomg()` | 执行入场、订单、撮合评分、价格和公共信息更新 |
| `MarketMaker.clear()` | 清算净需求并处理线性价格边界 |
| `GCOMGResult` | 保存市场时间序列、撮合计数和个体最终评分 |
| `autocorrelation_function()` | 用 FFT 计算归一化自相关 |
| `summarize_result()` | 计算 burn-in 后的市场和撮合统计 |
| `save_csv()` | 保存逐轮时间序列 |
| `save_diagnostics_plot()` | 绘制价格、活跃度、优先组、收益分布和自相关 |
| `save_price_change_distribution_plot()` | 绘制实际价格增量分布与正态 Q–Q 图 |
| `_parse_args()`、`main()` | 命令行入口和文件输出 |

设去重后的实际策略数为 `K <= N*s`。每轮主要计算量为 \(O(Ns+N)\)，动作映射存储为 \(O(KP)\)，个体评分为 \(O(Ns)\)。显式 FSS 路径另需完整策略矩阵。

## 9. 安装与运行

依赖 Python、NumPy 和 Matplotlib：

```bash
python -m pip install numpy matplotlib
```

直接采用命令行默认参数：

```bash
python gcomg_fss.py
```

显式 FSS 示例：

```bash
python gcomg_fss.py \
  --N 101 --m 4 --s 2 --r 4 --T 100 --L 0 \
  --steps 50000 --burn-in 5000 --seed 123 \
  --zero-score-tie random --outdir run_gcomg_m4
```

隐式 FSS 示例：

```bash
python gcomg_fss.py \
  --N 101 --m 5 --s 2 --r 4 --T 100 --L 0 \
  --steps 50000 --burn-in 5000 --seed 123 \
  --zero-score-tie random --outdir run_gcomg_m5
```

查看全部选项：

```bash
python gcomg_fss.py --help
```

### 参数表

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--N` | `501` | 代理人数 |
| `--m` | `3` | 公共信息记忆长度，支持 1–10 |
| `--s` | `2` | 每个代理持有的不同策略数 |
| `--r` | `4.0` | 个体最佳策略评分的入场阈值，可为负数 |
| `--T` | `100.0` | 评分遗忘时间尺度 |
| `--L` | `0.0` | 市场参照阈值 |
| `--lambda` | `10*N` | 市场深度 |
| `--x0` | `100.0` | 初始价格 |
| `--price-mode` | `linear` | `linear` 或 `log` |
| `--steps` | `100000` | 模拟轮数 |
| `--seed` | `123` | 随机种子 |
| `--mu0` | 随机 | 初始公共信息状态 |
| `--initial-score-offset` | `10.0` | 相对 `r` 的初始评分偏移 |
| `--initial-score` | 未设置 | 所有个体策略评分的绝对初值，覆盖偏移规则 |
| `--zero-score-tie` | `random` | 命令行零偏离评分规则 |
| `--burn-in` | `10000` | 诊断舍弃的初始轮数 |
| `--outdir` | `gcomg_output` | 输出目录 |

主要约束包括：`N >= 1`、`1 <= m <= 10`、`s >= 1` 且不超过完整策略数、`T >= 1`、`steps >= 1`、`x0 > 0`、`lambda > 0`。汇总统计要求 burn-in 后至少保留 4 个观测；命令行完整运行还会计算滞后 100 的自相关，因此至少应保留 101 个 burn-in 后观测。

## 10. 输出文件

### 10.1 `gcomg_timeseries.csv`

每行对应一轮，包含全部模拟轮次，不应用 burn-in。

| 字段 | 含义 |
|---|---|
| `t` | 从 0 开始的轮次 |
| `mu_t` | 本轮决策使用的公共信息状态 |
| `new_bit` | 本轮末尾追加的信息位 |
| `demand` | 净超额需求 \(D_t\) |
| `buy_orders`、`sell_orders` | 实际买入和卖出人数 |
| `active_agents` | 活跃代理数 |
| `majority_count`、`minority_count` | 活跃多数侧和少数侧人数 \(M_t,m_t\) |
| `priority_count` | 多数侧被随机抽中的优先代理数 |
| `majority_game_agents` | 使用 Majority-Game 符号的活跃代理数 |
| `minority_game_agents` | 使用 Minority-Game 符号的活跃残余代理数 |
| `majority_direction` | 买方多数 `+1`、卖方多数 `-1`、平衡 `0` |
| `price` | 本轮清算后的 \(x_{t+1}\) |
| `price_change` | 按价格模式保存冲击或实际增量 |
| `simple_return`、`log_return` | 简单和对数收益率 |
| `score_signal` | 本轮市场信号 \(\chi_t\) |

CSV 不保存每个代理的评分、持有策略或撮合身份，也不包含价格回退布尔数组；这些数据需从 Python 结果对象获取。

### 10.2 `gcomg_summary.json`

文件包含：

- `parameters`：完整 `GCOMGConfig`；
- `lambda_used`：实际市场深度；
- `diagnostics`：burn-in 后的统计量。

除需求、活跃度、对数收益率及其自相关外，还保存：

- `mean_majority_side_agents`；
- `mean_minority_side_agents`；
- `mean_priority_matched_agents`；
- `mean_majority_game_share_of_active`。

最后一项先在每个活跃轮次计算

\[
\frac{\texttt{majority\_game\_agents}[t]}
{\texttt{active\_agents}[t]},
\]

无人交易轮次记为 `0`，再对时间取平均。它不是两个均值直接相除。

收益率绝对值自相关使用 \(|r_t-\bar r|\)，不是原始 \(|r_t|\)；`acf_volume_*` 使用活跃代理数。自相关协方差估计统一除以样本数，再除以零滞后协方差；零方差序列可能产生 `NaN`。

### 10.3 图形输出

`gcomg_diagnostics.png` 包含：

1. 从模拟起点开始的前至多 3000 步价格；
2. 同时段的活跃代理数和多数侧优先撮合人数；
3. burn-in 后标准化对数收益率分布，纵轴为对数刻度；
4. burn-in 后滞后 1 至 100 的收益率、去均值收益率绝对值和活跃度自相关。

价格和活跃度子图不舍弃 burn-in；分布和自相关子图使用 burn-in 后数据。

`gcomg_price_change_distribution.png` 使用 burn-in 后实际价格增量

\[
\Delta x_t=x_{t+1}-x_t,
\]

绘制直方图和正态 Q–Q 图。它不直接使用含义随价格模式变化的 `price_change` 字段。

## 11. 测试

`test_gcomg_fss.py` 覆盖：

- 买方多数和卖方多数时的优先组人数及方向；
- 买卖平衡时不存在严格优先组；
- 显式 FSS 下的个体评分形状和种子可复现性；
- 非平衡与平衡情况下的评分身份人数恒等式；
- 隐式 FSS 的动作矩阵列数和个体评分形状。

运行：

```bash
python -m unittest -v test_gcomg_fss.py
```

这些测试主要检查撮合分组和数据结构，不等同于对价格边界、长期统计性质或模型经济含义的完整验证。

## 12. 当前实现边界

本版本把随机优先撮合用作策略评分符号的分配机制，不模拟：

- 历史持仓、现金账户和真实往复交易；
- 旧报价与新报价之间的逐笔成交价格；
- 多档限价订单簿、撤单和不同订单类型；
- 做市商库存及动态买卖价差。

因此，内部撮合和残余订单只改变评分规则；市场价格仍由全体活跃代理的净需求 \(D_t/\lambda\) 决定。
