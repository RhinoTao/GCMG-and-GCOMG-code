# GCMG-FSS 多代理金融市场模拟器

`gcmg_fss.py` 实现了基于公共信息的 Grand Canonical Minority Game（GCMG，巨正则少数派博弈），策略集合为 Full Strategy Space（FSS，完整策略空间）。程序支持 `1 <= m <= 10`，包含策略抽样、代理与市场动力学、最终评分计算、时间序列保存和统计诊断。

代码按 `m` 采用两种 FSS 表示：

- `m <= 4`：显式构造完整策略矩阵；
- `m >= 5`：不枚举完整策略池，只生成并保存代理实际持有的策略。

两种表示共用同一套策略选择、入场判断、评分信号、订单清算和公共信息更新规则。它们的区别主要在初始化、评分存储和最终结果的组织方式。

## 1. FSS 的定义与程序表示

### 1.1 策略和策略 ID

记忆长度为 `m` 时，公共信息状态数为

\[
P=2^m.
\]

一个确定性策略要对每个公共信息状态给出 `-1` 或 `+1`，所以完整策略池包含

\[
|\mathrm{FSS}|=2^P=2^{2^m}
\]

个策略。程序用长度为 `P` 的动作向量表示策略：

\[
A_R=(A_{R,0},\ldots,A_{R,P-1}),\qquad A_{R,\mu}\in\{-1,+1\}.
\]

策略 ID 与动作向量采用固定的一一对应关系。令 \(b_{R,\mu}\in\{0,1\}\)，则

\[
R=\sum_{\mu=0}^{P-1}b_{R,\mu}2^\mu,
\qquad
A_{R,\mu}=2b_{R,\mu}-1.
\]

也就是说，公共信息状态 `mu` 对应策略 ID 的第 `mu` 个二进制位，最低位对应 `mu=0`；位值 `0` 映射为卖出 `-1`，位值 `1` 映射为买入 `+1`。显式和隐式路径都使用这套编码。

`FSSStrategySpace` 保存 `m` 和可选的 `actions`：

- `P` 返回 \(2^m\)；
- `n_strategies` 返回 \(2^P\)，使用 Python 整数，因此 `m=10` 时也能精确表示 \(2^{1024}\)；
- `actions` 是完整动作矩阵，或隐式路径下的 `None`；
- `is_materialized` 表示完整矩阵是否已经生成。

### 1.2 `m <= 4`：显式 FSS

`build_fss(m)` 生成形状为

```text
(2**P, P)
```

的 `int8` 矩阵。第 `R` 行就是策略 ID `R` 对应的动作向量。

初始化时，每个代理通过

```python
rng.choice(n_total_strategies, size=s, replace=False)
```

独立抽取 `s` 个策略。同一代理内无重复，不同代理可以持有相同策略。模拟器再从完整矩阵中取出至少被一个代理持有的策略，形成实际参与在线计算的候选集合。

### 1.3 `m >= 5`：隐式 FSS

完整 FSS 无法直接存储。例如 `m=10` 时，`P=1024`，策略总数为 \(2^{1024}\)。但模拟只需要代理实际持有的策略，不需要先枚举其余策略。

`_sample_fss_holdings()` 直接为每条待抽策略生成 `P` 个独立公平比特。每个 `P` 位向量出现的概率均为 \(2^{-P}\)，因此单条策略等价于从完整 FSS 中均匀抽取。若同一代理抽到自己已经持有的向量，就拒绝并重抽；这实现了每个代理内部的均匀无放回抽样。

不同代理分别抽样，彼此之间不要求策略不同。随后程序对所有 `N*s` 条持有记录做全局去重：

- `unique_actions`：去重后的候选动作矩阵；
- `holdings_positions`：每个代理的持策略在候选矩阵中的行号；
- `unique_strategy_ids`：候选策略的完整 FSS ID；
- `holdings_strategy_ids`：每个代理实际持有策略的完整 FSS ID。

全局去重只是让共享策略复用同一份动作向量和评分，不改变各代理的持有关系。

策略 ID 在 `P <= 64` 时保存为 `uint64`；当 `P > 64` 时，改用保存 Python 任意精度整数的 `object` 数组。因此 `m>=7` 时不会发生 ID 截断。

初始化的主要存储规模为 \(O(NsP)\)，而不是完整 FSS 矩阵的 \(O(2^P P)\)。

## 2. 模拟状态与初始化

无论 FSS 是否显式生成，市场循环都使用以下状态：

| 变量 | 含义 |
|---|---|
| `candidate_actions` | 至少被一个代理持有的策略动作矩阵，共 `K` 行 |
| `candidate_scores` | 这 `K` 个候选策略的全局共享评分 |
| `holdings_positions` | `(N, s)`，代理持有策略在候选矩阵中的行号 |
| `mu` | 当前公共信息状态 |
| `market_maker.price` | 当前价格 \(x_t\) |

`K <= N*s`。不同代理共享同一策略时，该策略只占一个候选行，也只有一份评分。评分不是代理私有状态；所有代理读取同一张候选评分表，但只在自己持有的策略中选择。

持有策略在初始化后固定，不再重新抽样。未持有策略不会影响本次模拟的代理决策，因此在线循环只需更新候选策略评分。

初始评分为

\[
S_R(0)=
\begin{cases}
\text{initial\_score},&\text{若已指定},\\
r+\text{initial\_score\_offset},&\text{否则}.
\end{cases}
\]

默认 `r=4`、`initial_score_offset=10`，所以初始评分为 `14`，所有代理最初都满足入场条件。设置 `initial_score` 后，偏移量不再参与计算。

`mu0` 未指定时，程序从 `[0, P)` 中随机抽取初始公共信息。所有策略初始评分相同，首轮选择会在各代理持有的并列策略中随机进行。

## 3. 单轮 GCMG 动力学

第 `t` 轮开始时，已知候选评分、固定持有关系、公共信息 \(\mu_t\) 和价格 \(x_t\)。每轮按以下顺序执行。

### 3.1 选择各代理的最优持有策略

代理 \(i\) 的策略集合记为 \(\mathcal R_i\)：

\[
R_i^*(t)\in\arg\max_{R\in\mathcal R_i}S_R(t),
\qquad
B_i(t)=\max_{R\in\mathcal R_i}S_R(t).
\]

`_pick_best_strategies()` 对 `(N, s)` 的持有评分矩阵做向量化处理。最高评分并列时，只为并列策略生成有效随机数并选择最大者，等价于在并列项中随机选取。这里采用精确浮点相等判断，不使用 `tie_tolerance`。

### 3.2 入场判断与订单生成

代理只有在其最佳持有策略的评分严格大于 `r` 时才交易：

\[
I_i(t)=\mathbf 1\{B_i(t)>r\}.
\]

入场代理执行最优策略在当前公共信息下的动作，未入场代理订单为零：

\[
a_i(t)=I_i(t)A_{R_i^*(t),\mu_t}.
\]

`r` 可以为负数。负阈值降低入场门槛，但不改变严格大于的比较方式。

### 3.3 超额需求

\[
D_t=\sum_i a_i(t)=n_{\rm buy}(t)-n_{\rm sell}(t).
\]

每个活跃代理只提交一个 `+1` 或 `-1` 订单，因此

\[
n_{\rm buy}(t)+n_{\rm sell}(t)=\texttt{active\_agents}[t].
\]

### 3.4 评分信号与新公共信息位

计算需求冲击及其相对市场参照阈值的偏离：

\[
q_t=\frac{D_t}{\lambda},\qquad z_t=q_t-L.
\]

设 `tie_tolerance = eps`。在非零偏离区域：

| 条件 | `score_signal` \(g_t\) | `new_bit` \(b_t\) |
|---|---:|---:|
| \(z_t>\mathrm{eps}\) | `+1` | `1` |
| \(z_t<-\mathrm{eps}\) | `-1` | `0` |

当 \(|z_t|\leq\mathrm{eps}\) 时，`new_bit` 始终随机取 `0` 或 `1`，评分信号则由 `zero_score_tie` 控制：

- `zero`：\(g_t=0\)；
- `random`：独立随机取 `+1` 或 `-1`。

因此，零偏离时的公共信息位与评分信号不必对应。`zero_score_tie` 处理的是 \(D/\lambda-L\) 的零偏离，不是最优策略评分并列；两者是代码中不同的随机分支。

### 3.5 在线更新候选策略评分

评分方程为

\[
S_R(t+1)=\alpha S_R(t)-A_{R,\mu_t}g_t,
\qquad
\alpha=1-\frac1T.
\]

在线循环对所有候选策略执行该更新，包括本轮没有被任何代理选中的候选策略。更新只使用本轮开始时的 \(\mu_t\)，不使用刚生成的 `new_bit`。

该符号约定使与市场信号相反的动作加分。例如 \(g_t=+1\) 时，当前状态动作是 `-1` 的策略加分，动作是 `+1` 的策略减分；旧评分同时乘以 \(\alpha\)。

`T` 控制遗忘时间尺度，代码要求 `T >= 1`，所以 \(0\leq\alpha<1\)。

### 3.6 价格清算

定义

\[
h_t=\frac{D_t}{\lambda}.
\]

`price_mode="linear"` 时，只要候选价格为正，就使用

\[
x_{t+1}=x_t+h_t.
\]

若 \(x_t+h_t\leq0\)，仅当前步骤改用对数规则：

\[
x_{t+1}=x_t\exp(h_t).
\]

下一轮仍先尝试线性规则。`price_mode="log"` 时，每一步都使用对数规则，不触发这种线性边界回退。

各情况下的输出字段为：

| 模式与步骤 | `price_change` | `log_return` | `simple_return` |
|---|---|---|---|
| 普通线性步骤 | \(h_t\) | \(\log(x_{t+1}/x_t)\) | \(h_t/x_t\) |
| 线性越界回退步骤 | 实际增量 \(x_{t+1}-x_t\) | 通常为 \(h_t\) | 通常为 \(\exp(h_t)-1\) |
| 显式 `log` 模式 | \(h_t\) | \(h_t\) | \(\exp(h_t)-1\) |

`price_change` 的含义因此不完全统一：普通线性步骤和显式 `log` 模式保存 \(D/\lambda\)，线性回退步骤保存实际价格增量。实际价格增量应统一由 `np.diff(result.price)` 计算。

线性回退中，如果极大的负冲击使指数结果下溢到零，代码会把价格设为最小正浮点数，并根据实际新旧价格重新计算两种收益率。此时 `log_return` 不再等于原始的 \(h_t\)。

`result.price_fallback[t]` 记录第 `t` 步是否发生线性边界回退。该标记不写入 CSV；命令行程序在结束时报告回退步数。

### 3.7 更新公共信息并保存本轮结果

\[
\mu_{t+1}=(2\mu_t+b_t)\bmod P.
\]

该操作将历史左移一位、丢弃最旧位，并把最新位放在最低二进制位。

本轮记录中的 `mu_t` 是决策所用状态，`new_bit` 是本轮末尾追加的位，`price` 是清算后的 \(x_{t+1}\)。评分更新虽然写在价格清算之前，但二者都只依赖本轮已经确定的订单和信号。

## 4. 最终评分的两种计算方式

### 4.1 在线循环只维护实际持有策略

代理持有关系固定，未被任何代理持有的策略不会影响策略选择、订单或价格。因此，两种 FSS 路径在循环中都只更新 `K` 个候选策略，而不是逐步更新完整 FSS 的 \(2^P\) 个评分。

对于 `m >= 5`，模拟结束后直接返回这些候选策略的最终评分：

```text
final_scores[j]
final_score_strategy_ids[j]
sampled_strategy_actions[j, :]
```

三者在隐式路径下按同一个候选行号 `j` 对应，分别表示最终评分、完整 FSS 策略 ID 和动作向量。`final_holdings[i, :]` 则保存代理 `i` 持有的完整 FSS 策略 ID，不是候选矩阵行号。

### 4.2 `m <= 4` 在模拟结束后重建完整评分

显式路径需要返回完整 FSS 所有策略的最终评分，包括从未被持有的策略。未持有策略的评分也会受到历史信号影响，不能只对初始评分做衰减。

代码利用评分方程的闭式解。经过 `n` 轮后：

\[
S_R(n)=\alpha^nS_R(0)
-\sum_{t=0}^{n-1}\alpha^{n-1-t}A_{R,\mu_t}g_t.
\]

按公共信息状态聚合加权信号：

\[
C_\mu=\sum_{t:\mu_t=\mu}\alpha^{n-1-t}g_t,
\]

即可写成

\[
S_R(n)=\alpha^nS_R(0)-\sum_{\mu=0}^{P-1}A_{R,\mu}C_\mu.
\]

`_reconstruct_all_strategy_scores()` 先按 `global_info` 分组计算 `C`，再用完整动作矩阵与 `C` 做一次矩阵乘法，得到所有策略的最终评分。这与逐步评分方程一致，但避免在市场循环中更新完整 FSS。

显式路径下，`final_scores[R]` 对应完整 FSS 的策略 ID `R`，所以 `final_score_strategy_ids=None`。`final_holdings` 仍保存各代理持有的完整 FSS ID。

## 5. 零偏离与无人参与状态

当 `L=0`、`zero_score_tie="zero"` 且没有代理参与时，\(D_t=0\)、\(g_t=0\)，评分只按 \(\alpha\) 衰减，市场可能持续处于无人参与状态。公共信息位仍会随机更新。

主程序检查最后 10% 的轮次；如果活跃代理数始终为零，会输出警告。`random` 模式在零偏离时仍产生非零评分信号，可用于对照这种零值约定。

配置对象和命令行的默认约定不同：

- 直接使用 `GCMGConfig()` 时，`zero_score_tie="zero"`；
- 通过命令行运行时，`_parse_args()` 的实际默认值是 `random`。

进行可复现对照实验时，建议显式设置该参数，不要依赖入口默认值。

## 6. 代码结构

| 类、常量或函数 | 作用 |
|---|---|
| `MAX_FSS_M`、`DENSE_FSS_MAX_M` | 分别限定最大 `m=10` 和显式 FSS 的最大 `m=4` |
| `FSSStrategySpace` | 描述完整 FSS；完整动作矩阵只在显式路径存在 |
| `build_fss()` | 构造显式 FSS，或返回隐式 FSS 描述符 |
| `_bits_to_python_ids()` | 按小端位序把动作比特编码为精确策略 ID |
| `_sample_fss_holdings()` | 隐式均匀抽样、代理内去重和全局候选策略去重 |
| `GCMGConfig` | 保存模型参数、解析市场深度并检查配置 |
| `MarketMaker.clear()` | 清算需求、处理线性边界回退并计算收益率 |
| `_pick_best_strategies()` | 向量化选择最高评分的持有策略并随机处理并列 |
| `_reconstruct_all_strategy_scores()` | 在显式路径末尾重建完整 FSS 最终评分 |
| `simulate_gcmg()` | 初始化持有策略并执行全部市场轮次 |
| `GCMGResult` | 保存时间序列、最终评分、策略 ID 和价格回退标记 |
| `autocorrelation_function()` | 用 FFT 计算归一化自相关函数 |
| `lag_autocorrelation()` | 提取指定滞后的自相关系数 |
| `summarize_result()` | 计算 burn-in 后的矩和自相关诊断 |
| `save_csv()` | 保存逐轮时间序列 |
| `save_diagnostics_plot()` | 绘制价格、活跃度、收益率分布和自相关图 |
| `save_price_change_distribution_plot()` | 绘制实际价格增量分布和正态 Q–Q 图 |
| `_parse_args()`、`main()` | 命令行入口、文件输出和终端摘要 |

设 `K` 为全局去重后的实际持有策略数。市场循环每轮的主要计算量为 \(O(Ns+K)\)。隐式 FSS 初始化的存储规模为 \(O(NsP)\)；结果时间序列另需 \(O(\mathrm{steps})\) 内存。显式路径还需要完整 FSS 矩阵，并在末尾执行完整评分重建。

## 7. 安装与运行

依赖 Python、NumPy 和 Matplotlib：

```bash
python -m pip install numpy matplotlib
```

采用命令行默认参数运行：

```bash
python gcmg_fss.py
```

运行 `m=4` 的显式 FSS：

```bash
python gcmg_fss.py \
  --N 101 --m 4 --s 2 --r 4 --T 100 --L 0 \
  --steps 50000 --burn-in 5000 --seed 123 \
  --zero-score-tie random --outdir run_fss_m4
```

运行 `m=10` 的隐式 FSS：

```bash
python gcmg_fss.py \
  --N 101 --m 10 --s 2 --r 4 --T 100 --L 0 \
  --steps 50000 --burn-in 5000 --seed 123 \
  --zero-score-tie random --outdir run_fss_m10
```

负入场阈值和全程对数价格示例：

```bash
python gcmg_fss.py \
  --r -20 --price-mode log --steps 50000 \
  --burn-in 5000 --zero-score-tie zero \
  --outdir run_negative_r_log
```

查看全部选项：

```bash
python gcmg_fss.py --help
```

在相同代码和运行环境中，相同参数与随机种子可复现相应抽样路径和模拟轨迹。显式与隐式路径使用不同的抽样实现，不应把不同路径下的随机数消耗顺序视为相同。

## 8. 参数表

| 配置字段 | 命令行参数 | 默认值 | 含义 |
|---|---|---:|---|
| `N` | `--N` | `501` | 代理人数 |
| `m` | `--m` | `3` | 公共信息记忆长度，支持 1–10 |
| `s` | `--s` | `2` | 每个代理持有的不同策略数 |
| `r` | `--r` | `4.0` | 最佳持有策略评分的入场阈值，可为负数 |
| `T` | `--T` | `100.0` | 评分遗忘时间尺度 |
| `L` | `--L` | `0.0` | 市场参照阈值 |
| `lam` | `--lambda` | `None` | 市场深度；未设置时取 `10*N` |
| `x0` | `--x0` | `100.0` | 初始价格 |
| `price_mode` | `--price-mode` | `linear` | `linear` 或 `log` |
| `steps` | `--steps` | `100000` | 模拟轮数 |
| `seed` | `--seed` | `123` | 随机种子 |
| `mu0` | `--mu0` | `None` | 初始公共信息状态；未设置时随机抽取 |
| `initial_score_offset` | `--initial-score-offset` | `10.0` | 相对 `r` 的初始评分偏移 |
| `initial_score` | `--initial-score` | `None` | 绝对初始评分，设置后覆盖偏移规则 |
| `zero_score_tie` | `--zero-score-tie` | 配置对象 `zero`；命令行 `random` | 零偏离时的评分信号 |
| `tie_tolerance` | 无 | `1e-12` | 判断 \(D/\lambda-L\) 是否为零的容差 |
| 无 | `--burn-in` | `10000` | 诊断时舍弃的初始轮数 |
| 无 | `--outdir` | `gcmg_output` | 输出目录 |

主要约束包括：`N >= 1`、`1 <= m <= 10`、`s >= 1` 且不超过完整策略总数、`T >= 1`、`steps >= 1`、`x0 > 0`、`lambda > 0`、`mu0 ∈ [0, P)`。`burn_in` 必须为非负数；汇总统计要求其后至少保留 4 个观测值，诊断图还需要计算滞后 100 的自相关，因此命令行完整运行至少应保留 101 个 burn-in 后观测值。

## 9. 输出文件与诊断口径

### 9.1 `gcmg_timeseries.csv`

每行对应一轮，保存全部模拟轮次，不应用 `burn_in`。

| 字段 | 含义 |
|---|---|
| `t` | 从 0 开始的轮次 |
| `mu_t` | 本轮决策使用的公共信息状态 |
| `new_bit` | 本轮结束时追加的信息位 |
| `demand` | 超额需求 \(D_t\) |
| `buy_orders` | 买单数 |
| `sell_orders` | 卖单数 |
| `active_agents` | 本轮入场代理数 |
| `price` | 本轮清算后的价格 \(x_{t+1}\) |
| `price_change` | 按第 3.6 节所述模式保存冲击或实际价格增量 |
| `simple_return` | 简单收益率 |
| `log_return` | 对数收益率 |
| `score_signal` | 本轮评分信号 \(g_t\) |

CSV 不包含策略 ID、最终评分或价格回退标记；这些数据通过 `GCMGResult` 获取。

### 9.2 `gcmg_summary.json`

文件分为三部分：

- `parameters`：`GCMGConfig` 的完整配置；
- `lambda_used`：实际采用的市场深度；
- `diagnostics`：舍弃 `burn_in` 后的统计量。

诊断包括：

- 需求的均值、标准差和超额峰度；
- 活跃代理数的均值和标准差；
- 对数收益率的均值、标准差和超额峰度；
- 对数收益率的一阶自相关；
- 去均值对数收益率绝对值在滞后 1、2、5、10 处的自相关；
- 活跃代理数在滞后 1、2、5、10 处的自相关。

超额峰度按标准化四阶矩减 3 计算。`acf_abs_return_*` 使用 \(|r_t-\bar r|\)，不是原始 \(|r_t|\)；`acf_volume_*` 使用活跃代理数作为活跃度代理变量。

自相关通过 FFT 计算。序列先去均值，协方差估计统一除以样本数 `n`，再除以零滞后协方差。常数序列的零方差会使部分统计量或自相关系数成为 `NaN`。

### 9.3 `gcmg_diagnostics.png`

包含四幅图：

1. 从模拟起点开始的前至多 3000 步价格；
2. 从模拟起点开始的前至多 3000 步活跃代理数；
3. burn-in 后的标准化对数收益率分布，纵轴为对数刻度，并叠加标准正态密度；
4. burn-in 后滞后 1 至 100 的收益率、去均值收益率绝对值和活跃度自相关。

价格和活跃度子图没有舍弃 burn-in；分布和自相关子图使用 burn-in 后的数据。

### 9.4 `gcmg_price_change_distribution.png`

该图使用 burn-in 后的实际价格增量

\[
\Delta x_t=x_{t+1}-x_t,
\]

而不是直接使用 `price_change` 字段。左图是增量直方图，并叠加具有相同样本均值和标准差的正态密度；右图是正态 Q–Q 图，最多绘制 5000 个按顺序统计量抽取的点。

## 10. 作为 Python 模块使用

```python
from gcmg_fss import GCMGConfig, simulate_gcmg, summarize_result

config = GCMGConfig(
    N=501,
    m=10,
    s=2,
    r=4.0,
    T=100.0,
    steps=100_000,
    seed=123,
    zero_score_tie="random",
)
result = simulate_gcmg(config)
diagnostics = summarize_result(result, burn_in=10_000)
```

`result.price` 长度为 `steps+1`，其中 `price[0]` 是初始价格；其余市场时间序列和 `price_fallback` 的长度均为 `steps`。

读取策略相关结果时，应先区分路径：

- `m <= 4`：`final_scores` 按完整 FSS 策略 ID 排列，`final_holdings` 中的 ID 可直接作为完整评分和完整动作矩阵的行号；
- `m >= 5`：`final_scores` 只包含实际持有策略，应通过 `final_score_strategy_ids` 对照完整 FSS ID；`final_holdings` 保存的是完整 FSS ID，不能直接作为 `sampled_strategy_actions` 的行号。

这种结果结构保留了隐式抽样的可追溯性，同时避免为 \(2^P\) 个策略分配评分或动作数组。
