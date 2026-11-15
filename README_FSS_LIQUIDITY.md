# GCMG-FSS 模拟与流动性扫描

本项目包含一个 GCMG-FSS 基础模拟器，以及在其上实现的市场深度扫描和绘图程序。

| 文件 | 作用 |
|---|---|
| `gcmg_fss.py` | 单次 GCMG-FSS 模拟、时间序列输出和基础诊断 |
| `scan_liquidity_fss.py` | 扫描多组 \(\lambda/N\)，保存各种子统计量和代表性轨迹 |
| `plot_liquidity_fss.py` | 读取扫描 NPZ，绘制 2×2 流动性诊断图 |

阅读代码时可以分成两层：`gcmg_fss.py` 决定代理和市场动力学；`scan_liquidity_fss.py` 只改变市场深度、复用种子并组织实验数据。流动性扫描没有另写一套策略选择或评分模型。

## 1. 基础 GCMG-FSS 模型

### 1.1 公共信息和策略

记忆长度为 `m` 时，公共信息状态数为

\[
P=2^m,\qquad \mu_t\in\{0,\ldots,P-1\}.
\]

一个确定性策略对每个状态指定买入或卖出：

\[
A_R=(A_{R,0},\ldots,A_{R,P-1}),\qquad A_{R,\mu}\in\{-1,+1\}.
\]

完整策略空间 FSS 包含所有这样的动作向量，因此

\[
|\mathrm{FSS}|=2^P=2^{2^m}.
\]

每个代理初始化时从 FSS 中独立抽取 `s` 个不同策略，形成固定集合 \(\mathcal R_i\)。同一代理内无重复，不同代理可以持有相同策略；模拟过程中不重新抽取。

策略评分是全局共享状态。代理只在自己持有的策略中比较评分：

\[
R_i^*(t)\in\arg\max_{R\in\mathcal R_i}S_R(t),
\qquad
B_i(t)=\max_{R\in\mathcal R_i}S_R(t).
\]

最高评分并列时，程序在并列项中无偏向随机选择。这里使用精确浮点相等判断，与需求信号的 `tie_tolerance` 无关。

### 1.2 初始评分和公共信息

初始评分按以下规则设置：

\[
S_R(0)=
\begin{cases}
\text{initial\_score},&\text{若明确指定},\\
r+\text{initial\_score\_offset},&\text{否则}.
\end{cases}
\]

基础模拟器默认 `r=4`、`initial_score_offset=10`，所以初始评分为 `14`，所有代理最初都满足入场条件。

`mu0` 未指定时，程序根据当前随机种子抽取初始公共信息。由于所有策略初始评分相同，首轮策略选择从各代理的持有策略并列项中随机产生。

### 1.3 单轮执行顺序

第 `t` 轮开始时，已知策略评分、固定持有关系、公共信息 \(\mu_t\) 和价格 \(x_t\)。

**1. 选择策略并判断是否入场**

\[
I_i(t)=\mathbf 1\{B_i(t)>r\}.
\]

只有最佳持有策略的评分严格大于 `r` 时，代理才交易。`r` 可以为负数。入场代理执行所选策略的动作，未入场代理订单为零：

\[
a_i(t)=I_i(t)A_{R_i^*(t),\mu_t}.
\]

**2. 汇总清算前需求**

\[
D_t=\sum_i a_i(t)=n_{\rm buy}(t)-n_{\rm sell}(t).
\]

每个活跃代理提交一个 `+1` 或 `-1` 订单，因此买卖订单数之和等于 `active_agents`。

**3. 生成评分信号和公共信息位**

\[
q_t=\frac{D_t}{\lambda},\qquad z_t=q_t-L.
\]

设 `tie_tolerance = eps`。在非零偏离区域：

| 条件 | 评分信号 \(g_t\) | 新信息位 \(b_t\) |
|---|---:|---:|
| \(z_t>\mathrm{eps}\) | `+1` | `1` |
| \(z_t<-\mathrm{eps}\) | `-1` | `0` |

零偏离分支见第 1.4 节。

**4. 更新策略评分**

\[
S_R(t+1)=\alpha S_R(t)-A_{R,\mu_t}g_t,
\qquad
\alpha=1-\frac1T.
\]

在线循环更新所有至少被一个代理持有的策略，包括本轮未被选中的策略。`T` 控制评分遗忘时间尺度；代码要求 `T >= 1`，所以 \(0\leq\alpha<1\)。

**5. 清算价格**

默认线性规则为

\[
x_{t+1}=x_t+\frac{D_t}{\lambda}.
\]

价格边界回退和全程对数模式见第 3 节。价格不反馈到策略评分、订单或公共信息。

**6. 更新公共信息并记录结果**

\[
\mu_{t+1}=(2\mu_t+b_t)\bmod P.
\]

该操作丢弃最旧位，并将最新位放在 `m` 位历史的最低二进制位。结果中的 `global_info[t]` 是本轮决策所用状态，`appended_bit[t]` 是本轮末尾追加的位，`price[t+1]` 是本轮清算后的价格。

### 1.4 零偏离分支

当 \(|z_t|\leq\mathrm{eps}\) 时：

- 新公共信息位始终随机取 `0` 或 `1`；
- `zero_score_tie="zero"` 时，\(g_t=0\)；
- `zero_score_tie="random"` 时，\(g_t\) 独立随机取 `+1` 或 `-1`。

因此，`zero_score_tie` 只控制评分信号，不控制公共信息位，也不用于最优策略评分并列。

直接使用 `GCMGConfig()` 时，该字段默认是 `zero`；基础模拟器命令行入口和流动性扫描脚本的实际默认值均为 `random`。写实验配置时应显式记录该参数。

## 2. FSS 的显式与隐式实现

### 2.1 策略 ID 编码

策略 ID 与动作向量一一对应。令 \(b_{R,\mu}\in\{0,1\}\)，则

\[
R=\sum_{\mu=0}^{P-1}b_{R,\mu}2^\mu,
\qquad
A_{R,\mu}=2b_{R,\mu}-1.
\]

公共信息状态 `mu` 对应策略 ID 的第 `mu` 个二进制位；`0` 映射为 `-1`，`1` 映射为 `+1`。显式和隐式路径使用相同编码。

### 2.2 两条实现路径

| 范围 | 策略空间表示 | 代理抽样 |
|---|---|---|
| `m <= 4` | 显式生成 `(2**P, P)` 的完整 `int8` 动作矩阵 | 对每个代理调用 `rng.choice(..., replace=False)` |
| `m >= 5` | `FSSStrategySpace.actions=None`，不枚举完整策略池 | 直接生成独立公平比特串，代理内拒绝重复 |

隐式路径中，每个 `P` 位向量出现的概率均为 \(2^{-P}\)，所以单条策略等价于从完整 FSS 均匀抽取；代理内重复时拒绝重抽，实现均匀无放回抽样。它不是用任意指定的随机子集替代 FSS。

程序随后对所有 `N*s` 条持有记录全局去重：

- 每种实际持有策略只保存一份动作向量和一份评分；
- `holdings_positions` 保存代理到候选策略行的映射；
- 完整 FSS 策略 ID 另行保存，保证结果可追溯；
- 不同代理共享策略时，不改变各自的持有关系。

策略 ID 在 `P <= 64` 时使用 `uint64`，更宽时使用保存 Python 任意精度整数的 `object` 数组。`m=10` 时 \(P=1024\)，但动作存储只随实际持有策略数增长，初始化主要存储规模为 \(O(NsP)\)，而非 \(O(2^P P)\)。

### 2.3 在线评分与最终评分

持有关系固定，未被任何代理持有的策略不会影响本次市场轨迹。因此市场循环只更新 `K` 个去重后的候选策略，其中 `K <= N*s`。

对于 `m >= 5`，结果只返回这些候选策略的最终评分：

```text
final_scores[j]
final_score_strategy_ids[j]
sampled_strategy_actions[j, :]
```

三者按同一候选行号对应。`final_holdings[i, :]` 保存代理 `i` 的完整 FSS 策略 ID，不是候选矩阵行号。

对于 `m <= 4`，程序在模拟结束后重建完整 FSS 所有策略的最终评分。评分方程的闭式解为

\[
S_R(n)=\alpha^nS_R(0)
-\sum_{t=0}^{n-1}\alpha^{n-1-t}A_{R,\mu_t}g_t.
\]

令

\[
C_\mu=\sum_{t:\mu_t=\mu}\alpha^{n-1-t}g_t,
\]

则

\[
S_R(n)=\alpha^nS_R(0)-\sum_\mu A_{R,\mu}C_\mu.
\]

`_reconstruct_all_strategy_scores()` 先按公共信息状态聚合 \(C_\mu\)，再用完整动作矩阵做一次矩阵乘法。这样不必在每轮更新 \(2^P\) 个评分，同时仍能返回完整最终评分。此路径下 `final_scores[R]` 直接对应策略 ID `R`，`final_score_strategy_ids=None`。

## 3. 价格形成与边界处理

定义

\[
h_t=\frac{D_t}{\lambda}.
\]

`price_mode="linear"` 时，只要候选价格为正，就使用

\[
x_{t+1}=x_t+h_t.
\]

若 \(x_t+h_t\leq0\)，仅当前步骤改用

\[
x_{t+1}=x_t\exp(h_t),
\]

下一步重新尝试线性规则。`price_mode="log"` 时，每一步都使用对数规则。

各步骤的输出字段为：

| 模式与步骤 | `price_change` | `log_return` | `simple_return` |
|---|---|---|---|
| 普通线性步骤 | \(h_t\) | \(\log(x_{t+1}/x_t)\) | \(h_t/x_t\) |
| 线性越界回退步骤 | 实际增量 \(x_{t+1}-x_t\) | 通常为 \(h_t\) | 通常为 \(\exp(h_t)-1\) |
| 显式 `log` 模式 | \(h_t\) | \(h_t\) | \(\exp(h_t)-1\) |

线性回退若遇到极端指数下溢，基础模拟器将价格设为最小正浮点数，并根据实际新旧价格重算收益率。`GCMGResult.price_fallback[t]` 记录是否发生线性边界回退。

`price_change` 不是在所有模式下都等于实际价格增量。需要 \(\Delta x_t\) 时，应使用 `np.diff(result.price)`。

## 4. 基础模拟器的运行与输出

### 4.1 安装和单次运行

依赖 Python、NumPy 和 Matplotlib：

```bash
python -m pip install numpy matplotlib
```

运行默认单次模拟：

```bash
python gcmg_fss.py
```

显式 FSS 示例：

```bash
python gcmg_fss.py \
  --N 101 --m 4 --s 2 --r 4 --T 100 --L 0 \
  --steps 50000 --burn-in 5000 --seed 123 \
  --initial-score-offset 10 --zero-score-tie random \
  --outdir run_fss_m4
```

隐式 FSS 示例：

```bash
python gcmg_fss.py \
  --N 101 --m 10 --s 2 --r 4 --T 100 --L 0 \
  --steps 50000 --burn-in 5000 --seed 123 \
  --initial-score-offset 10 --zero-score-tie random \
  --outdir run_fss_m10
```

主要参数默认值如下：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--N` | `501` | 代理人数 |
| `--m` | `3` | 记忆长度，支持 1–10 |
| `--s` | `2` | 每个代理持有的策略数 |
| `--r` | `4.0` | 入场评分阈值，可为负数 |
| `--T` | `100.0` | 评分遗忘时间尺度 |
| `--L` | `0.0` | 市场参照阈值 |
| `--lambda` | `10*N` | 市场深度 |
| `--x0` | `100.0` | 初始价格 |
| `--price-mode` | `linear` | `linear` 或 `log` |
| `--steps` | `100000` | 模拟轮数 |
| `--seed` | `123` | 随机种子 |
| `--mu0` | 随机 | 初始公共信息状态 |
| `--initial-score-offset` | `10.0` | 相对 `r` 的初始评分偏移 |
| `--initial-score` | 未设置 | 绝对初始评分，覆盖偏移规则 |
| `--zero-score-tie` | `random` | 命令行零偏离评分规则 |
| `--burn-in` | `10000` | 基础诊断舍弃的初始轮数 |
| `--outdir` | `gcmg_output` | 输出目录 |

### 4.2 基础输出

| 文件 | 内容 |
|---|---|
| `gcmg_timeseries.csv` | 每轮公共信息、需求、买卖订单、活跃人数、清算后价格、价格字段和收益率 |
| `gcmg_diagnostics.png` | 初始价格和活跃度轨迹、burn-in 后标准化对数收益率分布及自相关 |
| `gcmg_price_change_distribution.png` | burn-in 后实际价格增量分布和正态 Q–Q 图 |
| `gcmg_summary.json` | 完整配置、实际市场深度和 burn-in 后统计量 |

基础诊断中的收益率自相关使用对数收益率；`|return|` 序列实际为去均值收益率的绝对值；`volume` 使用活跃代理数。自相关由 FFT 计算，协方差估计统一除以样本数，再除以零滞后协方差。零方差序列会产生 `NaN`。

## 5. 流动性扫描的实验设计

### 5.1 扫描变量和种子复用

扫描脚本定义无量纲市场深度

\[
\kappa=\frac{\lambda}{N},
\qquad
\lambda=\kappa N.
\]

命令行 `--lambda-over-n` 给出的是 \(\kappa\)，不是 \(\lambda\) 本身。默认值为

```text
10, 1, 0.1, 0.01
```

对每组 \(\kappa\)，程序按相同顺序使用完全相同的种子列表。每个 `(kappa, seed)` 组合独立调用基础模拟器，除 `lam` 外，其余模型参数保持一致。FSS 策略空间描述符只构造一次，并在全部运行中复用；代理持有策略仍由每次运行的种子生成。

这种设计用于区分两类变化：

- 相同种子、不同 \(\lambda\)：检查市场深度的直接缩放和数值边界；
- 不同种子、相同 \(\lambda\)：估计抽样和轨迹差异。

默认 4 组深度、10 个种子，共执行 40 次模拟。

### 5.2 默认 `L=0` 时的需求不变性

在通常数值范围内，当 \(L=0\) 且 \(\lambda>0\) 时，

\[
\operatorname{sgn}(D_t/\lambda)=\operatorname{sgn}(D_t).
\]

评分、公共信息和代理决策不依赖价格水平。因此，只要相同种子的零偏离分支保持一致，改变 \(\lambda\) 就不会改变：

- 策略持有和初始公共信息；
- 策略评分轨迹；
- 活跃代理和需求 \(D_t\)；
- 公共信息轨迹。

这一结论与选择线性还是对数价格无关，因为价格不反馈到订单动力学。极大的 \(\lambda\) 可能因 `tie_tolerance` 改变零偏离判定，不能把不变性无条件推广到所有浮点参数。

### 5.3 对数模式下的流动性缩放

对数模式中

\[
r_t=\ln(x_{t+1}/x_t)=D_t/\lambda.
\]

若需求轨迹不随 \(\lambda\) 改变，则

\[
r_t^{(\lambda)}=\frac{\lambda_0}{\lambda}r_t^{(\lambda_0)}.
\]

因此，对数收益率的均值和标准差按 \(1/\lambda\) 缩放，归一化自相关不变。固定 `N` 时，这就是图中波动率随 \(N/\lambda\) 的斜率一参考关系。

代表性种子的对数价格由对数收益率重建：

\[
\ell_t=\ln[x(t)/x(0)]=\sum_{\tau<t}r_\tau.
\]

累计需求定义为

\[
C_t=\sum_{\tau<t}D_\tau.
\]

在对数模式下，

\[
\lambda\ell_t=C_t.
\]

扫描保存的 `rescaled_log_price` 就是 \(\lambda\ell_t\)。不同深度的重标定轨迹重合，是对上述订单流不变性的直接数值检查，不是新的独立模型结果。

线性模式中 \(r_t\neq D_t/\lambda\)，且可能出现边界回退，所以这些等式不再成立。程序仍计算重标定差异作为诊断，但第四幅子图改画回退比例。

## 6. 对数价格的数值处理

极小 \(\lambda\) 会使 \(D/\lambda\) 很大。基础对数价格若反复执行 `price *= exp(D/lambda)`，绝对价格可能浮点溢出，尽管订单动力学本身仍然有效。

扫描器在对数模式下临时把 `MarketMaker.clear` 替换为 `_bounded_log_clear`：

1. 仍计算原始 `impact = D/lambda`；
2. 返回 `log_return = impact`；
3. 内部价格固定为 `1.0`；
4. `last_fallback=False`；
5. 模拟结束后用 `log_returns` 的累计和重建真正的相对对数价格。

绝对价格不进入评分、订单或公共信息，所以固定内部价格不会改变扫描保存的需求、活跃度和对数收益率。`price_change` 仍返回 `D/lambda`，简单收益率仍按 `expm1(impact)` 计算，但扫描不保存简单收益率。

该替换只作用于对数模式，并通过 `try/finally` 恢复原始方法。线性模式直接调用未修改的 `simulate_gcmg()`。当前实现是串行扫描，不适合在共享同一模块的并发任务中使用这种临时方法替换。

## 7. 扫描统计量与代表性轨迹

### 7.1 每个 `(lambda/N, seed)` 的统计量

所有统计量都舍弃前 `burn_in` 轮。设 burn-in 后的对数收益率为 \(r_t\)，需求为 \(D_t\)，活跃代理数为 \(V_t\)。

| NPZ 数组 | 单次运行计算 |
|---|---|
| `std_log_return` | `std(r)`，总体标准差 |
| `mean_log_return` | `mean(r)` |
| `acf_return_lag1` | 带符号对数收益率的一阶自相关 |
| `mean_demand` | `mean(D)` |
| `std_demand` | `std(D)`，总体标准差 |
| `acf_demand_lag1` | 需求的一阶自相关 |
| `mean_active_agents` | `mean(V)` |
| `fallback_rate` | burn-in 后 `price_fallback` 为真的时间比例 |

自相关复用基础模拟器的 `lag_autocorrelation()`：先去均值，再用 FFT 计算归一化自相关。零方差序列可能返回 `NaN`。这里的收益率 ACF 是带符号收益率的方向持续性，不是绝对收益率或波动率聚集。

### 7.2 代表性轨迹

程序只保存一个代表性种子的完整轨迹，默认是 `--seeds` 中的第一个种子。`--trajectory-seed` 可以指定另一种子，但必须包含在种子列表中。

对每组 \(\lambda/N\)，保存：

- `log_price`：\(\ell_t=\ln[x(t)/x(0)]\)；
- `cumulative_demand`：\(C_t=\sum_{\tau<t}D_\tau\)；
- `rescaled_log_price`：\(\lambda\ell_t\)。

统计量只使用 burn-in 后数据，轨迹则保留从 `t=0` 开始的完整路径，便于观察过渡期。

`--trajectory-stride k` 每隔 `k` 个点保存一次，最后一个 `t=steps` 始终包含。轨迹时间点保存在 `trajectory_time` 中。

### 7.3 扫描结束时的两个检查

`representative_rescale_error` 为

\[
\max_t|\lambda\ell_t-C_t|
\]

在全部深度上的最大值。对数模式下应接近浮点误差；线性模式下不要求为零。

`representative_demand_spread` 为每个时间点上不同深度的累计需求极差，再取所有时间的最大值：

\[
\max_t\left(\max_\lambda C_t^{(\lambda)}
-\min_\lambda C_t^{(\lambda)}\right).
\]

默认 `L=0` 时，该值可用于检查代表性需求轨迹是否真正不随市场深度改变。它只检查代表性种子，不代表所有种子的需求都相同。

## 8. 2×2 绘图的具体口径

`plot_liquidity_fss.py` 使用 `np.nanmean` 和 `np.nanstd` 跨种子聚合，误差棒为种子间总体标准差（`ddof=0`），不是标准误。涉及 \(N/\lambda\) 的子图先按逆深度升序排列，横轴使用对数刻度。

### (a) 代表性对数价格

绘制每个 \(\lambda/N\) 下同一代表性种子的

\[
\ln[x(t)/x(0)].
\]

burn-in 区间用灰色背景标出。该图直接显示较小市场深度对价格路径的放大作用。

### (b) 对数收益率波动率

横轴为 \(N/\lambda=1/\kappa\)，纵轴为各种子 `std_log_return` 的均值，误差棒为种子间 SD。

虚线满足

\[
\sigma_r\propto N/\lambda,
\]

并通过排序后第一个有效数据点确定截距。它是斜率一参考线，不是独立拟合，也不是对所有参数区间成立的理论定律。

### (c) 收益率一阶自相关

绘制各种子 `acf_return_lag1` 的均值和种子间 SD，并标出零线。该图用于观察方向持续性是否随市场深度改变。

### (d) 由价格模式决定

- `log`：绘制各深度的 \(\lambda\ln[x(t)/x(0)]\)，并报告这些重标定轨迹的最大极差；
- `linear`：绘制 burn-in 后价格回退比例的种子均值和 SD，用于检查数值价格边界是否随逆深度增加而变得重要。

默认输出为 300 DPI PNG，可通过 `--dpi` 修改，最低允许 50。

## 9. 流动性扫描与绘图命令

### 9.1 默认对数模式扫描

```bash
python scan_liquidity_fss.py
```

等价的主要显式参数为：

```bash
python scan_liquidity_fss.py \
  --N 101 --m 3 --s 2 --r 4 --T 100 --L 0 \
  --price-mode log --steps 50000 --burn-in 5000 \
  --lambda-over-n 10 1 0.1 0.01 \
  --seeds 123 124 125 126 127 128 129 130 131 132 \
  --initial-score-offset 10 --zero-score-tie random \
  --output liquidity_scan.npz
```

绘图：

```bash
python plot_liquidity_fss.py liquidity_scan.npz
```

默认图片名为 `liquidity_scan_2x2.png`。也可以明确指定：

```bash
python plot_liquidity_fss.py liquidity_scan.npz \
  --output liquidity_2x2.png --dpi 300
```

### 9.2 线性模式与边界诊断

```bash
python scan_liquidity_fss.py \
  --price-mode linear \
  --output liquidity_linear.npz

python plot_liquidity_fss.py liquidity_linear.npz \
  --output liquidity_linear_2x2.png
```

### 9.3 减少轨迹存储

```bash
python scan_liquidity_fss.py \
  --trajectory-seed 123 \
  --trajectory-stride 10 \
  --output liquidity_stride10.npz
```

输出路径没有 `.npz` 后缀时，程序会自动补上。扫描使用压缩 NPZ，不另写逐种子 CSV。

## 10. 流动性扫描参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--N` | `101` | 代理人数 |
| `--m` | `3` | 公共信息记忆长度 |
| `--s` | `2` | 每个代理持有的策略数 |
| `--r` | `4.0` | 入场评分阈值 |
| `--T` | `100.0` | 评分遗忘时间尺度 |
| `--L` | `0.0` | 市场参照阈值 |
| `--x0` | `100.0` | 基础模拟器初始价格 |
| `--price-mode` | `log` | `log` 或 `linear` |
| `--steps` | `50000` | 每次运行的轮数 |
| `--burn-in` | `5000` | 统计舍弃的初始轮数 |
| `--lambda-over-n` | `10 1 0.1 0.01` | 一组正的 \(\lambda/N\) |
| `--seeds` | `123,...,132` | 每组深度复用的不同种子 |
| `--trajectory-seed` | 第一个种子 | 保存完整轨迹的代表性种子 |
| `--trajectory-stride` | `1` | 轨迹抽样间隔 |
| `--mu0` | 未设置 | 固定初始公共信息；缺省时由种子生成 |
| `--initial-score-offset` | `10.0` | 相对 `r` 的初始评分偏移 |
| `--initial-score` | 未设置 | 绝对初始评分，覆盖偏移规则 |
| `--zero-score-tie` | `random` | 零偏离评分信号规则 |
| `--output` | `liquidity_scan.npz` | 扫描输出文件 |

扫描器要求所有 \(\lambda/N\) 有限且严格为正，种子互不重复，轨迹间隔至少为 1，并且 burn-in 后至少保留 4 个观测。其他模型参数继续由 `GCMGConfig` 校验。

## 11. `liquidity_scan.npz` 数据结构

设深度数为 `K`，种子数为 `S`，轨迹点数为 `T_traj`。

| 数组 | 形状 | 内容 |
|---|---|---|
| `lambda_over_n` | `(K,)` | 输入的 \(\kappa=\lambda/N\) |
| `lambda_values` | `(K,)` | 实际使用的 \(\lambda=\kappa N\) |
| `seeds` | `(S,)` | 种子列表 |
| `std_log_return` 等八个逐运行统计数组 | `(K,S)` | 第 7.1 节统计量 |
| `trajectory_time` | `(T_traj,)` | 代表性轨迹的时间下标 |
| `trajectory_seed` | 标量 | 代表性种子 |
| `log_price` | `(K,T_traj)` | 相对对数价格 |
| `rescaled_log_price` | `(K,T_traj)` | \(\lambda\) 乘相对对数价格 |
| `cumulative_demand` | `(K,T_traj)` | 累计需求 |
| `representative_rescale_error` | 标量 | 重标定轨迹与累计需求的最大差 |
| `representative_demand_spread` | 标量 | 不同深度累计需求的最大极差 |
| `metadata_json` | 字符串标量 | 完整扫描参数和指标定义 |

八个逐运行统计数组分别是 `std_log_return`、`mean_log_return`、`acf_return_lag1`、`mean_demand`、`std_demand`、`acf_demand_lag1`、`mean_active_agents` 和 `fallback_rate`。

metadata 中的 `initial_score` 保留用户输入，未指定时为 `null`；实际初始评分仍按 `r + initial_score_offset` 进入基础模拟器。NPZ 不保存每个种子的完整时间序列，也不保存最终策略评分。

## 12. 结果解读

1. 先检查 metadata，确认 `L`、价格模式、初始评分和零偏离规则；
2. 在默认 `L=0` 下，先看代表性累计需求是否跨深度重合，再讨论收益率缩放；
3. 对数模式中，重标定价格重合和波动率斜率一主要是同一需求冲击 \(D/\lambda\) 的数值表现；
4. 若 `L≠0`，\(\lambda\) 会进入 \(D/\lambda-L\)，需求动力学本身可以改变，不能继续套用纯缩放解释；
5. 线性模式需要结合第四幅子图的回退比例判断数值价格边界是否重要；
6. 代表性轨迹只对应一个种子，跨种子结论应结合统计数组和误差棒；
7. 若改变 `N`，\(\lambda=\kappa N\) 同时变化，不能把不同 `N` 的结果仅解释为同一代理动力学下的流动性变化。
