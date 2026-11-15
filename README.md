# GCMG-RSS 多代理金融市场模拟器

`gcmg_rss.py` 实现了基于全局信息的 Grand Canonical Minority Game（GCMG，巨正则少数派博弈），默认策略集合为 Reduced Strategy Space（RSS，约化策略空间）。程序包含策略空间构造、代理与市场动力学、时间序列保存，以及收益率和活跃度诊断。

代码分为两层：

- `build_rss()` 只负责生成策略表；
- `simulate_gcmg()` 只依赖策略表接口，不关心策略来自 RSS、FSS 还是其他构造。

因此，阅读市场动力学时可以先不看 Hadamard 矩阵；替换策略空间时也不需要修改代理、评分或价格模块。

## 1. 状态变量与数据结构

记忆长度为 `m` 时，公开信息共有

\[
P=2^m
\]

个状态。程序用整数 `mu` 表示最近 `m` 位市场历史，用二维数组 `actions` 表示策略表：

```text
actions.shape = (n_strategies, P)
actions[R, mu] ∈ {-1, +1}
```

`actions[R, mu]` 是策略 `R` 在信息状态 `mu` 下给出的行动：`-1` 表示卖出，`+1` 表示买入。RSS 下 `n_strategies = 2P`。

模拟中的主要状态为：

| 变量 | 形状 | 含义 |
|---|---|---|
| `holdings` | `(N, s)` | 每个代理固定持有的策略编号 |
| `scores` | `(n_strategies,)` | 所有代理共享的策略虚拟评分 |
| `mu` | 标量 | 当前公开信息状态 |
| `market_maker.price` | 标量 | 当前价格 \(x_t\) |

`holdings` 在整个模拟中不变。代理之间独立抽取策略，因此不同代理可以持有相同策略；同一代理的 `s` 个策略则通过无放回抽样保证互不重复。

评分不是代理私有状态。每个策略只有一份全局评分，代理根据自己持有的策略子集读取同一张评分表。

## 2. RSS 策略空间

### 2.1 构造方法

`build_rss(m)` 先构造 \(P\) 阶 Sylvester–Walsh–Hadamard 矩阵：

\[
H_1=[1],\qquad
H_{2k}=
\begin{bmatrix}
H_k&H_k\\
H_k&-H_k
\end{bmatrix},
\]

再将其与相反矩阵上下拼接：

\[
A=\begin{bmatrix}H_P\\-H_P\end{bmatrix}.
\]

代码中的 `actions` 就是矩阵 \(A\)。前 `P` 行两两正交，后 `P` 行依次为前 `P` 行的负策略：

\[
H_PH_P^{\mathsf T}=P I_P,\qquad
A_{R+P,\mu}=-A_{R,\mu}.
\]

默认 `m=3`，对应 `P=8` 个公开信息状态和 `16` 个策略。`build_rss()` 默认检查前半部分的正交性以及后半部分的逆相关关系。这里的 RSS 是一种标准代表，并不意味着 RSS 的构造唯一。

### 2.2 可替换的策略空间接口

`StrategySpace` 协议只要求两个属性：

```python
m: int
actions: np.ndarray
```

传入 `simulate_gcmg()` 的自定义策略空间必须满足：

- `strategy_space.m == config.m`；
- `actions` 是二维数组，列数为 `2**m`；
- 所有元素均为 `-1` 或 `+1`；
- 策略数不少于每个代理的持策略数 `s`。

模拟器会先将 `actions` 转为 `int8`，再按行读取策略、按当前的 `mu` 读取行动。

## 3. 初始化

`simulate_gcmg()` 按以下顺序初始化：

1. 检查策略表和配置参数；
2. 用 `numpy.random.default_rng(seed)` 创建随机数生成器；
3. 为每个代理无放回地抽取 `s` 个策略，形成 `holdings`；
4. 将所有策略评分初始化为同一个值；
5. 设置初始公开信息和初始价格。

默认初始评分为

\[
S_R(0)=r+\text{initial\_score\_offset}.
\]

默认 `r=-100`、`initial_score_offset=10`，所以所有策略从 `-90` 开始，且所有代理最初都满足入场条件。若设置 `initial_score`，则直接使用该绝对值，`initial_score_offset` 不再参与计算。

`mu0` 未指定时，程序从 `[0, P)` 中均匀随机抽取初始状态。所有评分相等时，首轮最优策略并列，代理由随机数生成器在自己持有的策略中等概率选取一个。

## 4. 单轮市场动力学

第 `t` 轮开始时，已知 `scores`、固定的 `holdings`、公开信息 \(\mu_t\) 和价格 \(x_t\)。代码严格按以下顺序推进。

### 4.1 选择策略并判断是否入场

代理 \(i\) 在自己持有的策略中选择评分最高者：

\[
R_i^*(t)\in\arg\max_{R\in\mathcal R_i}S_R(t),
\qquad
B_i(t)=\max_{R\in\mathcal R_i}S_R(t).
\]

若最高分策略并列，程序只对并列项随机排序并选取一个。这里使用精确浮点相等判断，`tie_tolerance` 不参与策略评分并列的判断。

GCMG 的入场条件为

\[
I_i(t)=\mathbf 1\{B_i(t)>r\}.
\]

注意是严格大于。未入场代理本轮订单为零；入场代理执行所选策略在当前公开信息下的行动：

\[
a_i(t)=I_i(t)\,A_{R_i^*(t),\mu_t}.
\]

这一步由 `_pick_best_strategies()` 和 `simulate_gcmg()` 主循环前半段完成。

### 4.2 汇总超额需求

\[
D_t=\sum_i a_i(t)=n_{\rm buy}(t)-n_{\rm sell}(t).
\]

代码同时保存买单数、卖单数和活跃代理数。由于每个活跃代理只提交一个 `+1` 或 `-1` 订单，

\[
n_{\rm buy}+n_{\rm sell}=\texttt{active\_agents}.
\]

### 4.3 生成评分信号和新的公开信息位

先计算相对市场参照阈值的偏离：

\[
q_t=\frac{D_t}{\lambda},\qquad z_t=q_t-L.
\]

设 `tie_tolerance = eps`。非零偏离时，评分信号 \(g_t\) 和新信息位 \(b_t\) 为：

| 条件 | `score_signal` \(g_t\) | `new_bit` \(b_t\) |
|---|---:|---:|
| \(z_t>\mathrm{eps}\) | `+1` | `1` |
| \(z_t<-\mathrm{eps}\) | `-1` | `0` |

当 \(|z_t|\leq\mathrm{eps}\) 时，新信息位始终独立随机取 `0` 或 `1`；评分信号由 `zero_score_tie` 决定：

- `random`：独立随机取 `+1` 或 `-1`，这是默认值；
- `zero`：取 `0`，对应字面上的 \(\operatorname{sign}(0)=0\)。

因此，在零偏离时，`new_bit` 与 `score_signal` 不必对应。`random` 模式下二者分别抽样。

### 4.4 更新全部策略评分

\[
S_R(t+1)=\alpha S_R(t)-A_{R,\mu_t}g_t,
\qquad
\alpha=1-\frac1T.
\]

更新覆盖策略空间中的全部策略，而不限于本轮被代理持有、选择或实际执行的策略。评分只读取本轮开始时的 \(\mu_t\)，不使用刚生成的新信息位。

该符号约定使与市场信号相反的行动加分。例如 \(g_t=+1\) 时，行动为 `-1` 的策略增加 `1` 分，行动为 `+1` 的策略扣除 `1` 分，同时原有评分乘以衰减系数 \(\alpha\)。

`T` 是评分遗忘时间尺度。`T` 越大，旧评分衰减越慢；代码要求 `T >= 1`，从而保证 \(0\leq\alpha<1\)。

### 4.5 清算订单并更新价格

定义价格冲击

\[
h_t=\frac{D_t}{\lambda}.
\]

`price_mode="linear"` 时，正常步骤采用线性价格规则：

\[
x_{t+1}=x_t+h_t.
\]

如果该步会得到非正价格，`MarketMaker.clear()` 不终止模拟，而是仅对该步改用对数价格规则：

\[
x_{t+1}=x_t\exp(h_t).
\]

下一步仍回到线性模式。`price_mode="log"` 时，所有步骤都采用对数价格规则。

收益率按实际采用的价格规则计算：

- 有效线性步骤：`simple_return = h_t / x_t`，`log_return = log(x_{t+1}/x_t)`；
- 对数步骤：`log_return = h_t`，`simple_return = exp(h_t)-1`。

`price_change` 保存的始终是 \(h_t=D_t/\lambda\)，不是无条件成立的 \(x_{t+1}-x_t\)。在线性越界回退步骤和全程 `log` 模式下，两者会不同；实际价格增量应从 `np.diff(result.price)` 计算。

### 4.6 更新公开信息并记录结果

新信息位追加到 `m` 位历史的末尾：

\[
\mu_{t+1}=(2\mu_t+b_t)\bmod P.
\]

这等价于历史左移一位、丢弃最旧位，并将最新位放在最低二进制位。

随后保存本轮数据。需要注意时间下标：CSV 中的 `mu_t` 是本轮决策所用状态，`new_bit` 是本轮末尾追加的位，`price` 则是清算后的 \(x_{t+1}\)。

## 5. 零偏离与无人参与状态

`zero_score_tie` 只影响评分信号，不影响公开信息的随机更新。

当 `L=0` 且使用 `zero_score_tie="zero"` 时，若所有代理都不参与，则 \(D_t=0\)、\(g_t=0\)，评分只按 \(\alpha\) 衰减。这种状态下市场可能持续无人参与。主程序会检查最后 10% 的轮次；若活跃代理数全部为零，会在终端输出警告。

默认的 `random` 模式在零偏离时仍给出非零评分信号，避免评分更新完全停止。它属于代码显式提供的零值正则化约定，进行模型对照时应与 `zero` 模式分开报告。

## 6. 代码结构

| 类或函数 | 作用 |
|---|---|
| `StrategySpace` | 定义模拟器所需的策略空间协议 |
| `RSSStrategySpace` | 保存 `m`、`actions`，并提供 `P` 和策略数属性 |
| `sylvester_hadamard()` | 递归构造 2 的幂阶 Hadamard 矩阵 |
| `build_rss()` | 生成 `[H; -H]` 策略表并验证结构 |
| `GCMGConfig` | 保存模型和模拟参数，解析市场深度，执行参数校验 |
| `MarketMaker.clear()` | 根据超额需求更新价格并计算两种收益率 |
| `_pick_best_strategies()` | 向量化选择各代理的最高分持有策略，随机处理并列 |
| `simulate_gcmg()` | 初始化状态并执行完整市场循环 |
| `GCMGResult` | 保存全部时间序列、最终评分和固定持策略表 |
| `autocorrelation_function()` | 用 FFT 计算归一化自相关函数 |
| `lag_autocorrelation()` | 提取指定滞后的自相关系数 |
| `summarize_result()` | 计算 burn-in 后的矩和自相关诊断 |
| `save_csv()` | 保存逐轮时间序列 |
| `save_diagnostics_plot()` | 绘制价格、活跃度、收益率分布和自相关图 |
| `save_price_change_distribution_plot()` | 绘制实际价格增量分布及正态 Q–Q 图 |
| `_parse_args()`、`main()` | 命令行入口、文件输出和终端摘要 |

主循环中，策略选择和订单汇总按代理向量化；代理初始化仍逐个调用无放回抽样。每轮的主要计算量为 \(O(Ns+n_{\rm strategies})\)，结果数组占用 \(O(\mathrm{steps})\) 量级内存。

## 7. 运行方法

依赖 Python、NumPy 和 Matplotlib。直接运行：

```bash
python gcmg_rss.py
```

自定义模型和模拟参数：

```bash
python gcmg_rss.py \
  --N 1001 --m 4 --s 2 \
  --r -100 --T 10000 --L 0 \
  --steps 200000 --burn-in 20000 \
  --seed 42 --outdir gcmg_output_custom
```

全程使用对数价格规则：

```bash
python gcmg_rss.py --price-mode log
```

采用字面上的零信号规则：

```bash
python gcmg_rss.py --zero-score-tie zero
```

指定所有策略的绝对初始评分：

```bash
python gcmg_rss.py --initial-score 0
```

查看全部命令行参数：

```bash
python gcmg_rss.py --help
```

在相同运行环境中，相同参数和随机种子可复现相同的随机抽取与模拟轨迹。

## 8. 参数表

| 配置字段 | 命令行参数 | 默认值 | 含义 |
|---|---|---:|---|
| `N` | `--N` | `501` | 代理人数 |
| `m` | `--m` | `3` | 公开信息记忆长度 |
| `s` | `--s` | `2` | 每个代理持有的不同策略数 |
| `r` | `--r` | `-100.0` | 最高策略评分的入场阈值，可为负数 |
| `T` | `--T` | `10000.0` | 评分遗忘时间尺度 |
| `L` | `--L` | `0.0` | 市场参照阈值 |
| `lam` | `--lambda` | `None` | 市场深度；未设置时取 `10*N` |
| `x0` | `--x0` | `100.0` | 初始价格 |
| `price_mode` | `--price-mode` | `linear` | `linear` 或 `log` |
| `steps` | `--steps` | `100000` | 模拟轮数 |
| `seed` | `--seed` | `123` | 随机种子 |
| `mu0` | `--mu0` | `None` | 初始信息状态；未设置时随机抽取 |
| `initial_score_offset` | `--initial-score-offset` | `10.0` | 相对 `r` 的初始评分偏移 |
| `initial_score` | `--initial-score` | `None` | 绝对初始评分，设置后覆盖偏移规则 |
| `zero_score_tie` | `--zero-score-tie` | `random` | 零偏离时的评分信号：`random` 或 `zero` |
| `tie_tolerance` | 无 | `1e-12` | 判断 \(D/\lambda-L\) 是否为零的容差 |
| 无 | `--burn-in` | `10000` | 诊断时舍弃的初始轮数 |
| 无 | `--outdir` | `gcmg_output` | 输出目录 |

主要约束包括：`N >= 1`、`m >= 1`、`s >= 1` 且不超过策略总数、`T >= 1`、`steps >= 1`、`x0 > 0`、`lambda > 0`、`mu0 ∈ [0, P)`。`burn_in` 必须为非负数，且其后至少保留 4 个观测值。

## 9. 输出与诊断口径

### 9.1 `gcmg_timeseries.csv`

每行对应一轮，包含全部模拟轮次，不应用 `burn_in`。

| 字段 | 含义 |
|---|---|
| `t` | 从 0 开始的轮次 |
| `mu_t` | 本轮决策使用的信息状态 |
| `new_bit` | 本轮结束时追加的信息位 |
| `demand` | 超额需求 \(D_t\) |
| `buy_orders` | 买单数 |
| `sell_orders` | 卖单数 |
| `active_agents` | 本轮入场代理数 |
| `price` | 本轮清算后的价格 \(x_{t+1}\) |
| `price_change` | 价格冲击 \(D_t/\lambda\) |
| `simple_return` | 简单收益率 |
| `log_return` | 对数收益率 |
| `score_signal` | 本轮评分信号 \(g_t\) |

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

超额峰度按标准化四阶矩减 3 计算。`acf_abs_return_*` 使用的是 \(|r_t-\bar r|\)，不是原始 \(|r_t|\)；`acf_volume_*` 使用活跃代理数作为交易量或活跃度的代理变量。

自相关通过 FFT 计算，序列先去均值，协方差估计统一除以样本数 `n`，再除以零滞后协方差。常数序列的零方差会使部分统计量或自相关系数成为 `NaN`。

### 9.3 `gcmg_diagnostics.png`

包含四幅图：

1. 从模拟起点开始的前至多 3000 步价格；
2. 从模拟起点开始的前至多 3000 步活跃代理数；
3. burn-in 后的标准化对数收益率分布，纵轴为对数刻度，并叠加标准正态密度；
4. burn-in 后滞后 1 至 100 的收益率、去均值收益率绝对值和活跃度自相关。

价格和活跃度子图展示的是初始轨迹，并未舍弃 burn-in；分布和自相关子图使用 burn-in 后的数据。

### 9.4 `gcmg_price_change_distribution.png`

横轴使用 burn-in 后的实际价格增量

\[
\Delta x_t=x_{t+1}-x_t,
\]

而非标准化收益率。左图是增量直方图，并叠加具有相同样本均值和标准差的正态密度；右图是正态 Q–Q 图，最多绘制 5000 个按顺序统计量抽取的点。该图在 `log` 模式下仍表示实际 \(\Delta x\)。

## 10. 作为 Python 模块使用

```python
from gcmg_rss import (
    GCMGConfig,
    build_rss,
    simulate_gcmg,
    summarize_result,
)

config = GCMGConfig(
    N=1001,
    m=4,
    s=2,
    r=-100.0,
    T=10_000.0,
    steps=200_000,
    seed=42,
)
strategy_space = build_rss(config.m)
result = simulate_gcmg(config, strategy_space)
diagnostics = summarize_result(result, burn_in=20_000)
```

`result.price` 长度为 `steps+1`，其中 `price[0]` 是初始价格；其余时间序列长度均为 `steps`。`final_scores` 保存最后一轮更新后的全局策略评分，`final_holdings` 保存固定的代理持策略表。

若使用其他策略空间，只需构造满足 `StrategySpace` 协议的对象并传给 `simulate_gcmg(config, strategy_space)`。市场深度默认规则、入场条件、评分更新、公开信息更新和价格形成逻辑均无需改动。
