# GCMG-RSS / FSS 超额需求方差扫描

本项目比较一般 GCMG 参数下，RSS 与 FSS 的超额需求方差随记忆长度 `m` 的变化。与固定 \(r=-\infty\)、\(T=+\infty\) 的原始 Minority Game 极限版本不同，本版将 `r`、`T`、`L` 和初始评分作为可配置参数，代理是否入场由评分阈值决定。

两个脚本分别负责计算和绘图：

```text
scan_sigma2.py
    调用 gcmg_rss.py / gcmg_fss.py
    逐策略空间、逐 m、逐 seed 模拟
    计算单次运行的需求方差与活跃度
    跨种子聚合并检查内部局部极小值
    写入 runs / summary / metadata

plot_sigma2.py
    读取扫描 CSV
    绘制均值、种子散点、误差棒和极小值标记
```

扫描器不重新实现 GCMG。`make_config()` 负责把命令行参数写入 RSS 或 FSS 的 `GCMGConfig`，市场循环仍由两个基础模拟器执行。

## 1. 单次模拟的核心模型

### 1.1 公共信息、策略与持有关系

记忆长度为 `m` 时，公共信息状态数为

\[
P=2^m,\qquad \mu_t\in\{0,\ldots,P-1\}.
\]

策略 \(R\) 对每个公共信息状态给出固定动作

\[
A_{R,\mu}\in\{-1,+1\},
\]

其中 `-1` 表示卖出，`+1` 表示买入。每个代理初始化时抽取 `s` 个不同策略，形成固定集合 \(\mathcal R_i\)。不同代理可以持有相同策略，模拟过程中不重新抽取。

策略评分是全局共享状态。代理只比较自己持有的策略，并选择评分最高者：

\[
R_i^*(t)\in\arg\max_{R\in\mathcal R_i}S_R(t),
\qquad
B_i(t)=\max_{R\in\mathcal R_i}S_R(t).
\]

最高评分并列时，基础模拟器在并列项中无偏向随机选择。这里使用精确浮点相等判断，与需求信号的 `tie_tolerance` 无关。

### 1.2 入场条件和超额需求

代理只有在最佳持有策略的评分严格大于 `r` 时才交易：

\[
I_i(t)=\mathbf 1\{B_i(t)>r\}.
\]

本轮订单为

\[
a_i(t)=I_i(t)A_{R_i^*(t),\mu_t}.
\]

未入场代理的订单为零。清算前净超额需求为

\[
D_t=\sum_i a_i(t)=n_{\rm buy}(t)-n_{\rm sell}(t).
\]

默认 `r=-100`，初始评分为 `r+10=-90`，所以所有代理在第一轮都满足入场条件；后续是否继续入场取决于评分演化。本版不保证 `active_agents=N`，零需求和灭绝状态都可能出现。

### 1.3 评分信号和评分更新

先计算需求冲击及其相对参照阈值的偏离：

\[
q_t=\frac{D_t}{\lambda},\qquad z_t=q_t-L.
\]

在非零偏离区域，评分信号为

\[
g_t=\operatorname{sgn}(z_t).
\]

所有候选策略按

\[
S_R(t+1)=\alpha S_R(t)-A_{R,\mu_t}g_t,
\qquad
\alpha=1-\frac1T
\]

更新，包括本轮没有被代理选中的已持有策略。RSS 基础模拟器更新完整策略评分表；FSS 基础模拟器在线更新实际持有策略的共享评分。

`T` 越大，旧评分衰减越慢。`r` 控制入场，`T` 控制评分记忆，`L` 控制需求参照点，三者作用于不同环节。

### 1.4 零偏离和公共信息更新

当

\[
|z_t|\leq\texttt{tie\_tolerance}
\]

时，基础模拟器进入零偏离分支：

- 新公共信息位 `new_bit` 始终随机取 `0` 或 `1`；
- `zero_score_tie="zero"` 时，评分信号 \(g_t=0\)；
- `zero_score_tie="random"` 时，评分信号独立随机取 `+1` 或 `-1`。

因此，`zero` 只表示字面上的 \(\operatorname{sgn}(0)=0\) 评分规则，不会让公共信息确定性追加 `0`。默认值为 `random`。两种约定可能产生不同轨迹，不应混合为同一组动力学结果。

非零偏离时，若 \(z_t>0\) 则 `new_bit=1`，若 \(z_t<0\) 则 `new_bit=0`。本轮末尾公共信息更新为

\[
\mu_{t+1}=(2\mu_t+\texttt{new\_bit}_t)\bmod P.
\]

这等价于丢弃最旧位，并把最新位追加到 `m` 位历史的最低位。

### 1.5 价格更新与需求统计的关系

扫描固定采用线性价格规则：

\[
x_{t+1}=x_t+\frac{D_t}{\lambda}.
\]

只有当候选价格非正时，当前步骤改用

\[
x_{t+1}=x_t\exp(D_t/\lambda),
\]

下一步恢复线性规则。扫描器不提供全程 `log` 价格选项。

价格不反馈到策略评分、订单或公共信息，因此价格回退不参与 \(\sigma_D^2\) 的计算。回退次数只作为运行诊断保存。

两个基础模块的回退记录接口不同：

- FSS 结果包含 `price_fallback`，扫描可以得到实际回退步数；
- RSS 结果不包含该字段，`calculate_observables()` 通过 `getattr(..., None)` 将回退计数写为 `0`。

所以 RSS 行中的 `price_fallback_steps_total=0` 和 `price_fallback_steps_after_burn=0` 是接口缺省值，不能用来证明没有发生价格回退。

## 2. RSS 与 FSS 的实现路径

### 2.1 RSS

RSS 使用 Sylvester–Walsh–Hadamard 矩阵 \(H_P\) 构造

\[
A=\begin{bmatrix}H_P\\-H_P\end{bmatrix}.
\]

策略总数为 `2P`。前半部分的 `P` 个策略两两正交，后半部分依次为对应的逆策略。扫描范围内每个 `m` 都显式生成 RSS 矩阵。

### 2.2 FSS

完整 FSS 包含所有长度为 `P` 的 \(\pm1\) 动作向量：

\[
|\mathrm{FSS}|=2^P=2^{2^m}.
\]

基础 FSS 模块按 `m` 采用两种表示：

- `m <= 4`：生成完整 FSS 矩阵，再为每个代理无放回抽取 `s` 个策略；
- `m >= 5`：直接生成长度为 `P` 的独立公平比特串，映射为 \(\pm1\) 动作；同一代理内拒绝重复，不同代理允许共享策略。

隐式路径等价于每个代理从完整 FSS 中均匀无放回抽样，但不需要保存 \(2^P\) 行矩阵。程序随后对所有代理的持有记录去重，使不同代理共享同一策略的动作向量和评分。全局去重只减少存储和在线更新量，不改变持有关系。

扫描器对每个策略空间和每个 `m` 只构造一次策略空间描述符，然后在各种子下复用。代理持有策略的抽样仍在各自的 `simulate_gcmg()` 中由对应种子决定。

## 3. 统计量与跨种子聚合

### 3.1 单次运行的需求方差

每次运行舍弃前 `burn_in` 轮，对剩余 `n` 个清算前需求计算

\[
\bar D=\frac1n\sum_tD_t,\qquad
\overline{D^2}=\frac1n\sum_tD_t^2,
\]

\[
\sigma_D^2=\overline{D^2}-\bar D^2,
\qquad
\sigma_D=\sqrt{\sigma_D^2}.
\]

这是分母为 `n` 的总体方差，不是 `ddof=1` 的样本方差，也没有除以 `N` 或 `N^2`。代码用 `max(0, ...)` 避免浮点舍入产生微小负方差。

单次结果还保存以下诊断：

| 字段 | 含义 |
|---|---|
| `n_postburn` | burn-in 后的需求观测数 |
| `mean_D` | \(\bar D\) |
| `mean_D2` | \(\overline{D^2}\) |
| `active_time_fraction` | 活跃代理数大于零的时间比例 |
| `mean_active_agent_fraction` | 平均每轮活跃代理数除以 `N` |
| `nonzero_demand_fraction` | \(D_t\neq0\) 的时间比例 |
| `extinct_after_burn` | burn-in 后是否每一轮都没有活跃代理 |
| `price_fallback_steps_total` | 全部轮次的价格回退计数，受 RSS 接口限制 |
| `price_fallback_steps_after_burn` | burn-in 后的价格回退计数，受 RSS 接口限制 |

`extinct_after_burn=1` 要求整个 burn-in 后区间都无人交易。只有部分轮次无人交易的运行仍属于非灭绝运行。

### 3.2 同一 `m` 下的种子聚合

`aggregate_runs()` 按 `(strategy_space, m)` 分组，分别对各种子已经算出的 `sigma2` 和 `sigma` 聚合：

\[
\operatorname{SD}=s_{\rm seeds}\quad(\texttt{ddof=1}),
\qquad
\operatorname{SEM}=\frac{\operatorname{SD}}{\sqrt{n_{\rm seeds}}}.
\]

每个统计量有两套口径：

- `*_all_seeds`：使用全部种子；
- `*_nonextinct`：只使用 `extinct_after_burn=0` 的种子。

主图和极小值检查使用 `nonextinct` 口径。只有一个有效种子时，SD 和 SEM 为 `NaN`；写入 JSON 时非有限数转为 `null`，绘图时误差棒按零处理。

聚合时需要注意：

- 平均 \(\sigma_D^2\) 是各种子方差的平均，不是把各种子需求合并后再求方差；
- 平均 \(\sigma_D\) 是各种子标准差的平均，不等于平均方差的平方根；
- SD 和 SEM 衡量种子间差异，不是单条时间序列内部的标准误；
- 灭绝过滤改变进入 `nonextinct` 统计的种子集合，应结合 `n_valid_seeds` 解读。

## 4. 内部极小值检查

`detect_minima()` 分别针对方差和标准差、RSS 和 FSS 执行相同检查。

一个 `m` 点只有满足以下条件才进入检查集合：

1. 该点所有种子都非灭绝，即 `n_valid_seeds == n_seeds`；
2. 对应均值为有限数；
3. 对内部极小值而言，相邻整数点 `m-1` 和 `m+1` 也满足前两项。

严格内部局部极小值要求

\[
y_m<y_{m-1},\qquad y_m<y_{m+1}.
\]

边界点不进入内部极小值列表。程序另行记录所有合格点中的最低值位置，因此边界最低值仍会出现在 `global_minimum_m_among_eligible_points` 中。

### 4.1 三点二次拟合

程序用 `(m-1,y_{m-1})`、`(m,y_m)`、`(m+1,y_{m+1})` 拟合二次函数。只有二次项系数为正且顶点位于左右节点之间时，才返回 `quadratic_vertex_estimate`。

该顶点只是离散曲线的探索性插值，不表示非整数 `m` 有模型含义，也不增加实际扫描分辨率。

### 4.2 合并 1-SEM 判据

定义中间点相对左右邻点的下降幅度：

\[
\Delta_-=y_{m-1}-y_m,\qquad
\Delta_+=y_{m+1}-y_m.
\]

对应合并标准误为

\[
e_-=\sqrt{\mathrm{SEM}_{m-1}^2+\mathrm{SEM}_m^2},
\qquad
e_+=\sqrt{\mathrm{SEM}_{m+1}^2+\mathrm{SEM}_m^2}.
\]

只有 \(\Delta_->e_-\) 且 \(\Delta_+>e_+\) 时，字段 `lower_than_both_neighbors_by_more_than_combined_1_SEM` 才为真。

`minimum_verified` 只表示至少存在一个严格内部局部极小值，不要求通过 1-SEM 判据。它不能解释为统计显著性或相变证明。

## 5. 绘图规则

`plot_sigma2.py` 必须读取：

- `sigma2_runs.csv`；
- `sigma2_summary.csv`。

若输入目录还包含 `sigma2_scan_metadata.json`，程序读取其中的 `N`，供可选 P39 参考线使用。

输出两张图：

| 文件 | 主曲线 |
|---|---|
| `sigma2_vs_m_RSS_FSS.png` | 非灭绝种子的平均 \(\sigma_D^2\) |
| `sigma_vs_m_RSS_FSS_P39_style.png` | 非灭绝种子的平均 \(\sigma_D\) |

主曲线误差棒是跨种子 **SD**，不是 SEM。图中其他标记为：

- 默认显示单种子散点，非灭绝种子为实心点，灭绝种子为 `x`；
- 只有部分种子非灭绝的 `m` 点仍绘制非灭绝均值，并用空心圈标记；
- 所有种子均灭绝的点不进入主曲线和极小值检查，另以 `x` 显示全种子均值；
- 若存在多个内部极小值，图中标注其中均值最低者，并注明是否通过合并 1-SEM 判据。

`--show-p39-reference` 在方差图中添加 \(N\)，在标准差图中添加 \(\sqrt N\)。它们只是 El Farol 讲义图的对照线，不是一般 GCMG 的理论预测。若没有 metadata，程序无法取得 `N`，即使指定该选项也不会画参考线。

## 6. 安装与运行

项目目录需要同时包含：

```text
gcmg_rss.py
gcmg_fss.py
scan_sigma2.py
plot_sigma2.py
```

依赖 Python、NumPy 和 Matplotlib：

```bash
python -m pip install numpy matplotlib
```

### 6.1 默认扫描

```bash
python scan_sigma2.py --outdir sigma2_scan
```

默认扫描 RSS 和 FSS 的 `m=1,...,10`，每个点使用种子 `123 456 789`，共执行

\[
(10+10)\times3=60
\]

次模拟，每次 50000 轮。

主要默认参数的显式写法为：

```bash
python scan_sigma2.py \
  --N 101 --s 2 --r -100 --T 10000 --L 0 \
  --steps 50000 --burn-in 5000 --seeds 123 456 789 \
  --rss-m-min 1 --rss-m-max 10 \
  --fss-m-min 1 --fss-m-max 10 \
  --initial-score-offset 10 --zero-score-tie random \
  --outdir sigma2_scan
```

扫描完成后绘图：

```bash
python plot_sigma2.py --input-dir sigma2_scan
```

添加 P39 参考线：

```bash
python plot_sigma2.py --input-dir sigma2_scan --show-p39-reference
```

隐藏单种子散点，并指定另一图片目录：

```bash
python plot_sigma2.py \
  --input-dir sigma2_scan \
  --outdir figures_sigma2 \
  --no-individual-seeds
```

### 6.2 小规模运行检查

```bash
python scan_sigma2.py \
  --N 101 --steps 1000 --burn-in 100 --seeds 123 \
  --rss-m-min 1 --rss-m-max 3 \
  --fss-m-min 1 --fss-m-max 3 \
  --outdir smoke_results
```

该命令只检查扫描、聚合和文件输出流程，不适合正式统计。正式运行需要检查过渡期、统计窗口和种子数是否足够。

扫描结果在全部运行结束后一次性写出，不是每完成一个种子就追加保存；中断运行不会留下本次扫描的部分 CSV。

## 7. 参数表

### 7.1 `scan_sigma2.py`

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--N` | `101` | 代理人数 |
| `--s` | `2` | 每个代理持有的不同策略数 |
| `--r` | `-100.0` | 最佳持有策略评分的入场阈值 |
| `--T` | `10000.0` | 评分遗忘时间尺度 |
| `--L` | `0.0` | 需求冲击的市场参照阈值 |
| `--lambda` | 未设置 | 市场深度；缺省时为 `10*N` |
| `--x0` | `100.0` | 初始价格 |
| `--steps` | `50000` | 每次运行的总轮数 |
| `--burn-in` | `5000` | 统计前舍弃的轮数 |
| `--seeds` | `123 456 789` | 一个或多个随机种子 |
| `--rss-m-min` / `--rss-m-max` | `1` / `10` | RSS 扫描范围 |
| `--fss-m-min` / `--fss-m-max` | `1` / `10` | FSS 扫描范围 |
| `--mu0` | 未设置 | 固定初始公共信息；缺省时由各次运行的种子随机生成 |
| `--initial-score-offset` | `10.0` | 未指定绝对初值时，所有策略从 `r+offset` 开始 |
| `--initial-score` | 未设置 | 所有策略的绝对初始评分；设置后覆盖 offset 规则 |
| `--zero-score-tie` | `random` | 零偏离时的评分信号；公共信息位始终随机 |
| `--outdir` | `sigma2_scan` | 扫描输出目录 |

扫描范围必须位于 `1 <= m <= 10`，并要求 `steps > burn-in + 1`，即至少留下两个统计观测。价格模式固定为线性加边界回退，不是命令行选项。

### 7.2 `plot_sigma2.py`

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--input-dir` | `sigma2_scan` | 扫描结果目录 |
| `--outdir` | 与输入目录相同 | 图片输出目录 |
| `--show-p39-reference` | 关闭 | 添加 \(N\) 或 \(\sqrt N\) 参考线 |
| `--no-individual-seeds` | 关闭 | 隐藏单种子散点 |

## 8. 输出文件

### 8.1 `sigma2_runs.csv`

每行对应一次 `(strategy_space, m, seed)` 运行，包含：

- 扫描标识：`strategy_space`、`m`、`seed`；
- 模型参数：`N`、`s`、`r`、`T`、`L`、`lambda_used`；
- 统计窗口：`steps`、`burn_in`、`n_postburn`；
- 实际初始评分 `initial_score` 和 `zero_score_tie`；
- 需求统计：`mean_D`、`mean_D2`、`sigma2`、`sigma`；
- 活跃度、零需求和灭绝诊断；
- 价格回退计数。

这里的 `initial_score` 是实际用于模拟的数值：若用户指定 `--initial-score`，保存该值；否则保存 `r + initial_score_offset`。文件采用 `utf-8-sig` 编码。

### 8.2 `sigma2_summary.csv`

每行对应一个 `(strategy_space, m)`，主要字段为：

- `n_seeds`、`n_valid_seeds`；
- `mean_sigma2_all_seeds`、`sd_sigma2_all_seeds`、`sem_sigma2_all_seeds`；
- `mean_sigma2_nonextinct`、`sd_sigma2_nonextinct`、`sem_sigma2_nonextinct`；
- `mean_sigma_all_seeds`、`sd_sigma_all_seeds`、`sem_sigma_all_seeds`；
- `mean_sigma_nonextinct`、`sd_sigma_nonextinct`、`sem_sigma_nonextinct`；
- `mean_active_time_fraction`、`mean_active_agent_fraction`、`mean_nonzero_demand_fraction`。

最后三项是各单次运行诊断的跨种子算术平均。

### 8.3 `sigma2_scan_metadata.json`

保存：

- 需求与方差的定义；
- 本次扫描的 `N`、`s`、`r`、`T`、`L`、市场深度和初始价格；
- `steps`、`burn_in`、种子列表和扫描范围；
- `initial_score_offset` 与用户传入的 `initial_score`；
- 零偏离规则和价格规则；
- `minimum_checks_sigma2` 与 `minimum_checks_sigma`；
- 有限系统结果的解释提醒。

metadata 中的 `parameters.initial_score` 保留用户输入，未指定时为 `null`，不会像逐运行 CSV 那样替换为 `r+offset`。非有限浮点数在 JSON 中同样转换为 `null`。

## 9. 阅读结果的顺序

1. 先看活跃度、零需求比例和 `n_valid_seeds`，确认曲线是否受到无人交易或灭绝运行影响；
2. 核对 `r`、`T`、`L`、实际初始评分和 `zero_score_tie`，避免把不同动力学的结果直接比较；
3. 查看单种子散点和种子间 SD，再解读均值曲线；
4. 区分边界最低值与严格内部局部极小值；
5. 分开理解“存在离散极小值”和“下降幅度超过合并 1-SEM”；
6. 如需讨论相变，进一步增加种子和模拟长度，并检查不同 `N` 下极小值位置与深度的稳定性。

离散方差曲线的极小值是有限样本、有限系统下的数值特征，不能单独用来证明热力学相变。
