# RSS / FSS 的需求方差扫描：原始 Minority Game 极限

本项目比较 RSS 与 FSS 在讲义 P37 极限条件下的超额需求波动：

\[
L=0,\qquad r\to-\infty,\qquad T\to+\infty.
\]

扫描对象是需求方差 \(\sigma_D^2\) 随记忆长度 `m` 的变化，同时给出标准差 \(\sigma_D\)。程序直接用 `r=-inf` 和 `T=inf` 实现极限，不用大的有限数近似。

两个脚本分别负责计算和绘图：

```text
scan_sigma2.py
    调用 gcmg_rss.py / gcmg_fss.py
    逐策略空间、逐 m、逐 seed 模拟
    计算单次统计量并跨种子聚合
    检查内部局部极小值
    写入 runs / summary / metadata

plot_sigma2.py
    读取扫描结果
    绘制均值、种子散点、误差棒和极小值标记
```

`scan_sigma2.py` 不重新实现 GCMG，而是通过 `make_config()` 固定极限参数，再调用基础模拟器。策略选择、评分、订单和价格行为以这两个基础模块为准。

## 1. 单次模拟的核心动力学

### 1.1 公共信息与策略

记忆长度为 `m` 时，公共信息状态数为

\[
P=2^m,\qquad \mu_t\in\{0,\ldots,P-1\}.
\]

策略 \(R\) 在每个状态下给出固定动作

\[
A_{R,\mu}\in\{-1,+1\},
\]

其中 `-1` 表示卖出，`+1` 表示买入。每个代理在初始化时抽取 `s` 个不同策略，持有集合随后保持不变；不同代理可以持有相同策略。

每轮开始时，代理选择持有策略中评分最高者：

\[
R_i^*(t)\in\arg\max_{R\in\mathcal R_i}S_R(t).
\]

若最高分并列，基础模拟器在并列项中无偏向随机选择。所有策略使用同一份全局评分表，代理只在自己的持有子集上比较评分。

### 1.2 P37 极限如何落到代码上

`make_config()` 对 RSS 和 FSS 都设置：

```python
r=-math.inf
T=math.inf
L=0.0
price_mode="linear"
```

这三个模型条件的直接后果是：

- `r=-inf`：有限的最高评分总满足 `best_score > r`，所以每个代理每轮都交易，`active_agents=N`；
- `T=inf`：基础模拟器计算得到 \(\alpha=1-1/T=1\)，评分不衰减；
- `L=0`：评分和公共信息只由超额需求相对零点的符号驱动。

所有活跃代理各提交一个 `+1` 或 `-1` 订单，清算前净超额需求为

\[
D_t=\sum_i A_{R_i^*(t),\mu_t}
=n_{\rm buy}(t)-n_{\rm sell}(t).
\]

评分更新为

\[
S_R(t+1)=S_R(t)-A_{R,\mu_t}g_t.
\]

在非零需求且超出基础模拟器零值容差时，

\[
g_t=\operatorname{sgn}(D_t/\lambda)=\operatorname{sgn}(D_t),
\]

因为 \(\lambda>0\)。策略评分与价格水平无关，价格也不会反馈到订单或公共信息。

本轮结束后，公共信息按

\[
\mu_{t+1}=(2\mu_t+b_t)\bmod P
\]

更新，其中 \(b_t\) 是本轮追加的二进制位。

### 1.3 零需求分支的实际行为

基础模拟器在 \(|D_t/\lambda|\leq\texttt{tie\_tolerance}\) 时进入零偏离分支。公共信息位和评分信号的处理需要分开看：

- 新公共信息位 `new_bit` 始终随机取 `0` 或 `1`；
- `zero_score_tie="zero"` 时，评分信号 \(g_t=0\)；
- `zero_score_tie="random"` 时，评分信号独立随机取 `+1` 或 `-1`。

因此，`zero` 并不会让公共信息确定性追加 `0`，它只选择字面上的 \(\operatorname{sgn}(0)=0\) 评分规则。扫描脚本默认使用 `random`。

默认 `N=101` 为奇数，所有代理始终各提交一个 \(\pm1\) 订单，所以 \(D_t\) 不可能为零，上述分支不会触发。只有使用偶数 `N`，或数值上因极大的 `lambda` 落入零值容差时，该设置才可能改变轨迹。若要在偶数 `N` 下严格执行零需求不评分，应显式使用：

```bash
--zero-score-tie zero
```

### 1.4 价格规则及其在本扫描中的作用

扫描固定采用线性价格：

\[
x_{t+1}=x_t+\frac{D_t}{\lambda}.
\]

若候选价格非正，当前步骤改用

\[
x_{t+1}=x_t\exp(D_t/\lambda),
\]

下一步恢复线性规则。该回退不改变需求、评分信号或公共信息规则，也不参与 \(\sigma_D^2\) 的计算。

`calculate_observables()` 会读取结果中的 `price_fallback` 并记录回退步数，但两个基础模块的接口不同：

- FSS 结果提供 `price_fallback`，总回退步数和 burn-in 后回退步数有实际含义；
- RSS 结果没有该字段，扫描通过 `getattr(..., None)` 将两项写为 `0`。

所以 CSV 中 RSS 的回退次数只是接口缺省值，不能据此判断 RSS 是否发生过价格回退。

## 2. RSS 与 FSS 的实现路径

### 2.1 RSS

RSS 使用 Sylvester–Walsh–Hadamard 矩阵 \(H_P\)，构造

\[
A=\begin{bmatrix}H_P\\-H_P\end{bmatrix}.
\]

策略总数为 `2P`。前半部分的 `P` 个策略两两正交，后半部分是对应的逆策略。扫描范围内每个 `m` 都显式生成 RSS 矩阵。

### 2.2 FSS

完整 FSS 包含所有长度为 `P` 的 \(\pm1\) 动作向量：

\[
|\mathrm{FSS}|=2^P=2^{2^m}.
\]

基础 FSS 模块按 `m` 切换表示：

- `m <= 4`：生成完整策略矩阵，再为每个代理无放回抽样；
- `m >= 5`：直接生成独立公平比特串，等价于从完整 FSS 均匀抽样；同一代理内拒绝重复，不同代理允许共享策略。

隐式路径只存储实际持有策略，并对所有代理的持有记录去重，以共享动作向量和评分。它没有用任意指定的随机子集替代 FSS；每个代理的持有集合仍相当于从完整策略池均匀无放回抽取。

扫描器对每个策略空间和每个 `m` 只调用一次 `build_rss()` 或 `build_fss()`，随后在不同种子下复用策略空间描述符。代理持有策略的随机抽取仍在各自的 `simulate_gcmg()` 中完成。

## 3. 统计量与跨种子聚合

### 3.1 单次运行

对每次运行，程序舍弃前 `burn_in` 轮，使用剩余 `n` 个需求观测：

\[
\bar D=\frac1n\sum_tD_t,\qquad
\overline{D^2}=\frac1n\sum_tD_t^2,
\]

\[
\sigma_D^2=\overline{D^2}-\bar D^2,
\qquad
\sigma_D=\sqrt{\sigma_D^2}.
\]

这是分母为 `n` 的总体方差，不是样本方差，也没有除以 `N` 或 `N^2`。代码用 `max(0, ...)` 防止浮点舍入产生微小负方差。

单次结果还保存以下审计量：

| 字段 | 计算方式 |
|---|---|
| `n_postburn` | burn-in 后的需求观测数 |
| `mean_D` | \(\bar D\) |
| `mean_D2` | \(\overline{D^2}\) |
| `active_time_fraction` | 活跃代理数大于零的时间比例 |
| `mean_active_agent_fraction` | 平均每轮活跃代理数除以 `N` |
| `nonzero_demand_fraction` | \(D_t\neq0\) 的时间比例 |
| `extinct_after_burn` | burn-in 后活跃代理数是否始终为零 |
| `price_fallback_steps_total` | 全部轮次的价格回退步数，受上述 RSS 接口限制 |
| `price_fallback_steps_after_burn` | burn-in 后的价格回退步数，受上述 RSS 接口限制 |

在预期 P37 极限下，前两个活跃度比例都应为 `1`，`extinct_after_burn` 应为 `0`。默认奇数 `N` 下，`nonzero_demand_fraction` 也应为 `1`。

### 3.2 同一 `m` 下的种子聚合

`aggregate_runs()` 按 `(strategy_space, m)` 分组，分别对每次运行已经算出的 `sigma2` 和 `sigma` 做跨种子统计：

\[
\operatorname{SD}=s_{\rm seeds}\quad(\texttt{ddof=1}),
\qquad
\operatorname{SEM}=\frac{\operatorname{SD}}{\sqrt{n_{\rm seeds}}}.
\]

这里有两套均值：

- `*_all_seeds`：使用全部种子；
- `*_nonextinct`：只使用 `extinct_after_burn=0` 的种子。

主图和极小值检查使用 `nonextinct` 口径。单个种子时，SD 和 SEM 为 `NaN`，JSON 中写成 `null`，绘图时误差棒按零处理。

需要注意：

- \(\sigma_D^2\) 的组均值是各种子方差的平均，不是把各种子需求合并后再求方差；
- \(\sigma_D\) 的组均值是各种子标准差的平均，不等于 \(\sqrt{\overline{\sigma_D^2}}\)；
- SD 和 SEM 描述种子间差异，不是单条时间序列内部的标准误。

## 4. 极小值检查

`detect_minima()` 分别对 `sigma2` 和 `sigma`、RSS 和 FSS 执行相同检查。

一个 `m` 点只有满足以下条件才进入候选集合：

1. 该点所有种子都非灭绝，即 `n_valid_seeds == n_seeds`；
2. 对应均值为有限数；
3. 相邻整数点 `m-1` 和 `m+1` 也满足前两项。

内部局部极小值要求严格满足

\[
y_m<y_{m-1},\qquad y_m<y_{m+1}.
\]

边界点不进入内部极小值列表。程序另外记录所有合格点中的最小值位置，因此边界最低值仍会出现在 `global_minimum_m_among_eligible_points` 中。

### 4.1 三点二次拟合

对每个内部极小值，程序用 `(m-1,y_{m-1})`、`(m,y_m)`、`(m+1,y_{m+1})` 拟合二次函数。只有二次项系数为正且顶点位于左右节点之间时，才返回 `quadratic_vertex_estimate`。

该顶点只是离散扫描的探索性插值，不表示模型允许非整数记忆长度，也不增加实际扫描分辨率。

### 4.2 合并 1-SEM 判据

左右差距分别定义为

\[
\Delta_-=y_{m-1}-y_m,\qquad
\Delta_+=y_{m+1}-y_m.
\]

程序计算

\[
e_-=\sqrt{\mathrm{SEM}_{m-1}^2+\mathrm{SEM}_m^2},
\qquad
e_+=\sqrt{\mathrm{SEM}_{m+1}^2+\mathrm{SEM}_m^2},
\]

只有 \(\Delta_->e_-\) 且 \(\Delta_+>e_+\) 时，字段 `lower_than_both_neighbors_by_more_than_combined_1_SEM` 才为真。

`minimum_verified` 只表示至少找到一个严格内部局部极小值，不要求通过上述 1-SEM 判据。它不能解释为统计显著性或相变证明。

## 5. 绘图脚本如何读取结果

`plot_sigma2.py` 必须读取：

- `sigma2_runs.csv`；
- `sigma2_summary.csv`。

若目录中还有 `sigma2_scan_metadata.json`，程序读取其中的 `N`，供可选参考线使用。绘图输出两张图：

| 文件 | 纵轴 |
|---|---|
| `sigma2_vs_m_RSS_FSS.png` | 非灭绝种子的平均 \(\sigma_D^2\) |
| `sigma_vs_m_RSS_FSS_P39_style.png` | 非灭绝种子的平均 \(\sigma_D\) |

主曲线的误差棒是跨种子 **SD**，不是 SEM。图中还包含：

- 默认显示各个种子的散点，非灭绝种子为实心点，灭绝种子为 `x`；
- 部分种子灭绝的 `m` 点仍绘制非灭绝种子均值，并用空心圈标记；
- 所有种子均灭绝的点不参与主曲线和极小值检查，另以 `x` 显示全种子均值；
- 若存在多个内部极小值，图中标注其中均值最低者，并注明是否通过合并 1-SEM 判据。

`--show-p39-reference` 增加讲义对照线：方差图为 \(N\)，标准差图为 \(\sqrt N\)。这只是 El Farol P39 参考线，不是对数据的标准化。若没有 metadata，程序无法取得 `N`，即使指定该选项也不会画出参考线。

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

### 6.1 默认完整扫描

```bash
python scan_sigma2.py --outdir sigma2_scan
```

默认扫描 RSS 和 FSS 的 `m=1,...,10`，每个点使用种子 `123 456 789`，共执行

\[
(10+10)\times3=60
\]

次模拟，每次 50000 轮。

扫描完成后绘图：

```bash
python plot_sigma2.py --input-dir sigma2_scan
```

添加 P39 参考线：

```bash
python plot_sigma2.py --input-dir sigma2_scan --show-p39-reference
```

隐藏单种子散点，或指定另一绘图目录：

```bash
python plot_sigma2.py \
  --input-dir sigma2_scan \
  --outdir figures_sigma2 \
  --no-individual-seeds
```

### 6.2 小规模运行检查

```bash
python scan_sigma2.py \
  --N 101 --steps 1200 --burn-in 100 --seeds 123 \
  --rss-m-min 1 --rss-m-max 3 \
  --fss-m-min 1 --fss-m-max 3 \
  --outdir smoke_results
```

该命令只用于检查流程，不适合正式统计。正式扫描应保证过渡期和统计窗口足够长，并检查 `m=10` 时窗口相对于 \(2^{10}=1024\) 是否充分。

扫描结果在主循环全部结束后一次性写出，不是每完成一个种子就追加保存；中断运行不会留下部分 CSV 结果。

## 7. 参数表

### 7.1 固定模型条件

| 条件 | 程序值 | 说明 |
|---|---|---|
| `L` | `0.0` | 固定市场参照阈值 |
| `r` | `-inf` | 所有代理始终参与 |
| `T` | `inf` | 评分不衰减 |
| `price_mode` | `linear` | 仅价格非正时单步对数回退 |

这些条件不作为扫描脚本的命令行参数。

### 7.2 `scan_sigma2.py`

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--N` | `101` | 代理人数 |
| `--s` | `2` | 每个代理持有的策略数 |
| `--lambda` | 未设置 | 市场深度；缺省时为 `10*N` |
| `--x0` | `100.0` | 初始价格 |
| `--steps` | `50000` | 每次运行的总轮数 |
| `--burn-in` | `5000` | 统计前舍弃的轮数 |
| `--seeds` | `123 456 789` | 一个或多个随机种子 |
| `--rss-m-min` / `--rss-m-max` | `1` / `10` | RSS 扫描范围 |
| `--fss-m-min` / `--fss-m-max` | `1` / `10` | FSS 扫描范围 |
| `--mu0` | 未设置 | 固定初始公共信息；缺省时由各次运行的种子随机生成 |
| `--initial-score` | `0.0` | 所有策略的初始评分 |
| `--zero-score-tie` | `random` | 零偏离时的评分信号；公共信息位始终随机 |
| `--outdir` | `sigma2_scan` | 扫描输出目录 |

扫描范围必须位于 `1 <= m <= 10`，并要求 `steps > burn-in + 1`，即至少留下两个统计观测。

### 7.3 `plot_sigma2.py`

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
- 模型配置：`N`、`s`、`r`、`T`、`L`、`lambda_used`、`initial_score`、`zero_score_tie`；
- 统计窗口：`steps`、`burn_in`、`n_postburn`；
- 需求统计：`mean_D`、`mean_D2`、`sigma2`、`sigma`；
- 活跃度与零需求诊断；
- 价格回退计数。

CSV 中的 `r` 和 `T` 分别写成字符串 `-inf` 和 `inf`。文件采用 `utf-8-sig` 编码。

### 8.2 `sigma2_summary.csv`

每行对应一个 `(strategy_space, m)`，主要字段为：

- `n_seeds`、`n_valid_seeds`；
- `mean/sd/sem_sigma2_all_seeds`；
- `mean/sd/sem_sigma2_nonextinct`；
- `mean/sd/sem_sigma_all_seeds`；
- `mean/sd/sem_sigma_nonextinct`；
- 三类时间比例或活跃比例的跨种子均值。

这里的 `mean/sd/sem` 表示三个独立字段前缀，实际列名如 `mean_sigma2_nonextinct`、`sd_sigma2_nonextinct` 和 `sem_sigma2_nonextinct`。

### 8.3 `sigma2_scan_metadata.json`

保存：

- 统计量和需求的定义；
- 固定极限参数及实际市场深度；
- 初始评分、零偏离规则和种子列表；
- RSS/FSS 扫描范围；
- `minimum_checks_sigma2` 和 `minimum_checks_sigma`；
- 有限系统结果的解释提醒。

非有限浮点数在写入 JSON 前转换为 `null`。

## 9. 结果检查顺序

阅读扫描结果时，建议先检查模型是否符合预期，再看曲线：

1. 确认 `mean_active_agent_fraction=1`、`active_time_fraction=1`，且没有灭绝运行；
2. 偶数 `N` 下检查零需求比例及 `zero_score_tie`，明确采用的是哪一种评分约定；
3. 检查各种子散点和种子间 SD，避免只看三个种子的均值；
4. 区分边界最小值与严格内部局部极小值；
5. 分开查看“存在离散极小值”和“超过合并 1-SEM”两个结论；
6. 用更多种子、更长窗口和不同 `N` 检查极小值位置是否稳定。

离散扫描中的极小值是有限样本、有限系统下的数值特征，不能单独用来证明热力学相变。
