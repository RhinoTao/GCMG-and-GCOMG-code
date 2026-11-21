# GCOMGv2-Frozen-FSS：冻结实际市场多数方向的期望得分版本

本程序在上一版 `GCOMGv2.1-FSS` 基础上构造一个严格的 Frozen-counterfactual 对照模型。保留：

- FSS 显式策略空间（`m<=4`）与隐式抽样策略空间（`m>=5`）；
- 每个 agent 持有 `s` 个策略，按最高分策略决策；
- 有限 Grand-Canonical 参与门槛 `r`；
- 有限分数记忆时间 `T`；
- 原 GCMG 的订单汇总、线性/对数价格更新与滚动公共信息更新；
- agent-specific strategy scores，形状为 `(N,s)`；
- 可选 epsilon-greedy 策略探索，但默认严格设为 `epsilon=0`；
- 不包含 score redistribution。

## 1. 与原 GCOMGv2 的区别

原 GCOMGv2 对每个候选策略分别构造反事实订单簿：活跃 agent 替换自己的实际订单，非活跃 agent 则向实际订单簿加入一张假想订单；随后重新计算该策略自己的 `D_cf, M_cf, n_cf` 和多数方向。

Frozen-counterfactual 版本不再进行这些逐策略的订单簿重算。每轮先由实际订单确定一次：

\[
D_t=N_{buy}(t)-N_{sell}(t),\qquad
M_t=\max(N_{buy},N_{sell}),\qquad
n_t=\min(N_{buy},N_{sell}),
\]

\[
d_t=\operatorname{sgn}(D_t),\qquad
\chi_t=\operatorname{sgn}\!\left(\frac{D_t}{\lambda}-L\right),\qquad
pt_t=\left|\frac{D_t}{\lambda}\right|.
\]

然后把这组实际市场状态冻结，所有 agent 的所有候选策略都面对同一个 `D_t, M_t, n_t, d_t, chi_t, pt_t`。

对于当前公共信息 `mu_t` 下的候选行动 \(a_{iR}^{\mu_t}\)：

\[
\boxed{
 g_{iR}(t+1)=
 \begin{cases}
 \displaystyle \frac{n_t-M_t}{M_t}\,a_{iR}^{\mu_t}\,\chi_t\,pt_t,
 &D_t\ne0\ \text{且}\ a_{iR}^{\mu_t}=d_t,\\[6pt]
 0,&\text{otherwise}.
 \end{cases}}
\]

当 `L=0` 时，符合实际市场多数方向的策略得到非正得分，反方向策略得零分。若实际订单簿平衡或为空，则两种候选行动均得零分。

**核心差别：候选策略不会加入/移除订单，也不会改变本轮市场的多数方向。** 这正是用来检验原 GCOMGv2 的 `D=+/-1` 锁定是否来自“候选行动自己重新定义多数方向”的机制。

## 2. 分数和参与规则

\[
S_{iR}(t+1)=\left(1-\frac1T\right)S_{iR}(t)+g_{iR}(t+1).
\]

每个 agent 以最高分策略参与，条件仍为：

\[
\max_R S_{iR}(t)>r.
\]

未入场 agent 的所有候选策略仍按当前真实市场的冻结状态评分；因此，若市场实际为空，`D=0`，所有策略本轮得分为 0，只有分数衰减项继续作用。

## 3. epsilon 的含义

`--strategy-exploration` 默认值为 `0.0`。当 epsilon 为零时，agent 总是选择当前最高分策略（若最高分并列，仍按原程序随机打破并列；这不是 epsilon-greedy 探索）。

若要做探索对照，可以显式设置：

```powershell
--strategy-exploration 0.005
```

## 4. 推荐运行：对应 N=101, m=4, s=2, T=100, seed=123 的对照

下面命令补充指定了 `r=-0.15`、`initial-score=0`、`lambda=10N=1010`。这是一个示例测试点，不代表唯一或最佳参数；若你此前三种相使用了不同 `r` 或初始分数，应把它们原样代入以做严格比较。

```powershell
python .\gcomg_v2_frozen_fss.py --N 101 --m 4 --s 2 --T 100 --r -0.15 --L 0 --lambda 1010 --x0 100 --price-mode linear --initial-score 0 --strategy-exploration 0 --steps 10000 --seed 123 --burn-in 1000 --outdir run_frozen_N101_m4_s2_T100_seed123
```

建议保持其他参数不变，额外测试多个 seed。不要只凭一条价格路径判断是否形成波动聚集；重点比较收益自相关、绝对收益自相关、需求自相关、活跃 agent 数量和需求同号连续区间长度。

## 5. 输出

- `gcomgv2_frozen_timeseries.csv`：每轮实际需求、买卖数量、活跃人数、冻结市场多数/少数侧人数、与冻结多数方向一致的候选策略比例、均值策略得分、价格和收益。
- `gcomgv2_frozen_diagnostics.png`：价格、活跃人数、标准化收益分布及收益/绝对收益 ACF。
- `gcomgv2_frozen_price_change_distribution.png`：价格增量 PDF 和正态 Q-Q 图。
- `gcomgv2_frozen_summary.json`：参数和诊断汇总。

## 6. 解释限制

Frozen 版本只消除“单个候选行动在自己的反事实订单簿里翻转多数方向”这一反馈通道。它并不保证没有趋势，也不保证出现 volatility clustering。若 `D=+/-1` 仍长时间持续，可能还涉及有限策略持仓、公共信息状态饱和、历史分数差异和有限记忆时间等机制。
