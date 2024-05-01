# linearmodels 6.0
面板数据模型、系统回归、工具变量和资产定价的估计与推断。
最新版本的稳定文档位于 `doc`。最新发展的文档位于 `devel`。

[linearmodels 6.0 (bashtage.github.io)](https://bashtage.github.io/linearmodels/)

> 这个三方库极大的弥补了statsmodel的缺陷，两个搭配起来使用完美！

一些在statsmodels中缺失的常见线性模型的估计和推断

##  linearmodels 概要
### 面板数据模型
- 固定效应 (PanelOLS)
- 随机效应 (RandomEffects)
- 第一差分 (FirstDifferenceOLS)
- 组间估计 (BetweenOLS)
- 混合OLS (PooledOLS)
- Fama-MacBeth 估计 (FamaMacBeth)
### 高维回归
- 吸收最小二乘法 (AbsorbingLS)
### 单方程工具变量(IV)模型
- 二阶段最小二乘法 (2SLS, IV2SLS)
- 有限信息最大似然 (LIML, IVLIML)
- 广义矩估计 (GMM, IVGMM)
- 持续更新GMM (CUE-GMM, IVGMMCUE)
### 系统回归估计量
- 似不相关的回归 (SUR, SUR)
- 三阶段最小二乘法 (3SLS, IV3SLS)
- 广义矩系统估计量 (GMM, IVSystemGMM)
### 资产定价模型估计和测试
- 线性因子模型 (2-step, 适用于交易或非交易因子) (LinearFactorModel)
- 线性因子模型 (GMM, 适用于交易或非交易因子) (LinearFactorModelGMM)
- 线性因子模型 (1-step SUR, 仅适用于交易因子) (TradedFactorModel)

# Panel Data Model Estimation


# 引言
面板数据包括对多个实体（个体、公司、国家）在多个时间段的观察。在面板数据的经典应用中，实体的数量 N 是较大的，而时间段的数量 T 是较小的（通常在 2 到 5 之间）。这些估计器的大多数渐近理论是在假设 N 会发散而 T 是固定的条件下发展起来的。
大多数面板模型旨在估计一个模型的参数，该模型可以描述为：
$$
y_{it} = x_{it}\beta + \alpha_i + \epsilon_{it}
$$
其中 $i$  表示实体，$t$  表示时间。$\beta$ 包含了感兴趣的参数。$\alpha_i$ 是特定于实体的组成部分，通常在标准设置中不识别，因此不能一致地估计，并且是与 \( x_{it} \) 和协变量无关的特定误差。
所有模型都需要两个输入：
- 因变量（dependent）：模型中的被建模变量
- 自变量（exog）：模型中的回归变量
并使用不同的技术来解决 \( \alpha_i \) 的存在。
特别是：
- PanelOLS 使用固定效应（即，实体效应）来消除特定于实体的组成部分。这在数学上等同于为每个实体包含一个虚拟变量，尽管出于性能原因实现时不这样做。
- BetweenOLS 在实体内部求平均，然后使用 OLS 对时间平均值进行回归。
- FirstDifferenceOLS 采用一阶差分来消除特定于实体的效应。
- RandomEffects 使用准差分来有效估计当实体效应与回归变量独立时的情况。但是，当实体效应与回归变量之间存在依赖时，它并不一致。
- PooledOLS 忽略了实体效应，在效应与回归变量独立时是一致的，但在效应与回归变量独立时效率低下。

PanelOLS 比其他估计器更通用，并且可以用来模拟 2 效应（例如，实体效应和时间效应）。
模型规范类似于 statsmodels。这个例子估计了一个固定效应回归，用于一个工作男性的工资面板，将工资的对数建模为经验的平方、一个表示男性是否已婚的虚拟变量，以及一个表示男性是否是工会成员的虚拟变量的函数。

```python
from linearmodels.panel import PanelOLS
from linearmodels.datasets import wage_panel
import statsmodels.api as sm

data = wage_panel.load()
data = data.set_index(['nr','year'])
dependent = data.lwage
exog = sm.add_constant(data[['expersq','married','union']])
mod = PanelOLS(dependent, exog, entity_effects=True)
res = mod.fit(cov_type='unadjusted')
print(res)