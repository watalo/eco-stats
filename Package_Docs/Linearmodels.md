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


```

## Examples

这些示例涵盖了用于估计面板模型的模型。最初的示例都忽略了协方差选项，因此使用了适用于同方差数据的默认经典协方差。其他协方差选项在本文档末尾描述。

### Loading data

这些示例都使用了 F. Vella 和 M. Verbeek (1998) 的工资面板数据，该数据集包含了1980年代男性的工资和特征。实体标识符是 `nr`，时间标识符是 `year`。这个数据在 Jeffrey Wooldridge 的《计量经济学导论》第14章中广泛使用。

这里使用了一个 MultiIndex DataFrame 来以面板格式保存数据。在设置索引之前，创建了一个年份的分类变量，这有助于生成虚拟变量。

```python
import pandas as pd
from linearmodels.datasets import wage_panel
data = wage_panel.load()
year = pd.Categorical(data.year)
data = data.set_index(["nr", "year"])
data["year"] = year
print(wage_panel.DESCR)
print(data.head())
```

### Basic regression on panel data

PooledOLS 就是普通的 OLS，它理解不同的面板数据结构。作为一个基础模型，这很有用。这里使用所有变量和时间虚拟变量对对数工资进行建模。

```python
import statsmodels.api as sm
from linearmodels.panel import PooledOLS
exog_vars = ["black", "hisp", "exper", "expersq", "married", "educ", "union", "year"]
exog = sm.add_constant(data[exog_vars])
mod = PooledOLS(data.lwage, exog)
pooled_res = mod.fit()
print(pooled_res)
```

### Estimating parameters with uncorrelated effects

在建模面板数据时，通常要考虑 OLS 无法有效估计的模型。最常见的是误差分量模型，它在标准 OLS 模型中添加了一个额外的项，该项影响实体 i 的所有值。当与中的回归变量不相关时，可以使用随机效应模型来有效估计此模型的参数。

#### Random effects

随机效应模型与 Pooled OLS 模型几乎相同，但它考虑了模型的结构，因此更有效。随机效应使用一种准差分策略，通过减去实体内部值的时间平均值来考虑共同冲击。

```python
from linearmodels.panel import RandomEffects
mod = RandomEffects(data.lwage, exog)
re_res = mod.fit()
print(re_res)
```

### The between estimator

另一种选择是 between estimator，通常效率较低，但可以用来估计模型参数。它非常简单，首先计算和的时间平均值，然后使用这些平均值运行一个简单的回归。

由于平均值消除了年份的差异，所以去掉了年份虚拟变量。由于与 exper 相当共线性，exper 也被去掉了。这些结果与之前的模型大致相似。

```python
from linearmodels.panel import BetweenOLS
exog_vars = ["black", "hisp", "exper", "married", "educ", "union"]
exog = sm.add_constant(data[exog_vars])
mod = BetweenOLS(data.lwage, exog)
be_res = mod.fit()
print(be_res)
```

### Handling correlated effects

当效应与回归变量相关时，RE 和 BE estimators 是不一致的。通常的解决方案是使用固定效应，这在 PanelOLS 中可用。固定效应称为 entity_effects 应用于实体，time_effects 应用于时间维度：
#### Including fixed effects

通过设置 `entity_effects=True` 来包含实体效应。这等同于为每个实体包含虚拟变量。在这个面板中，这将添加 545 个虚拟变量，模型的估计将大大变慢。PanelOLS 实际上并不使用虚拟变量，而是使用组内差分来达到相同的效果。

#### Time-invariant Variables

使用实体效应时不能包含时间不变变量，因为一旦差分，这些将全部为 0。由于'exper' 在加入实体效应和时间虚拟变量后将完全共线性，因此 'exper' 也被排除在外。

```python
from linearmodels.panel import PanelOLS
exog_vars = ["expersq", "union", "married", "year"]
exog = sm.add_constant(data[exog_vars])
mod = PanelOLS(data.lwage, exog, entity_effects=True)
fe_res = mod.fit()
print(fe_res)
```

### Time Effects

可以使用 `time_effects=True` 添加时间效应。在这里，时间虚拟变量被移除了。请注意，核心系数是相同的。唯一的变化在于可混性检验的统计量，因为现在“效应”包括了实体和时间，而之前只包括了实体。

```python
exog_vars = ["expersq", "union", "married"]
exog = sm.add_constant(data[exog_vars])
mod = PanelOLS(data.lwage, exog, entity_effects=True, time_effects=True)
fe_te_res = mod.fit()
print(fe_te_res)
```

### Other Options

当使用 PanelOLS 和 effects 时，还有一些额外的选项可用。`use_lsdv` 可以用来强制模型使用虚拟变量进行估计。这在大多数情况下是不必要的，并且会更慢。这个选项主要是为了测试。

### Other Effects

PanelOLS 可以用来估计具有多达 2 个效应固定效应的模型。这些可以包括任何组合的：

- 实体效应
- 时间效应
- 其他效应

其他效应是使用 PanelOLS 的 `other_effects` 输入指定的。输入应该与 exog 有相同数量的观测值，并且可以有 1 或两列。下面我们使用其他效应仅设置时间效应来重现模型。

```python
from linearmodels.panel.data import PanelData
exog_vars = ["expersq", "union", "married"]
exog = sm.add_constant(data[exog_vars])
# 使用 `PanelData` 结构来简化构建时间 ID
time_ids = pd.DataFrame(PanelData(data.lwage).time_ids, index=data.index, columns=["Other Effect"])
mod = PanelOLS(data.lwage, exog, entity_effects=True, other_effects=time_ids)
fe_oe_res = mod.fit()
print(fe_oe_res)
```

### Using first differences

当可能存在相关性时，一阶差分是使用固定效应的替代方法。在使用一阶差分时，必须排除时间不变变量。此外，只能包括一个线性时间趋势变量，因为这将看起来像一个常数。这个变量将吸收数据中的所有时间趋势，因此这些变量的解释可能具有挑战性。

```python
from linearmodels.panel import FirstDifferenceOLS
exog_vars = ["exper", "expersq", "union", "married"]
exog = data[exog_vars]
mod = FirstDifferenceOLS(data.lwage, exog)
fd_res = mod.fit()
print(fd_res)
```

### Comparing models

可以使用 `compare` 对模型结果进行比较。`compare` 接受结果的列表，或者是一个结果的字典，其中键被解释为模型名称。

```python
from linearmodels.panel import compare
print(compare({"BE": be_res, "RE": re_res, "Pooled": pooled_res}))
```

### Covariance options

#### Heteroskedasticity Robust Covariance

通过设置 `cov_type="robust"`，可以使用 White 的稳健协方差。这个估计器增加了对某些类型规范问题的鲁棒性，但当使用固定效应（实体效应）时不应使用，因为它不再稳健。相反，需要使用聚类协方差。

#### Clustered by Entity

通常要聚类的变量是实体或实体和时间。可以使用 `cov_type="clustered"` 并使用额外的关键字参数 `cluster_entity=True` 和/或 `cluster_time=True` 来实现。

这个下一个示例通过两者进行聚类。

```python
exog_vars = ["black", "hisp", "exper", "expersq", "married", "educ", "union"]
exog = sm.add_constant(data[exog_vars])
mod = PooledOLS(data.lwage, exog)
robust = mod.fit(cov_type="robust")
clust_entity = mod.fit(cov_type="clustered", cluster_entity=True)
clust_entity_time = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
```

#### An OrderedDict is used to hold the results for comparing models

使用 OrderedDict 来保存比较模型的结果。这样可以命名模型，并且可以指定模型的顺序。一个标准字典将产生有效的随机顺序。

聚类到实体降低了整体的 t 统计量。这表明每个实体的残差中存在重要的相关性。聚类到两者也都降低了 t 统计量，这表明数据中存在横截面依赖性。注意：通过实体聚类解决了时间上的相关性，而通过时间聚类则控制了时间段内实体之间的相关性。

```python
from collections import OrderedDict
res = OrderedDict()
res["Robust"] = robust
res["Entity"] = clust_entity
res["Entity-Time"] = clust_entity_time
print(compare(res))
```