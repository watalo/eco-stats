# linearmodels 6.0
面板数据模型、系统回归、工具变量和资产定价的估计与推断。最新版本的稳定文档位于 `doc`。最新发展的文档位于 `devel`。
- 地址：[linearmodels 6.0 (bashtage.github.io)](https://bashtage.github.io/linearmodels/)

> 这个三方库极大的弥补了statsmodel的缺陷，两个搭配起来使用完美！包含了一些在statsmodels中缺失的常见线性模型的估计和推断

#  概要
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

# Chapter 1 面板数据模型估计

## 1.引言
面板数据包括对多个实体（个体、公司、国家）在多个时间段的观察。在面板数据的经典应用中，实体的数量 N 是较大的，而时间段的数量 T 是较小的（通常在 2 到 5 之间）。这些估计法的大多数渐近理论是在假设 N 会发散而 T 是固定的条件下发展起来的。

大多数面板模型旨在估计一个模型的参数，该模型可以描述为：
$$
y_{it} = x_{it}\beta + \alpha_i + \epsilon_{it}
$$
其中 $i$  表示实体，$t$  表示时间。$\beta$ 包含了感兴趣的参数。$\alpha_i$ 是特定于实体的组成部分，通常在标准设置中不识别，因此不能一致地估计，并且是与$x_{it}$ 和协变量无关的特定误差。

所有模型都需要两个输入：
- 因变量（`dependent`）：模型中的被解释变量
- 自变量（`exog`）：模型中的解释变量

并使用不同的技术来解决 $\alpha_i$ 的存在。
特别是：
- `PanelOLS` 使用固定效应（即，实体固定效应）来消除特定于实体的组成部分。这在数学上等同于为每个实体包含一个虚拟变量，尽管出于性能原因实现时不这样做。  ——LSDV法
- `BetweenOLS` 在实体内部求平均，然后使用 OLS 对时间平均值进行回归。——组间估计法
- `FirstDifferenceOLS` 采用一阶差分来消除个体固定效应。
- `RandomEffects` 使用准差分来有效估计当实体效应与回归变量独立时的情况。但是，当实体效应与回归变量之间存在依赖时，它并不一致。
- `PooledOLS` 忽略了实体效应，在效应与回归变量独立时是一致的，但在效应与回归变量独立时效率低下。

`PanelOLS` 比其他估计函数更通用，并且可以用来模拟 2 效应（例如，实体效应和时间效应）。
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
## 2.数据面板分析的数据格式

表达数据有两种主要方法：
- 使用`MultiIndex DataFrames`，其中外索引是实体，内索引是时间索引。这需要使用pandas。
- 3D结构，其中维度0（外）是变量，维度1是时间索引，维度2是实体索引。也可以使用2D数据结构，其维度为(t, n)，被视为具有维度(1, t, n)的3D数据结构。这些3D数据结构可以是pandas、NumPy或xarray。
### 2.1.MultiIndex DataFrames

使用最精确的数据格式是`MultiIndex DataFrame`。这是最精确的，因为只有单列可以保留面板中的所有类型。例如，使用`pandas.Panel`模块时，不可能将单个分类变量跨越多列。

以下示例使用职业培训数据构建`MultiIndex DataFrame`，使用`set_index`命令。实体索引是`'fcode'`，时间索引是`'year'`。

```python
from linearmodels.datasets import jobtraining
data = jobtraining.load()
print(data.head())
```

这里使用`set_index`设置`MultiIndex`设置'`fcode`'（实体）和'`year`'（时间）。

```python
mi_data = data.set_index(["fcode", "year"])
print(mi_data.head())
```

`MultiIndex DataFrame`可用于初始化模型。当仅引用单个系列时，可以使用MultiIndex Series表示。

```python
from linearmodels import PanelOLS
mod = PanelOLS(mi_data.lscrap, mi_data.hrsemp, entity_effects=True)
print(mod.fit())
```

### 2.2.NumPy 数组

3D NumPy数组可用于处理面板数据，其中三个轴分别为0（项目）、1（时间）和2（实体）。NumPy数组通常不是数据的最佳格式，因为所有结果都使用通用变量名。

Pandas在0.25版本中放弃了对Panel的支持。

```python
import numpy as np
np_data = np.asarray(mi_data)
np_lscrap = np_data[:, mi_data.columns.get_loc("lscrap")]
np_hrsemp = np_data[:, mi_data.columns.get_loc("hrsemp")]
nentity = mi_data.index.levels[0].shape[0]
ntime = mi_data.index.levels[1].shape[0]
np_lscrap = np_lscrap.reshape((nentity, ntime)).T
np_hrsemp = np_hrsemp.reshape((nentity, ntime)).T
np_hrsemp.shape = (1, ntime, nentity)
```

```python
res = PanelOLS(np_lscrap, np_hrsemp, entity_effects=True).fit()
print(res)
```

### 2.3.xarray DataArrays

xarray是用于数据结构的相对较新的软件包。`xarray`提供的数据结构与面板模型相关，因为`pandas.Panel`计划在未来移除，因此唯一可行的3D数据格式将是`xarray`。`DataArray.DataArrays`类似于`pandas.Panel`，尽管`DataArrays`使用一些不同的表示法。原则上，可以在`DataArray`中表达与`Panel`相同的信息。

```python
da = mi_data.to_xarray()
da.keys()
```

```python
res = PanelOLS(da["lscrap"].T, da["hrsemp"].T, entity_effects=True).fit()
print(res)
```

### 2.4.将分类和字符串转换为虚拟变量

分类或字符串变量被视为因子，因此被转换为虚拟变量。始终删除第一类。如果不希望这样做，你应该在估计模型之前手动将数据转换为虚拟变量。

```python
import pandas as pd
year_str = mi_data.reset_index()["year"].astype("str")
# year_cat = pd.Categorical(year_str.iloc[:, 0]) # 原文中使用报错
year_cat = pd.Categorical(year_str.iloc[:])
year_str.index = mi_data.index
year_cat.index = mi_data.index
mi_data["year_str"] = year_str
mi_data["year_cat"] = year_cat
```

在这里，`'year'`字段已经被转换成了字符串，随后在模型中使用这些字符串来生成年份虚拟变量。

```python
print("Exogenous variables")
print(mi_data[["hrsemp", "year_str"]].head())
print(mi_data[["hrsemp", "year_str"]].dtypes)
res = PanelOLS(mi_data[["lscrap"]], mi_data[["hrsemp", "year_str"]], entity_effects=True).fit()
print(res)
```

使用分类变量具有相同的效果。

```python
print("Exogenous variables")
print(mi_data[["hrsemp", "year_cat"]].head())
print(mi_data[["hrsemp", "year_cat"]].dtypes)
res = PanelOLS(mi_data[["lscrap"]], mi_data[["hrsemp", "year_cat"]], entity_effects=True).fit()
print(res)
```


## 3.示例

这些示例涵盖了用于估计面板模型的模型。最初的示例都忽略了协方差选项，因此使用了适用于同方差数据的默认经典协方差。其他协方差选项在本文档末尾描述。

### 3.1.载入数据

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

### 3.2.基础回归

PooledOLS 就是普通的 OLS，它理解不同的面板数据结构。作为一个基础模型，这很有用。这里使用所有变量和时间虚拟变量对`'lawge'`进行建模。

```python
import statsmodels.api as sm
from linearmodels.panel import PooledOLS
exog_vars = ["black", "hisp", "exper", "expersq", "married", "educ", "union", "year"]
exog = sm.add_constant(data[exog_vars])
mod = PooledOLS(data.lwage, exog)
pooled_res = mod.fit()
print(pooled_res)
```
### 3.3.随机效应的参数估计
Estimating parameters with uncorrelated effects

在建模面板数据时，通常要考虑 OLS 无法有效估计的模型。最常见的是误差分量模型，它在标准 OLS 模型中添加了一个额外的项，该项影响实体 i 的所有值。当与中的回归变量不相关时，可以使用随机效应模型来有效估计此模型的参数。

### 随机效应
Random effects

随机效应模型与 Pooled OLS 模型几乎相同，但它考虑了模型的结构，因此更有效。随机效应使用一种准差分策略，通过减去实体内部值的时间平均值来考虑共同冲击。

```python
from linearmodels.panel import RandomEffects
mod = RandomEffects(data.lwage, exog)
re_res = mod.fit()
print(re_res)
```

## 组间估计器
The between estimator

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

## Handling correlated effects

当效应与回归变量相关时，RE 和 BE estimators 是不一致的。通常的解决方案是使用固定效应，这在 PanelOLS 中可用。固定效应称为 entity_effects 应用于实体，time_effects 应用于时间维度：
### Including fixed effects

通过设置 `entity_effects=True` 来包含实体效应。这等同于为每个实体包含虚拟变量。在这个面板中，这将添加 545 个虚拟变量，模型的估计将大大变慢。PanelOLS 实际上并不使用虚拟变量，而是使用组内差分来达到相同的效果。

### Time-invariant Variables

使用实体效应时不能包含时间不变变量，因为一旦差分，这些将全部为 0。由于'exper' 在加入实体效应和时间虚拟变量后将完全共线性，因此 'exper' 也被排除在外。

```python
from linearmodels.panel import PanelOLS
exog_vars = ["expersq", "union", "married", "year"]
exog = sm.add_constant(data[exog_vars])
mod = PanelOLS(data.lwage, exog, entity_effects=True)
fe_res = mod.fit()
print(fe_res)
```

## Time Effects

可以使用 `time_effects=True` 添加时间效应。在这里，时间虚拟变量被移除了。请注意，核心系数是相同的。唯一的变化在于可混性检验的统计量，因为现在“效应”包括了实体和时间，而之前只包括了实体。

```python
exog_vars = ["expersq", "union", "married"]
exog = sm.add_constant(data[exog_vars])
mod = PanelOLS(data.lwage, exog, entity_effects=True, time_effects=True)
fe_te_res = mod.fit()
print(fe_te_res)
```

## Other Options

当使用 PanelOLS 和 effects 时，还有一些额外的选项可用。`use_lsdv` 可以用来强制模型使用虚拟变量进行估计。这在大多数情况下是不必要的，并且会更慢。这个选项主要是为了测试。

## Other Effects

PanelOLS 可以用来估计具有多达 2 个效应固定效应的模型。这些可以包括任何组合的：

- 实体效应
- 时间效应
- 其他效应

其他效应是使用 PanelOLS 的 `other_effects` 输入指定的。输入应该与 exog 有相同数量的观测值，并且可以有 1 或两列。下面我们使用 `other_effects` 仅设置时间效应来重现模型。

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

## Using first differences

当可能存在相关性时，一阶差分是使用固定效应的替代方法。在使用一阶差分时，必须排除时间不变变量。此外，只能包括一个线性时间趋势变量，因为这将看起来像一个常数。这个变量将吸收数据中的所有时间趋势，因此这些变量的解释可能具有挑战性。

```python
from linearmodels.panel import FirstDifferenceOLS
exog_vars = ["exper", "expersq", "union", "married"]
exog = data[exog_vars]
mod = FirstDifferenceOLS(data.lwage, exog)
fd_res = mod.fit()
print(fd_res)
```

## Comparing models

可以使用 `compare` 对模型结果进行比较。`compare` 接受结果的列表，或者是一个结果的字典，其中键被解释为模型名称。

```python
from linearmodels.panel import compare
print(compare({"BE": be_res, "RE": re_res, "Pooled": pooled_res}))
```

## Covariance options

### Heteroskedasticity Robust Covariance

通过设置 `cov_type="robust"`，可以使用 White 的稳健协方差。这个估计器增加了对某些类型规范问题的鲁棒性，但当使用固定效应（实体效应）时不应使用，因为它不再稳健。相反，需要使用聚类协方差。

### Clustered by Entity

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

### An OrderedDict is used to hold the results for comparing models

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

以下是文档的翻译，使用Markdown格式，并保留了文中的代码：

---

# 使用公式指定模型

所有模型都可以使用公式来指定。这里使用的公式语法与statsmodels中的类似。单变量回归的基本公式语法是：

在使用BetweenOLS、PooledOLS和RandomEffects时，使用的公式是完全标准的，与statsmodels中的相同。FirstDifferenceOLS几乎相同，但需要注意的是，模型不能包含截距。

PanelOLS实现了效果（实体、时间或其他），在公式中有一个小扩展，允许将实体效果或时间效果（或两者）作为公式的一部分来指定。虽然不可能使用公式接口指定其他效果，但可以在使用公式时作为可选参数包含它们。

## 加载和准备数据

当使用公式时，需要一个MultiIndex pandas dataframe，其中索引是实体-时间。这里使用来自statsmodels的“企业投资的决定因素”提供的Grunfeld数据来说明公式的使用。该数据集包含有关企业投资、市值和厂房资本存量的数据。

使用set_index使用数据集中的变量设置索引。

```python
from statsmodels.datasets import grunfeld
data = grunfeld.load_pandas().data
data = data.set_index(["firm", "year"])
print(data.head())
```

## PanelOLS带有实体效果

使用特殊命令EntityEffects来指定实体效果。默认情况下不包括常数，因此如果需要常数，应在公式中包含1+。当包含效果时，无论是否包含常数，模型和拟合都是相同的。

### PanelOLS带有实体效果和常数

可以使用1+符号显式包含常数。当在模型中包含常数时，会施加一个额外的约束，即效果的数量为0。这允许使用因变量和回归变量的总均值来识别常数。

```python
from linearmodels import PanelOLS
mod = PanelOLS.from_formula("invest ~ value + capital + EntityEffects", data=data)
print(mod.fit())
```

## Panel带有实体和时间效果

同样，可以使用TimeEffects来包含时间效果。在许多模型中，时间效果可以一致地估计，因此它们可以等价地作为分类变量包含在回归器集合中。

```python
mod = PanelOLS.from_formula("invest ~ 1 + value + capital + EntityEffects + TimeEffects", data=data)
print(mod.fit())
```

## Between OLS

其他面板模型是直接的，并包含在内以确保完整性。

```python
from linearmodels import BetweenOLS, FirstDifferenceOLS, PooledOLS
mod = BetweenOLS.from_formula("invest ~ 1 + value + capital", data=data)
print(mod.fit())
```

## First Difference OLS

首次差分模型绝不应包含常数，因为这在差分后是不可知的。

```python
mod = FirstDifferenceOLS.from_formula("invest ~ value + capital", data=data)
print(mod.fit())
```

## Pooled OLS

混合OLS估计器是PanelOLS的一个特例，当没有效果时。它实际上与statsmodels中的OLS（或WLS）相同，但为了完整性而包含。

```python
mod = PooledOLS.from_formula("invest ~ 1 + value + capital", data=data)
print(mod.fit())
```


以下是文档的翻译，使用Markdown格式，并保留了文中的代码：

---

# 与pandas PanelOLS和FamaMacBeth的比较

pandas在0.18版本中弃用了PanelOLS (`pandas.stats.plm.PanelOLS`) 和 FamaMacBeth (`pandas.stats.plm.FamaMacBeth`)，并在0.20版本中完全移除了它们。`linearmodels.panel.model.PanelOLS` 和 `linearmodels.panel.model.FamaMacBeth` 提供了类似的功能集，但有一些值得注意的区别：

1. 当使用MultiIndex DataFrame时，这个包期望MultiIndex是实体-时间的形式。pandas使用的是时间-实体。使用一行代码可以简单地将一个转换为另一个。

```python
data = data.reset_index().set_index(['entity','time'])
```

2. 在linearmodels中，效果是通过差分实现的，因此即使是非常大的模型（100000个实体+）也可以快速估计。pandas中的版本使用了LSDV，在大模型中是不可行的，并且在中等大小的模型中可能会很慢。

3. 效果不是显式估计的，也不会在模型摘要中报告。效果通常不一致（例如，大N面板中的实体效果），因此通常没有意义将它们与参数估计一起报告。可以一致估计的效果可以作为虚拟变量包含（例如，在固定T但大N的模型中的时间效果）。

4. R-squared的定义不同。linearmodels中的默认R-squared报告了在移除包含的效果后的拟合。PanelOLS还提供了一组使用替代模型测量模型参数拟合度的R-squared度量。这些仅对仅包含实体效果的模型有意义。在`rsquared_inclusive`属性中提供了一个与pandas中R-squared类似定义的R-squared度量。

5. 模型首先被指定，然后使用`fit()`方法明确拟合。拟合选项（如协方差估计器的选择）在拟合模型时提供。

6. 如果需要截距，必须显式包含。如果估计了一个带有效果而没有截距的模型，那么将包含所有效果。如果估计了一个带有截距的模型，那么估计的模型将通过使用效果总和为0的限制来调整，以便截距有意义。

7. 其他统计数据（如F统计量）不同，因为不一致的效果没有包含在测试统计量中。



这里使用投资的经典Grunfeld数据展示了差异。

```python
import numpy as np
from statsmodels.datasets import grunfeld
data = grunfeld.load_pandas().data
data.year = data.year.astype(np.int64)
from linearmodels import PanelOLS
etdata = data.set_index(['firm','year'])
PanelOLS(etdata.invest, etdata[['value','capital']], entity_effects=True).fit()
```

PanelOLS估计摘要：


参数估计：


已弃用的pandas PanelOLS的调用类似。注意使用时间-实体数据格式。

输出格式非常不同。

```python
etdata = data.set_index(['year','firm'])
from pandas.stats import plm
plm.PanelOLS(etdata['invest'], etdata[['value','capital']], entity_effects=True)
```

回归分析摘要：

以下是文档的翻译，使用Markdown格式，并保留了文中的代码和超链接：

---

# 模块引用

## 面板数据模型

- `PanelOLS(dependent, exog, *[, weights, ...])`: 用于面板数据的一维和二维固定效应估计器。
- `RandomEffects(dependent, exog, * [, weights, ...])`: 用于面板数据的一维随机效应模型。
- `BetweenOLS(dependent, exog, *[, weights, ...])`: 用于面板数据的Between估计器。
- `FirstDifferenceOLS(dependent, exog, *[, ...])`: 用于面板数据的首次差分模型。
- `PooledOLS(dependent, exog, *[, weights, ...])`: 用于面板数据的混合系数估计器。
- `FamaMacBeth(dependent, exog, *[, weights, ...])`: 用于面板数据的混合系数估计器。

## 估计结果

- `FamaMacBethResults(res)`: Fama MacBeth面板数据模型的结果容器。
- `PanelResults(res)`: 不包括效应的面板数据模型的结果容器。
- `PanelEffectsResults(res)`: 包括效应的面板数据模型的结果容器。
- `RandomEffectsResults(res)`: 随机效应面板数据模型的结果容器。
- `PanelModelComparison(results, *[, ...])`: 比较多个模型。
- `compare(results, *[, precision, stars])`: 比较多个模型的结果。

## 面板模型协方差估计器

- `HomoskedasticCovariance(y, x, params, [..., ...])`: 同方差协方差估计。
- `HeteroskedasticCovariance(y, x, params, ...)`: 使用White估计器的异方差协方差估计。
- `ClusteredCovariance(y, x, params, [..., ...])`: 一维（Rogers）或二维聚类协方差估计。
- `DriscollKraay(y, x, params, entity_ids, ...)`: Driscoll-Kraay异方差自相关稳健协方差估计。
- `ACCovariance(y, x, params, entity_ids, ...)`: 自相关稳健协方差估计。
- `FamaMacBethCovariance(y, x, params, [..., ...])`: Fama-MacBeth估计器的HAC估计器。

## 面板数据结构

- `PanelData(x[, var_name, convert_dummies, ...])`: 处理面板数据的替代格式的抽象。
- `_Panel(df)`: 将MI DataFrame转换为3维结构，其中列是项目。

## 测试数据生成

- `generate_panel_data([nentity, ntime, nexog, ...])`: 为测试模拟面板数据。
- `PanelModelData(data, weights, other_effects, ...)`: 用于持有模拟面板数据的类型化命名元组。

---

[Module Reference - linearmodels 6.0](https://bashtage.github.io/linearmodels/panel/reference.html)

