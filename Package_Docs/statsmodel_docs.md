# 文档摘要

--- 
 statsmodels 0.14.1

> 使用kimi.ai生成的翻译文档

---
## 用户指南

#### 背景
- **内生变量和外生变量**：介绍了统计建模中的基本概念，如内生变量（endog）和外生变量（exog）。
#### 导入路径和结构
- 讨论了如何导入statsmodels库的不同模块和结构。
#### 使用R风格的公式拟合模型
- 解释了如何使用类似于R语言的公式语法来拟合统计模型。
#### 陷阱
- 提供了一些在使用statsmodels时可能遇到的问题和注意事项。
#### 回归和线性模型
- **线性回归**：介绍了线性回归模型的基本概念和使用方法。
- **广义线性模型**：讨论了广义线性模型（GLMs）的特点和应用。
- **广义估计方程**：介绍了使用广义估计方程（GEE）的模型。
- **广义可加模型（GAM）**：解释了广义可加模型（GAM）的构建和拟合。
- **稳健线性模型**：讨论了稳健线性模型的特点和使用场景。
- **线性混合效应模型**：介绍了线性混合效应模型的理论和实现。
- **离散因变量回归**：讨论了当因变量是离散的时候使用回归模型的方法。
- **广义线性混合效应模型**：介绍了广义线性混合效应模型的理论和实现。
- **方差分析（ANOVA）**：提供了方差分析的使用方法和解释。
#### 其他模型
- **时间序列分析**：介绍了时间序列分析的方法，包括`tsa`模块和状态空间方法。
- **向量自回归**：讨论了`tsa.vector_ar`模块，用于向量自回归模型。
- **生存和持续时间分析的方法**：介绍了非参数方法和广义矩估计（GMM）等用于生存和持续时间分析的技术。
- **多变量统计**：讨论了多变量统计方法和工具。
#### 统计和工具
- **统计**：提供了统计测试和模型检验的工具。
- **列联表**：介绍了列联表的分析方法。
- **链式方程多重插补**：讨论了使用链式方程进行多重插补的技术。
- **处理效应**：介绍了处理效应分析的方法。
- **经验似然**：讨论了经验似然方法的应用。
- **图形**：提供了数据可视化的工具。
- **输入输出**：介绍了数据的输入输出方法。
#### 处理大数据集
- **优化**：讨论了在处理大数据集时的优化技术。
- **数据集**：介绍了`datasets`包，用于加载和使用标准数据集。
- **沙盒**：提供了一个实验性功能和代码的测试区域。
---
## 快速入门

这是一个非常简单的案例研究，旨在帮助你快速上手statsmodels。
### 加载模块和函数
安装statsmodels及其依赖项后，我们需要加载一些模块和函数：

```python
import statsmodels.api as sm
import pandas
from patsy import dmatrices
```

- pandas基于numpy数组提供丰富的数据结构和数据分析工具。pandas.DataFrame函数提供带有标签的数组（可能是异构的），类似于R语言中的“data.frame”。pandas.read_csv函数可用于将逗号分隔值文件转换为DataFrame对象。
- patsy是一个Python库，用于描述统计模型并使用R语言风格的公式构建设计矩阵。

### 数据
我们下载了Guerry数据集，这是一组用于支持安德烈-米歇尔·古埃里1833年《法国道德统计论文》的历史数据。数据集由Rdatasets仓库以逗号分隔值格式（CSV）在线托管。我们本可以本地下载文件，然后使用read_csv加载它，但pandas为我们自动处理了这一切：

```python
df = sm.datasets.get_rdataset("Guerry", "HistData").data
```

我们选择感兴趣的变量，并查看底部的5行：

```python
df = df[['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']]
df = df.dropna()
```

### 动机和模型
我们想要了解法国86个省的识字率是否与1820年代皇家彩票的人均赌注有关。我们需要控制每个省的财富水平，并希望在我们的回归方程的右侧包括一系列虚拟变量，以控制由于地区效应导致的未观察到的异质性。
使用普通最小二乘回归（OLS）估计模型。

### 设计矩阵（内生和外生）
为了拟合statsmodels覆盖的大多数模型，你需要创建两个设计矩阵。第一个是内生变量矩阵（即依赖的、响应的、回归量等）。第二个是外生变量矩阵（即独立的、预测的、回归因子等）。OLS系数估计如常计算：

```python
vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
df = df[vars]
y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')
```

### 模型拟合和摘要
在statsmodels中拟合模型通常涉及3个简单步骤：

1. 使用模型类描述模型
2. 使用类方法拟合模型
3. 使用摘要方法检查结果

```python
mod = sm.OLS(y, X)  # 描述模型
res = mod.fit()  # 拟合模型
print(res.summary())  # 总结模型
```

OLS回归结果摘要：

```
Dep. Variable: Lottery R-squared: 0.338
Model: OLS Adj. R-squared: 0.287
Method: Least Squares F-statistic: 6.636
Date: Thu, 14 Dec 2023 Prob (F-statistic): 1.07e-05
Time: 14:55:33 Log-Likelihood: -375.30
No. Observations: 85 AIC: 764.6 Df Residuals: 78 BIC: 781.7 Df Model: 6 Covariance Type: nonrobust
...
```

### 诊断和规范检验
statsmodels允许你进行一系列有用的回归诊断和规范检验。例如，应用Rainbow检验线性（零假设是关系正确建模为线性）：

```python
res.rsquared  # R-squared值
sm.stats.linear_rainbow(res)  # 应用Rainbow检验
```

statsmodels还提供图形函数。例如，我们可以绘制一组回归因子的偏回归图：

```python
sm.graphics.plot_partregress('Lottery', 'Wealth', ['Region', 'Literacy'], data=df, obs_labels=False)
```

---
# OLS模型拟合结果的属性与方法
```python
results = sm.OLS(y,X).fit()
```
## 方法

| 方法                                                  | 描述                                             |
| --------------------------------------------------- | ---------------------------------------------- |
| `compare_f_test(restricted)`                        | 使用F检验来测试限制模型是否正确。                              |
| `compare_lm_test(restricted[, demean, use_lr])`     | 使用拉格朗日乘数检验来测试一组线性限制。                           |
| `compare_lr_test(restricted[, large_sample])`       | 进行似然比检验以判断限制模型是否正确。                            |
| `conf_int([alpha, cols])`                           | 计算拟合参数的置信区间。                                   |
| `conf_int_el(param_num[, sig, upper_bound, ...])`   | 使用经验似然方法计算置信区间。                                |
| `cov_params([r_matrix, column, scale, cov_p, ...])` | 计算方差/协方差矩阵。                                    |
| `el_test(b0_vals, param_nums[, ...])`               | 使用经验似然方法测试单个或联合假设。                             |
| `f_test(r_matrix[, cov_p, invcov])`                 | 计算联合线性假设的F检验。                                  |
| `get_influence()`                                   | 计算影响和异常值指标。                                    |
| `get_prediction([exog, transform, weights, ...])`   | 计算预测结果。                                        |
| `get_robustcov_results([cov_type, use_t])`          | 创建一个以稳健协方差为默认的新结果实例。                           |
| `info_criteria(crit[, dk_params])`                  | 为模型返回一个信息准则。                                   |
| `initialize(model, params, **kwargs)`               | 初始化（可能重新初始化）一个结果实例。                            |
| `load(fname)`                                       | 加载一个pickled的结果实例。                              |
| `normalized_cov_params()`                           | 查看特定模型类的文档字符串。                                 |
| `outlier_test([method, alpha, labels, order, ...])` | 根据方法测试观测值是否为异常值。                               |
| `predict([exog, transform])`                        | 使用`self.model.predict`与`self.params`作为第一个参数调用。 |
| `remove_data()`                                     | 从结果和模型中移除数据数组和所有nobs数组。                        |
| `save(fname[, remove_data])`                        | 保存这个实例的pickle。                                 |
| `scale()`                                           | 作为协方差矩阵的尺度因子。                                  |
| `summary([yname, xname, title, alpha, slim])`       | 总结回归结果。                                        |
| `summary2([yname, xname, title, alpha, ...])`       | 实验性总结函数，用于总结回归结果。                              |
| `t_test(r_matrix[, cov_p, use_t])`                  | 为形式为Rb = q的每个线性假设计算t检验。                        |
| `t_test_pairwise(term_name[, method, alpha, ...])`  | 执行具有多重测试校正p值的成对t检验。                            |
| `wald_test(r_matrix[, cov_p, invcov, use_f, ...])`  | 计算联合线性假设的Wald检验。                               |
| `wald_test_terms([skip_single, ...])`               | 计算跨越多列项的一系列Wald检验。                             |

## 属性
| 属性                 | 描述                                     |
| ------------------ | -------------------------------------- |
| `HC0_se`           | White's (1980) 异方差稳健标准误差。              |
| `HC1_se`           | MacKinnon 和 White 的 (1985) 异方差稳健标准误差。  |
| `HC2_se`           | MacKinnon 和 White 的 (1985) 异方差稳健标准误差。  |
| `HC3_se`           | MacKinnon 和 White 的 (1985) 异方差稳健标准误差。  |
| `aic`              | 赤池信息准则（Akaike's information criteria）。 |
| `bic`              | 贝叶斯信息准则（Bayes' information criteria）。  |
| `bse`              | 参数估计的标准误差。                             |
| `centered_tss`     | 围绕均值的总（加权）平方和。                         |
| `condition_number` | 返回外生矩阵的条件数。                            |
| `cov_HC0`          | 异方差稳健协方差矩阵。                            |
| `cov_HC1`          | 异方差稳健协方差矩阵。                            |
| `cov_HC2`          | 异方差稳健协方差矩阵。                            |
| `cov_HC3`          | 异方差稳健协方差矩阵。                            |
| `eigenvals`        | 返回按降序排列的特征值。                           |
| `ess`              | 解释的平方和。                                |
| `f_pvalue`         | F统计量的p值。                               |
| `fittedvalues`     | 原始（未漂白）设计的预测值。                         |
| `fvalue`           | 完全指定模型的F统计量。                           |
| `llf`              | 模型的对数似然。                               |
| `mse_model`        | 模型的均方误差。                               |
| `mse_resid`        | 残差的均方误差。                               |
| `mse_total`        | 总均方误差。                                 |
| `nobs`             | 观测数n。                                  |
| `pvalues`          | 参数t统计量的双尾p值。                           |
| `resid`            | 模型的残差。                                 |
| `resid_pearson`    | 归一化以具有单位方差的残差。                         |
| `rsquared`         | 模型的R平方。                                |
| `rsquared_adj`     | 调整后的R平方。                               |
| `ssr`              | （漂白的）残差平方和。                            |
| `tvalues`          | 给定参数估计的t统计量。                           |
| `uncentered_tss`   | 未中心化的平方和。                              |
| `use_t`            | 标志，表示在推断中是否使用学生分布。                     |
| `wresid`           | 转换/漂白的回归变量和回归因子的残差。                    |
### `t_test()`的用法
计算 Rb = q 形式的每个线性假设的 t 检验。b代表回归参数。
##### 参数
**r_matrix**：{array_like, `str`，`tuple`}
- 其中：
	- array：如果给出一个数组R， $p \times k$ 的2维数组或长度 k 的1维数组用于明确线性限制。
		- 这个线性组合Rb=0
			- 例如，有3个回归系数，b=(b_1，b_2, b_3)，那么：
			- 参数可以使用，`R = [0, 1, 0]`
			- 等于$H_0:b_2=0$
			- 相当于`r_matrix = (R, q=0)`
	- str ：要测试的完整假设可以作为字符串给出。 请参阅示例。
	- tuple ：形式为 （R， q） 的数组元组。如果给出 q， 可以是标量或长度 p 行向量。

**cov_p**：array_like，可选参数
- 参数协方差矩阵的替代估计值。 如果给出 None，则使用 self.normalized_cov_params。

**use_t**：bool，可选参数
- 如果use_t为None，则使用模型的默认值。
- 如果use_t为True，则 p 值基于 t 分布。
- 如果use_t为False，则 p 值基于正态值分配。
#### 返回
ContrastResults
- 测试的结果是此结果实例的属性。 可用结果具有与参数表相同的元素在`summary()`中。

### `f_test()`的用法
用于检验线性假设的方法

参数与t_test()一致

返回值只有一个数值。

### `get_robustcov_results()`的用法

```python
OLSResults.get_robustcov_results(
	 cov_type='HC1', 
	 use_t=None, 
	 **kwargs
 )
```

创建一个具有稳健协方差矩阵的新结果实例作为默认值。
#### 参数 (Parameters)
- `cov_type`: str
  使用哪种稳健的三明治估计器。详见下面的注释(Notes)。

- `use_t`: bool
  如果为真，则使用t分布进行推断。如果为假，则使用正态分布。如果`use_t`为`None`，则使用适当的默认值，当`cov_type`是非稳健的，`use_t`为真；在所有其他情况下为假。

- `**kwargs`: 
  稳健协方差计算所需的必需或可选参数。详见下面的注释(Notes)。
#### 返回值 (Returns)
- `RegressionResults`
  该方法创建一个新的结果实例，将请求的稳健协方差作为参数的默认协方差。像p值和假设检验这样的推断统计将基于这个协方差矩阵。
#### 注释 (Notes)
- 目前可用的协方差类型和所需的或可选的参数如下：
  - `'fixed scale'` 使用预定义的尺度
    - `scale`: float, 可选
      设置尺度的参数。默认为1。
  - `'HC0'`, `'HC1'`, `'HC2'`, `'HC3'`: 异方差性稳健协方差
    - 无需关键词参数
  - `'HAC'`: 异方差性-自相关稳健协方差
    - `maxlags`: 整数, 必需
      使用的滞后数
    - `kernel`: {可调用对象, str}, 可选
      目前可用的核函数是 `['bartlett', 'uniform']`，默认是Bartlett
    - `use_correction`: bool, 可选
      如果为真，使用小样本校正
  - `'cluster'`: 聚类协方差估计器
    - `groups`: array_like[int], 必需
      聚类或组的整数值索引
    - `use_correction`: bool, 可选
      如果为真，三明治协方差计算时使用小样本校正。如果为假，计算时不使用小样本校正
    - `df_correction`: bool, 可选
      如果为真（默认），则推断统计和假设检验（例如p值，f_pvalue, conf_int, 和 t_test 和 f_test）的自由度基于组数减一，而不是总观测值数减去解释变量数。结果实例的df_resid也将调整。当`use_t`也为真时，使用校正值使用学生t分布计算p值。如果组数很小，这可能与基于正态分布的p值有较大差异。如果为假，则结果实例的df_resid不调整。
  - `'hac-groupsum'`: Driscoll和Kraay的面板数据异方差性和自相关稳健协方差
    - 这里需要更多的选项
    - `time`: array_like, 必需
      时间周期的索引
    - `maxlags`: 整数, 必需
      使用的滞后数
    - `kernel`: {可调用对象, str}, 可选
      可用的核函数是 `['bartlett', 'uniform']`，默认是Bartlett
    - `use_correction`: {False, 'hac', 'cluster'}, 可选
      如果为假，则计算三明治协方差时不使用小样本校正。如果`use_correction`为 `'cluster'`（默认），则使用与`cov_type='cluster'`相同的小样本校正
    - `df_correction`: bool, 可选
      df_resid的调整，见上面的`'cluster'`协方差类型
  - `'hac-panel'`: 面板数据中的异方差性和自相关稳健标准误。在这种情况下，数据需要被排序，每个面板单元或聚类的时序需要被堆叠。个体或组的时间序列成员资格可以通过组指示符或通过递增的时间周期指定。需要组或时间之一。
    - `groups`: array_like[int]
      组的指标
    - `time`: array_like[int]
      时间周期的索引
    - `maxlags`: int, 必需
      使用的滞后数
    - `kernel`: {可调用对象, str}, 可选
      可用的核函数是 `['bartlett', 'uniform']`，默认是Bartlett
    - `use_correction`: {False, 'hac', 'cluster'}, 可选
      如果为假，计算三明治协方差时不使用小样本校正
    - `df_correction`: bool, 可选
      df_resid的调整，见上面的`'cluster'`协方差类型

提醒：在`'hac-groupsum'`和`'hac-panel'`中，`use_correction`不是布尔值，需要在 {False, ‘hac’, ‘cluster’} 中。

### `OLSResults.HC0_se` 

是一个用于计算异方差稳健标准误的方法，它基于White(1980)提出的一种技术。这种标准误的计算方法可以对存在异方差性（即误差项的方差不是恒定的）的回归模型进行校正。

具体来说，`HC0_se`的计算公式定义为：

`sqrt(diag(X.T X)^(-1)X.T diag(e_i^(2)) X(X.T X)^(-1))`

其中，e_i 是模型的残差，即 `resid[i]`。在这个公式中，X 是设计矩阵（包含了自变量和截距项），e_i^(2) 是残差的平方。

当调用` HC0_se `或 `cov_HC0` 方法时，`RegressionResults` 实例将会新增一个属性` het_scale`。在这个情况下，`het_scale` 仅仅是残差的平方（resid**2）。这个属性可以用来获取模型残差的平方，进而用于计算异方差稳健标准误。

与此类似的还有`HC1`，`HC2`，`HC3`

---
# 异方差

## BP检验

> statsmodels.stats.diagnostic.het_breuschpagan 
### 概述
`statsmodels.stats.diagnostic.het_breuschpagan`是用于检测异方差性的Breusch-Pagan Lagrange Multiplier检验。该检验的假设是残差方差不依赖于解释变量x。

异方差性意味着残差方差是恒定的。如果存在异方差性，那么在小样本或中等大小的样本中，该检验可能会夸大结果的显著性。在这种情况下，F统计量更为可取。
### 参数
- `resid`: array_like
  - 对于Breusch-Pagan检验，这应该是回归的残差。如果`exog`中给出了数组，那么残差是通过OLS回归或`resid on exog`计算的。在这种情况下，`resid`应包含因变量。`exog`可以与x相同。

- `exog_het`: array_like
  - 包含怀疑与残差异方差性相关的变量。

- `robust`: bool, 默认为True
  - 标志，指示是否使用Koenker版本的检验（默认），它假设误差项是独立同分布的，或者是原始的Breusch-Pagan版本，它假设残差正态分布。
### 返回值
- `lm`: float
  - Lagrange乘数统计量。

- `lm_pvalue`: float
  - Lagrange乘数检验的p值。

- `fvalue`: float
  - 假设误差方差不依赖于x的F统计量。

- `f_pvalue`: float
  - F统计量的p值。
### 注意事项
- 假设x包含常数项（用于计算自由度和R^2）。
### 验证
- Chisquare检验统计量与R-stats中的`bptest`函数的结果是完全一致的（默认studentize=True）。
### 实现

- 这是使用Greene书中的通用LM检验公式计算的（第17.6节），而不是使用显式公式（第11.4.3节），除非将`robust`设置为False。p值的自由度假设x是满秩的。
### 参考文献

1. Greene, W. H. 《Econometric Analysis》. New Jersey: Prentice Hall; 第5版. (2002).
2. Breusch, T. S.; Pagan, A. R. (1979). “A Simple Test for Heteroskedasticity and Random Coefficient Variation”. Econometrica. 47 (5): 1287–1294.
3. Koenker, R. (1981). “A note on studentizing a test for heteroskedasticity”. Journal of Econometrics 17 (1): 107–112.

---
# 自相关
## 自相关图

`statsmodels.graphics.tsaplots.plot_acf` 
- 是 `statsmodels` 库中的一个函数
- 用于绘制时间序列数据的自相关函数（ACF）。
### 函数签名
```python
statsmodels.graphics.tsaplots.plot_acf(
    x,
    ax=None,
    lags=None,
    *,
    alpha=0.05,
    use_vlines=True,
    adjusted=False,
    fft=False,
    missing='none',
    title='Autocorrelation',
    zero=True,
    auto_ylims=False,
    bartlett_confint=True,
    vlines_kwargs=None,
    **kwargs
)
```
### 参数说明
- `x`: 
	- 数组类型，时间序列值的数组。
- `ax`: 
	- `AxesSubplot`，可选，如果提供，将在这个子图上绘制而不是创建新的图形。
- `lags`:
	- {int, 数组类型}, 可选，一个整数或滞后值数组，用于水平轴。如果 `lags` 是整数，则使用 `np.arange(lags)`。如果不提供，默认使用 `lags=np.arange(len(corr))`。
- `alpha`:
	- 标量，可选，如果给出一个数字，将返回给定置信水平的置信区间。例如，如果 `alpha=0.05`，则返回95%的置信区间，其中标准差根据 Bartlett 公式计算。如果为 `None`，则不绘制置信区间。
- `use_vlines`: 
	- 布尔值，可选，如果为 `True`，则绘制垂直线和标记。如果为 `False`，则仅绘制标记。默认标记是 `‘o’`；可以使用 `marker` 关键字参数覆盖。
- `adjusted`:
	- 布尔值，如果为 `True`，则自相关系数的分母为 `n-k`，否则为 `n`。
- `fft`:
	- 布尔值，可选，如果为 `True`，则通过 FFT 计算 ACF。
- `missing`:
	- 字符串，可选，一个字符串 `[‘none’, ‘raise’, ‘conservative’, ‘drop’]` 指定如何处理 NaN。`‘none’` 表示不采取任何措施，`‘raise’` 表示抛出异常，`‘conservative’` 表示删除包含 NaN 的行，`‘drop’` 表示删除 NaN。
- `title`:
	- 字符串，可选，放置在图形上的标题。默认是 `‘Autocorrelation’`。
- `zero`: 
	- 布尔值，可选，标志是否包括零滞后的自相关。默认是 `True`。
- `auto_ylims`: 
	- 布尔值，可选，如果为 `True`，则根据 ACF 值自动调整 y 轴的界限。
- `bartlett_confint`: 
	- 布尔值，默认为 `True`，表示置信区间为 ACF 值的 2 个标准误差。如果自相关被用来作为 ARIMA 程序中测试残差随机性的一部分，则假定残差是白噪声，标准误差的近似公式为每个 `r_k` 的标准误差 `= 1/sqrt(N)`。有关 `1/sqrt(N)` 结果的更多细节，请参阅 [1] 的第 9.4 节。有关更基础的讨论，请参阅 [2] 的第 5.3.2 节。对于原始数据的 ACF，滞后 k 的标准误差是按照假设正确的模型是 MA(k-1) 来找到的。这允许可能的解释，即如果所有超过某个滞后的自相关都在限制之内，模型可能是由最后一个显著自相关的顺序定义的 MA。在这种情况下，假设数据的移动平均模型，并使用 Bartlett 公式生成置信区间的标准误差。有关 Bartlett 公式结果的更多细节，请参阅 [1] 的第 7.2 节。
- `vlines_kwargs`:
	- 字典，可选，传递给 `vlines` 的可选关键字参数字典。
- `**kwargs`:
	- 关键字参数，可选，直接传递给 Matplotlib 的 `plot` 和 `axhline` 函数的关键字参数。
### 返回值
- `Figure`: 如果 `ax` 是 `None`，则创建的图形。否则，连接到 `ax` 的图形。
### 注意
- 该函数改编自 `matplotlib` 的 `xcorr`。
- 数据被绘制为 `plot(lags, corr, **kwargs)`。
- `kwargs` 用于将 Matplotlib 可选参数传递给追踪自相关的线以及在 0 处的水平线。这些选项必须对 `Line2D` 对象有效。
- `vlines_kwargs` 用于将额外的可选参数传递给连接每个自相关到轴的垂直线。这些选项必须对 `LineCollection` 对象有效。
### 参考文献
1. Brockwell and Davis, 1987. Time Series Theory and Methods
2. Brockwell and Davis, 2010. Introduction to Time Series and Forecasting, 2nd edition.
### 示例
```python
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 加载太阳黑子数据集
dta = sm.datasets.sunspots.load_pandas().data
# 设置索引为时间序列
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
# 删除不必要的列
del dta["YEAR"]
# 绘制自相关图，滞后40
sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40)
# 显示图形
plt.show()
```

![[Pasted image 20240420024558.png]]


 最后更新：2023年12月14日

---
## 自相关 Q检验

> statsmodels.stats.diagnostic.acorr_ljungbox - statsmodels 0.14.1
### statsmodels.stats.diagnostic.acorr_ljungbox
```python
statsmodels.stats.diagnostic.acorr_ljungbox(
	x, 
	lags=None, 
	boxpierce=False, 
	model_df=0, 
	period=None, 
	return_df=True, 
	auto_lag=False
)
```

Ljung-Box自相关残差检验。
### 参数
- `x`: 数组类型
  数据序列。在计算检验统计量之前，数据会被去均值。
- `lags`: {整型, 数组类型}, 默认为None
  如果lags是整数，则被认为是包含的最大滞后，测试结果会报告所有较小滞后长度的滞后。如果lags是列表或数组，则包括列表中的最大滞后的所有滞后，但只报告列表中滞后的测试。如果lags为None，则默认maxlag是min(10, nobs // 5)。如果设置了period，则默认滞后数会改变。
- `boxpierce`: 布尔值，默认为False
  如果为真，则除了Ljung-Box检验的结果外，还会返回Box-Pierce检验的结果。
- `model_df`: 整型，默认为0
  模型消耗的自由度数量。在ARMA模型中，这个值通常是p+q，其中p是AR阶数，q是MA阶数。这个值会从测试中使用的自由度中减去，以便统计量的调整后的自由度为l`ags - model_df`。如果`lags - model_df <= 0`，则返回NaN。
- `period`: 整型，默认为`None`
  季节性时间序列的周期。用于计算季节性数据的最大滞后，如果设置了，则使用`min(2*period, nobs // 5)`。如果为None，则使用默认规则设置滞后数。设置时，必须 >= 2。
- `auto_lag`: 布尔值，默认为`False`
  标志，指示是否基于最大相关值的阈值自动确定最优滞后长度。
### 返回
- `DataFrame`
  带有以下列的框架：
  - `lb_stat` - Ljung-Box检验统计量。
  - `lb_pvalue` - 基于卡方分布的p值。p值计算为1 - chi2.cdf(lb_stat, dof)，其中dof是lags - model_df。如果lags - model_df <= 0，则返回NaN作为p值。
  - `bp_stat` - Box-Pierce检验统计量。
  - `bp_pvalue` - Box-Pierce检验基于卡方分布的p值。p值计算为1 - chi2.cdf(bp_stat, dof)，其中dof是lags - model_df。如果lags - model_df <= 0，则返回NaN作为p值。

### 注意
Ljung-Box和Box-Pierce统计量在自相关函数的缩放上有所不同。Ljung-Box检验具有更好的有限样本属性。
### 另见
- `statsmodels.regression.linear_model.OLS.fit`: 回归模型拟合。
- `statsmodels.regression.linear_model.RegressionResults`: 线性回归模型的结果。
- `statsmodels.stats.stattools.q_stat`: 从估计的自相关计算的Ljung-Box检验统计量。
### 参考文献
- [*] Green, W. “Econometric Analysis,” 5th ed., Pearson, 2003.
- [†] J. Carlos Escanciano, Ignacio N. Lobato “An automatic Portmanteau test for serial correlation.”, Volume 151, 2009.
### 示例
```python
import statsmodels.api as sm

# 加载数据集
data = sm.datasets.sunspots.load_pandas().data
# 使用ARMA模型拟合太阳黑子活动数据
res = sm.tsa.ARMA(data["SUNACTIVITY"], (1,1)).fit(disp=-1)
# 进行Ljung-Box检验，设置滞后为10，并返回DataFrame
print(sm.stats.acorr_ljungbox(res.resid, lags=[10], return_df=True))
```

最后更新2023年12月14日

---
## 处理方法：HAC稳健标准误

Newey-West方法需要采用下面这个方法中一种。
使用 [[#`get_robustcov_results()`的用法|get_robustcov_results()]] 实现。
### 案例：

假设我们有一组时间序列数据，我们怀疑存在异方差性和自相关性，我们想要使用`NeweyWest`方法来计算稳健的标准误。以下是如何使用`statsmodels`库进行此计算的示例：

```python
import numpy as np
import statsmodels.api as sm

# 模拟一些时间序列数据
np.random.seed(123)
x = np.random.randn(100, 2)  # 自变量
y = np.dot(x, [1.5, -0.5]) + np.random.randn(100)  # 因变量
# 添加常数项
x = sm.add_constant(x)
# 进行OLS回归
model = sm.OLS(y, x).fit()
# 使用Newey-West方法计算稳健协方差矩阵
nw_cov = model.get_robustcov_results(cov_type='HAC', maxlags=1, use_correction=True)
# 从稳健协方差矩阵中提取标准误
nw_se = np.sqrt(np.diag(nw_cov))
# 打印稳健标准误
print("Newey-West 稳健标准误:", nw_se)
```

在这个案例中，我们首先生成了一些模拟的时间序列数据，然后使用OLS进行了回归。
接着，我们使用`get_robustcov_results`方法计算了`NeweyWest`稳健协方差矩阵，并从中提取了标准误。`maxlags=1`表示我们使用了1个滞后，`use_correction=True`表示我们在估计中使用了小样本校正。

以下是`statsmodels.stats.diagnostic.linear_reset`函数的文档内容翻译，以及一个Markdown格式的示例案例：

#  函数形式检验（RESET检验）

#### statsmodels.stats.diagnostic.linear_reset
```python
statsmodels.stats.diagnostic.linear_reset(
	res, 
	power=3, 
	test_type='fitted', 
	use_f=False, 
	cov_type='nonrobust', 
	cov_kwargs=None
)
```
Ramsey的遗漏非线性检验。
#### 参数
- `res`: RegressionResults
  线性回归的结果实例。
- `power`: `{int, List[int]}`, 默认为3
  如果为整数，则表示在模型中包含的最大幂次，包括幂次2, 3, ..., power。如果为整数列表，则包含列表中的所有幂次。
- `test_type`: str, 默认为"fitted"
  使用的增强类型：
  - "fitted": 默认，用拟合值的幂次增强回归变量。
  - "exog": 用外生变量的幂次增强外生变量，排除二元回归变量。
  - "princomp": 用外生变量的首个主成分的幂次增强外生变量。
- `use_f`: bool, 默认为False
  标志，指示是否使用F检验（True）或卡方检验（False）。
- `cov_type`: str, 默认为"nonrobust"
  协方差类型。默认为“nonrobust”，使用经典的OLS协方差估计器。可以指定“HC0”，“HC1”，“HC2”，“HC3”之一来使用White的协方差估计器。OLS.fit支持的所有协方差类型都被接受。
- `cov_kwargs`: dict, 默认为None
  传递给OLS.fit的协方差选项字典。详见OLS.fit的文档。
#### 返回值
ContrastResults
Ramsey重置检验的测试结果。有关实现细节，请参见注释。
#### 注释
RESET检验使用增强回归的形式，其中Z是以下之一的一组回归变量：
- 原始回归的拟合值的幂次。
- 外生变量的幂次，不包括常数和二元回归变量。
- 外生变量的首个主成分的幂次。如果模型包含常数，此列在计算主成分之前将被删除。在任一情况下，主成分是从剩余列的相关矩阵中提取的。

检验是关于零假设的Wald检验。如果`use_f`为True，则二次型检验统计量除以限制数量，使用F分布来计算临界值。
#### 示例案例

```python
import statsmodels.api as sm
import pandas as pd

# 假设df是包含自变量和因变量的DataFrame
df = pd.DataFrame({
    'y': [3, -1.2, 4.5, 2.8, 5],
    'x1': [1, 2, 3, 4, 5],
    'x2': [2, 4, 6, 8, 10]
})

# 定义因变量和自变量
y = df['y']
X = df[['x1', 'x2']]

# 添加常数项并拟合线性回归模型
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# 使用linear_reset进行RESET检验
reset_results = sm.stats.diagnostic.linear_reset(model, power=[1, 2, 3])

# 输出检验结果
print(reset_results)
```

在这个示例中，我们首先创建了一个包含因变量`y`和自变量`x1`、`x2`的`DataFrame`。然后，我们为自变量添加了一个常数项，并使用最小二乘法拟合了一个线性回归模型。接着，我们调用`linear_reset`函数对模型进行RESET检验，`power=[1, 2, 3]`意味着我们考虑了原始回归变量及其平方和立方的幂次作为潜在的遗漏非线性项。

请注意，具体的实现和参数可能会根据`statsmodels`的版本有所不同。你应该查阅你所使用的版本的官方文档以获取最准确的信息。

# 多重共线性检验

#### statsmodels.stats.outliers_influence.variance_inflation_factor
```python
statsmodels.stats.outliers_influence.variance_inflation_factor(
	exog, 
	exog_idx
)
```  
方差膨胀因子（Variance Inflation Factor，简称VIF）是衡量当将一个额外的变量（由`exog_idx`指定）添加到线性回归中时，参数估计量方差的增加程度的指标。它是设计矩阵`exog`多重共线性的度量。
如果VIF大于5，则表明由`exog_idx`指定的解释变量与其他解释变量高度共线性，因此参数估计将因此具有较大的标准误差。
#### 参数
- `exog`: {ndarray, DataFrame}
  回归中使用的所有解释变量的设计矩阵。
- `exog_idx`: int
  在`exog`列中的外生变量的索引。
#### 返回
- float
  方差膨胀因子。
#### 注意事项
此函数不保存辅助回归。
### 参考
- [方差膨胀因子 - Wikipedia](https://en.wikipedia.org/wiki/Variance_inflation_factor)
#### 示例
假设我们有一个设计矩阵`X`和一个我们想要检查共线性的变量索引`i`，我们可以使用以下代码计算VIF：

```python
import statsmodels.api as sm
# 假设X是一个NumPy数组或pandas DataFrame，包含了我们的解释变量
X = ...  # 你的数据
# 我们想要检查的变量的索引
i = ...
# 计算VIF
vif = sm.stats.outliers_influence.variance_inflation_factor(X, i)

print(f"Variance Inflation Factor for variable at index {i}: {vif}")
```
请注意，这只是一个示例，实际使用时需要替换`X`和`i`为你自己的数据和变量索引。

# 离散因变量的回归

用于有限和定性因变量的回归模型。当前模块允许估计具有二元（Logit, Probit）、名义（MNLogit）或计数（Poisson, NegativeBinomial）数据的模型。

从0.9版本开始，这还包括新的计数模型，这些模型在0.9版本中仍然是实验性的，包括NegativeBinomialP、GeneralizedPoisson和零膨胀模型，ZeroInflatedPoisson、ZeroInflatedNegativeBinomialP和ZeroInflatedGeneralizedPoisson。

查看 [Module Reference](https://www.statsmodels.org/stable/discretemod.html#module-reference)以获取命令和参数。

## 示例
```python
# 从Spector和Mazzeo (1980)加载数据
import statsmodels.api as sm
spector_data = sm.datasets.spector.load_pandas()
spector_data.exog = sm.add_constant(spector_data.exog)
# Logit模型
logit_mod = sm.Logit(spector_data.endog, spector_data.exog)
logit_res = logit_mod.fit()
print(logit_res.summary())
```
Logit回归结果
```
==============================================================================
Dep. Variable: GRADE   No. Observations: 32
Model: Logit   Df Residuals: 28
Method: MLE   Df Model: 3
Date: Thu, 14 Dec 2023
Pseudo R-squ.: 0.3740
Time: 14:49:28
Log-Likelihood: -12.890
converged: True
LL-Null: -20.592
Covariance Type: nonrobust
LLR p-value: 0.001502
==============================================================================
coef    std err     z    P>|z| [0.025     0.975]
------------------------------------------------------------------------------------------
const    -13.0213     4.931    -2.641      0.008    -22.687     -3.356
GPA       2.8261     1.263      2.238      0.025      0.351      5.301
TUCE      0.0952     0.142      0.672      0.501     -0.182      0.373
PSI       2.3787     1.065      2.234      0.025      0.292      4.465
```

详细示例可以在此处找到：
- [Overview](https://www.statsmodels.org/stable/examples/notebooks/generated/discrete_choice_overview.html)
- [Examples](https://www.statsmodels.org/stable/examples/notebooks/generated/discrete_choice_example.html)
## 技术文档

目前所有模型都是通过最大似然估计，并假设误差独立同分布。
所有离散回归模型定义了相同的方法，并遵循相同的结构，这与回归结果类似，但具有一些特定于离散模型的方法。
此外，其中一些包含额外的模型特定方法和属性。
## 参考文献

这类模型的一般参考文献是：
```
A.C. Cameron and P.K. Trivedi.  `Regression Analysis of Count Data`.
    Cambridge, 1998

G.S. Madalla. `Limited-Dependent and Qualitative Variables in Econometrics`.
    Cambridge, 1983.

W. Greene. `Econometric Analysis`. Prentice Hall, 5th. edition. 2003.
```

## 模块引用
特定模型类包括：

| 模型名称                           | 参数列表示例                                 | 描述                   |
| ------------------------------ | -------------------------------------- | -------------------- |
| Logit                          | (endog, exog[, offset, check_rank])    | Logit模型              |
| Probit                         | (endog, exog[, offset, check_rank])    | Probit模型             |
| MNLogit                        | (endog, exog[, check_rank])            | 多项式Logit模型           |
| Poisson                        | (endog, exog[, offset, exposure, ...]) | Poisson模型            |
| NegativeBinomial               | (endog, exog[, ...])                   | 负二项模型                |
| NegativeBinomialP              | (endog, exog[, p, offset, ...])        | 广义负二项模型（NB-P）        |
| GeneralizedPoisson             | (endog, exog[, p, offset, ...])        | 广义Poisson模型          |
| ZeroInflatedPoisson            | (endog, exog[, ...])                   | Poisson零膨胀模型         |
| ZeroInflatedNegativeBinomialP  | (endog, exog[, ...])                   | 零膨胀广义负二项模型           |
| ZeroInflatedGeneralizedPoisson | (endog, exog)                          | 零膨胀广义Poisson模型       |
| HurdleCountModel               | (endog, exog[, offset, ...])           | 计数数据的障碍模型            |
| TruncatedLFNegativeBinomialP   | (endog, exog[, ...])                   | 计数数据的截断广义负二项模型       |
| TruncatedLFPoisson             | (endog, exog[, offset, ...])           | 计数数据的截断Poisson模型     |
| ConditionalLogit               | (endog, exog[, missing])               | 拟合分组数据的条件逻辑回归模型      |
| ConditionalMNLogit             | (endog, exog[, missing])               | 拟合分组数据的条件多项式Logit模型  |
| ConditionalPoisson             | (endog, exog[, missing])               | 拟合分组数据的条件Poisson回归模型 |

目前，针对序数因变量的累积链接模型在miscmodels中，因为它是GenericLikelihoodModel的子类。这将在未来的版本中改变。

| 模型名称      | 参数列表示例                | 描述                   |
|---------------|----------------------------|------------------------|
| OrderedModel  | (endog, exog[, offset, distr]) | 基于逻辑或正态分布的序数模型 |

OrderedModel (endog, exog[, offset, distr]) 基于逻辑或正态分布的序数模型


特定结果类包括：


DiscreteModel是所有离散回归模型的超类。估计结果作为DiscreteResults的一个子类的实例返回。每个模型类别，二元、计数和多项式，都有自己的中间级别的模型和结果类。这些中间类主要是为了便于实现由DiscreteModel和DiscreteResults定义的方法和属性。

| 模型名称             | 参数列表示例                         | 描述                             |
|----------------------|------------------------------------|----------------------------------|
| DiscreteModel        | (endog, exog[, check_rank])         | 离散选择模型的抽象类            |
| DiscreteResults      | (model, mle_fit[, cov_type, ...])  | 离散因变量模型的结果类          |
| BinaryModel          | (endog, exog[, offset, check_rank]) | 二元数据的模型                  |
| BinaryResults        | (model, mle_fit[, cov_type, ...])  | 二元数据的结果类                |
| CountModel           | (endog, exog[, offset, exposure, ...]) | 计数数据的模型                  |
| MultinomialModel    | (endog, exog[, offset, ...])       | 多项式数据的模型                |
| GenericZeroInflated  | (endog, exog[, ...])               | 通用零膨胀模型                  |


