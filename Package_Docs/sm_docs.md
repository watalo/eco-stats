
# Statsmodels官方文档摘要

--- 
 statsmodels 0.14.1

> 使用kimi.ai生成的翻译文档

---

### 用户指南

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
### 快速入门

这是一个非常简单的案例研究，旨在帮助你快速上手statsmodels。
#### 加载模块和函数
安装statsmodels及其依赖项后，我们需要加载一些模块和函数：

```python
import statsmodels.api as sm
import pandas
from patsy import dmatrices
```

- pandas基于numpy数组提供丰富的数据结构和数据分析工具。pandas.DataFrame函数提供带有标签的数组（可能是异构的），类似于R语言中的“data.frame”。pandas.read_csv函数可用于将逗号分隔值文件转换为DataFrame对象。
- patsy是一个Python库，用于描述统计模型并使用R语言风格的公式构建设计矩阵。

#### 数据
我们下载了Guerry数据集，这是一组用于支持安德烈-米歇尔·古埃里1833年《法国道德统计论文》的历史数据。数据集由Rdatasets仓库以逗号分隔值格式（CSV）在线托管。我们本可以本地下载文件，然后使用read_csv加载它，但pandas为我们自动处理了这一切：

```python
df = sm.datasets.get_rdataset("Guerry", "HistData").data
```

我们选择感兴趣的变量，并查看底部的5行：

```python
df = df[['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']]
df = df.dropna()
```

#### 动机和模型
我们想要了解法国86个省的识字率是否与1820年代皇家彩票的人均赌注有关。我们需要控制每个省的财富水平，并希望在我们的回归方程的右侧包括一系列虚拟变量，以控制由于地区效应导致的未观察到的异质性。
使用普通最小二乘回归（OLS）估计模型。

#### 设计矩阵（内生和外生）
为了拟合statsmodels覆盖的大多数模型，你需要创建两个设计矩阵。第一个是内生变量矩阵（即依赖的、响应的、回归量等）。第二个是外生变量矩阵（即独立的、预测的、回归因子等）。OLS系数估计如常计算：

```python
vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
df = df[vars]
y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')
```

#### 模型拟合和摘要
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

#### 诊断和规范检验
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
### OLS模型拟合结果的属性与方法
```python
results = sm.OLS(y,X).fit()
```
#### 方法

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

#### 属性
| 方法                 | 描述                                     |
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
#### `t_test()`的用法
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
##### 返回
ContrastResults
- 测试的结果是此结果实例的属性。 可用结果具有与参数表相同的元素在`summary()`中。

#### `f_test()`的用法
用于检验线性假设的方法

参数与t_test()一致

返回值只有一个数值。

#### `OLSResults.HC0_se` 

是一个用于计算异方差稳健标准误的方法，它基于White(1980)提出的一种技术。这种标准误的计算方法可以对存在异方差性（即误差项的方差不是恒定的）的回归模型进行校正。

具体来说，`HC0_se`的计算公式定义为：

`sqrt(diag(X.T X)^(-1)X.T diag(e_i^(2)) X(X.T X)^(-1))`

其中，e_i 是模型的残差，即 `resid[i]`。在这个公式中，X 是设计矩阵（包含了自变量和截距项），e_i^(2) 是残差的平方。

当调用` HC0_se `或 `cov_HC0` 方法时，`RegressionResults` 实例将会新增一个属性` het_scale`。在这个情况下，`het_scale` 仅仅是残差的平方（resid**2）。这个属性可以用来获取模型残差的平方，进而用于计算异方差稳健标准误。

与此类似的还有`HC1`，`HC2`，`HC3`