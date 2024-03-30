
# Statsmodels官方文档摘要

--- 
 statsmodels 0.14.1

> 使用kimi.ai生成的翻译文档

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