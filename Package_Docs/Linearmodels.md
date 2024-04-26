# linearmodels 6.0
面板数据模型、系统回归、工具变量和资产定价的估计与推断。
最新版本的稳定文档位于 `doc`。最新发展的文档位于 `devel`。

[linearmodels 6.0 (bashtage.github.io)](https://bashtage.github.io/linearmodels/)

> 这个三方库极大的弥补了statsmodel的缺陷，两个搭配起来使用完美！
## 一些在statsmodels中缺失的常见线性模型的估计和推断：

### 面板数据模型
- 固定效应 (PanelOLS)
- 随机效应 (RandomEffects)
- 第一差分 (FirstDifferenceOLS)
- 介于估计 (BetweenOLS)
- 汇集OLS (PooledOLS)
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

