# 第6章 大样本OLS

## 6.1 为何需要大样本理论

> #定义 大样本理论（渐近理论）
> 研究当样本容量n趋向无穷大时统计量的性质。

大样本理论成为主流的原因：
- 小样本理论的假设过强。
	1. 严格外生性假设要求解释变量与所有的扰动项正交（不相关）。时间序列的自相关太常见了
	2. 假定扰动项为正态分布，大样本理论无此限制。绝大部分经济变量不符合正态分布
- 小样本下，必须研究统计量的精确分布，大样本只需要渐近分布，这个更易推导。
- 大样本的代价是要求样本容量够大，至少30个，最好是100以上，这点在现代很容易实现。
## 6.2 随机收敛
### 1.确定性序列的收敛

确定性序列$\{a_n\}_{n=1}^{\infty}=\{a_1,a_2,a_3,\cdots \}$收敛于常数 $a$，记为$$\lim_{n\to \infty}a_n = a \quad or \quad a_n \to a$$如果对于任意小的正数$\epsilon > 0$，都存在$N > 0$，只要$n>N$，就有$|a_n-a| < \epsilon$ 。

> 只要n足够大，序列总会落到以a为中心的极小区间里面。

### 2.随机序列的收敛

#定义 **依概率收敛 (Convergence in Probability)**
随机序列 $\lbrace x_n​ \rbrace_{n=1}^{\infty}$ 依概率收敛于常数 a，记为 $p\lim_{n\to \infty}x_n​=a$，如果对于任意 $ϵ>0$，当 $n \to \infty$ 时，都有 $$\lim_{n\to\infty} P(∣xn​−a∣>ϵ)=0 \quad or\quad x_n \xrightarrow{p} a$$
任意给定很小的正数$\epsilon>0$，当n越来越大时，随机变量落入$(a-\epsilon,a+\epsilon)$之外的概率趋近于0。

#定理 **连续映射定理**$$p\lim_{n\to\infty}g(x) =g(p\lim_{n\to\infty}x_n)$$
### 3.依均方的收敛

#定义 **依均方收敛 (Convergence in Mean Square)**
如果随机序列 $\lbrace x_n​ \rbrace_{n=1}^{\infty}$ 的期望收敛于 a，方差收敛于0，即 $\lim_{n→∞​}E(x_n​)=a$，且$\lim_{n\to\infty}Var(x_n​)$ 收敛于0，则称 ${x_n​}$ 依均方收敛于常数 a，记为$x_n \xrightarrow{ms} a$。

> 依均方收敛意味着依概率收敛，依均方收敛通常比证明依概率收敛要容易，所以常被用来证明依概率收敛。
### 4.依分布的收敛
#定义 **依分布收敛 (Convergence in Distribution)**
如果随机序列${x_n​}$ 的累积分布函数$F_n​(x)$对于任意给定的x 都收敛于某个随机变量 X 的累积分布函数$F(x)$，即$$\lim_{n→∞}​F_n​(x)=F(x)$$则称$\lbrace x_n​ \rbrace$ 依分布收敛于随机变量 x，记为$x_n ​​\xrightarrow{d} x$。
那么，$x$的分布就是$x_n$的渐近分布。

> 依分布收敛关注的是随机变量序列的分布函数的收敛性，而不关心随机变量序列本身的具体取值。
## 6.3 大数定律与中心极限定理

### 1.切比雪夫不等式
#定义 
设服从任意分布的随机变量X的随机变量X的数学期望$E(X)=\mu$，方差$D(X)=\mu^2$，则：$$P(|X-\mu|\le k\sigma) \ge 1 - \frac{1}{k^2},k \gt1$$
利用切比雪夫不等式，可以在随机变量X的分布未知的情况下，对事件$|X-\mu|\lt k\sigma$的概率作出估计。

> - 例：对于任意一个分布而言，观测值落在偏离均值正负3个标准差内的概率最下位多少？
>      - 解析：根据切比雪夫不等式：$$P(|X-\mu| \le 3\sigma)\ge 1-\frac{1}{3^2} \approx 89\% $$
### 2.大数定律
#定义 
设随机变量$X_1,X_2,…,X_n$独立同分布（i.i.d）
 - 期望为$\mu$
 - $S_n=X_1+X_2+\cdots+X_n = \sum_{i=1}^nx_i$
 - 则$\frac{S_n}{n}$收敛于$\mu$ : $$\lim \limits_{n \to \infty}\overline X = \mu$$

> 样本容量n足够大，样本均值就趋于总体均值
### 3.中心极限定理
#定义 
设随机变量$X_1,X_2,…,X_n$独立同分布，且具有有限的数学期望和方差：$E (X_k) = μ$，$D(X_k) =σ^2 > 0$，当n充分大时，样本均值近似服从正态分布，即：$$\overline X \sim N(\mu,\frac{\sigma^2}{n})$$
## 6.4 使用蒙特卡罗模拟中心极限定理

> [[Chapter_06.ipynb]]
## 6.5 统计量的大样本性质
### 1.一致估计量
#定义 **一致估计量（consistent estimator）**
估计量$\hat\beta_n$ 是参数 $\beta$ 的<span style="color:#00b0f0">一致估计量</span>，有：$$p\lim_{n\to\infty} \hat\beta_n = \beta$$
### 2.渐近正态分布与渐近方差
#定义 **asymptotically normal**
如果$\sqrt{n}(\hat\beta_n-\beta) \xrightarrow{d} N(0, \sigma^2)$， 则
- $\hat\beta_n$ 是<span style="color:#00b0f0">渐近正态分布</span>
- $\sigma^2$ 为其<span style="color:#00b0f0">渐近方差</span>，记为$Avar(\hat\beta_n)$
### 3.渐近有效
#定义 **asymptotically more efficient** 
假设 $\hat\beta$ 和 $\tilde \beta$ 都是渐近正态分布：
- 如果 $Avar(\hat\beta) \le Avar(\tilde\beta)$，则称设 $\hat\beta$ 比 $\tilde \beta$ 更<span style="color:#00b0f0">渐近有效</span>。
## 6.6 随机过程的性质

随机序列有个更好听的名字，<span style="color:#00b0f0">随机过程(stochastic process)</span>
- 如果下标是时间，则称为<span style="color:#00b0f0">时间序列(time series)</span>
### 1.严格平稳过程

#定义 **严格平稳过程(strictly stationary process)**
如果对任意m个时期的时间集合$\{t_1,t_2,\cdots , t_m\}$，随机向量$\{x_{t_1},x_{t_2},\cdots,x_{t_m}\}$的联合分布等于随机向量$\{x_{t_{1+k}},x_{t_{2+k}},\cdots,x_{t_{m+k}}\}$的联合分布。
- k为任意整数

平稳过程的联合分布
- 不随时间下标变化 —— t
- 只与时间长度相关 —— m

#定义 **弱平稳过程(strictly stationary process)**
随机过程x_t是弱平稳过程（weakly stationary process），或协方差平稳过程（covariance stationary process）：
- E(x_t) 不依赖于t， 是个常数
- Cov(x_t,x_{t+k})只依赖于k，

#定义 白噪声过程

### 2.渐近独立性



## 6.7 大样本OLS的假定

## 6.8 OLS的大样本性质

## 6.9 大样本统计推断

## 6.10 大样本OLS的python命令及实例

## 6.11 大样本理论的蒙特卡罗模拟


---
---
## 习题
