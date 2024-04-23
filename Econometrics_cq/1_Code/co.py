# 使用CO方法 需要多次迭代
import os 
import statsmodels.api as sm
import pandas as pd
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

# 读取原始数据
icecream = pd.read_stata(r'E:\Eco-stats\eco-stats\Econometrics_CQ\2_Data\Data-2e\icecream.dta')
icecream = icecream.drop(columns=['time'])
X = icecream[['temp','price','income']]
y = icecream['consumption']
X = sm.add_constant(X)

# 初始OLS回归:得到 resid_0 
model = sm.OLS(y,X)
results = model.fit()
resid = pd.DataFrame()
resid['lag_init'] = results.resid
print(resid)
# Cochrane-Orcutt迭代过程
converged = False
iterations = 0
max_iterations = 10  # 设置最大迭代次数
tolerance = 0.001  # 设置收敛容差
hat_rho = [0]
results_list = []

while not converged and iterations < max_iterations:
    # 整理OLS的残差数据，：dataframe格式
    ## 第一轮迭代时，残差数据为初始OLS获得
    ## 第二轮迭代以后，残差数据为上一轮CO估计的残差数据,需要反算
    
    # 残差数据处理
    resid_af = resid.copy()
    resid_af['lag_0'] = resid_af['lag_init'].shift(0)
    resid_af['lag_1'] = resid_af['lag_init'].shift(1)
    resid_afj = resid_af.dropna()  # 去除nan值
    print(resid_afj)
    
    resid_X = resid_afj['lag_1']
    resid_y = resid_afj['lag_0']
    # X = sm.add_constant(X) # 不加入常数项效果更好
    rho_fit = sm.OLS(resid_y[1:], resid_X[1:]).fit()
    hat_rho.append(rho_fit.params.iloc[0])
    print(rho_fit.params.iloc[0])
    
    # 使用Cochrane-Orcutt变换调整误差项
    X_adj = X.copy() - X.copy().shift(1)*hat_rho[-1]
    y_adj = y.copy() - y.copy().shift(1)*hat_rho[-1]
    X_adj = sm.add_constant(X_adj)
    
    # 重新进行OLS回归，去除t0项，t0项应为nan
    results_new = sm.OLS(y_adj[1:], X_adj[1:]).fit()
    
    # 回归完之后的残差是mu_t = resid_t - \resid_{t-1}*rho需要解方程组得出resid序列
    # mu = results_new.resid
    # resid_af['mu']=mu
    # resid['lag_init']=resid_af['mu']+ rho_fit.params.iloc[0]* resid_af['lag_1']
    # resid.loc[0, "lag_init"]  = resid_af['lag_0'].iloc[0]
    # print(resid)
    # 检查是否收敛
    if abs(hat_rho[-1]-hat_rho[-2]) < tolerance:
        converged = True
    else:
        results_list.append(results_new)
        results = results_new  # 更新模型为新迭代的结果
        iterations += 1
        print(f"迭代 {iterations}: rho = {hat_rho[-1]:.4f}")

print('【Cochrane-Orcutt】迭代结果:')
# print(f'迭代次数: {iterations}次，模型呈现{iterations-1}阶自相关。') #这句话不对，CO方法好像只涉及到1阶自相关。。
print(results_list[-1].summary()) 