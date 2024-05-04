#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cq.py
@Time    :   2024/04/16 22:17:45
@Author  :   watalo 
@Version :   1.0
@Contact :   watalo@163.com
'''

import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from math import log, exp
from statsmodels.stats.diagnostic import het_breuschpagan,het_white
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class Wls:
    def __init__(self, data, X_ls: list, y: str, X_aux: list):
        '''wls方法及实例化
        Arguments:
            data -- dataframe,原始数据集
            X_ls -- list,解释变量表头列表
            y -- str,因变量表头
            x_aux -- list,辅助回归表头列表
        '''
        self.data = data
        self.X_ls = X_ls
        self.y = y
        self.X_aux = X_aux
    
    @property  
    def results_ols(self):
        X = self.data[self.X_ls]
        y = self.data[self.y]
        X = sm.add_constant(X)
        model = sm.OLS(y,X)
        results = model.fit()
        return results
    
    
    @property 
    def results_fwls(self):
        X = self.data[self.X_ls]
        y = y = self.data[self.y]
        X = sm.add_constant(X)
        model_fwls = sm.WLS(y,X,weights=1/self.aw_fwls)
        results_fwls = model_fwls.fit()
        return results_fwls
    
    @property
    def aw_fwls(self):
        '''aw_fwls 返回辅助回归后的加权回归系数
        分为:
        - 有常数情况:R平方,aw
        - 无常数情况:R平方,aw
        返回值:元组
        '''
        # 不带常数项的辅助回归，通常R2更高
        X = self.data[self.X_aux]
        y_lne2 = self.results_ols.resid.apply(lambda x: log(float(x)**2))        
        model_lne2 = sm.OLS(y_lne2,X)
        results_lne2 = model_lne2.fit()
        R2 = results_lne2.rsquared
        aw = results_lne2.fittedvalues.apply(lambda x: exp(x))
        
        # 带常数项的辅助回归，通常R2不高，但是如果较高，首选这个
        X1 = sm.add_constant(X) # 无常数项，拟合优度更高
        model_lne2_const = sm.OLS(y_lne2,X1)
        results_lne2_const = model_lne2_const.fit()
        R2_const = results_lne2_const.rsquared
        aw_const = results_lne2_const.fittedvalues.apply(lambda x: exp(x))
        
        # 给出推荐结果
        if float(R2_const) > 0.5:
            return aw_const
        else:
            if float(R2) > 0.5:
                return aw
            else:
                print('拟合优度较差，建议更换辅助回归方程')
                return None
    
    
    def bp_test(self, model='ols'):
        X = sm.add_constant(self.data[self.X_ls])
        if model == 'ols':
            results = het_breuschpagan(resid=self.results_ols.resid,exog_het=X)
            return results
        elif model == 'wls':
            results = het_breuschpagan(resid=self.results_wls.resid,exog_het=X)
            return results
        else:
            raise ValueError('model参数错误,请选择ols或wls')
    
    def white_test(self,model='ols'):
        X = sm.add_constant(self.data[self.X_ls])
        if model == 'ols':
            results = het_white(resid=self.results_ols.resid,exog=X)
            return results
        elif model == 'wls':
            results = het_white(resid=self.results_wls.resid,exog=X)
            return results
        else:
            raise ValueError('model参数错误,请选择ols或wls')
        
        
def describe_bcmodel(df, frequency, target=None, condition_col=None, condition=None):
    '''describe_bcmodel 二值模型的描述性统计，返回原始数据
    Arguments:
        df:dataframe -- 含有频次的数据集
        frequency:str --  频次的字段名

    Keyword Arguments:
        target:str -- 观测对象的字段名 (default: {None})
        condition_col:str -- 条件变量的字段名 (default: {None})
        condition:any -- 条件值 (default: {None})

    Returns:
         -- 按频次还原后的数据集
    '''
    result = df.loc[np.repeat(df.index.values, df[frequency])].drop(frequency, axis=1).reset_index(drop=True)
    if (target is None) and (condition_col is None): 
        print(result.describe().T) 
    else:
        result = result[[target, condition_col]][result[condition_col] == condition].drop(condition_col, axis=1).reset_index(drop=True)
        print(f'when {condition_col} is {condition}:')
        print(result.describe().T)
        return result
    return result


def odds_ratio(results):
    odds_ratios = np.exp(results.params)
    result_logit_or = pd.DataFrame({'odds ratio': odds_ratios, 
                                    'std err': results.bse,
                                    'z':results.tvalues,
                                    'p>|z|':results.pvalues,
                                    }, 
                                index=results.params.index)
    pd.set_option('display.float_format', '{:.4f}'.format)
    return result_logit_or

## Ch12：时间序列中用到的函数

def acfgram(time_series,lags=10):
    '''acgram 绘制时间序列的自相关图和偏自相关图,并返回acf和pacf的结果
        达到类似与其他统计软件一样的效果
    Arguments:
        time_series -- pd.Series,array-like, 时间序列
        
    Keyword Arguments:
        lags -- int, 最大滞后阶数 (default: {10})
        
    Returns:
        1.plot:绘制时间序列的序列图、acf图和pacf图
        2.dataframe:返回字段命为lags acf pacf Q和Prob(Q)的数据
            - Q、Prob(Q) -- acf的统计量
    '''
    # 计算自相关系数
    acf_result = sm.tsa.acf(time_series,
                            nlags = lags,
                            qstat=True,
                            fft=False)
    # 计算偏自相关系数
    pacf_result = sm.tsa.pacf(time_series, nlags=lags)
    
    # 创建DataFrame来存储结果
    result_df = pd.DataFrame({
        'Lags': np.arange(1,lags+1),
        'ACF': acf_result[0][1:],
        'PACF': pacf_result[1:],
        'Q':acf_result[1],
        'Prob(Q)': acf_result[2]  
    })
    
    # 绘制自相关图
    _, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), dpi=400)
    ## 分别画出3个图
    time_series.plot(ax=axes[0])
    plot_acf(time_series, lags=lags,ax=axes[1])
    plot_pacf(time_series,lags=lags,ax=axes[2])
    ## 设置图标题
    axes[0].set_title('Time-Series') 
    axes[1].set_title('Autocorrelation') 
    axes[2].set_title('Partial-Autocorrelation') 
    plt.show()
    
    return result_df


def estimate_p(data, col, lags):
    '''estimate_p 用于估计AR模型的p值
        集中显示k阶ar模型的p值,AIC,BIC
    Arguments:
        data -- dataframe: 包含时间序列数据的
        col  -- str:       时间序列的列名
        lags -- int:       假设的AR模型最大阶数
    returns:
        df -- dataframe: 包含lags阶AR模型的p值,AIC,BIC
                         - 每阶只显示最大阶数的值
    '''
    
    def _AR_p(data, col, lag):
        endog = data[col]
        exog_var =[]
        for i in range(1,lag+1):
            data[f'{col}_l{i}']=data['dlny'].shift(i)
            exog_var.append(f'{col}_l{i}')     
        exog = sm.add_constant(data[exog_var])  
        res=sm.OLS(endog=endog, exog=exog, missing='drop').fit(cov_type='HC1')
        return res.params.index[-1], res.nobs, res.pvalues[-1], res.aic, res.bic
    
    
    df = pd.DataFrame({'index':['nobs', 'p-value', 'AIC', 'BIC']})
    df.set_index('index', inplace=True)
    for i in range(lags):
        _ = _AR_p(data, col, i+1)
        df[_[0]] = _[1:]
        
    df =  df.T
    min_ = df[['AIC', 'BIC']].min()
    for col,row in df[['p-value','AIC', 'BIC']].iterrows():
        if row['AIC'] == min_['AIC']:
            df.loc[col, 'AIC'] = f"{row['AIC']:.4f}[min]"
        if row['BIC'] == min_['BIC']:
            df.loc[col, 'BIC'] = f"{row['BIC']:.4f}[min]"
        if row['p-value'] > 0.05:
            df.loc[col, 'p-value'] = f"{row['p-value']:.5f}[>0.05]"
            
    return df

if __name__ == '__main__':
   pass