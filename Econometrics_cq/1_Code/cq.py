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
from math import log, exp
from statsmodels.stats.diagnostic import het_breuschpagan,het_white

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


# class Panel:
    
#     def __init__(self,data):
#         self.data = data






if __name__ == '__main__':
   pass