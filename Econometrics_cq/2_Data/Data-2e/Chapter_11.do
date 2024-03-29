** 第11章，《计量经济学及Stata应用》第2版

* 11.1

twoway function Probit=normal(x),range(-5 5) || function Logit=exp(x)/(1+exp(x)),range(-5 5) lp(dash) ytitle("累积分布函数")

* 11.9

use titanic.dta,clear
list 
sum [fweight=freq]
sum survive if child [fweight=freq]
sum survive if female [fweight=freq]
sum survive if class1 [fweight=freq]
sum survive if class2 [fweight=freq]
sum survive if class3 [fweight=freq]
sum survive if class4 [fweight=freq]

reg survive child female class1 class2 class3 [fweight=freq],r
logit survive child female class1 class2 class3 [fweight=freq],nolog 
logit survive child female class1 class2 class3 [fweight=freq],nolog r
logit survive child female class1 class2 class3 [fweight=freq],or nolog

margins,dydx(*)
margins,dydx(*) atmeans

estat clas

predict prob
list prob survive freq if class1==1 & child==0 & female==1
list prob survive freq if class3==1 & child==0 & female==0

probit survive child female class1 class2 class3 [fweight=freq],nolog
margins,dydx(*)
estat clas
predict prob1
corr prob prob1 [fweight=freq]

exit




