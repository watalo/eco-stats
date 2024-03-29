** 第8章，《计量经济学及Stata应用》第2版

* 8.5

use icecream.dta,clear
tsset time 
twoway connect consumption time,msymbol(circle) yaxis(1) || connect temp time, msymbol(triangle) yaxis(2)

reg consumption temp price income
predict e1,r
twoway scatter e1 L.e1 || lfit e1 L.e1
twoway scatter e1 L2.e1 || lfit e1 L2.e1
ac e1

estat bgodfrey
estat bgodfrey,nomiss0
wntestq e1
corrgram e1
estat dwatson

newey consumption temp price income,lag(3)
newey consumption temp price income,lag(6)

prais consumption temp price income,corc
prais consumption temp price income,nolog

reg consumption temp L.temp price income
estat bgodfrey
estat dwatson


exit




