*** 第19章，《计量经济学及Stata应用》第2版

** Chapter 19 Regression Control Method

* ssc install rcm, all replace

sysuse growth,clear     
xtset region time 
label list

* Political Integration

display tq(1997q3)
display tq(2003q4)

rcm gdp, trunit(9) trperiod(150) counit(4 10 12 13 14 19 20 22 23 25) postperiod(150/175)

* Economic Integration

display tq(2002q1)

rcm gdp, trunit(9) trperiod(176) placebo(unit cut(2) period(168))

exit




