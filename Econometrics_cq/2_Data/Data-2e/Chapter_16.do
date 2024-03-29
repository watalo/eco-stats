*** 第16章，《计量经济学及Stata应用》第2版

* install rdrobust, rddensity and lpdensity from Github

sysuse rdrobust_senate.dta,clear
sum 

rdplot vote margin

rdrobust vote margin
rdrobust vote margin,all

rdrobust vote margin,kernel(uniform) all

rdrobust vote margin,bwselect(msetwo) all
rdrobust vote margin,bwselect(cerrd) all
rdrobust vote margin,bwselect(certwo) all

rdrobust vote margin,all covs(class termshouse termssenate)

rdrobust population margin,all


* Manipulation Test

rddensity margin,plot
