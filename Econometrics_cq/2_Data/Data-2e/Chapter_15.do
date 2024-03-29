*** 第15章，《计量经济学及Stata应用》第2版

** 15.6

* Using experimental data

use nsw_dw.dta, clear
bysort treat: sum
* global cov age education black hispanic married nodegree re74 re75 c.age#c.age c.education#c.education c.re74#c.re74 c.re75#c.re75
global cov age education black hispanic married nodegree re74 re75 
tabstat $cov,by(treat) 
tabstat $cov,by(treat) stat(mean sd) nototal 
logit treat $cov,r nolog

reg re78 treat,r
reg re78 treat $cov, r

* Using composite data (experimental treatment group and nonexperimental comparison group from PSID)

drop if treat == 0
append using cps_controls.dta
tabstat $cov,by(treat) nototal 

reg re78 treat,r
reg re78 treat $cov, r

* One to one matching for ATT
teffects psmatch (re78) (treat $cov), atet 

* One to four matching for ATET with gen()
teffects psmatch (re78) (treat $cov),atet nn(4) gen(match) 
list match* in 1/2
predict pscore,ps tlevel(1)
sum pscore
dis 0.25*r(sd)

* One to one caliper matching for ATET
teffects psmatch (re78) (treat $cov),atet nn(4) caliper(0.0135) osample(outside) 
teffects psmatch (re78) (treat $cov) if outside==0,atet nn(4) 

qui teffects psmatch (re78) (treat $cov),atet nn(4) 
teoverlap,ptlevel(1) name(overlap,replace) xtitle(propensity score) 
hist pscore if treat == 0,fraction name(pscore_control,replace) xtitle(propensity score for the control group)
hist pscore if treat == 1,fraction name(pscore_treat,replace) xtitle(propensity score for the treatment group)

qui teffects psmatch (re78) (treat $cov),atet nn(4) 
tebalance summarize
tebalance box,name(box,replace) 
tebalance density,name(density,replace) 

* Manually draw balance plot for the density of propensity score

* raw
twoway kdensity pscore if treat == 0, lp(dash) xtitle(Propensity Score) yaxis(1) || kdensity pscore if treat == 1, yaxis(2) 
ytitle(Density)  title(Raw) legend(label(1 Control) label(2 Treated))

* match
gen pscore_match = pscore[match1]
list pscore pscore_match match1 in 1/5
list pscore in 470

twoway kdensity pscore_match if treat==1,lp(dash) xtitle(Propensity Score) ||  kdensity pscore if treat==1, ytitle(Density) title(Matched) 
legend(label(1 Control) label(2 Treated)) 

sum pscore pscore_match if treat==1

tebalance box education,name(box_educ,replace) 
tebalance density education,name(density_educ,replace) 

exit