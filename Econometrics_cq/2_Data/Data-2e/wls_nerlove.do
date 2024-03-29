* WLS for Nerlove(1963)
capture log close
log using wls_nerlove.smcl,replace
set more off
use nerlove.dta, clear
reg lntc lnq lnpl lnpk lnpf
predict e1,res
gen e2=e1^2
gen lne2=log(e2)
reg lne2 lnq,noc 
predict lne2f
gen e2f=exp(lne2f)
* Weighted least square regression
reg lntc lnq lnpl lnpk lnpf [aw=1/e2f]
reg lntc lnq lnpl lnpk lnpf [aw=1/e2f],r
log close
exit
