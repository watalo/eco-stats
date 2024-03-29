* 第2章，《计量经济学及Stata应用》第2版

use grilic_small.dta,clear
describe
list s lnw

set more off
set more on

list s lnw in 1/5
list s lnw in 11/15
list s lnw if s>=16

* drop if s>=16
* keep if s>=16

sort s
list

gsort -s
list

histogram s, width(1) frequency	
help histogram

scatter lnw s
gen n=_n
scatter lnw s,mlabel(n)

summarize s
sum
tabulate s
pwcorr lnw s expr,sig star(.05) 

generate lns=log(s)
gen s2=s^2
gen exprs=s*expr
gen w=exp(lnw)

gen colleg=(s>=16)
rename colleg college
* drop college
* gen college=(s>=15)
replace college=(s>=15)

display log(2)
dis 2^0.5

log using today.smcl,replace
log off
log on
log close

sysdir

exit







