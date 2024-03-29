** 第3章，《计量经济学及Stata应用》第2版

* 3.2

twoway function y=x^2,range(-1 1) xline(0) yline(0) lwidth(thick) xtitle(x1) ytitle(f(x1))
twoway function y=-x^2,range(-1 1) xline(0) yline(0) lwidth(thick) xtitle(x1) ytitle(f(x1))

* 3.5

use grilic.dta,clear
describe

sum
sum lnw,detail
hist lnw,width(0.1)
kdensity lnw,normal normop(lpattern(dash))

gen wage=exp(lnw)
kdensity wage

kdensity lnw if s==16

twoway kdensity lnw || kdensity lnw if s==16,lpattern(dash)
twoway (kdensity lnw) (kdensity lnw if s==16,lpattern(dash))

* 3.6

sum lnw
sum lnw if s==16
twoway (kdensity lnw if s==12) (kdensity lnw if s==16,lpattern(dash))

sum lnw if rns==0
sum lnw if rns==1
dis 5.725644*(554/(554+204))+5.581083*(204/(554+204))
sum lnw

* 3.8

dis normal(1.96)

twoway function y=normalden(x),range(-5 5) xline(0) ytitle("概率密度")
twoway function y=normalden(x),range(-5 10) || function z=normalden(x,1,2),range(-5 10) lpattern(dash) ytitle("概率密度")
twoway function chi3=chi2den(3,x),range(0 20) || function chi5=chi2den(5,x),range(0 20) lp(dash) ytitle("概率密度")
twoway function t1=tden(1,x),range(-5 5) || function t5=tden(5,x),range(-5 5) lp(dash) ytitle("概率密度")
twoway function F20=Fden(10,20,x),range(0 5) || function F5=Fden(10,5,x),range(0 5) lp(dash) ytitle("概率密度")

exit

