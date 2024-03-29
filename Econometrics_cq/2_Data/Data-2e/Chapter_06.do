** 第6章，《计量经济学及Stata应用》第2版

* 6.1

use grilic.dta,clear
gen wage=exp(lnw)
twoway kdensity wage,xaxis(1) yaxis(1) xvarlab(wage) || kdensity lnw,xaxis(2) yaxis(2) xvarlab(ln(wage)) lp(dash)

gen lns=log(s)
twoway kdensity s,xaxis(1) yaxis(1) xvarlab(s) || kdensity lns,xaxis(2) yaxis(2) xvarlab(lns) lp(dash)

* 6.2

twoway function N=normal(x) ,range(-5 5) || function t1=t(1,x),range(-5 5)  lp(dash) || function t5=t(5,x),range(-5 5)  lp(shortdash) ytitle("累积分布函数")
twoway function N=normalden(x) ,range(-5 5) || function t1=tden(1,x),range(-5 5) lp(dash) || function t5=tden(5,x),range(-5 5)  lp(shortdash) ytitle("概率密度")

* 6.4

program onesample,rclass                              
  drop _all				     
  set obs 30					  
  gen x=runiform()		      
  sum x					    	  
  return scalar mean_sample=r(mean)  
end						    	  
set more off                   
simulate xbar=r(mean_sample),seed(101) reps(10000) nodots: onesample
hist xbar,normal 

* 6.5

twoway function y1=normalden(x,4,.2),range(3 8) || function y2=normalden(x,4.5,.3),range(3 8)  || function y3= normalden(x,5,.5),range(3 8) || function y4=normalden(x,5.5,.7),range(3 8) xline(4) ytitle("概率密度")

* 6.6

use price_retail.dta,clear
graph twoway connect price year,yline(100,lp(dash))

* 6.10

use nerlove.dta,clear
reg lntc lnq lnpl lnpk lnpf
display 1/_b[lnq]
test lnq=1

reg lntc lnq lnpl lnpk lnpf,r
test lnq=1

* 6.11

* sample size = 20
program chi2data_20,rclass           
  drop _all								
  set obs 20					          	
  gen x = rchi2(1)			         	
  gen y = 1 + 2*x + rchi2(10)-10	    	
  reg y x								
  return scalar b=_b[x]   	         
end										
set more off   
simulate bhat=r(b),reps(10000) seed(10101) nodots:chi2data_20  
sum bhat
hist bhat,normal

* sample size = 100
program chi2data_100,rclass           
  drop _all								
  set obs 100					          	
  gen x = rchi2(1)			         	
  gen y = 1 + 2*x + rchi2(10)-10	    	
  reg y x								
  return scalar b=_b[x]   	         
end										
simulate bhat=r(b),reps(10000) seed(10101) nodots:chi2data_100  
sum bhat
hist bhat,normal

* sample size = 1000
program chi2data_1000,rclass           
  drop _all								
  set obs 1000					          	
  gen x = rchi2(1)			         	
  gen y = 1 + 2*x + rchi2(10)-10	    	
  reg y x								
  return scalar b=_b[x]   	         
end										
simulate bhat=r(b),reps(10000) seed(10101) nodots:chi2data_1000  
sum bhat
hist bhat,normal

exit

