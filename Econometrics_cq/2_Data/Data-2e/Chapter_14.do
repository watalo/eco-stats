** 第14章，《计量经济学及Stata应用》第2版

* 14.4

program randwalk,rclass     
  drop _all					
  set obs 1000		   		
  gen eps=rnormal()			
  gen y=sum(eps)				
  gen t=_n 					
  tsset t 					   
  reg y L.y 					
  return scalar b1=_b[L.y]	
end                          
simulate beta=r(b1),seed(10101) reps(1000): randwalk
kdensity beta 


drop _all          
set obs 10000     
set seed 1234    
gen u=rnormal()  
gen y=sum(u)     
set seed 12345    
gen v=rnormal()  
gen x=sum(v)      
reg y x
reg u v
gen t=_n   
line y x t,lp(dash)

* 14.6

use nelson_plosser.dta,clear
tsline lrgnp lun if year>=1890,lp(dash) xlabel(1890(10)1970)
dfuller lrgnp,trend
di 12*(62/100)^(1/4)
dfuller lrgnp,lags(9) trend reg
dfuller lrgnp,lags(1) trend reg
dfuller d.lrgnp

dfuller lun,lags(3) reg

* 14.7

use macro_3e.dta,clear
tsset time
tsline fygm3 fygt1,lp(dash)

* 14.9

use mpyr.dta,clear
tsline logmr logy r,lp(solid dash shortdash) xlabel(1900(10)1990)
varsoc logmr logy r
vecrank logmr logy r,lags(2) trend(trend) max
vec logmr logy r,lags(2) rank(1)

veclmar
vecstable,graph

reg logmr logy r

exit




