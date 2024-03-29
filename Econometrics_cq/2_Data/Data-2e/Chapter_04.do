** 第4章，《计量经济学及Stata应用》第2版

* 4.1

use grilic.dta,clear
list s lnw in 1/10
twoway scatter lnw s || lfit lnw s

* 4.7

use grilic.dta,clear
reg lnw s
reg lnw s,noc

* 4.8

use grilic.dta,clear
sum s
return list
display r(sd)/r(mean)

reg lnw s
ereturn list

* 4.9

clear                       
set obs 20	               
set seed 10101		        
gen x = rnormal(3,4)	    
gen e = rnormal(0,9)       
gen y = 1 + 2*x + e	        
reg y x	
twoway function PRF=1+2*x,range(-5 15)  || scatter y x || lfit y x,lp(dash) 

exit

