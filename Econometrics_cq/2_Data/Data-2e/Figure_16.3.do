* Program for 图16.3

clear
set obs 500
gen x = 250*runiform()
gen u = rnormal()
gen y = (x-50)^2 + 500*u
* twoway (scatter y x,msymbol(oh) msize(small) xline(0)) (lfit y x if x<0) (function y=-300+130*x,range(0 50)) 

twoway (scatter y x,msymbol(oh) msize(small) xline(150) xtitle("running variable(x)") ytitle("outcome variable(y)") ylabel("")) (lfit y x if x<150) (lfit y x if x>=150)

