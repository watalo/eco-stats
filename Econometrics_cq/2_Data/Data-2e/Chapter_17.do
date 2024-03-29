*** 第17章，《计量经济学及Stata应用》第2版

** Chapter 17 Difference in Differences

use cao_chen.dta, clear
xtset county year

** Transforming the Dependent Variable

tab rebel_num
gen rebel_density = rebel_num / (pop1600/1000000)
sum rebel_density
hist rebel_density,fraction
twoway function asinh=asinh(x),range(-20 20) xline(0,lp(dot)) yline(0,lp(dot)) || function log=log(x),range(-20 20) lp(dash) 
gen rebel = asinh(rebel_density)
sum rebel

** Paralell Trend Plot for Each Year

preserve 
collapse (mean) mean_year=rebel,by(canal year)
twoway (connect mean_year year if canal==1,msize(small)) (connect mean_year year if canal==0,lp(dash) msize(small) xline(1825,lp(dash)) legend(label(1 Canal Counties) label(2 Non-canal Counties)))
restore  

* Paralell Trend Plot for Each Decade

gen period=floor((year-1826)/10)*10 
replace period=-60 if period<-60
tabstat year,by(period) stat(min max) nototal

tab period, gen(period)
describe period*

preserve 
collapse (mean) mean_decade=rebel if period<80,by(canal period)
twoway (connect mean_decade period if canal==1,xtitle(Number of years since the 1826 reform)) (connect mean_decade period if canal==0,lp(dash) xline(-5,lp(dash)) xlabel(-60 "-60" -50 "-50" -40 "-40" -30 "-30" -20 "-20" -10 "-10" 0 "10" 10 "20" 20 "30" 30 "40" 40 "50" 50 "60" 60 "70" 70 "80") legend(label(1 Canal Counties) label(2 Non-canal Counties)))
restore  

** Paralell Trend Test restricting period<80

* ssc install reghdfe, replace

global cov larea_after rug_after disaster disaster_after flood drought flood_after drought_after popden1600_after maize maize_after sweetpotato sweetpotato_after wheat_after rice_after

reghdfe rebel c.canal#(c.period2-period14) $cov if period<80, absorb(i.county i.year c.prerebel#i.year i.prov#i.year) cluster(county) 

test c.canal#c.period2 c.canal#c.period3 c.canal#c.period4 c.canal#c.period5 c.canal#c.period6 

* ssc install coefplot,replace

coefplot,vertical keep(c.canal*) msymbol(circle_hollow) ciopts(lp(dash) recast(rcap)) addplot(line @b @at) xtitle(Number of years since the 1826 Reform)  xlabel(1 "-50" 2 "-40" 3 "-30" 4 "-20" 5 "-10" 6 "10" 7 "20" 8 "30" 9 "40" 10 "50" 11 "60" 12 "70" 13 "80") xline(5.5, lp(dash) lwidth(vthin)) ytitle(Coefficients)  ylabel(-0.1(0.05)0.3) yline(0,lp(dash) lwidth(vthin)) 


** Table 3: Baseline Results

reghdfe rebel canal_post, absorb(i.county i.year) cluster(county)
est sto fe

reghdfe rebel canal_post, absorb(i.county i.year c.prerebel#i.year) cluster(county)
est sto prerebel

reghdfe rebel canal_post, absorb(i.county i.year c.prerebel#i.year i.prov#i.year) cluster(county)
est sto prov

reghdfe rebel canal_post, absorb(i.county i.year c.prerebel#i.year i.prov#i.year i.pref#c.year) 
est sto pref

reghdfe rebel canal_post $cov, absorb(i.county i.year c.prerebel#i.year i.prov#i.year i.pref#c.year) cluster(county)
est sto cov

esttab fe prerebel prov pref cov, se r2 mtitle nogap star(* 0.1 ** 0.05 *** 0.01) 

exit
