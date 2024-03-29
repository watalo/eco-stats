** 第9章，《计量经济学及Stata应用》第2版

* 9.4

use icecream.dta,clear
quietly reg consumption temp price income
estat ic
 
qui reg consumption temp L.temp price income
estat ic

qui reg consumption temp L.temp L2.temp price income
estat ic

reg consumption temp L.temp L2.temp price income
reg consumption temp L.temp price income

* 9.5

use grilic.dta,clear 
qui reg lnw s expr tenure smsa rns
estat ovtest
estat ovtest,rhs

gen expr2=expr^2
reg lnw s expr expr2 tenure smsa rns
estat ovtest,rhs

* 9.6

twoway function VIF=1/(1-x),xtitle(R2) xline(0.9,lp(dash)) yline(10,lp(dash)) xlabel(0.1(0.1)1) ylabel(10 100 200 300)

use grilic.dta,clear 
qui reg lnw s expr tenure smsa rns
estat vif

gen s2=s^2
reg lnw s s2 expr tenure smsa rns
estat vif
reg s2 s

sum s
gen sd=(s-r(mean))/r(sd)
gen sd2=sd^2
reg lnw sd sd2 expr tenure smsa rns
estat vif
reg sd2 sd

reg lnw sd expr tenure smsa rns
dis .2290816/2.231828
reg lnw s expr tenure smsa rns

* 9.7

use nerlove.dta,clear 
reg lntc lnq lnpl lnpk lnpf
replace lnq=lnq*100 if _n==1
reg lntc lnq lnpl lnpk lnpf
reg lntc lnq lnpl lnpk lnpf if _n>1

use nerlove.dta,clear 
qui reg lntc lnq lnpl lnpk lnpf
predict lev,leverage
sum lev
dis r(max)/r(mean)
gsort -lev
list lev in 1/3

replace lnq=lnq*100 if _n==1
qui reg lntc lnq lnpl lnpk lnpf
predict lev1,lev
sum lev1
dis r(max)/r(mean)

* 9.9

use consumption.dta,clear
twoway connect c y year,msymbol(circle) msymbol(triangle) 
twoway connect c y year,msymbol(circle) msymbol(triangle) xlabel(1980(10)2010) xline(1992)

reg c y
scalar ssr=e(rss)
reg c y if year<1992
scalar ssr1=e(rss)
reg c y if year>=1992
scalar ssr2=e(rss)
di ((ssr-ssr1-ssr2)/2)/((ssr1+ssr2)/40)

gen d=(year>1991)
gen yd=y*d
reg c y d yd
test d yd

qui reg c y
estat imtest,white
tsset year
estat bgodfrey
dis 44^(1/4)
newey c y d yd,lag(3)
test d yd

* 9.10

use consumption.dta,clear
gen y1=y
replace y1=. if year==1980 | year==1990 | year==2000 | year==2010
ipolate y1 year,gen(y2)
gen lny1=log(y1)
ipolate lny1 year,gen(lny3)
gen y3=exp(lny3)
list year y y2 y3 if year==1980 | year==1990 | year==2000 | year==2010

exit




