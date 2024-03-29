** 第13章，《计量经济学及Stata应用》第2版

* 13.1

use gdp_china.dta,clear
tsset year
tsline y,xlabel(1980(10)2010)

gen lny=log(y)
tsline lny,xlabel(1980(10)2010)

gen dlny=d.lny
tsline dlny,xlabel(1980(10)2010)

gen g=(y-l.y)/l.y
tsline dlny g,xlabel(1980(10)2010) lp(dash)

corrgram dlny
ac dlny,lags(20)

* 13.2

reg dlny l.dlny if year<2013,r
predict dlny1
list dlny1 if year==2021
dis exp(lny[35]+dlny1[36])
dis y[36]
dis y[36]-exp(lny[35]+dlny1[36])

* 13.3

quietly reg dlny l.dlny if year<2013,r
estat ic

reg dlny l(1/2).dlny if year<2013,r
estat ic

reg dlny l(1/3).dlny if year<2013,r
estat ic

quietly reg dlny l(1/2).dlny if year<2013,r
predict dlny2
dis exp(lny[35]+dlny2[36])
dis y[36]-exp(lny[35]+dlny2[36])

* 13.4

use border.dta, clear
tsset decade
reg border l(1/2).border l.drought diff age rival wall unified,r
dis -.6333046/(1-1.518284+.5586965)

* 13.7

use gdp_china.dta,clear
gen lny=log(y)
gen dlny=d.lny
varbasic dlny if year<2013,lags(1) irf
varbasic dlny if year<2013,irf

* 13.11

use macro_swatson.dta,clear
tsline dinf unem,lp(solid dash)
varsoc dinf unem
var dinf unem,lags(1/2)
varwle
varlmar
varstable,graph
vargranger

irf create iu, set(macro) replace
irf graph oirf,yline(0)
irf create ui, order(unem dinf) replace
irf graph oirf,i(unem) r(dinf) yline(0)
irf graph oirf,i(dinf) r(unem) yline(0)
varbasic dinf unem if quarter<tq(1999q1),lags(1/2) nograph
fcast compute f_,step(13)
fcast graph f_dinf f_unem,observed lp(dash)

* 13.12

clear
set obs 100
gen t=_n
gen t2=t^2
corr t t2

use gdp_china.dta,clear
gen t=_n
gen lny=log(y)
reg lny l(1/2).lny t if year<2013,r
predict lny3
dis exp(lny3[36])
dis y[36]-exp(lny3[36])

* 13.13

use airpassengers.dta,clear
tsset time
tsline airpassengers
gen month=month((dofm(time)))
tab month,gen(m)
reg airpassengers m2-m12
predict air_sa,r
sum airpassengers
gen airpassengers_sa = air_sa+r(mean)
tsline airpassengers_sa airpassengers,lp(dash)

exit




