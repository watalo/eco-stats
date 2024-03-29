** 第12章，《计量经济学及Stata应用》第2版

* 12.13

use lin_1992.dta,clear
xtset province year
xtdes
xtsum ltvfo ltlan ltwlab ltpow ltfer hrs mipric1 giprice mci ngca
xtline ltvfo

reg ltvfo ltlan ltwlab ltpow ltfer hrs mipric1 giprice mci ngca,vce(cluster province) 
estimates store OLS
reg ltvfo ltlan ltwlab ltpow ltfer hrs mipric1 giprice mci ngca

xtreg ltvfo ltlan ltwlab ltpow ltfer hrs mipric1 giprice mci ngca,fe r
estimates store FE_robust

xtreg ltvfo ltlan ltwlab ltpow ltfer hrs mipric1 giprice mci ngca,fe
estimates store FE

reg ltvfo ltlan ltwlab ltpow ltfer hrs mipric1 giprice mci ngca i.province,vce(cluster province)
estimates store LSDV

xtserial ltvfo ltlan ltwlab ltpow ltfer hrs mipric1 giprice mci ngca,output
estimates store FD

xtreg ltvfo ltlan ltwlab ltpow ltfer hrs mipric1 giprice mci ngca t,fe r
estimates store FE_trend

tab year,gen(year)
xtreg ltvfo ltlan ltwlab ltpow ltfer hrs mci ngca year2-year18,fe r
estimates store FE_TW
test year2 year3 year4 year5 year6 year7 year8 year9 year10 year12 year13 year14 year15 year16 year17 year18
xtreg ltvfo ltlan ltwlab ltpow ltfer hrs mci ngca i.year,fe r

xtreg ltvfo ltlan ltwlab ltpow ltfer hrs mci ngca,re r theta
estimates store RE_robust
xttest0

xtreg ltvfo ltlan ltwlab ltpow ltfer hrs mci ngca,re 
estimates store RE

xtreg ltvfo ltlan ltwlab ltpow ltfer hrs mci ngca,mle nolog
estimates store MLE

hausman FE RE,constant sigmamore
* ssc install xtoverid      
quietly xtreg ltvfo ltlan ltwlab ltpow ltfer hrs mci ngca,r
xtoverid

xtreg ltvfo ltlan ltwlab ltpow ltfer hrs mci ngca,be

esttab OLS FE_robust FE_trend FE RE,b se mtitle

exit




