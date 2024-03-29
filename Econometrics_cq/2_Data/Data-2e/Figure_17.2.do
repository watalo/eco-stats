* Ashenfelter's dip

use ashenfelter_dip.dta,clear

twoway (connect trainee year,msize(small) xline(1964)) (connect comparison year,lp(dash) msize(small) xlabel(1959(1)1969))