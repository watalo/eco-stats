*** 第18章，《计量经济学及Stata应用》第2版

** Chapter 18 Synthetic Control Method

* ssc install synth, all replace

sysuse synth_smoking,clear     
xtset state year  
synth cigsale lnincome age15to24 retprice beer cigsale(1988) cigsale(1980) cigsale(1975), trunit(3) trperiod(1989) ///
xperiod(1980(1)1988) figure nested sigf(6)

sum cigsale if state==3 & year<1989

* ssc install synth2, all replace

* In-space placebo test and leave-one-out robustness test
synth2 cigsale lnincome age15to24 retprice beer cigsale(1988) cigsale(1980) cigsale(1975), trunit(3) trperiod(1989) /// 
xperiod(1980(1)1988) nested sigf(6) placebo(unit cut(2)) loo

* In-time placebo test
synth2 cigsale lnincome age15to24 retprice beer cigsale(1980) cigsale(1975), trunit(3) trperiod(1989) xperiod(1980(1)1984) /// 
nested sigf(6) placebo(period(1985))

* Mixed placebo test for fake treatment time 1985 with cut(10)
synth2 cigsale lnincome age15to24 retprice beer cigsale(1980) cigsale(1975), trunit(3) trperiod(1985) xperiod(1980(1)1984) /// 
postperiod(1985(1)1988) nested placebo(unit cut(10)) sigf(6)

exit