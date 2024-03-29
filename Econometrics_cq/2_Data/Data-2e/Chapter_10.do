** 第10章，《计量经济学及Stata应用》第2版

* 10.9

use grilic.dta,clear
reg lnw s expr tenure rns smsa,r 
reg lnw s iq expr tenure rns smsa,r
ivregress 2sls lnw s expr tenure rns smsa (iq=med kww),r first

estat overid

quietly ivregress 2sls lnw s expr tenure rns smsa (iq=med kww)	
estat firststage
ivregress liml lnw s expr tenure rns smsa (iq=med kww),r

qui reg lnw iq s expr tenure rns smsa
estimates store ols
qui ivregress 2sls lnw s expr tenure rns smsa (iq=med kww)
estimates store iv
hausman iv ols,constant sigmamore 

estat endogenous

qui reg lnw s expr tenure rns smsa,r
est sto ols_no_iq
qui reg lnw iq s expr tenure rns smsa,r
est sto ols_with_iq
qui ivregress 2sls lnw s expr tenure rns smsa (iq=med kww),r
est sto tsls
qui ivregress liml lnw s expr tenure rns smsa (iq=med kww),r
est sto liml
estimates table ols_no_iq ols_with_iq tsls liml,b se 
estimates table ols_no_iq ols_with_iq tsls liml,star(0.1 0.05 0.01) 

* ssc install estout 
esttab ols_no_iq ols_with_iq tsls liml,se r2 mtitle star(* 0.1 ** 0.05 *** 0.01) 
esttab ols_no_iq ols_with_iq tsls liml using iv.rtf,se r2 mtitle star(* 0.1 ** 0.05 *** 0.01) replace

exit




