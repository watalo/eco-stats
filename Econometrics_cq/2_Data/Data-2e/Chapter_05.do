** 第5章，《计量经济学及Stata应用》第2版

* 5.1

use cobb_douglas.dta, clear
list
reg lny lnk lnl
predict lny1
predict e,residual
list lny lny1 e
line lny lny1 year,lp(solid dash)

* 5.12

use grilic.dta,clear
reg lnw s expr tenure smsa rns
vce
reg lnw s expr tenure smsa rns,noc

reg lnw s expr tenure smsa if rns
reg lnw s expr tenure smsa if ~rns
reg lnw s expr tenure smsa if !rns

reg lnw s expr tenure smsa rns if s>=12
reg lnw s expr tenure smsa if s>=12 & rns

quietly reg lnw s expr tenure smsa rns
predict lnw1
predict e,residual

test s=0.1
dis ttail(752,0.45188757)*2
dis ttail(752,0.45188757)
test expr=tenure
test expr+tenure=s

exit

