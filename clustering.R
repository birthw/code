

d=read.table("EM_output.txt", as.is=T, header=T)

d$se.paternal <- ifelse(d$x2_full_vs_no_paternal < .1, NA, abs(d$beta_full_paternal/sqrt(d$x2_full_vs_no_paternal)))
d$se.maternal <- ifelse(d$x2_full_vs_no_maternal < .1, NA, abs(d$beta_full_maternal/sqrt(d$x2_full_vs_no_maternal)))
d$se.non_trans <- ifelse(d$x2_full_vs_no_non_trans < .1, NA, abs(d$beta_full_non_trans/sqrt(d$x2_full_vs_no_non_trans)))

d$z.paternal <- ifelse(d$x2_full_vs_no_paternal < .00001, NA, abs(sqrt(d$x2_full_vs_no_paternal)))
d$z.maternal <- ifelse(d$x2_full_vs_no_maternal < .00001, NA, abs(sqrt(d$x2_full_vs_no_maternal)))
d$z.non_trans <- ifelse(d$x2_full_vs_no_non_trans < .00001, NA, abs(sqrt(d$x2_full_vs_no_non_trans)))
d$z=apply(d[,c("z.non_trans","z.paternal","z.maternal")],1,FUN=max,na.rm=TRUE)


se.ratio <- apply(subset(d, !is.na(se.paternal) & !is.na(se.maternal) & !is.na(se.non_trans))[,c("se.paternal", "se.maternal", "se.non_trans")], 2, mean)
se.ratio <- se.ratio/mean(se.ratio)
se.factor <- apply(t(d[,c("se.paternal", "se.maternal", "se.non_trans")])/se.ratio, 2, mean, na.rm=TRUE)


##################  P  M  N
classes <- matrix(c(1, 1, 0,
                    1, 0, 0,
                    0, 1, 0,
                   0, 1, 1,
                    1, 1/2, -1/2,
                    -1/2,1/2,1,
                    1/2, 1, 1/2,
                    1,0,1 ), byrow=TRUE, ncol=3)

class.prob <- matrix(NA, nrow=nrow(d), ncol=nrow(classes))
beta.max <- apply(d[,c("beta_full_paternal", "beta_full_maternal", "beta_full_non_trans")], 1, max)
beta.min <- apply(d[,c("beta_full_paternal", "beta_full_maternal", "beta_full_non_trans")], 1, min)
beta <- ifelse(abs(beta.max) < abs(beta.min), abs(beta.min), beta.max)
for (j in 1:20) {
    for (i in 1:nrow(d))
        for (k in 1:nrow(classes))
            class.prob[i, k] <- prod(dnorm(unlist(d[i,c("beta_full_paternal", "beta_full_maternal", "beta_full_non_trans")]), mean=beta[i]*classes[k,], sd=se.factor[i]*se.ratio))
    class.prob <- class.prob/apply(class.prob, 1, sum)

    Emu <- class.prob%*%classes
    Emu2 <- class.prob%*%(classes^2)
    beta <- apply(Emu*as.matrix(d[,c("beta_full_paternal", "beta_full_maternal", "beta_full_non_trans")]), 1, sum)/apply(Emu2, 1, sum)
}

d$z=apply(d[,c("z.non_trans","z.paternal","z.maternal")],1,FUN=max, na.rm=T)
dd=data.frame(class.prob)

library(dplyr)
library(mgcv)
library(tibble)
library(tidyr)

d$id=c(1:dim(d)[1])
dd2=na.omit(dd)
ddd=dd2 %>%
 rownames_to_column('id') %>%  # creates an ID number
  gather(dept, cnt, X1:X8) %>%
  group_by(as.numeric(id)) %>%
  arrange(id)   %>%
  slice(which.max(cnt))%>% as.data.frame()
dd2=na.omit(dd)
dat=merge(d,ddd,by="id")
d4=cbind(dat,na.omit(class.prob))




