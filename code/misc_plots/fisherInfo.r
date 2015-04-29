
cut.vals <- seq(-6,6,.1)

tail.prob <- EY <- VarY <- NULL
i <- 40
for(i in 1:length(cut.vals)) {
    cut <- cut.vals[i]
    tail.prob[i] <- integrate(dnorm,cut,Inf)$value
    EY[i] <- integrate(function(y) y*dnorm(y)/tail.prob[i],cut,Inf)$value
    VarY[i] <- integrate(function(y) (y-EY[i])^2*dnorm(y)/tail.prob[i],cut,Inf)$value
}

def.par <- par(no.readonly=TRUE)

pdf("informationUnivar.pdf",height=4,width=4)
par(mar=c(4.1,4.1,3.1,1.1))
plot(3-cut.vals,VarY,type="l",xlab=expression(mu),
     ylab="Information",
     main="Leftover Fisher Information",
     yaxt="n")
axis(2,c(0,.5,1))
dev.off()

pdf("informationBivar.pdf",height=4,width=4)
par(mar=c(4.1,4.1,3.1,1.1))
plot(3-cut.vals,1+VarY,type="l",ylim=c(0,2),
     xlab=expression(mu),
     ylab="Information",
     main="Leftover Fisher Information",
     yaxt="n")
axis(2,0:2)
abline(1,0,lty=2)
legend("bottomleft",legend=c("Data Splitting","Data Carving"),lty=2:1,bty="n")
dev.off()


pdf("information.pdf",height=4,width=7)
layout(matrix(c(1,1,2,3),2,2,byrow=TRUE),heights=c(1,7))
#par(mfrow=c(1,2),mar=c(2.1,3.1,3.1,0.1))
par(mar = c(0,0,0,0))
plot(0:1,0:1,type="n",xaxt="n",yaxt="n",ann = F, bty = 'n')
text(.5,.5,"Leftover Information",cex=1.5)
par(mar = c(4.1,4.1,3.1,0.1))
plot(3-cut.vals,VarY,type="l",xlab=expression(mu),ylab="Fisher Information",
     main=expression(paste(N(mu,1),"1{Y>3}")),
     yaxt="n")
axis(2,c(0,.5,1))
par(mar=c(4.1,3.1,3.1,1.1))
plot(3-cut.vals,1+VarY,type="l",ylim=c(0,2),
     xlab=expression(mu),ylab="Fisher Information",
     main=expression(paste(N(mu*1[2],I[2])*1,group("{",Y[1]>3,"}"))),
     yaxt="n")
axis(2,0:2)
abline(1,0,lty=2)
legend("bottomleft",legend=c("Sample Splitting","Sample Carving"),lty=2:1,bty="n")
dev.off()


rt.ar <- function() arrows(.95,.5,1,.5,.1)
lt.ar <- function() arrows(.05,.5,0,.5,.1)
up.ar <- function() arrows(.5,.9,.5,1,.1)
dn.ar <- function() arrows(.5,.05*14/8.5,.5,0,.1)
num <- function(n)
    text(0,.9,
         paste(n,
               c(". Draw ",". Describe ")[n %% 2 + 1],
               ifelse(n>1,n-1,"a scene"),
               sep=""),
         pos=4)

panel <- function(n) {
    plot(0:1,0:1,type="n",xaxt="n",yaxt="n",ann = F)
    num(n)
    if(n < 10) {
        if(n %% 4 == 1) rt.ar()
        if(n %% 2 == 0) dn.ar()
        if(n %% 4 == 3) lt.ar()
    }
}


pdf("pictionary.pdf",height=14,width=8.5)
layout(matrix(
       c(1,2,
         4,3,
         5,6,
         8,7,
         9,10),5,byrow=TRUE))
op <- par(mar=c(0,0,0,0))
for(n in 1:10) panel(n)
dev.off()
