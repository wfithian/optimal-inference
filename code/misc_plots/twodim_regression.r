
# Fig 5a
pdf("fullvred.pdf",height=4.3,width=4)
par(mar=c(4.1,4.1,3.1,0.1))
y <- t(c(2.9,2.5))
plot(y,xlim=c(-5,5),ylim=c(-5,5),xlab=expression(Y[1]),ylab=expression(Y[2]),
     main="Conditioning Sets")#"Full vs. Reduced Model: First Step",asp=1)
polygon(c(0,10,10),c(0,10,-10),lty=2,col="#F4E918")
polygon(c(0,-10,-10),c(0,10,-10),lty=2,col="#F4E918")
abline(h=0)
abline(v=0)
text(2,.5,"A")
text(y+c(.3,-.4),labels="Y")
lines(c(y[2],10),c(y[2],y[2]),lwd=2,col="brown")
lines(c(-y[2],-10),c(y[2],y[2]),lwd=2,col="brown")
points(y,pch=16)
dev.off()

# Fig 5b
pdf("fullvredNulls.pdf",height=4.3,width=4)
par(mar=c(4.1,4.1,3.1,0.1),yaxs="i")
x <- seq(-6,6,.01)
plot(x,(abs(x)>2.5)*dnorm(x)/2/pnorm(-2.5),ylim=c(0,1.4),lty=1,
      col="brown",type="l",
      main="Conditional Null Distributions",
      ylab="Density",xlab=expression(Y[1]))
polygon(c(x,0),c(abs(x)*dnorm(x)/2/integrate(function(u) u*dnorm(u),0,10)$value,0),lty=2,col="#F4E918")
lines(x,(abs(x)>2.5)*dnorm(x)/2/pnorm(-2.5),col="brown")
legend("topleft",legend=c("Saturated Model","Selected Model","Observed Value"),lty=1:3,bg="white", col=c("brown","black","black"))
#norm.y <- sqrt(sum(y^2))
#curve((abs(x)>2.5)*dbeta((x/norm.y)^2,.5,.5)*abs(x/norm.y)/norm.y/2,-norm.y,norm.y,add=T)
abline(v=2.9,lty=3)
dev.off()

# p-values for selected and saturated models
integrate(function(x) abs(x)*dnorm(x)/integrate(function(u) u*dnorm(u), 0, 10)$value, 2.9,10)
pnorm(-2.9)/pnorm(-2.5)

B <- 10000
mu <- c(5,5)
pvals <- NULL
for(b in 1:B) {
    y <- mu + rnorm(2)
    if(abs(y[1]) > abs(y[2])) {
        pvals <- rbind(pvals, c(
            integrate(function(x) abs(x)*dnorm(x)/integrate(function(u) u*dnorm(u), 0, 10)$value, abs(y[1]),10)$value,
            pnorm(-abs(y[1]))/pnorm(-abs(y[2]))
            ))
    }
}
mean(pvals[,1]<.05)
mean(pvals[,2]<.05)

#hist(cos(2*pi*runif(1000000)),freq=F,breaks=seq(-1,1,.025))
#curve(dbeta(x^2,.5,.5)*abs(x),-1,1,add=T)

pdf("fullvredXty.pdf",height=4.3,width=4)
par(mar=c(4.1,4.1,2.1,0.1))
y <- t(c(2.9,2.5))
plot(y,xlim=c(-5,5),ylim=c(-5,5),
     xlab=expression(paste(X[1],"' Y",sep="")),
     ylab=expression(paste(X[2],"' Y",sep="")),
     main="Full vs. Reduced Model: First Step",asp=1)
polygon(c(0,10,10),c(0,10,-10),lty=2,col="#F0F0FF")
polygon(c(0,-10,-10),c(0,10,-10),lty=2,col="#F0F0FF")
abline(h=0)
abline(v=0)
text(2,.5,"A")
text(y+c(.3,-.4),labels="X'Y")
lines(c(y[2],10),c(y[2],y[2]),lwd=2,col="blue")
lines(c(-y[2],-10),c(y[2],y[2]),lwd=2,col="blue")
points(y)
dev.off()
