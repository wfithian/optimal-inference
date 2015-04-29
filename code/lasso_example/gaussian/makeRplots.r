
data <- read.table("summary.csv",sep=",",header=T)

pdf("tradeoff1_will.pdf",height=3.5,width=3.5)
par(yaxs="i",mar=c(4.1,4.1,0.7,0.2))
matplot(100*data$split,data[,c(6,5,4)],type="l",
        main="",xlab="# data points used for selection",ylab="Probability",
        lty=3:1,col=1,xlim=c(0,100),ylim=c(0,1))
legend(-7,.85,legend=c("Screening","Power, Carving","Power, Splitting"),bty="n",lty=c(3,1,2),
       cex=1)
dev.off()

pdf("tradeoff2_will.pdf",height=3.5,width=3.5)
par(yaxs="i",mar=c(4.1,4.1,0.7,0.2))
matplot(100*data$split,data[,6]*data[,c(5,4)],type="l",
        main="",xlab="# data points used for selection",ylab="Screening x Power",
        lty=2:1,col=1,xlim=c(0,100),ylim=c(0,1))
legend("topleft",legend=c("Data Carving","Data Splitting"),bty="n",lty=c(1,2),
       cex=1)
dev.off()
