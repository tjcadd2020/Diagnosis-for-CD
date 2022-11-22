library(circlize)
############################ plot

pdf('snv.pdf')

circos.clear()
circos.initialize(factors=c("chr1"), 
xlim=matrix(c(0,max(cov$end)), ncol=2))
col_text <- "grey40"



# genomes x axis
brk <- seq(0,6000,500)*10^3 

lb = c()
for(i in 1:length(round(brk/10^3, 1))){
    c1 = as.character(round(brk/10^3, 1)[i])
    c2 = 'kb'
    c3 = paste(c1,c2,sep = '')
    lb = c(lb,c3)
    
}


circos.track(ylim=c(0, 1), panel.fun=function(x, y) {
    
chr=CELL_META$sector.index
xlim=CELL_META$xlim
ylim=CELL_META$ylim
    
# circos.text(mean(xlim), ylim[1] + .1, lb, cex=1, col=col_text, facing="clockwise", niceFacing=TRUE,adj = c(0, 90))
circos.axis(h="top", major.at=brk, labels=lb, labels.cex=1.5, col='black', labels.col='black', lwd=1.1)
}, bg.col="#F5E8C7", bg.border=T, track.height=0.1)


# gc content
circos.track(factors=gc$chr, x=gc$start, y=gc$value1, panel.fun=function(x, y) {
circos.lines(x, y, col="#1687A7", lwd=1.2) #035397 
circos.segments(x0=0, x1=max(gc$end), y0=0.3, y1=0.3, lwd=0.6, lty="11", col="grey90")
circos.segments(x0=0, x1=max(gc$end), y0=0.5, y1=0.5, lwd=0.6, lty="11", col="grey90")
circos.segments(x0=0, x1=max(gc$end), y0=0.7, y1=0.7, lwd=0.6, lty="11", col="grey90")
}, ylim=range(gc$value1), track.height=0.15, bg.border=F)
# gc y axis
circos.yaxis(at=c(0.3, 0.5, 0.7), labels.cex=0.25, lwd=0, tick.length=0, labels.col="grey80", col="#FFFFFF")


# coverage
circos.genomicTrack(data=cov, panel.fun=function(region, value, ...) {
circos.genomicLines(region, value, type="l", col="#6E85B7", lwd=1.2) #377D71
circos.segments(x0=0, x1=max(cov$end), y0=0, y1=0, lwd=0.6, lty="11", col="grey90")
circos.segments(x0=0, x1=max(cov$end), y0=50, y1=50, lwd=0.6, lty="11", col="grey90")
    circos.segments(x0=0, x1=max(cov$end), y0=100, y1=100, lwd=0.6, lty="11", col="grey90")
#circos.segments(x0=0, x1=max(cov$end), y0=500, y1=500, lwd=0.6, lty="11", col="grey90")
}, track.height=0.15, bg.border=F)
circos.yaxis(at=c(0, 50, 100), labels.cex=0.25, lwd=0, tick.length=0, labels.col="grey80", col="#FFFFFF")





# # coverage的散点图
circos.genomicTrack(cov_marker, 
                    panel.fun = function(region, value, ...) {
                      circos.genomicPoints(region, value, pch = 16, cex = 0.4, col = "#AD8B73", ...)}, track.height=0.08, bg.border=F)


# gene labels
circos.genomicLabels(gene, labels.column=5, cex=1, col='grey40', line_lwd=0.8, line_col="grey40", 
side="inside", connection_height=0.05, labels_height=0.04)


dev.off()