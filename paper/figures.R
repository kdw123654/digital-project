args <- commandArgs(trailingOnly = TRUE)
root <- normalizePath(if (length(args)) args[1] else '.', mustWork = TRUE)
dir.create(file.path(root, 'figures'), showWarnings = FALSE)
metrics <- read.csv(file.path(root, 'metrics.csv'), stringsAsFactors = FALSE)
classes <- read.csv(file.path(root, 'class_ap.csv'), stringsAsFactors = FALSE)
if (.Platform$OS.type == 'windows') windowsFonts(Arial = windowsFont('Arial'))
ink <- '#243241'; blue <- '#235E83'; orange <- '#B76124'; gray <- '#66727E'
device <- function(name, height) {
  png(file.path(root, 'figures', paste0(name, '.png')), width = 3.49,
      height = height, units = 'in', res = 600, type = 'cairo', pointsize = 10)
  par(family = 'Arial', col = ink, col.axis = ink, col.lab = ink,
      bg = 'white', las = 1, mgp = c(1.7, .35, 0), tcl = -.2)
}

# Fig. 1: cohort selection and the two training routes. All counts are observed.
device('fig1_pipeline', 2.46)
par(mar = c(.1, .1, .1, .1), xaxs = 'i', yaxs = 'i')
plot.new(); plot.window(xlim = c(0, 1), ylim = c(0, 1))
box_text <- function(x, y, w, h, label, fill = '#F2F5F7', cex = .93) {
  rect(x-w/2, y-h/2, x+w/2, y+h/2, col = fill, border = '#A7B3BF', lwd = .7)
  text(x, y, label, cex = cex)
}
arr <- function(x1, y1, x2, y2) arrows(x1, y1, x2, y2, length=.045, lwd=.8, col=gray)
text(.5, .958, 'KNHANES 2020-2024', font=2, cex=1.04)
box_text(.5, .83, .92, .17, 'Confirmed status and eligible fasting labs\nAge 19-39: 5,229 / age 20-39: 5,055', cex=.90)
arr(.5, .743, .5, .698)
box_text(.5, .622, .92, .15, 'Nutrition participants: 4,508\n15 predictors -> 45 encoded columns', cex=.90)
arr(.5, .546, .5, .50)
box_text(.5, .44, .92, .12, 'Shared PSU folds; fit-only preprocessing', '#E9F0F5', .90)
arr(.33, .379, .27, .326); arr(.67, .379, .73, .326)
box_text(.26, .232, .45, .182, 'Direct classification\nLR / RF / XGB\nLGBM / MLP', cex=.90)
box_text(.74, .232, .45, .182, 'Joint distribution\njLinear / jMLP\n/ jEPF', cex=.90)
arr(.26, .14, .40, .08); arr(.74, .14, .60, .08)
text(.5, .035, 'Same five groups; weighted OOF evaluation', cex=.90, font=2)
dev.off()

# Fig. 2: zoomed scale shows the observed differences without hiding intervals.
device('fig2_macro_ap', 2.49)
par(mar=c(3.05, 4.0, .35, .45))
ord <- match(c('LR','RF','XGB','LGBM','MLP','jLinear','jMLP','jEPF'), metrics$model)
d <- metrics[ord, ]; y <- rev(seq_len(nrow(d)))
plot(d$macro_ap, y, type='n', axes=FALSE, xlim=c(.328,.373), ylim=c(.5,8.5),
     xlab='Weighted macro AP', ylab='')
rect(.328,.5,.373,8.5,col='#FAFBFC',border=NA)
abline(v=seq(.33,.37,.01), col='#E1E5E9', lwd=.65)
axis(1, at=seq(.33,.37,.01), labels=sprintf('%.2f',seq(.33,.37,.01)), cex.axis=.95)
axis(2, at=y, labels=d$model, tick=FALSE, cex.axis=.99)
segments(d$ci_low,y,d$ci_high,y,col=gray,lwd=1.05)
segments(d$ci_low,y-.09,d$ci_low,y+.09,col=gray,lwd=.8)
segments(d$ci_high,y-.09,d$ci_high,y+.09,col=gray,lwd=.8)
points(d$macro_ap,y,pch=21,bg=blue,col=blue,cex=.85)
box(col='#CAD2DA',lwd=.6)
dev.off()

# Fig. 3: every model is shown for each abnormal group. Horizontal jitter is
# visual only. Dashed baselines are weighted class prevalence, not thresholds.
device('fig3_class_ap', 2.57)
par(mar=c(2.7, 3.7, .5, .3))
cn <- c('High_BP','High_Glu','Dyslipid','Complex')
labels <- c('BP only','Glucose\nonly','Lipid\nonly','Multiple')
plot(1:4, rep(0,4), type='n', axes=FALSE, xlim=c(.72,4.28), ylim=c(.025,.47),
     xlab='', ylab='Weighted class AP')
rect(.72,.025,4.28,.47,col='#FAFBFC',border=NA)
abline(h=seq(.1,.4,.1), col='#E1E5E9', lwd=.65)
axis(1,at=1:4,labels=FALSE,tick=FALSE)
text(1:4, par('usr')[3]-.018, labels=labels, adj=c(.5,1), xpd=NA, cex=.90)
axis(2,at=seq(.1,.4,.1),labels=sprintf('%.1f',seq(.1,.4,.1)),cex.axis=.94)
model_order <- c('LR','RF','XGB','LGBM','MLP','jLinear','jMLP','jEPF')
jit <- seq(-.11,.11,length.out=8)
for (i in seq_along(model_order)) {
  rows <- classes[classes$model==model_order[i], ]
  values <- rows$ap[match(cn,rows$class)]
  if (!(model_order[i] %in% c('LR','LGBM','jEPF'))) {
    points((1:4)+jit[i],values,pch=1,col='#B3BBC3',cex=.62)
  }
}
for (m in c('LR','LGBM','jEPF')) {
  rows <- classes[classes$model==m, ]; values <- rows$ap[match(cn,rows$class)]
  i <- match(m,model_order)
  color <- c(LR=gray,LGBM=blue,jEPF=orange)[m]
  symbol <- c(LR=15,LGBM=16,jEPF=17)[m]
  points((1:4)+jit[i],values,pch=symbol,col=color,cex=.83)
}
prev <- classes[classes$model=='LR', ]
prev <- prev$prevalence[match(cn,prev$class)]
segments((1:4)-.19,prev,(1:4)+.19,prev,col=gray,lty=2,lwd=.9)
legend('topleft',c('LR','LGBM','jEPF'),pch=c(15,16,17),col=c(gray,blue,orange),
       horiz=TRUE,bty='n',cex=.90,x.intersp=.65,y.intersp=.8)
box(col='#CAD2DA',lwd=.6)
dev.off()
writeLines(c(R.version.string, capture.output(sessionInfo())), file.path(root,'R_SESSION.txt'))
cat('Saved three publication figures with R\n')
