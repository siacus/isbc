suppressPackageStartupMessages({library(rpart); library(rpart.plot)})
p <- 'runs/pheme_embedding_comparison/engagement/qwen_log_outcomes'
source('analysis/pheme_tree_labels.R')
for(path in list.files(p,pattern='^pheme_log_tree_.*\\.rds$',full.names=TRUE)) {
 fit <- readRDS(path);nodes <- as.integer(rownames(fit$frame));cut <- nodes[nodes>=8 & nodes<16 & fit$frame$var!='<leaf>']
 shown <- if(length(cut)) snip.rpart(fit,toss=cut) else fit
 shown$frame$yval <- round(shown$frame$yval,3)
 stem <- sub('\\.rds$','_overview',path)
 for(ext in c('pdf','png')) {
  if(ext=='pdf') pdf(paste0(stem,'.pdf'),width=14,height=6) else png(paste0(stem,'.png'),width=2800,height=1200,res=200)
  rpart.plot::prp(shown,type=4,extra=101,fallen.leaves=TRUE,branch=1,box.palette='GnBu',nn=FALSE,varlen=0,faclen=0,tweak=1.0,digits=3,under.percent=2,split.fun=pretty_tree_split_labels)
  dev.off()
 }
}
