#!/usr/bin/env Rscript
# Descriptive trees on the same standardized log scale as the predictive models.
source('analysis/pheme_design.R')
output_dir <- file.path(root,'runs/pheme_embedding_comparison/engagement/qwen_log_outcomes')
meta <- jsonlite::fromJSON(file.path(output_dir,'metadata.json'))
for(nm in names(paths)) stopifnot(unname(tools::md5sum(paths[[nm]]))==meta$inputs_md5[[nm]])
source('analysis/pheme_tree_helpers.R')
frames <- list(); splits <- list(); usage <- list(); preds <- list()
for(variant in names(tree_designs)) for(outcome in c('retweet_count','favorite_count')) {
 x <- tree_designs[[variant]]
 train <- as.data.frame(x[train_index,,drop=FALSE]); test <- as.data.frame(x[test_index,,drop=FALSE])
 yy <- log1p(data[[outcome]]); zz <- (yy-mean(yy[train_index]))/sd(yy[train_index]);train$.outcome <- zz[train_index]
 fit <- rpart(.outcome~.,data=train,method='anova',control=rpart.control(cp=.0005,maxdepth=8L,minsplit=30L,minbucket=15L,xval=0),model=TRUE)
 stem <- paste0('pheme_log_tree_',variant,'_',outcome)
 saveRDS(fit,file.path(output_dir,paste0(stem,'.rds')))
 title <- paste(if(outcome=='retweet_count')'Retweets' else 'Favorites','|',switch(variant,bits_only='Bits only',bits_controls='Bits + controls',all_variables='Bits + labels + controls'))
 for(ext in c('pdf','png')) {
  if(ext=='pdf') pdf(file.path(output_dir,paste0(stem,'.pdf')),width=14,height=9) else png(file.path(output_dir,paste0(stem,'.png')),width=2800,height=1800,res=200)
  rpart.plot::prp(fit,type=4,extra=101,fallen.leaves=TRUE,branch=1,box.palette='GnBu',nn=TRUE,varlen=0,faclen=0,tweak=.84,digits=3,under.percent=2,split.fun=pretty_tree_split_labels,main=paste(title,'\nNode means: standardized log(1 + count)'))
  dev.off()
 }
 f <- fit$frame;ids <- as.integer(rownames(f)); key <- paste(variant,outcome)
 frames[[key]] <- data.frame(variant=variant,outcome=outcome,node=ids,feature=f$var,n=f$n,mean_z=f$yval,rss=f$dev)
 inner <- which(f$var!='<leaf>')
 ss <- data.frame(variant=variant,outcome=outcome,node=ids[inner],feature=f$var[inner],improvement=vapply(inner,function(j) f$dev[j]-f$dev[match(2*ids[j],ids)]-f$dev[match(2*ids[j]+1,ids)],numeric(1)))
 splits[[key]] <- ss
 for(j in inner) {
  path <- path.rpart(fit,nodes=ids[j],print.it=FALSE)[[1]]
  # Record paths to both children: direction of bit activation remains explicit.
  for(child in c(2*ids[j],2*ids[j]+1)) {
   childpath <- path.rpart(fit,nodes=child,print.it=FALSE)[[1]]
   usage[[length(usage)+1L]] <- data.frame(variant=variant,outcome=outcome,node=child,path=paste(childpath,collapse='; '),n=f$n[match(child,ids)],mean_z=f$yval[match(child,ids)])
  }
 }
 preds[[key]] <- data.frame(id=data$id[test_index],variant=variant,outcome=outcome,observed_z=zz[test_index],predicted_z=as.numeric(predict(fit,newdata=test)))
 message(title,' root: ',f$var[1])
}
fwrite(rbindlist(frames),file.path(output_dir,'tree_nodes.csv'))
fwrite(rbindlist(splits),file.path(output_dir,'tree_split_improvement.csv'))
fwrite(rbindlist(usage),file.path(output_dir,'tree_paths.csv'))
fwrite(rbindlist(preds),file.path(output_dir,'tree_predictions.csv'))
