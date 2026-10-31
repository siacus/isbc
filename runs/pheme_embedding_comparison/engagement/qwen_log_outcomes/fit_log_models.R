#!/usr/bin/env Rscript
# Reuse the exact saved design and split; no encoder fitting and no test tuning.
source('analysis/pheme_design.R')
set.seed(seed)
boot <- matrix(sample.int(sum(test_index), sum(test_index)*bootstrap_draws, replace=TRUE), nrow=bootstrap_draws, ncol=sum(test_index))
saveRDS(boot,file.path(output_dir,'bootstrap_test_indices.rds'))
inputs_md5 <- as.list(vapply(paths,function(p)unname(tools::md5sum(p)),character(1)))
ci <- function(v) as.numeric(quantile(rowMeans(matrix(v[boot],nrow=nrow(boot))),c(.025,.975),names=FALSE))
metrics <- list();preds <- list();contrasts <- list();importance <- list();coef_rows <- list();scales <- list();pos <- 0L
for (oi in seq_along(c('retweet_count','favorite_count'))) {
 outcome <- c('retweet_count','favorite_count')[oi]
 logy <- log1p(data[[outcome]]);mu <- mean(logy[train_index]);sdev <- sd(logy[train_index]);z <- (logy-mu)/sdev
 yt <- z[test_index];base_loss <- yt^2
 scales[[oi]] <- data.frame(outcome=outcome,training_log_mean=mu,training_log_sd=sdev)
 base_ci <- ci(base_loss)
 metrics[[length(metrics)+1L]] <- data.frame(outcome=outcome,specification='constant',estimator='Constant',mse=mean(base_loss),ci_low=base_ci[1],ci_high=base_ci[2],relative_reduction_pct=0,oob_mse=NA_real_)
 losses <- list()
 for (si in seq_along(specifications)) {
  spec <- names(specifications)[si]
  x <- safe_design(specifications[[spec]],include_controls=specification_uses_controls[[spec]])
  xtr <- x[train_index,,drop=FALSE];xte <- x[test_index,,drop=FALSE]
  fitols <- lm.fit(cbind('(Intercept)'=1,xtr),z[train_index]);coefs <- fitols$coefficients
  coefs[is.na(coefs)] <- 0
  predols <- as.numeric(cbind('(Intercept)'=1,xte)%*%coefs)
  coef_rows[[length(coef_rows)+1L]] <- data.frame(outcome=outcome,specification=spec,term=names(fitols$coefficients),coefficient=fitols$coefficients,aliased=is.na(fitols$coefficients))
  message(sprintf('%s | %s | variance-split RF on standardized log outcome',outcome,spec))
  train <- as.data.frame(xtr);train$.outcome <- z[train_index]
  model_seed <- 1234L+100L*oi+si
  rf <- ranger(.outcome~.,data=train,num.trees=1000L,mtry=ncol(xtr),min.node.size=10L,splitrule='variance',importance='permutation',num.threads=n_cores,seed=model_seed,respect.unordered.factors='order')
  predrf <- as.numeric(predict(rf,data=as.data.frame(xte))$predictions)
  importance[[length(importance)+1L]] <- data.frame(outcome=outcome,specification=spec,predictor=names(rf$variable.importance),importance=as.numeric(rf$variable.importance))
  for (est in c('OLS','RF log')) {
   pp <- if(est=='OLS') predols else predrf
   loss <- (yt-pp)^2;cc <- ci(loss)
   metrics[[length(metrics)+1L]] <- data.frame(outcome=outcome,specification=spec,estimator=est,mse=mean(loss),ci_low=cc[1],ci_high=cc[2],relative_reduction_pct=100*(1-mean(loss)/mean(base_loss)),oob_mse=if(est=='RF log')rf$prediction.error else NA_real_)
   preds[[length(preds)+1L]] <- data.frame(id=data$id[test_index],outcome=outcome,specification=spec,estimator=est,observed_count=data[[outcome]][test_index],observed_z_log1p=yt,predicted_z_log1p=pp,squared_error=loss)
   losses[[paste(spec,est,sep='|')]] <- loss
   diff <- loss-base_loss;dc <- ci(diff)
   contrasts[[length(contrasts)+1L]] <- data.frame(outcome=outcome,comparison=paste(spec,est,'minus constant'),difference=mean(diff),ci_low=dc[1],ci_high=dc[2])
  }
  diff <- (yt-predrf)^2-(yt-predols)^2;dc <- ci(diff)
  contrasts[[length(contrasts)+1L]] <- data.frame(outcome=outcome,comparison=paste(spec,'RF minus OLS'),difference=mean(diff),ci_low=dc[1],ci_high=dc[2])
  fwrite(rbindlist(metrics),file.path(output_dir,'metrics.csv'))
 }
 for(est in c('OLS','RF log')) for(pair in list(c('veracity_plus_bits','veracity'),c('rumour_plus_bits','rumour'),c('event_plus_bits','event'),c('all_three_labels_plus_bits','all_three_labels'),c('bits','bits_only_no_controls'))) {
  diff <- losses[[paste(pair[1],est,sep='|')]]-losses[[paste(pair[2],est,sep='|')]];dc <- ci(diff)
  contrasts[[length(contrasts)+1L]] <- data.frame(outcome=outcome,comparison=paste(est,pair[1],'minus',pair[2]),difference=mean(diff),ci_low=dc[1],ci_high=dc[2])
 }
}
fwrite(rbindlist(metrics),file.path(output_dir,'metrics.csv'))
fwrite(rbindlist(preds),file.path(output_dir,'test_predictions.csv'))
fwrite(rbindlist(contrasts),file.path(output_dir,'paired_contrasts.csv'))
fwrite(rbindlist(importance),file.path(output_dir,'rf_permutation_importance.csv'))
fwrite(rbindlist(coef_rows),file.path(output_dir,'ols_coefficients.csv'))
fwrite(rbindlist(scales),file.path(output_dir,'outcome_scales.csv'))
write_json(list(created=as.character(Sys.time()),n_train=sum(train_index),n_test=sum(test_index),bootstrap_draws=nrow(boot),bootstrap_file=file.path(output_dir,'bootstrap_test_indices.rds'),outcome='log1p(count), standardized using training mean and sample SD',rf=list(num_trees=1000,min_node_size=10,mtry='all included predictors',splitrule='variance',seed='1234 + 100*outcome_index + specification_index',importance='out-of-bag permutation increase in standardized log-outcome MSE',tuning='none'),ols='ordinary least squares on standardized log1p outcomes; training mean and sample SD',predictors='same ten specifications and untransformed raw controls as saved analysis',inputs_md5=inputs_md5,random_forest_threads=n_cores,R_version=R.version.string,ranger_version=as.character(packageVersion('ranger'))),file.path(output_dir,'metadata.json'),pretty=TRUE,auto_unbox=TRUE)
writeLines(capture.output(sessionInfo()),file.path(output_dir,'sessionInfo.txt'))
print(rbindlist(metrics)[specification %in% c('constant','bits_only_no_controls','bits')])
