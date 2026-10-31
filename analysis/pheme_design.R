#!/usr/bin/env Rscript

# Qwen-only PHEME engagement models, with each taxonomy tested separately.
# This intentionally reads saved hard codes and existing outcome/split artifacts only.

suppressPackageStartupMessages({
  library(data.table)
  library(ranger)
  library(rpart)
  library(rpart.plot)
  library(ggplot2)
  library(jsonlite)
})

options(stringsAsFactors = FALSE)
# Avoid scientific notation in the printed node means of engagement trees.
options(scipen = 999)

root <- normalizePath(getwd())
output_dir <- file.path(root, "runs/pheme_embedding_comparison/engagement/qwen_log_outcomes")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

seed <- 1234L
bootstrap_draws <- 2000L
n_cores <- as.integer(Sys.getenv("ISBC_R_THREADS", "1"))
stopifnot(n_cores >= 1L)

paths <- list(
  labels = file.path(root, "data/pheme/pheme_labels.csv"),
  ready = file.path(root, "data/pheme/pheme_ready.csv"),
  engagement = file.path(root, "runs/pheme_embedding_comparison/engagement/pheme_engagement_extracted.csv"),
  split = file.path(root, "runs/pheme_embedding_comparison/engagement/pheme_engagement_split.csv"),
  codes = file.path(root, "runs/pheme_qwen_fixed_balance/encoded_pheme_ready_best.csv")
)
stopifnot(all(file.exists(unlist(paths))))

# Lock this analysis to the consolidated Qwen 16/6 representation.  The output
# directory deliberately does not accept a backbone/run argument.
qwen_metadata_path <- file.path(root, "runs/pheme_qwen_fixed_balance/metadata.json")
if (!file.exists(qwen_metadata_path)) stop("Missing Qwen run metadata")
qwen_metadata <- jsonlite::fromJSON(qwen_metadata_path)
if (!identical(as.integer(qwen_metadata$k_factors), 16L) ||
    !isTRUE(all.equal(as.numeric(qwen_metadata$target_active), 6))) {
  stop("This analysis requires the Qwen K=16 / target-active=6 run")
}

sha256 <- function(path) unname(tools::md5sum(path)) # compact file fingerprint for local provenance
as_id <- function(x) as.character(x)

labels <- fread(paths$labels, colClasses = list(character = "id"))
ready <- fread(paths$ready, colClasses = list(character = "id"))
engagement <- fread(paths$engagement, colClasses = list(character = "id"))
split <- fread(paths$split, colClasses = list(character = "id"))
codes <- fread(paths$codes, colClasses = list(character = "id"))

setnames(labels, "id", "id")
for (object in list(labels, ready, engagement, split, codes)) {
  if (anyDuplicated(object$id)) stop("Duplicate PHEME document identifiers detected")
}
bit_source <- sprintf("bit_%02d", 0:15)
bit_names <- sprintf("bit%02d", 0:15)
if (!all(bit_source %in% names(codes))) stop("Expected Qwen hard-bit columns are absent")

data <- merge(labels[, .(id, label, veracity, event)], ready[, .(id, text)], by = "id", all = FALSE)
data <- merge(data, engagement[, .(id, retweet_count, favorite_count, followers_count, text_length)], by = "id", all = FALSE)
data <- merge(data, split[, .(id, predictive_split)], by = "id", all = FALSE)
data <- merge(data, codes[, c("id", bit_source), with = FALSE], by = "id", all = FALSE)
setorderv(data, "id")
if (nrow(data) != 5802L || !identical(sort(unique(data$predictive_split)), c("test", "train"))) {
  stop("PHEME analysis input no longer matches the fixed 5,802-document split")
}
if (sum(data$predictive_split == "train") != 4641L || sum(data$predictive_split == "test") != 1161L) {
  stop("Expected 4,641 training and 1,161 test documents")
}
for (name in bit_source) {
  if (!all(data[[name]] %in% c(0, 1))) stop(sprintf("%s is not binary", name))
}
setnames(data, bit_source, bit_names)
data[, followers_count := fifelse(is.na(followers_count), 0, as.numeric(followers_count))]
data[, text_length := fifelse(is.na(text_length), median(text_length, na.rm = TRUE), as.numeric(text_length))]
train_index <- data$predictive_split == "train"
test_index <- !train_index

dummy_block <- function(x, prefix) {
  value <- factor(as.character(x))
  levels_value <- sort(levels(value))
  if (length(levels_value) < 2L) return(data.frame(row.names = seq_along(value)))
  value <- factor(value, levels = levels_value)
  out <- sapply(levels_value[-1], function(level) as.numeric(value == level))
  if (is.null(dim(out))) out <- matrix(out, ncol = 1L)
  out <- as.data.frame(out, check.names = FALSE)
  names(out) <- paste0(prefix, "_", make.names(levels_value[-1]))
  out
}

rumour_block <- data.frame(label_rumour = as.numeric(data$label == "rumour"))
veracity_block <- dummy_block(data$veracity, "veracity")
event_block <- dummy_block(data$event, "event")
bits_block <- as.data.frame(data[, ..bit_names])
controls_block <- data.frame(
  control_followers_count = data$followers_count,
  control_text_length = data$text_length
)
all_labels_block <- cbind(rumour_block, veracity_block, event_block)

specifications <- list(
  bits = bits_block,
  bits_only_no_controls = bits_block,
  veracity = veracity_block,
  veracity_plus_bits = cbind(veracity_block, bits_block),
  rumour = rumour_block,
  rumour_plus_bits = cbind(rumour_block, bits_block),
  event = event_block,
  event_plus_bits = cbind(event_block, bits_block),
  all_three_labels = all_labels_block,
  all_three_labels_plus_bits = cbind(all_labels_block, bits_block)
)
specification_labels <- c(
  bits = "Bits + controls",
  bits_only_no_controls = "Bits only (no controls)",
  veracity = "Veracity",
  veracity_plus_bits = "Veracity + bits",
  rumour = "Rumour status",
  rumour_plus_bits = "Rumour status + bits",
  event = "Event",
  event_plus_bits = "Event + bits",
  all_three_labels = "All three labels",
  all_three_labels_plus_bits = "All three labels + bits"
)
specification_uses_controls <- c(
  bits = TRUE,
  bits_only_no_controls = FALSE,
  veracity = TRUE,
  veracity_plus_bits = TRUE,
  rumour = TRUE,
  rumour_plus_bits = TRUE,
  event = TRUE,
  event_plus_bits = TRUE,
  all_three_labels = TRUE,
  all_three_labels_plus_bits = TRUE
)

safe_design <- function(block, include_controls = TRUE) {
  x <- if (include_controls) cbind(block, controls_block) else block
  x <- as.data.frame(x, check.names = FALSE)
  active <- vapply(x, function(z) length(unique(z[train_index])) > 1L, logical(1))
  if (!all(active)) x <- x[, active, drop = FALSE]
  as.matrix(x)
}

