source('analysis/pheme_design.R')
source('analysis/pheme_tree_labels.R')
# Unconditional descriptive profiles of all bits: P(label category | bit = 1)
# on the complete, fixed 5,802-document PHEME sample.  These tables are not
# selected by tree/forest use and do not concern held-out prediction.
conditional_label_rows <- list()
for (label_column in c("label", "veracity", "event")) {
  categories <- sort(unique(as.character(data[[label_column]])))
  for (bit in bit_names) {
    active <- data[[bit]] == 1
    active_n <- sum(active)
    for (category in categories) {
      conditional_label_rows[[length(conditional_label_rows) + 1L]] <- data.frame(
        bit = pretty_bit_name(bit), label_family = label_column,
        category = pretty_category(category),
        active_documents = active_n, active_share_all_documents = mean(active),
        probability_label_given_bit_active = if (active_n > 0) mean(as.character(data[[label_column]][active]) == category) else NA_real_,
        probability_label_unconditional = mean(as.character(data[[label_column]]) == category),
        stringsAsFactors = FALSE
      )
    }
  }
}
conditional_label_distribution <- rbindlist(conditional_label_rows)
fwrite(conditional_label_distribution, file.path(output_dir, "all_bits_label_distribution_given_active.csv"))

