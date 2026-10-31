pretty_feature_names <- function(x) {
  x <- gsub("^bit([0-9]{2})$", "Bit \\1", x)
  x <- gsub("^label_rumour$", "Rumour status = rumour", x)
  x <- gsub("^veracity_", "Veracity = ", x)
  x <- gsub("^event_", "Event = ", x)
  x <- gsub("^control_followers_count$", "followers_count", x)
  x <- gsub("^control_text_length$", "text_length", x)
  x <- gsub("not\\.applicable", "not applicable", x)
  x <- gsub("germanwings\\.crash", "Germanwings crash", x)
  x <- gsub("charliehebdo", "Charlie Hebdo", x)
  x <- gsub("ottawashooting", "Ottawa shooting", x)
  x <- gsub("sydneysiege", "Sydney siege", x)
  x <- gsub("ferguson", "Ferguson", x)
  x
}
pretty_bit_name <- function(x) sub("^bit([0-9]{2})$", "Bit \\1", x)
pretty_category <- function(x) {
  x <- as.character(x)
  x <- gsub("not_applicable", "not applicable", x, fixed = TRUE)
  x <- gsub("charliehebdo", "Charlie Hebdo", x, fixed = TRUE)
  x <- gsub("germanwings-crash", "Germanwings crash", x, fixed = TRUE)
  x <- gsub("ottawashooting", "Ottawa shooting", x, fixed = TRUE)
  x <- gsub("sydneysiege", "Sydney siege", x, fixed = TRUE)
  x <- gsub("ferguson", "Ferguson", x, fixed = TRUE)
  x <- gsub("non-rumour", "non-rumour", x, fixed = TRUE)
  x
}
pretty_tree_split_labels <- function(x, labs, digits, varlen, faclen) {
  labs <- gsub("control_followers_count", "followers_count", labs, fixed = TRUE)
  labs <- gsub("control_text_length", "text_length", labs, fixed = TRUE)
  labs <- gsub("label_rumour", "Rumour status = rumour", labs, fixed = TRUE)
  labs <- gsub("veracity_", "Veracity = ", labs, fixed = TRUE)
  labs <- gsub("event_", "Event = ", labs, fixed = TRUE)
  labs <- gsub("bit([0-9]{2})", "Bit \\1", labs)
  labs <- gsub("not\\.applicable", "not applicable", labs)
  labs <- gsub("charliehebdo", "Charlie Hebdo", labs)
  labs <- gsub("germanwings\\.crash", "Germanwings crash", labs)
  labs <- gsub("ottawashooting", "Ottawa shooting", labs)
  labs <- gsub("sydneysiege", "Sydney siege", labs)
  labs <- gsub("ferguson", "Ferguson", labs)
  labs
}
tree_designs <- list(
  all_variables = safe_design(cbind(all_labels_block, bits_block)),
  bits_controls = safe_design(bits_block),
  bits_only = safe_design(bits_block, include_controls = FALSE)
)
