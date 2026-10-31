# Run explicitly with Rscript; nothing is installed by the analysis scripts.
packages <- c('data.table','ranger','rpart','rpart.plot','ggplot2','jsonlite')
missing <- packages[!vapply(packages,requireNamespace,logical(1),quietly=TRUE)]
if(length(missing)) install.packages(missing,repos='https://cloud.r-project.org')
# Versions used for the paper are recorded in reproduction/reference/pheme/sessionInfo.txt.
