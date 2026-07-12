BiocManager::install("TCGAbiolinks", ask = FALSE, update = FALSE)
library(TCGAbiolinks)


pam50 <- TCGAquery_subtype(tumor = "BRCA")
pam50_labels <- pam50[, c("patient", "BRCA_Subtype_PAM50")]

View(pam50)

write.csv(pam50_labels, "TCGA_BRCA_PAM50_labels.csv", row.names = FALSE)
