
source("https://bioconductor.org/biocLite.R")
biocLite("DESeq2")

library("DESeq2")

human_cts <- read.csv("../../../data/03__rna_seq/02__for_DESeq2/hESC_all.counts.txt", sep="\t", row.names="gene_id")
human_cts <- as.matrix(human_cts)
head(human_cts)

mouse_cts <- read.csv("../../../data/03__rna_seq/02__for_DESeq2/mESC_all.counts.txt", sep="\t", row.names="gene_id")
mouse_cts <- as.matrix(mouse_cts)
head(mouse_cts)

human_cols <- read.csv("../../../data/03__rna_seq/02__for_DESeq2/hESC_all.cols.txt", sep="\t", row.names="column")
head(human_cols)

mouse_cols <- read.csv("../../../data/03__rna_seq/02__for_DESeq2/mESC_all.cols.txt", sep="\t", row.names="column")
head(mouse_cols)

orth_cts <- read.csv("../../../data/03__rna_seq/02__for_DESeq2/orths.counts.txt", sep="\t", row.names="orth_id")
orth_cts <- as.matrix(orth_cts)
head(orth_cts)

orth_cols <- read.csv("../../../data/03__rna_seq/02__for_DESeq2/orths.cols.txt", sep="\t", row.names="column")
head(orth_cols)

dds <- DESeqDataSetFromMatrix(countData = human_cts,
                              colData = human_cols,
                              design = ~ condition)

dds <- estimateSizeFactors(dds)

human_counts <- counts(dds, normalized=TRUE)
head(human_counts)

dds <- DESeqDataSetFromMatrix(countData = mouse_cts,
                              colData = mouse_cols,
                              design = ~ condition)

dds <- estimateSizeFactors(dds)

mouse_counts <- counts(dds, normalized=TRUE)
head(mouse_counts)

dds <- DESeqDataSetFromMatrix(countData = orth_cts,
                              colData = orth_cols,
                              design = ~ condition)
dds$condition <- factor(dds$condition, levels = c("mESC","hESC"))
dds

dds <- DESeq(dds)

res <- results(dds)

head(res)

counts <- counts(dds, normalized=TRUE)
head(counts)

write.table(human_counts, file = "../../../data/03__rna_seq/03__diff_expr/hESC.tpm.txt", sep = "\t",
            quote = FALSE)

write.table(mouse_counts, file = "../../../data/03__rna_seq/03__diff_expr/mESC.tpm.txt", sep = "\t",
            quote = FALSE)

write.table(res, file = "../../../data/03__rna_seq/03__diff_expr/orth.DESeq2.txt", sep = "\t",
            quote = FALSE)

write.table(counts, file = "../../../data/03__rna_seq/03__diff_expr/orth.tpm.txt", sep = "\t",
            quote = FALSE)
