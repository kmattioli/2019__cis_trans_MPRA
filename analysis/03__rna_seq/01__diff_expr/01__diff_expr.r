
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

options(install.packages.compile.from.source="interactive")

# BiocManager::install("GenomicRanges")

# BiocManager::install("DESeq2")

suppressMessages(library("DESeq2"))

human_cts <- read.csv("../../../data/03__rna_seq/00__HUES64/01__featurecounts/hESC_all.counts.txt", sep="\t", row.names="gene_id")
human_cts <- as.matrix(human_cts)
head(human_cts)

mouse_cts <- read.csv("../../../data/03__rna_seq/01__mESC/01__featurecounts/mESC_all.counts.txt", sep="\t", row.names="gene_id")
mouse_cts <- as.matrix(mouse_cts)
head(mouse_cts)

human_cols <- read.csv("../../../data/03__rna_seq/00__HUES64/01__featurecounts/hESC_all.cols.txt", sep="\t", row.names="column")
head(human_cols)

mouse_cols <- read.csv("../../../data/03__rna_seq/01__mESC/01__featurecounts/mESC_all.cols.txt", sep="\t", row.names="column")
head(mouse_cols)

orth_cts <- read.csv("../../../data/03__rna_seq/02__orths/01__featurecounts/orths.counts.txt", sep="\t", row.names="orth_id")
orth_cts <- as.matrix(orth_cts)
head(orth_cts)

orth_cols <- read.csv("../../../data/03__rna_seq/02__orths/01__featurecounts/orths.cols.txt", sep="\t", row.names="column")
head(orth_cols)

orth_cols_new <- read.csv("../../../data/03__rna_seq/02__orths/01__featurecounts/orths.cols.voom.samples.txt", sep="\t", row.names="column")
head(orth_cols_new)

dds <- DESeqDataSetFromMatrix(countData = human_cts,
                              colData = human_cols,
                              design = ~ condition)

dds <- estimateSizeFactors(dds)

human_counts <- counts(dds, normalized=TRUE)
head(human_counts)

dds <- DESeqDataSetFromMatrix(countData = mouse_cts,
                              colData = mouse_cols,
                              design = ~ condition )

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

counts <- counts(dds, normalized=TRUE)
head(counts)

write.table(human_counts, file = "../../../data/03__rna_seq/00__HUES64/02__tpms/hESC.tpm.txt", sep = "\t",
            quote = FALSE)

write.table(mouse_counts, file = "../../../data/03__rna_seq/01__mESC/02__tpms/mESC.tpm.txt", sep = "\t",
            quote = FALSE)

write.table(counts, file = "../../../data/03__rna_seq/02__orths/02__tpms/orth.tpm.txt", sep = "\t",
            quote = FALSE)

orth_cts_all <- read.csv("../../../data/03__rna_seq/02__orths/01__featurecounts/orths.counts.voom.txt", sep="\t", row.names="orth_id")
orth_cts_all <- as.matrix(orth_cts_all)
head(orth_cts_all)

# BiocManager::install("edgeR")

suppressMessages(library(edgeR))
suppressMessages(library(limma))

# install.packages("statmod")

suppressMessages(library(statmod))

options(stringsAsFactors = F)
set.seed(123)

## 1st do the analysis correcting for individual, using Transfected and Untransfected samples - Complex model
dge <- DGEList(counts = orth_cts_all)

dge <- calcNormFactors(dge)
dim(dge)

cutoff <- 1
drop <- which(apply(cpm(dge), 1, max) < cutoff)
d <- dge[-drop,] 
dim(d) # number of genes left

sample <- as.factor(orth_cols_new$sample)
treatment <- as.factor(orth_cols_new$treatment)
condition <- as.factor(orth_cols_new$condition)

design <- model.matrix(~treatment + treatment:condition + condition)
par(mfrow = c(1, 1))
v <- voom(d, design, plot=TRUE)

plotMDS(v)

cor <- duplicateCorrelation(v, design, block = orth_cols_new$sample)
vobj = voom( d, design, plot=TRUE, block=orth_cols_new$sample, correlation=cor$consensus.correlation)   

plotMDS(vobj)

fitDupCor <- lmFit(vobj, design, block=orth_cols_new$sample, correlation=cor$consensus.correlation)

fit <- eBayes(fitDupCor)

res <- decideTests(fit)
summary(res)

tt <- topTable(fit, coef = 2, n = Inf) 

DEgenes <- rownames(tt)[tt$adj.P.Val < 0.01]

tt <- topTable(fit, coef = "conditionmESC", n = Inf)

write.table(tt, file = "../../../data/03__rna_seq/03__DE/orth.voom.tt_table.with_interaction.txt", sep = "\t",
            quote = FALSE)

write.table(DEgenes, file = "../../../data/03__rna_seq/03__DE/DEgenes_treatment.voom.txt", sep = "\t",
            quote = FALSE)
