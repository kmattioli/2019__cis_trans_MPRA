suppressMessages(library("DESeq2"))
library(ggplot2)

gene_lengths <- readRDS('../../../misc/00__ensembl_orthologs/orth_gene_lengths.rds')
head(gene_lengths)

gene_lengths$median_hESC <- as.numeric(gene_lengths$median_hESC)
gene_lengths$median_mESC <- as.numeric(gene_lengths$median_mESC)

orth_tf_expr = read.table("../../../data/03__rna_seq/04__TF_expr/orth_TF_expression.txt", sep='\t', header=T)
head(orth_tf_expr)

nrow(orth_tf_expr)

de_genes <- as.vector(orth_tf_expr$gene_id_human[orth_tf_expr$sig == 'sig'])
length(de_genes)

ggplot(gene_lengths, aes(x=human_length, y = mouse_length)) + geom_point() +
       ylim(0,30000) + xlim(0,30000) + 
       geom_point(data = gene_lengths[gsub('_.*','',rownames(gene_lengths)) %in% de_genes,], color='red')

ggsave("FigS11B.pdf")

suppressMessages(library(edgeR))
suppressMessages(library(limma))

install.packages("statmod")

suppressMessages(library(statmod))

library(RColorBrewer)

orth_cts_all <- read.csv("../../../data/03__rna_seq/02__orths/01__featurecounts/orths.counts.voom.txt", sep="\t", row.names="orth_id")
orth_cts_all <- as.matrix(orth_cts_all)

options(stringsAsFactors = F)
set.seed(123)

dge <- DGEList(counts = orth_cts_all)

dge <- calcNormFactors(dge)
dim(dge)

cutoff <- 1
drop <- which(apply(cpm(dge), 1, max) < cutoff)
d <- dge[-drop,] 
dim(d) # number of genes left

### Plot log(cpm) distribution of each sample ####
lcpm <- cpm(dge, log = T)
col <- brewer.pal(12, "Paired")
par(mar=c(8,8,4,2))
pdf("FigS11A.pdf")
boxplot(lcpm, las=2, col=col, main='')
title(main = 'Normalized data', ylab = 'log-cpm')
dev.off()

tt <- read.csv("../../../data/03__rna_seq/03__DE/orth.voom.tt_table.with_interaction.txt", sep="\t", header=T)
head(tt)

housekeeping <- c('ENSG00000116459','ENSG00000102144','ENSG00000111640','ENSG00000166710','ENSG00000075624')
genes <- c(de_genes, housekeeping)

tt$human_gene <- gsub('__.*','', rownames(tt)) 
gene <- as.data.frame((tt[tt$human_gene %in% genes,c('logFC','human_gene')]))
all_genes <- gene

all_genes$status[all_genes$human_gene %in% housekeeping] <- 'housekeeping' 
all_genes$status[all_genes$human_gene %in% de_genes] <- 'differentially expressed TF'

all_genes$logFC <- as.numeric(abs(all_genes$logFC))

ggplot(all_genes, aes(x=reorder(status,human_gene),y = logFC,fill=status)) + geom_boxplot(outlier.shape = NA)  + 
  #geom_point(aes(fill=cell),color=all_genes$color) + 
  theme_bw() + scale_fill_manual(values = c('#E99675','#72B6A1')) +
  scale_color_manual(values = c('#E99675','#72B6A1')) +
  #geom_vline(xintercept=5.5, color="grey", size=1) +
  #scale_y_continuous(trans='log10') +
  theme(text = element_text(size = 18), axis.text = element_text(size = 20), legend.position = 'none') + 
  ylab('logFC')+ xlab('')+ylim(0,12)+
  theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())

ggsave("FigS11C.pdf")


