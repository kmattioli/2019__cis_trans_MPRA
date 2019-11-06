
# # install MPRAnalyze
# if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install("MPRAnalyze", version = "3.8")

# # install RCurl
# install.packages("RCurl")

# # install biocparallel
# if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")
# BiocManager::install("BiocParallel")

# load the package
library(MPRAnalyze)
library(tidyr)

library(BiocParallel)

dna_counts_depth <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/dna_counts.for_depth_estimation.mpranalyze.txt", sep="\t", header=TRUE)

# since we only have 1 dna replicate -- add another so code doesn't crash (expects matrix)
dna_counts_depth["dna_2"] <- dna_counts_depth["dna_1"]

row.names(dna_counts_depth) <- dna_counts_depth$element
dna_counts_depth <- dna_counts_depth[ , !(names(dna_counts_depth) %in% c("element")), drop=FALSE]
dna_counts_depth <- as.matrix(dna_counts_depth)

rna_counts_depth <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/rna_counts.for_depth_estimation.mpranalyze.txt", sep="\t", header=TRUE)
row.names(rna_counts_depth) <- rna_counts_depth$element
rna_counts_depth <- rna_counts_depth[ , !(names(rna_counts_depth) %in% c("element")), drop=FALSE]
rna_counts_depth <- as.matrix(rna_counts_depth)

dna_cols_depth <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/dna_col_ann.for_depth_estimation.mpranalyze.txt", sep="\t", header=TRUE)
names(dna_cols_depth) <- c("id", "condition", "sample")

# add second row to dna_cols_depth
row2 <- data.frame(id="dna_2", condition="dna", sample="2")
dna_cols_depth <- rbind(dna_cols_depth, row2)
row.names(dna_cols_depth) <- dna_cols_depth$id

rna_cols_depth <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/rna_col_ann.for_depth_estimation.mpranalyze.txt", sep="\t", header=TRUE)
names(rna_cols_depth) <- c("id", "condition", "sample")
row.names(rna_cols_depth) <- rna_cols_depth$id
rna_cols_depth

# make sure everything is a factor
dna_cols_depth$condition <- as.factor(dna_cols_depth$condition)
rna_cols_depth$condition <- as.factor(rna_cols_depth$condition)
rna_cols_depth$sample <- as.factor(rna_cols_depth$sample)
rna_cols_depth

all_comp_dna_counts <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/dna_counts.all_comp.mpranalyze.txt", sep="\t", header=TRUE)
row.names(all_comp_dna_counts) <- all_comp_dna_counts$comp_id
all_comp_dna_counts <- all_comp_dna_counts[ , !(names(all_comp_dna_counts) %in% c("comp_id"))]
all_comp_dna_counts <- as.matrix(all_comp_dna_counts)

all_comp_dna_cols <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/dna_col_ann.all_comp.mpranalyze.txt", sep="\t", header=TRUE)
row.names(all_comp_dna_cols) <- all_comp_dna_cols$X
head(all_comp_dna_cols)

all_comp_dna_cols$barcode <- as.factor(all_comp_dna_cols$barcode)
all_comp_dna_cols$seq <- as.factor(all_comp_dna_cols$seq)
all_comp_dna_cols$condition <- as.factor(all_comp_dna_cols$condition)
all_comp_dna_cols

all_comp_ctrls <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/ctrl_status.all_comp.mpranalyze.txt", sep="\t", header=TRUE)
all_comp_ctrls <- as.logical(all_comp_ctrls$ctrl_status)
head(all_comp_ctrls)

length(all_comp_ctrls)

native_rna_counts <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/native_rna_counts.seq_comp.mpranalyze.txt", sep="\t", header=TRUE)
row.names(native_rna_counts) <- native_rna_counts$comp_id
native_rna_counts <- native_rna_counts[ , !(names(native_rna_counts) %in% c("comp_id"))]
native_rna_counts <- as.matrix(native_rna_counts)

native_rna_cols <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/native_col_ann.seq_comp.mpranalyze.txt", sep="\t", header=TRUE)
row.names(native_rna_cols) <- native_rna_cols$X
head(native_rna_cols)

# make sure everything is a factor
native_rna_cols$barcode <- as.factor(native_rna_cols$barcode)
native_rna_cols$seq <- as.factor(native_rna_cols$seq)
native_rna_cols$condition <- as.factor(native_rna_cols$condition)
head(native_rna_cols)

all_rna_counts <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/all_rna_counts.seq_comp.mpranalyze.txt", sep="\t", header=TRUE)
row.names(all_rna_counts) <- all_rna_counts$comp_id
all_rna_counts <- all_rna_counts[ , !(names(all_rna_counts) %in% c("comp_id"))]
all_rna_counts <- as.matrix(all_rna_counts)

all_rna_cols <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/all_col_ann.seq_comp.mpranalyze.txt", sep="\t", header=TRUE)
row.names(all_rna_cols) <- all_rna_cols$X

# make sure everything is a factor
all_rna_cols$barcode <- as.factor(all_rna_cols$barcode)
all_rna_cols$seq <- as.factor(all_rna_cols$seq)
all_rna_cols$condition <- as.factor(all_rna_cols$condition)
head(all_rna_cols)

# for seq comparisons, load each cell line data separately
hues64_rna_counts <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/HUES64_rna_counts.seq_comp.mpranalyze.txt", sep="\t", header=TRUE)
row.names(hues64_rna_counts) <- hues64_rna_counts$comp_id
hues64_rna_counts <- hues64_rna_counts[ , !(names(hues64_rna_counts) %in% c("comp_id"))]
hues64_rna_counts <- as.matrix(hues64_rna_counts)

mesc_rna_counts <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/mESC_rna_counts.seq_comp.mpranalyze.txt", sep="\t", header=TRUE)
row.names(mesc_rna_counts) <- mesc_rna_counts$comp_id
mesc_rna_counts <- mesc_rna_counts[ , !(names(mesc_rna_counts) %in% c("comp_id"))]
mesc_rna_counts <- as.matrix(mesc_rna_counts)

hues64_rna_cols <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/HUES64_col_ann.seq_comp.mpranalyze.txt", sep="\t", header=TRUE)
row.names(hues64_rna_cols) <- hues64_rna_cols$X

mesc_rna_cols <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/mESC_col_ann.seq_comp.mpranalyze.txt", sep="\t", header=TRUE)
row.names(mesc_rna_cols) <- mesc_rna_cols$X

# make sure everything is a factor
hues64_rna_cols$barcode <- as.factor(hues64_rna_cols$barcode)
mesc_rna_cols$barcode <- as.factor(mesc_rna_cols$barcode)

hues64_rna_cols$seq <- as.factor(hues64_rna_cols$seq)
mesc_rna_cols$seq <- as.factor(mesc_rna_cols$seq)

hues64_rna_cols$condition <- as.factor(hues64_rna_cols$condition)
mesc_rna_cols$condition <- as.factor(mesc_rna_cols$condition)

head(hues64_rna_cols)

# for cell comparisons, load each cell line data separately
human_rna_counts <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/human_rna_counts.cell_comp.mpranalyze.txt", sep="\t", header=TRUE)
row.names(human_rna_counts) <- human_rna_counts$comp_id
human_rna_counts <- human_rna_counts[ , !(names(human_rna_counts) %in% c("comp_id"))]
human_rna_counts <- as.matrix(human_rna_counts)

mouse_rna_counts <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/mouse_rna_counts.cell_comp.mpranalyze.txt", sep="\t", header=TRUE)
row.names(mouse_rna_counts) <- mouse_rna_counts$comp_id
mouse_rna_counts <- mouse_rna_counts[ , !(names(mouse_rna_counts) %in% c("comp_id"))]
mouse_rna_counts <- as.matrix(mouse_rna_counts)

human_rna_cols <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/human_col_ann.cell_comp.mpranalyze.txt", sep="\t", header=TRUE)
row.names(human_rna_cols) <- human_rna_cols$X

mouse_rna_cols <- read.table("../../../data/02__mpra/01__counts/mpranalyze_files/mouse_col_ann.cell_comp.mpranalyze.txt", sep="\t", header=TRUE)
row.names(mouse_rna_cols) <- mouse_rna_cols$X

# make sure everything is a factor
human_rna_cols$barcode <- as.factor(human_rna_cols$barcode)
mouse_rna_cols$barcode <- as.factor(mouse_rna_cols$barcode)

human_rna_cols$seq <- as.factor(human_rna_cols$seq)
mouse_rna_cols$seq <- as.factor(mouse_rna_cols$seq)

human_rna_cols$condition <- as.factor(human_rna_cols$condition)
mouse_rna_cols$condition <- as.factor(mouse_rna_cols$condition)

head(human_rna_cols)

# create MPRA object
depth_obj <- MpraObject(dnaCounts = dna_counts_depth, rnaCounts = rna_counts_depth, 
                        dnaAnnot = dna_cols_depth, rnaAnnot = rna_cols_depth)

# estimate depth factors using uq -- here, a sample/condition pair == 1 library
depth_obj <- estimateDepthFactors(depth_obj, lib.factor = c("sample", "condition"),  depth.estimator='uq',
                                  which.lib = "dna")
depth_obj <- estimateDepthFactors(depth_obj, lib.factor = c("id"),  
                                  depth.estimator='uq', which.lib = "rna")

rna_depths <- rnaDepth(depth_obj)
rna_depths

rna_cols_depth$depth <- rna_depths
rna_cols_depth

nrow(rna_cols_depth)

nrow(native_rna_cols)

head(native_rna_cols)

# first need to set the dnadepths and rnadepths manually
all_comp_dna_cols$depth <- rep(1, nrow(all_comp_dna_cols))

# note 13 will change depending how many barcodes there are per element
native_rna_cols$depth <- rep(rna_depths, each=13)

# create MPRA object
obj <- MpraObject(dnaCounts = all_comp_dna_counts, rnaCounts = native_rna_counts, 
                  dnaAnnot = all_comp_dna_cols, rnaAnnot = native_rna_cols, controls = all_comp_ctrls,
                  BPPARAM = SnowParam(workers=16,type="SOCK"))

obj <- setDepthFactors(obj, dnaDepth = all_comp_dna_cols$depth, rnaDepth = native_rna_cols$depth)

obj <- analyzeComparative(obj = obj, 
                          dnaDesign = ~ barcode, 
                          rnaDesign = ~ seq, 
                          reducedDesign = ~ 1) 

native_res <- testLrt(obj)
head(native_res)

hist(native_res[all_comp_ctrls,]$pval)

hist(native_res[!all_comp_ctrls,]$pval)

write.table(native_res, file = "../../../data/02__mpra/02__activs/native_results.txt", sep = "\t",
            quote = FALSE)

# note 13 will change depending how many barcodes there are per element
all_rna_cols$depth <- rep(rna_depths, each=26)

# create MPRA object
obj1 <- MpraObject(dnaCounts = all_comp_dna_counts, rnaCounts = all_rna_counts, 
                   dnaAnnot = all_comp_dna_cols, rnaAnnot = all_rna_cols, controls = all_comp_ctrls,
                   BPPARAM = SnowParam(workers=16,type="SOCK"))

obj1 <- setDepthFactors(obj1, dnaDepth = all_comp_dna_cols$depth, rnaDepth = all_rna_cols$depth)

head(all_rna_cols)

obj1 <- analyzeComparative(obj = obj1, 
                           dnaDesign = ~ barcode, 
                           rnaDesign = ~ seq + condition+ seq:condition, 
                           reducedDesign = ~ seq + condition) 

int_res <- testLrt(obj1)
head(int_res)

hist(int_res[all_comp_ctrls,]$pval)

hist(int_res[!all_comp_ctrls,]$pval)

write.table(int_res, file = "../../../data/02__mpra/02__activs/cis_trans_interaction_results.txt", sep = "\t",
            quote = FALSE)

rna_cols_depth

hues64_depths <- rna_depths[1:3]
hues64_depths

# note 13 will change depending how many barcodes there are per element
hues64_rna_cols$depth <- rep(hues64_depths, each=26)

# create MPRA object
obj2 <- MpraObject(dnaCounts = all_comp_dna_counts, rnaCounts = hues64_rna_counts, 
                   dnaAnnot = all_comp_dna_cols, rnaAnnot = hues64_rna_cols, controls = all_comp_ctrls,
                   BPPARAM = SnowParam(workers=16,type="SOCK"))

obj2 <- setDepthFactors(obj2, dnaDepth = all_comp_dna_cols$depth, rnaDepth = hues64_rna_cols$depth)

obj2 <- analyzeComparative(obj = obj2, 
                           dnaDesign = ~ barcode, 
                           rnaDesign = ~ seq, 
                           reducedDesign = ~ 1) 

hues64_res <- testLrt(obj2)
head(hues64_res)

hist(hues64_res[all_comp_ctrls,]$pval)

hist(hues64_res[!all_comp_ctrls,]$pval)

write.table(hues64_res, file = "../../../data/02__mpra/02__activs/HUES64_cis_results.txt", sep = "\t",
            quote = FALSE)

# note 13 will change depending how many barcodes there are per element
mesc_depths <- rna_depths[4:6]
mesc_rna_cols$depth <- rep(mesc_depths, each=26)

# create MPRA object
obj3 <- MpraObject(dnaCounts = all_comp_dna_counts, rnaCounts = mesc_rna_counts, 
                   dnaAnnot = all_comp_dna_cols, rnaAnnot = mesc_rna_cols, controls = all_comp_ctrls,
                   BPPARAM = SnowParam(workers=16,type="SOCK"))

obj3 <- setDepthFactors(obj3, dnaDepth = all_comp_dna_cols$depth, rnaDepth = mesc_rna_cols$depth)

obj3 <- analyzeComparative(obj = obj3, 
                           dnaDesign = ~ barcode, 
                           rnaDesign = ~ seq, 
                           reducedDesign = ~ 1) 

mesc_res <- testLrt(obj3)
head(mesc_res)

hist(mesc_res[all_comp_ctrls,]$pval)

hist(mesc_res[!all_comp_ctrls,]$pval)

write.table(mesc_res, file = "../../../data/02__mpra/02__activs/mESC_cis_results.txt", sep = "\t",
            quote = FALSE)

# note 13 will change depending how many barcodes there are per element
human_rna_cols$depth <- rep(rna_depths, each=13)

# create MPRA object
obj4 <- MpraObject(dnaCounts = all_comp_dna_counts, rnaCounts = human_rna_counts, 
                   dnaAnnot = all_comp_dna_cols, rnaAnnot = human_rna_cols, controls = all_comp_ctrls,
                   BPPARAM = SnowParam(workers=16,type="SOCK"))

obj4 <- setDepthFactors(obj4, dnaDepth = all_comp_dna_cols$depth, rnaDepth = human_rna_cols$depth)

obj4 <- analyzeComparative(obj = obj4, 
                           dnaDesign = ~ barcode, 
                           rnaDesign = ~ condition, 
                           reducedDesign = ~ 1) 

human_trans_res <- testLrt(obj4)
head(human_trans_res)

hist(human_trans_res[all_comp_ctrls,]$pval)

hist(human_trans_res[!all_comp_ctrls,]$pval)

write.table(human_trans_res, file = "../../../data/02__mpra/02__activs/human_trans_results.txt", sep = "\t",
            quote = FALSE)

# note 13 will change depending how many barcodes there are per element
mouse_rna_cols$depth <- rep(rna_depths, each=13)

# create MPRA object
obj5 <- MpraObject(dnaCounts = all_comp_dna_counts, rnaCounts = mouse_rna_counts, 
                   dnaAnnot = all_comp_dna_cols, rnaAnnot = mouse_rna_cols, controls = all_comp_ctrls,
                   BPPARAM = SnowParam(workers=16,type="SOCK"))

obj5 <- setDepthFactors(obj5, dnaDepth = all_comp_dna_cols$depth, rnaDepth = mouse_rna_cols$depth)

obj5 <- analyzeComparative(obj = obj5, 
                           dnaDesign = ~ barcode, 
                           rnaDesign = ~ condition, 
                           reducedDesign = ~ 1) 

mouse_trans_res <- testLrt(obj5)
head(mouse_trans_res)

hist(mouse_trans_res[all_comp_ctrls,]$pval)

hist(mouse_trans_res[!all_comp_ctrls,]$pval)

write.table(mouse_trans_res, file = "../../../data/02__mpra/02__activs/mouse_trans_results.txt", sep = "\t",
            quote = FALSE)
