
# coding: utf-8

# ## 00__preprocess
# 
# in this notebook, i take the output from feature counts (counts per gene in HUES64 and mESCs) and prepare it for DESeq2 analysis.

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import itertools
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
import sys

from scipy.stats import spearmanr

# import utils
sys.path.append("../../../utils")
from plotting_utils import *

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
mpl.rcParams['figure.autolayout'] = False


# In[2]:


sns.set(**PAPER_PRESET)
fontsize = PAPER_FONTSIZE


# In[3]:


np.random.seed(2019)


# ## variables

# In[4]:


## note: should probably re-run hESC samples w/o lifted gencode annotations...


# In[5]:


rna_seq_dir = "../../../data/03__rna_seq/02__for_DESeq2"


# In[6]:


## CHANGE THESE WHEN FILES MOVE
hESC_rep1_f = "../../../data/03__rna_seq/00__HUES64/04__featurecounts/hESC_rep1.counts.txt"
hESC_rep2_f = "../../../data/03__rna_seq/00__HUES64/04__featurecounts/hESC_rep2.counts.txt"
mESC_rep1_f = "../../../data/03__rna_seq/01__mESC/04__featurecounts/mESC_rep1.counts.txt"
mESC_rep2_f = "../../../data/03__rna_seq/01__mESC/04__featurecounts/mESC_rep2.counts.txt"
mESC_rep3_f = "../../../data/03__rna_seq/01__mESC/04__featurecounts/mESC_rep3.counts.txt"


# In[11]:


orth_f = "../../../misc/00__ensembl_orthologs/ensembl96_human_mouse_orths.txt.gz"


# ## 1. import data

# In[8]:


hESC_rep1 = pd.read_table(hESC_rep1_f, skiprows=1)
hESC_rep2 = pd.read_table(hESC_rep2_f, skiprows=1)
hESC_rep1.head()


# In[9]:


mESC_rep1 = pd.read_table(mESC_rep1_f, skiprows=1)
mESC_rep2 = pd.read_table(mESC_rep2_f, skiprows=1)
mESC_rep3 = pd.read_table(mESC_rep3_f, skiprows=1)
mESC_rep1.head()


# In[12]:


orth = pd.read_table(orth_f)
orth.head()


# ## 2. join counts files

# In[13]:


hESC = hESC_rep1[["Geneid", 
                  "../00__HUES64/03__alignments/hESC_rep1/accepted_hits.bam"]].merge(hESC_rep2[["Geneid",
                                                                                                "../00__HUES64/03__alignments/hESC_rep2/accepted_hits.bam"]],
                                                                                     on="Geneid")
hESC.columns = ["long_gene_id", "rep1", "rep2"]
hESC["gene_id"] = hESC["long_gene_id"].str.split(".", expand=True)[0]
hESC.sample(5)


# In[14]:


mESC = mESC_rep1[["Geneid", 
                  "../01__mESC/03__alignments/mESC_rep1/accepted_hits.bam"]].merge(mESC_rep2[["Geneid",
                                                                                              "../01__mESC/03__alignments/mESC_rep2/accepted_hits.bam"]],
                                                                                   on="Geneid").merge(mESC_rep3[["Geneid",
                                                                                                                 "../01__mESC/03__alignments/mESC_rep3/accepted_hits.bam"]],
                                                                                                      on="Geneid")
mESC.columns = ["long_gene_id", "rep1", "rep2", "rep3"]
mESC["gene_id"] = mESC["long_gene_id"].str.split(".", expand=True)[0]
mESC.sample(5)


# ## 3. do some very quick QC

# In[15]:


orth["Mouse homology type"].value_counts()


# In[16]:


orth[orth["Gene name"].isin(["POU5F1", "NANOG", 
                             "SOX2", "XIST", "EOMES"])][["Gene stable ID", "Gene name", 
                                                         "Mouse gene stable ID", "Mouse gene name"]].drop_duplicates()


# In[17]:


hESC[hESC["gene_id"].isin(["ENSG00000181449", "ENSG00000204531", "ENSG00000163508", "ENSG00000111704"])]


# In[18]:


mESC[mESC["gene_id"].isin(["ENSMUSG00000074637", "ENSMUSG00000024406", "ENSMUSG00000032446", "ENSMUSG00000012396"])]


# In[19]:


xist_human = "ENSG00000229807"
sry_human = "ENSG00000184895"


# In[20]:


xist_mouse = "ENSMUSG00000086503"
sry_mouse = "ENSMUSG00000069036"


# In[21]:


hESC[hESC["gene_id"].isin([xist_human, sry_human])]


# In[22]:


mESC[mESC["gene_id"].isin([xist_mouse, sry_mouse])]


# ## 4. prepare counts for DESeq2 library estimation (all genes)

# In[23]:


hESC = hESC[["gene_id", "rep1", "rep2"]].drop_duplicates(subset="gene_id")
print(len(hESC))
print(len(hESC.gene_id.unique()))
hESC.head()


# In[24]:


mESC = mESC[["gene_id", "rep1", "rep2", "rep3"]].drop_duplicates()
print(len(mESC))
print(len(mESC.gene_id.unique()))
mESC.head()


# In[25]:


hESC_cols = {"rep1": ["rep1"], "rep2": ["rep2"]}
hESC_cols = pd.DataFrame.from_dict(hESC_cols, orient="index").reset_index()
hESC_cols.columns = ["column", "condition"]
hESC_cols.head()


# In[26]:


mESC_cols = {"rep1": ["rep1"], "rep2": ["rep2"], "rep3": ["rep3"]}
mESC_cols = pd.DataFrame.from_dict(mESC_cols, orient="index").reset_index()
mESC_cols.columns = ["column", "condition"]
mESC_cols.head()


# ## 5. prepare counts for DESeq2 differential expression analysis

# In[27]:


orth["Mouse homology type"].value_counts()


# In[28]:


# remove many2many orths
orth_sub = orth[orth["Mouse homology type"] != "ortholog_many2many"]
orth_sub.head()


# In[29]:


# subset to genes only
orth_genes = orth_sub[["Gene stable ID", "Gene name", "Mouse gene stable ID", "Mouse gene name"]].drop_duplicates()
orth_genes.columns = ["gene_id_human", "gene_name_human", "gene_id_mouse", "gene_name_mouse"]
orth_genes.head()


# In[30]:


orth_genes["orth_id"] = orth_genes["gene_id_human"] + "__" + orth_genes["gene_id_mouse"]
orth_genes.head()


# In[31]:


hESC_orth = hESC.merge(orth_genes, left_on="gene_id", right_on="gene_id_human")
hESC_orth.head()


# In[32]:


hESC_orth = hESC_orth[["orth_id", "rep1", "rep2"]].drop_duplicates()
print(len(hESC_orth))
hESC_orth.head()


# In[33]:


mESC_orth = mESC.merge(orth_genes, left_on="gene_id", right_on="gene_id_mouse")
mESC_orth.head()


# In[34]:


mESC_orth = mESC_orth[["orth_id", "rep1", "rep2", "rep3"]].drop_duplicates()
print(len(mESC_orth))
mESC_orth.head()


# In[35]:


orth_counts = hESC_orth.merge(mESC_orth, on="orth_id", suffixes=("_x", "_y"))
print(len(orth_counts))
print(len(orth_counts.orth_id.unique()))
orth_counts.columns = ["orth_id", "hESC_rep1", "hESC_rep2", "mESC_rep1", "mESC_rep2", "mESC_rep3"]
orth_counts.head()


# In[36]:


orth_cols = {"hESC_rep1": ["hESC", "sample1"], "hESC_rep2": ["hESC", "sample2"], 
             "mESC_rep1": ["mESC", "sample1"], "mESC_rep2": ["mESC", "sample2"], "mESC_rep3": ["mESC", "sample3"]}
orth_cols = pd.DataFrame.from_dict(orth_cols, orient="index").reset_index()
orth_cols.columns = ["column", "condition", "sample"]
orth_cols.head()


# ## 6. write files

# In[37]:


hESC.to_csv("%s/hESC_all.counts.txt" % rna_seq_dir, sep="\t", index=False)
mESC.to_csv("%s/mESC_all.counts.txt" % rna_seq_dir, sep="\t", index=False)
hESC_cols.to_csv("%s/hESC_all.cols.txt" % rna_seq_dir, sep="\t", index=False)
mESC_cols.to_csv("%s/mESC_all.cols.txt" % rna_seq_dir, sep="\t", index=False)


# In[38]:


orth_counts.to_csv("%s/orths.counts.txt" % rna_seq_dir, sep="\t", index=False)
orth_cols.to_csv("%s/orths.cols.txt" % rna_seq_dir, sep="\t", index=False)

