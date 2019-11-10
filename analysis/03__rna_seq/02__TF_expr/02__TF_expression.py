
# coding: utf-8

# # 03__TF_expr

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


# ## functions

# In[4]:


def is_sig(row):
    if row["adj.P.Val"] < 0.01 and np.abs(row["logFC"]) >= 1:
        return "sig"
    else:
        return "not sig"


# ## variables

# In[5]:


hESC_expr_f = "../../../data/03__rna_seq/00__HUES64/02__tpms/hESC.tpm.txt"
mESC_expr_f = "../../../data/03__rna_seq/01__mESC/02__tpms/mESC.tpm.txt"
orth_expr_f = "../../../data/03__rna_seq/02__orths/02__tpms/orth.tpm.txt"
orth_de_f = "../../../data/03__rna_seq/03__DE/orth.voom.tt_table.with_interaction.txt"


# In[6]:


orth_f = "../../../misc/00__ensembl_orthologs/ensembl96_human_mouse_orths.txt.gz"
human_gene_map_f = "../../../misc/00__ensembl_orthologs/gencode.v25lift37.GENE_ID_TO_NAME_AND_BIOTYPE_MAP.txt"
mouse_gene_map_f = "../../../misc/00__ensembl_orthologs/gencode.vM13.GENE_ID_TO_NAME_AND_BIOTYPE_MAP.txt"


# In[7]:


motif_info_dir = "../../../misc/01__motif_info"
motif_map_f = "%s/00__lambert_et_al_files/00__metadata/curated_motif_map.txt" % motif_info_dir
motif_info_f = "%s/00__lambert_et_al_files/00__metadata/motif_info.txt" % motif_info_dir


# ## 1. import data

# In[8]:


hESC_expr = pd.read_table(hESC_expr_f).reset_index()
mESC_expr = pd.read_table(mESC_expr_f).reset_index()
hESC_expr.head()


# In[9]:


orth_expr = pd.read_table(orth_expr_f).reset_index()
orth_expr.head()


# In[10]:


orth_de = pd.read_table(orth_de_f).reset_index()
orth_de.head()


# In[11]:


orth = pd.read_table(orth_f)
orth.head()


# In[12]:


human_gene_map = pd.read_table(human_gene_map_f, header=None)
human_gene_map.columns = ["gene_id", "biotype", "gene_name"]
human_gene_map.head()


# In[13]:


mouse_gene_map = pd.read_table(mouse_gene_map_f, header=None)
mouse_gene_map.columns = ["gene_id", "biotype", "gene_name"]
mouse_gene_map.head()


# In[14]:


motif_info = pd.read_table(motif_info_f)
motif_info.head()


# In[15]:


motif_info["short_id"] = motif_info["CIS-BP ID"].str.split(".", expand=True)[0]
motif_info = motif_info[motif_info["Best Motif(s)? (Figure 2A)"] == True]
motif_map = motif_info[["short_id","Ensembl ID", "HGNC symbol"]]
motif_map.columns = ["index", "gene_id", "gene_name"]
motif_map = motif_map.sort_values(by="index")
print(len(motif_map))
motif_map.head()


# In[16]:


motif_map[motif_map["gene_name"] == "ASCL2"]


# ## 2. do some QC on RNA-seq

# In[17]:


human_gene_map["index"] = human_gene_map["gene_id"].str.split(".", expand=True)[0]
mouse_gene_map["index"] = mouse_gene_map["gene_id"].str.split(".", expand=True)[0]
mouse_gene_map.head()


# In[18]:


hESC_expr = hESC_expr.merge(human_gene_map, on="index", how="left")
hESC_expr.sample(5)


# In[19]:


mESC_expr = mESC_expr.merge(mouse_gene_map, on="index", how="left")
mESC_expr.sample(5)


# In[20]:


human_genes_to_check = ["XIST", "SRY", "RPS4Y1", "DDX3Y", "POU5F1", "NANOG", "SOX2", "EOMES", "SOX17", "FOXA2"]


# In[21]:


human_sub = hESC_expr[hESC_expr["gene_name"].isin(human_genes_to_check)]
human_sub = pd.melt(human_sub[["gene_name", "rep1", "rep2", "rep3"]], id_vars="gene_name")
human_sub.head()


# In[22]:


fig = plt.figure(figsize=(4, 1))

ax = sns.barplot(data=human_sub, x="gene_name", y="value", hue="variable", palette="Paired", 
                 order=human_genes_to_check)
#ax.set_yscale('symlog')
ax.set_xticklabels(human_genes_to_check, va="top", ha="right", rotation=50)
ax.set_ylabel("tpm")
ax.set_title("expression of human genes in hESCs")
ax.set_xlabel("")

plt.legend(loc=2, bbox_to_anchor=(1.05, 1))


# In[23]:


mouse_genes_to_check = ["Xist", "Sry", "Eif2s3y", "Ddx3y", "Pou5f1", "Nanog", "Sox2", "Eomes", "Sox17", "Foxa2"]


# In[24]:


mouse_sub = mESC_expr[mESC_expr["gene_name"].isin(mouse_genes_to_check)]
mouse_sub = pd.melt(mouse_sub[["gene_name", "rep1", "rep2", "rep3"]], id_vars="gene_name")
mouse_sub.head()


# In[25]:


mouse_sub.gene_name.unique()


# In[26]:


fig = plt.figure(figsize=(4, 1))

ax = sns.barplot(data=mouse_sub, x="gene_name", y="value", hue="variable", palette="Paired", 
                 order=mouse_genes_to_check)
#ax.set_yscale('symlog')
ax.set_xticklabels(mouse_genes_to_check, va="top", ha="right", rotation=50)
ax.set_ylabel("tpm")
ax.set_title("expression of mouse genes in mESCs")
ax.set_xlabel("")

plt.legend(loc=2, bbox_to_anchor=(1.05, 1))


# ## 3. look at expression of TFs in hESCs

# In[27]:


uniq_human_TFs = motif_map["gene_name"].unique()
print(len(uniq_human_TFs))

TFs_in_seq = [x for x in uniq_human_TFs if x in list(hESC_expr["gene_name"])]
print(len(TFs_in_seq))

TFs_missing = [x for x in uniq_human_TFs if x not in list(hESC_expr["gene_name"])]
print(len(TFs_missing))


# In[28]:


TFs_missing


# the above TFs are missing from the RNA-seq so will not be included (I hand checked a few and couldn't find easy aliases, might look more later)

# In[29]:


hESC_TFs = hESC_expr[hESC_expr["gene_name"].isin(uniq_human_TFs)].drop_duplicates(subset=["index", "rep1", "rep2", "rep3",
                                                                                          "biotype", "gene_name"])
print(len(hESC_TFs))
hESC_TFs.head()


# In[30]:


tmp = hESC_TFs.groupby("gene_name")["index"].agg("count").reset_index()
tmp.sort_values(by="index", ascending=False).head()


# one of these TFs have multiple gene_ids (probably from using the lifted gencode v25 instead of gencode v25 in hg38). manually fix this guy

# In[31]:


hESC_TFs = hESC_TFs[hESC_TFs["index"] != "ENSG00000273439"]
len(hESC_TFs)


# In[32]:


fig = plt.figure(figsize=(2, 1))

ax = sns.distplot(np.log10(hESC_TFs["rep1"]+0.001), label="rep 1", color=sns.color_palette("Set2")[0], hist=False)
sns.distplot(np.log10(hESC_TFs["rep2"]+0.001), label="rep 2", color=sns.color_palette("Set2")[1], hist=False)
sns.distplot(np.log10(hESC_TFs["rep3"]+0.001), label="rep 3", color=sns.color_palette("Set2")[2], hist=False)

ax.set_xlabel("log10(tpm + 0.001)")
ax.set_ylabel("density")
ax.set_title("hESCs")

plt.legend(loc=2, bbox_to_anchor=(1.05, 1))


# In[33]:


hESC_TFs["mean_tpm"] = hESC_TFs[["rep1", "rep2", "rep3"]].mean(axis=1)
hESC_TFs.head()


# In[34]:


hESC_TFs_expr = list(hESC_TFs[hESC_TFs["mean_tpm"] > 1]["gene_name"])
len(hESC_TFs_expr)


# ## 4. look at expression of orthologous TFs in mouse

# In[35]:


human_mouse_TFs = hESC_TFs[["index", "gene_name", "mean_tpm"]]
print(len(human_mouse_TFs))
human_mouse_TFs = human_mouse_TFs.merge(orth[["Gene stable ID", 
                                              "Mouse gene stable ID", 
                                              "Gene name",
                                              "Mouse gene name"]].drop_duplicates(),
                                        left_on=["index", "gene_name"],
                                        right_on=["Gene stable ID", "Gene name"])
human_mouse_TFs.drop(["Gene stable ID", "Gene name"], axis=1, inplace=True)
human_mouse_TFs.columns = ["gene_id_human", "gene_name_human", "mean_tpm_human", "gene_id_mouse", "gene_name_mouse"]
print(len(human_mouse_TFs))
human_mouse_TFs.head()


# In[36]:


mESC_expr["mean_tpm_mouse"] = mESC_expr[["rep1", "rep2", "rep3"]].mean(axis=1)
mESC_expr.head()


# In[37]:


human_mouse_TFs = human_mouse_TFs.merge(mESC_expr[["index", "gene_name", "mean_tpm_mouse"]],
                                        left_on=["gene_id_mouse", "gene_name_mouse"],
                                        right_on=["index", "gene_name"])
human_mouse_TFs.drop(["index", "gene_name"], axis=1, inplace=True)
print(len(human_mouse_TFs))
human_mouse_TFs.head()


# In[38]:


human_mouse_TFs[human_mouse_TFs["gene_name_mouse"] == "Zfy2"]


# In[39]:


mESC_TFs_expr = list(human_mouse_TFs[human_mouse_TFs["mean_tpm_mouse"] > 1]["gene_name_mouse"].unique())
len(mESC_TFs_expr)


# ## 5. look at orthologous expression

# In[40]:


orth_expr["gene_id_human"] = orth_expr["index"].str.split("__", expand=True)[0]
orth_expr["gene_id_mouse"] = orth_expr["index"].str.split("__", expand=True)[1]
orth_expr.head()


# In[41]:


orth_sub = orth[["Gene stable ID", "Mouse gene stable ID", "Gene name", "Mouse gene name"]].drop_duplicates()
orth_sub.columns = ["gene_id_human", "gene_id_mouse", "gene_name_human", "gene_name_mouse"]
orth_expr = orth_expr.merge(orth_sub, on=["gene_id_human", "gene_id_mouse"]).drop_duplicates()
orth_expr.head()


# In[42]:


orth_expr["mean_tpm_hESC"] = orth_expr[["hESC_rep1", "hESC_rep2", "hESC_rep3"]].mean(axis=1)
orth_expr["mean_tpm_mESC"] = orth_expr[["mESC_rep1", "mESC_rep2", "mESC_rep3"]].mean(axis=1)
orth_expr.head()


# In[43]:


print(len(orth_expr))
orth_expr = orth_expr.merge(orth_de, on="index")
print(len(orth_expr))
orth_expr.head()


# In[44]:


orth_expr["sig"] = orth_expr.apply(is_sig, axis=1)
orth_expr.sig.value_counts()


# In[45]:


fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

ax.scatter(np.log10(orth_expr["mean_tpm_hESC"]+0.001), 
           np.log10(orth_expr["mean_tpm_mESC"]+0.001),
           color="gray", alpha=0.25, s=10, rasterized=True)


# In[46]:


print(len(human_mouse_TFs))
orth_tf_expr = human_mouse_TFs.merge(orth_expr, on=["gene_id_human", "gene_name_human", 
                                                    "gene_id_mouse", "gene_name_mouse"]).drop_duplicates()
print(len(orth_tf_expr))
orth_tf_expr.head()


# In[47]:


orth_tf_expr = orth_tf_expr[["gene_id_human", "gene_name_human", "mean_tpm_human", "gene_id_mouse", "gene_name_mouse",
                             "mean_tpm_mouse", "AveExpr", "logFC", "adj.P.Val", "sig"]].drop_duplicates()
len(orth_tf_expr)


# In[48]:


# remove any orth pair that maps to more than one gene
tmp = orth_tf_expr.groupby("gene_name_human")["gene_name_mouse"].agg("count").reset_index()
human_dupe_orths = tmp[tmp["gene_name_mouse"] > 1]
print(len(human_dupe_orths))
human_dupe_orths


# In[49]:


# remove any orth pair that maps to more than one gene
tmp = orth_tf_expr.groupby("gene_name_mouse")["gene_name_human"].agg("count").reset_index()
mouse_dupe_orths = tmp[tmp["gene_name_human"] > 1]
print(len(mouse_dupe_orths))
mouse_dupe_orths.head()


# In[50]:


orth_tf_expr = orth_tf_expr[~orth_tf_expr["gene_name_human"].isin(human_dupe_orths["gene_name_human"])]
orth_tf_expr = orth_tf_expr[~orth_tf_expr["gene_name_mouse"].isin(mouse_dupe_orths["gene_name_mouse"])]
len(orth_tf_expr)


# In[51]:


fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

ax.scatter(orth_tf_expr["mean_tpm_human"], 
           orth_tf_expr["mean_tpm_mouse"],
           color=sns.color_palette("Set2")[2], alpha=0.75, s=10, 
           linewidths=0.5, edgecolors="white")
ax.set_xscale("symlog")
ax.set_yscale("symlog")
ax.plot([-0.75, 200000], [-0.75, 200000], "k", linestyle="dashed")
ax.set_xlim((-0.75, 200000))
ax.set_ylim((-0.75, 200000))

ax.set_xlabel("human TF tpm in hESC")
ax.set_ylabel("mouse TF tpm in mESC")

# annotate corr
no_nan = orth_tf_expr[(~pd.isnull(orth_tf_expr["mean_tpm_human"])) & 
                      (~pd.isnull(orth_tf_expr["mean_tpm_mouse"]))]
r, p = spearmanr(no_nan["mean_tpm_human"], no_nan["mean_tpm_mouse"])
ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.90, "n = %s" % (len(no_nan)), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
fig.savefig("TF_human_v_mouse_scatter.pdf", dpi="figure", bbox_inches="tight")


# In[52]:


orth_tf_expr["neg_log_padj"] = -np.log10(orth_tf_expr["adj.P.Val"])
orth_tf_expr.head()


# In[53]:


fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

sig = orth_tf_expr[orth_tf_expr["sig"] == "sig"]
not_sig = orth_tf_expr[orth_tf_expr["sig"] == "not sig"]

ax.scatter(sig["mean_tpm_human"], 
           sig["mean_tpm_mouse"],
           color=sns.color_palette("Set2")[2], alpha=0.75, s=10, 
           linewidths=0.5, edgecolors="white")

ax.scatter(not_sig["mean_tpm_human"], 
           not_sig["mean_tpm_mouse"],
           color="gray", alpha=0.9, s=10, 
           linewidths=0.5, edgecolors="white")

ax.set_xscale("symlog")
ax.set_yscale("symlog")
ax.plot([-0.75, 400000], [-0.75, 400000], "k", linestyle="dashed")
ax.set_xlim((-0.75, 400000))
ax.set_ylim((-0.75, 400000))

ax.set_xlabel("human TF tpm in hESC")
ax.set_ylabel("mouse TF tpm in mESC")

# annotate corr
no_nan = orth_tf_expr[(~pd.isnull(orth_tf_expr["mean_tpm_human"])) & 
                      (~pd.isnull(orth_tf_expr["mean_tpm_mouse"]))]
r, p = spearmanr(no_nan["mean_tpm_human"], no_nan["mean_tpm_mouse"])
ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.90, "# sig = %s" % (len(sig)), ha="left", va="top", fontsize=fontsize, 
        color=sns.color_palette("Set2")[2],
        transform=ax.transAxes)
ax.text(0.05, 0.83, "# not sig = %s" % (len(not_sig)), ha="left", va="top", fontsize=fontsize, color="gray",
        transform=ax.transAxes)
fig.savefig("TF_human_v_mouse_scatter.w_sig_outline.pdf", dpi="figure", bbox_inches="tight")


# In[54]:


sig.sort_values(by="logFC", ascending=False).head()


# In[55]:


sig.sort_values(by="logFC", ascending=True).head()


# In[64]:


fig, ax = plt.subplots(figsize=(2.5, 2), nrows=1, ncols=1)

ax.scatter(sig["logFC"], 
           sig["neg_log_padj"],
           color=sns.color_palette("Set2")[2], alpha=0.75, s=15, 
           linewidths=0.5, edgecolors="white")
ax.scatter(not_sig["logFC"], 
           not_sig["neg_log_padj"],
           color="gray", alpha=0.9, s=15, 
           linewidths=0.5, edgecolors="white")

# annotate top genes
ZFP57 = sig[sig["gene_name_human"] == "ZFP57"].iloc[0]
GTF2I = sig[sig["gene_name_human"] == "GTF2I"].iloc[0]
POU3F1 = sig[sig["gene_name_human"] == "POU3F1"].iloc[0]
SOX3 = sig[sig["gene_name_human"] == "SOX3"].iloc[0]

ax.scatter(ZFP57["logFC"], ZFP57["neg_log_padj"], color=sns.color_palette("Set2")[2], alpha=0.75, s=15,
           linewidths=0.5, edgecolors="black")
ax.text(ZFP57.logFC+0.99, ZFP57.neg_log_padj+0.2, "ZFP57", ha="left", va="center", fontsize=fontsize-1)

ax.scatter(GTF2I["logFC"], GTF2I["neg_log_padj"], color=sns.color_palette("Set2")[2], alpha=0.75, s=15,
           linewidths=0.5, edgecolors="black")
ax.text(GTF2I.logFC+0.55, GTF2I.neg_log_padj-0.25, "GTF2I", ha="left", va="top", fontsize=fontsize-1)

ax.scatter(POU3F1["logFC"], POU3F1["neg_log_padj"], color=sns.color_palette("Set2")[2], alpha=0.75, s=15,
           linewidths=0.5, edgecolors="black")
ax.text(POU3F1.logFC+0.2, POU3F1.neg_log_padj+0.65, "POU3F1", ha="right", va="center", fontsize=fontsize-1)

ax.scatter(SOX3["logFC"], SOX3["neg_log_padj"], color=sns.color_palette("Set2")[2], alpha=0.75, s=15,
           linewidths=0.5, edgecolors="black")
ax.text(SOX3.logFC, SOX3.neg_log_padj-0.5, "SOX3", ha="right", va="top", fontsize=fontsize-1)


ax.set_xlim((-20, 25))
ax.set_xlabel("log2([mESCs/hESCs])")
ax.set_ylabel("-log10(adjusted p-value)")
ax.axvline(x=0, linestyle="dashed", color="black")


fig.savefig("TF_human_v_mouse_volcano.pdf", dpi="figure", bbox_inches="tight")


# ## 6. write files

# In[57]:


orth_tf_expr.head()


# In[58]:


orth_tf_expr[orth_tf_expr["gene_name_human"] == "ZNF524"]


# In[59]:


orth_tf_expr_f = "../../../data/03__rna_seq/04__TF_expr/orth_TF_expression.txt"
orth_tf_expr.to_csv(orth_tf_expr_f, sep="\t", index=False)


# In[60]:


hESC_TFs = hESC_TFs[["index", "gene_name", "mean_tpm"]].drop_duplicates()
len(hESC_TFs)


# In[61]:


hESC_TF_expr_f = "../../../data/03__rna_seq/04__TF_expr/hESC_TF_expression.txt"
hESC_TFs.to_csv(hESC_TF_expr_f, sep="\t", index=False)


# In[62]:


mESC_TFs = human_mouse_TFs[["gene_id_human", "gene_name_human", "gene_id_mouse", "gene_name_mouse", "mean_tpm_mouse"]].drop_duplicates()
len(mESC_TFs)


# In[63]:


mESC_TF_expr_f = "../../../data/03__rna_seq/04__TF_expr/mESC_TF_expression.txt"
mESC_TFs.to_csv(mESC_TF_expr_f, sep="\t", index=False)

