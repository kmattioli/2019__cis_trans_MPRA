
# coding: utf-8

# # TF DEA from Chiara samples

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


# ## Functions

# In[108]:


def is_sig(row):
    if row.padj < 0.01 and np.abs(row.log2FoldChange) >= 1:
        return "sig"
    else:
        return "not sig"


# In[109]:


def is_sig_voom(row):
    if row['adj.P.Val']< 0.01 and np.abs(row.logFC) >= 1:
        return "sig"
    else:
        return "not sig"


# ## variables

# In[51]:


hESC_expr_f = "../../../../for_winona/RNA-Seq/diff_expr/hESC.tpm.txt"
mESC_expr_f = "../../../../for_winona/RNA-Seq/diff_expr/mESC.tpm.txt"
orth_expr_f = "../../../../for_winona/RNA-Seq/diff_expr/orth.tpm.txt"
orth_de_f = "../../../../for_winona/RNA-Seq/diff_expr/orth.DESeq2.txt"


# In[85]:


ort_counts_voom = "../../../../for_winona/RNA-Seq/diff_expr/orth.cpm.txt"


# In[7]:


orth_de_f_s = "../../../../for_winona/RNA-Seq/diff_expr/orth.voom.tt_table.simple_model.txt"
orth_de_f_1 = "../../../../for_winona/RNA-Seq/diff_expr/orth.voom.tt_table.with_interaction.txt"
DEG_genes_transfected = "../../../../for_winona/RNA-Seq/diff_expr/DEgenes_treatment.voom.txt"


# In[8]:


orth_f = "../../../misc/00__ensembl_orthologs/ensembl96_human_mouse_orths.txt.gz"
human_gene_map_f = "../../../misc/00__ensembl_orthologs/gencode.v25lift37.GENE_ID_TO_NAME_AND_BIOTYPE_MAP.txt"
mouse_gene_map_f = "../../../misc/00__ensembl_orthologs/gencode.vM13.GENE_ID_TO_NAME_AND_BIOTYPE_MAP.txt"


# In[9]:


motif_info_dir = "../../../misc/01__motif_info"
motif_map_f = "%s/00__lambert_et_al_files/00__metadata/curated_motif_map.txt" % motif_info_dir
motif_info_f = "%s/00__lambert_et_al_files/00__metadata/motif_info.txt" % motif_info_dir


# ## 1. import data

# In[10]:


hESC_expr = pd.read_table(hESC_expr_f).reset_index()
mESC_expr = pd.read_table(mESC_expr_f).reset_index()
hESC_expr.head()


# In[136]:


orth_expr = pd.read_table(orth_expr_f).reset_index()
orth_expr.head()


# In[198]:


orth_expr_voom = pd.read_table(ort_counts_voom).reset_index()
orth_expr_voom.head()


# In[135]:


orth_de = pd.read_table(orth_de_f).reset_index()
orth_de.head()


# In[14]:


orth_de_1 = pd.read_table(orth_de_f_1).reset_index()
orth_de_1.head()


# In[15]:


orth_de_Simple = pd.read_table(orth_de_f_s).reset_index()
orth_de_Simple.head()


# In[16]:


DEGenes_trans = pd.read_table(DEG_genes_transfected)
DEGenes_trans.head()


# In[17]:


orth = pd.read_table(orth_f)
orth.head()


# In[18]:


human_gene_map = pd.read_table(human_gene_map_f, header=None)
human_gene_map.columns = ["gene_id", "biotype", "gene_name"]
human_gene_map.head()


# In[19]:


mouse_gene_map = pd.read_table(mouse_gene_map_f, header=None)
mouse_gene_map.columns = ["gene_id", "biotype", "gene_name"]
mouse_gene_map.head()


# In[20]:


motif_info = pd.read_table(motif_info_f)
motif_info.head()


# In[21]:


motif_info["short_id"] = motif_info["CIS-BP ID"].str.split(".", expand=True)[0]
motif_info = motif_info[motif_info["Best Motif(s)? (Figure 2A)"] == True]
motif_map = motif_info[["short_id","Ensembl ID", "HGNC symbol"]]
motif_map.columns = ["index", "gene_id", "gene_name"]
motif_map = motif_map.sort_values(by="index")
print(len(motif_map))
motif_map.head()


# ## 2. QC on RNA-Seq

# In[22]:


human_gene_map["index"] = human_gene_map["gene_id"].str.split(".", expand=True)[0]
mouse_gene_map["index"] = mouse_gene_map["gene_id"].str.split(".", expand=True)[0]
mouse_gene_map.head()


# In[23]:


hESC_expr = hESC_expr.merge(human_gene_map, on="index", how="left")
hESC_expr.sample(5)


# In[24]:


mESC_expr = mESC_expr.merge(mouse_gene_map, on="index", how="left")
mESC_expr.sample(5)


# In[25]:


human_genes_to_check = ["XIST", "SRY", "RPS4Y1", "DDX3Y", "POU5F1", "NANOG", "SOX2", "EOMES", "SOX17", "FOXA2"]


# In[27]:


human_sub = hESC_expr[hESC_expr["gene_name"].isin(human_genes_to_check)]
human_sub = pd.melt(human_sub[["gene_name", "rep1", "rep2","rep3",]], id_vars="gene_name")
human_sub.head()


# In[28]:


fig = plt.figure(figsize=(4, 1))

ax = sns.barplot(data=human_sub, x="gene_name", y="value", hue="variable", palette="Paired", 
                 order=human_genes_to_check)
#ax.set_yscale('symlog')
ax.set_xticklabels(human_genes_to_check, va="top", ha="right", rotation=50)
ax.set_ylabel("tpm")
ax.set_title("expression of human genes in hESCs")
ax.set_xlabel("")

plt.legend(loc=2, bbox_to_anchor=(1.05, 1))


# In[29]:


mouse_genes_to_check = ["Xist", "Sry", "Eif2s3y", "Ddx3y", "Pou5f1", "Nanog", "Sox2", "Eomes", "Sox17", "Foxa2"]


# In[30]:


mouse_sub = mESC_expr[mESC_expr["gene_name"].isin(mouse_genes_to_check)]
mouse_sub = pd.melt(mouse_sub[["gene_name", "rep1", "rep2","rep3"]], id_vars="gene_name")
mouse_sub.head()


# In[31]:


fig = plt.figure(figsize=(4, 1))

ax = sns.barplot(data=mouse_sub, x="gene_name", y="value", hue="variable", palette="Paired", 
                 order=mouse_genes_to_check)
#ax.set_yscale('symlog')
ax.set_xticklabels(mouse_genes_to_check, va="top", ha="right", rotation=50)
ax.set_ylabel("tpm")
ax.set_title("expression of mouse genes in mESCs")
ax.set_xlabel("")

plt.legend(loc=2, bbox_to_anchor=(1.05, 1))


# ## 3. Look at TF expression

# In[32]:


uniq_human_TFs = motif_map["gene_name"].unique()
print(len(uniq_human_TFs))

TFs_in_seq = [x for x in uniq_human_TFs if x in list(hESC_expr["gene_name"])]
print(len(TFs_in_seq))

TFs_missing = [x for x in uniq_human_TFs if x not in list(hESC_expr["gene_name"])]
print(len(TFs_missing))


# In[33]:


## Differentially expressed genes between transfected and untransfected samples
DEGenes_trans_human = pd.DataFrame(columns=['index'])
DEGenes_trans_mouse = pd.DataFrame(columns=['index'])
DEGenes_trans_human['index'] = DEGenes_trans.apply(lambda row: row['x'].split('__')[0], axis=1)
DEGenes_trans_mouse['index'] = DEGenes_trans.apply(lambda row: row['x'].split('__')[1], axis=1)
print(len(DEGenes_trans_human))
print(len(DEGenes_trans_mouse))


# In[34]:


DEGenes_trans_human = DEGenes_trans_human.merge(human_gene_map, how='left', on='index')
DEGenes_trans_mouse = DEGenes_trans_mouse.merge(mouse_gene_map, how='left', on='index')
DEGenes_trans_human.head()


# In[35]:


DEGenes_trans_human_list = DEGenes_trans_human['gene_name'].values.tolist()
DEGenes_trans_mouse_list = DEGenes_trans_mouse['gene_name'].values.tolist()

TFs_DE_trans = [x for x in DEGenes_trans_human_list if x in uniq_human_TFs]
print(len(TFs_DE_trans))


# In[36]:


TFs_DE_trans


# In[37]:


uniq_human_TFs_filtered = set(uniq_human_TFs)-set(TFs_DE_trans)
len(uniq_human_TFs_filtered)


# In[38]:


hESC_TFs = hESC_expr[hESC_expr["gene_name"].isin(uniq_human_TFs_filtered)].drop_duplicates(subset=["index", "rep1", "rep2",
                                                                                          "biotype", "gene_name"])
print(len(hESC_TFs))
hESC_TFs.head()


# In[39]:


tmp = hESC_TFs.groupby("gene_name")["index"].agg("count").reset_index()
tmp.sort_values(by="index", ascending=False).head()


# In[40]:


hESC_TFs = hESC_TFs[hESC_TFs["index"] != "ENSG00000214652"]
len(hESC_TFs)


# In[41]:


fig = plt.figure(figsize=(2, 1))

ax = sns.distplot(np.log10(hESC_TFs["rep1"]+0.001), label="rep 1", color=sns.color_palette("Set2")[0], hist=False)
sns.distplot(np.log10(hESC_TFs["rep2"]+0.001), label="rep 2", color=sns.color_palette("Set2")[1], hist=False)
sns.distplot(np.log10(hESC_TFs["rep3"]+0.001), label="rep 3", color=sns.color_palette("Set2")[2], hist=False)


ax.set_xlabel("log10(tpm + 0.001)")
ax.set_ylabel("density")
ax.set_title("hESCs")

plt.legend(loc=2, bbox_to_anchor=(1.05, 1))


# In[42]:


hESC_TFs["mean_tpm"] = hESC_TFs[["rep1", "rep2","rep3"]].mean(axis=1)
hESC_TFs.head()


# In[43]:


hESC_TFs_expr = list(hESC_TFs[hESC_TFs["mean_tpm"] > 1]["gene_name"])
len(hESC_TFs_expr)


# ## 4. Look at expression of orthologous TF

# In[44]:


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


# In[45]:


mESC_expr["mean_tpm_mouse"] = mESC_expr[["rep1", "rep2", "rep3"]].mean(axis=1)
mESC_expr.head()


# In[46]:


human_mouse_TFs = human_mouse_TFs.merge(mESC_expr[["index", "gene_name", "mean_tpm_mouse"]],
                                        left_on=["gene_id_mouse", "gene_name_mouse"],
                                        right_on=["index", "gene_name"])
human_mouse_TFs.drop(["index", "gene_name"], axis=1, inplace=True)
print(len(human_mouse_TFs))
human_mouse_TFs.head()


# In[47]:


human_mouse_TFs[human_mouse_TFs["gene_name_mouse"] == "Zfy2"]


# In[48]:


mESC_TFs_expr = list(human_mouse_TFs[human_mouse_TFs["mean_tpm_mouse"] > 1]["gene_name_mouse"].unique())
len(mESC_TFs_expr)


# ## 5. Look at orthologous expression

# In[137]:


orth_expr["gene_id_human"] = orth_expr["index"].str.split("__", expand=True)[0]
orth_expr["gene_id_mouse"] = orth_expr["index"].str.split("__", expand=True)[1]
orth_expr.head()


# In[180]:


orth_expr_voom["gene_id_human"] = orth_expr_voom["index"].str.split("__", expand=True)[0]
orth_expr_voom["gene_id_mouse"] = orth_expr_voom["index"].str.split("__", expand=True)[1]
orth_expr_voom.head()


# In[138]:


orth_sub = orth[["Gene stable ID", "Mouse gene stable ID", "Gene name", "Mouse gene name"]].drop_duplicates()
orth_sub.columns = ["gene_id_human", "gene_id_mouse", "gene_name_human", "gene_name_mouse"]
orth_expr = orth_expr.merge(orth_sub, on=["gene_id_human", "gene_id_mouse"]).drop_duplicates()
orth_expr.head()


# In[181]:


orth_sub = orth[["Gene stable ID", "Mouse gene stable ID", "Gene name", "Mouse gene name"]].drop_duplicates()
orth_sub.columns = ["gene_id_human", "gene_id_mouse", "gene_name_human", "gene_name_mouse"]
orth_expr_voom = orth_expr_voom.merge(orth_sub, on=["gene_id_human", "gene_id_mouse"]).drop_duplicates()
orth_expr_voom.head()


# In[139]:


orth_expr["mean_tpm_hESC"] = orth_expr[["hESC_rep1", "hESC_rep2","hESC_rep3"]].mean(axis=1)
orth_expr["mean_tpm_mESC"] = orth_expr[["mESC_rep1", "mESC_rep2", "mESC_rep3"]].mean(axis=1)
orth_expr.head()


# In[182]:


orth_expr_voom["mean_tpm_hESC"] = orth_expr_voom[["hESC_rep1", "hESC_rep2","hESC_rep3"]].mean(axis=1)
orth_expr_voom["mean_tpm_mESC"] = orth_expr_voom[["mESC_rep1", "mESC_rep2", "mESC_rep3"]].mean(axis=1)
orth_expr_voom.head()


# In[140]:


orth_expr= orth_expr.merge(orth_de, on="index")
orth_expr.head()


# In[183]:


orth_expr_voom= orth_expr_voom.merge(orth_de_1, on="index")
orth_expr_voom.head()


# In[142]:


orth_expr["sig"] = orth_expr.apply(is_sig, axis=1)
orth_expr.sig.value_counts()


# In[184]:


orth_expr_voom["sig"] = orth_expr_voom.apply(is_sig_voom, axis=1)
orth_expr_voom.sig.value_counts()


# In[145]:


fig = plt.figure(figsize=(2, 1))

ax = sns.distplot(np.log10(orth_expr["baseMean"]+0.001), label="rep 1", color=sns.color_palette("Set2")[2], hist=False)

ax.set_xlabel("log10(base mean tpm + 0.001)")
ax.set_ylabel("density")

plt.legend(loc=2, bbox_to_anchor=(1.05, 1))


# In[146]:


orth_expr_filt = orth_expr[orth_expr["baseMean"] >= 1]
len(orth_expr_filt)


# In[147]:


fig, ax = plt.subplots(figsize=(2.2, 1.2), nrows=1, ncols=1)

ax.scatter(np.log10(orth_expr_filt[orth_expr_filt["sig"] == "not sig"]["baseMean"]+0.001), 
           orth_expr_filt[orth_expr_filt["sig"] == "not sig"]["log2FoldChange"],
           color="gray", alpha=0.75, s=10, rasterized=True)
ax.scatter(np.log10(orth_expr_filt[orth_expr_filt["sig"] == "sig"]["baseMean"]+0.001), 
           orth_expr_filt[orth_expr_filt["sig"] == "sig"]["log2FoldChange"],
           color="firebrick", alpha=0.75, s=10, rasterized=True)


# In[148]:


fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

ax.scatter(np.log2(orth_expr_filt["mean_tpm_hESC"]+0.001), 
           np.log2(orth_expr_filt["mean_tpm_mESC"]+0.001),
           color="gray", alpha=0.25, s=10, rasterized=True)


# In[149]:


orth_tf_expr = human_mouse_TFs.merge(orth_expr, on=["gene_id_human", "gene_name_human", 
                                                    "gene_id_mouse", "gene_name_mouse"]).drop_duplicates()
print(len(orth_tf_expr))
orth_tf_expr.head()


# In[185]:


orth_tf_expr_voom = human_mouse_TFs.merge(orth_expr_voom, on=["gene_id_human", "gene_name_human", 
                                                    "gene_id_mouse", "gene_name_mouse"]).drop_duplicates()
print(len(orth_tf_expr_voom))
orth_tf_expr_voom.head()


# In[151]:


orth_tf_expr = orth_tf_expr[["gene_id_human", "gene_name_human", "mean_tpm_hESC", "gene_id_mouse", "gene_name_mouse",
                             "mean_tpm_mESC", "baseMean", "log2FoldChange", "lfcSE", "padj", "sig"]].drop_duplicates()
len(orth_tf_expr)


# In[186]:


orth_tf_expr_voom = orth_tf_expr_voom[["gene_id_human", "gene_name_human", "mean_tpm_hESC", "gene_id_mouse", "gene_name_mouse",
                             "mean_tpm_mESC", "AveExpr", "logFC", "adj.P.Val", "sig"]].drop_duplicates()
len(orth_tf_expr_voom)


# In[154]:


# remove any orth pair that maps to more than one gene
tmp = orth_tf_expr.groupby("gene_name_human")["gene_name_mouse"].agg("count").reset_index()
human_dupe_orths = tmp[tmp["gene_name_mouse"] > 1]
print(len(human_dupe_orths))
human_dupe_orths


# In[155]:


# remove any orth pair that maps to more than one gene
tmp = orth_tf_expr.groupby("gene_name_mouse")["gene_name_human"].agg("count").reset_index()
mouse_dupe_orths = tmp[tmp["gene_name_human"] > 1]
print(len(mouse_dupe_orths))
mouse_dupe_orths.head()


# In[156]:


orth_tf_expr = orth_tf_expr[~orth_tf_expr["gene_name_human"].isin(human_dupe_orths["gene_name_human"])]
orth_tf_expr = orth_tf_expr[~orth_tf_expr["gene_name_mouse"].isin(mouse_dupe_orths["gene_name_mouse"])]
len(orth_tf_expr)


# In[187]:


orth_tf_expr_voom = orth_tf_expr_voom[~orth_tf_expr_voom["gene_name_human"].isin(human_dupe_orths["gene_name_human"])]
orth_tf_expr_voom = orth_tf_expr_voom[~orth_tf_expr_voom["gene_name_mouse"].isin(mouse_dupe_orths["gene_name_mouse"])]
len(orth_tf_expr_voom)


# In[159]:


fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

ax.scatter(orth_tf_expr["mean_tpm_hESC"], 
           orth_tf_expr["mean_tpm_mESC"],
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
no_nan = orth_tf_expr[(~pd.isnull(orth_tf_expr["mean_tpm_hESC"])) & 
                      (~pd.isnull(orth_tf_expr["mean_tpm_mESC"]))]
r, p = spearmanr(no_nan["mean_tpm_hESC"], no_nan["mean_tpm_mESC"])
ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.90, "n = %s" % (len(no_nan)), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
fig.savefig("../../../../for_winona/RNA-Seq/diff_expr/TF_human_v_mouse_scatter-NOut.pdf", dpi="figure", bbox_inches="tight")


# In[162]:


fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

sig = orth_tf_expr[orth_tf_expr["sig"] == "sig"]
not_sig = orth_tf_expr[orth_tf_expr["sig"] == "not sig"]

ax.scatter(sig["mean_tpm_hESC"], 
           sig["mean_tpm_mESC"],
           color=sns.color_palette("Set2")[2], alpha=0.75, s=10, 
           linewidths=0.5, edgecolors="white")

ax.scatter(not_sig["mean_tpm_hESC"], 
           not_sig["mean_tpm_mESC"],
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
no_nan = orth_tf_expr[(~pd.isnull(orth_tf_expr["mean_tpm_hESC"])) & 
                      (~pd.isnull(orth_tf_expr["mean_tpm_mESC"]))]
r, p = spearmanr(no_nan["mean_tpm_hESC"], no_nan["mean_tpm_mESC"])
ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.90, "# sig = %s" % (len(sig)), ha="left", va="top", fontsize=fontsize, 
        color=sns.color_palette("Set2")[2],
        transform=ax.transAxes)
ax.text(0.05, 0.83, "# not sig = %s" % (len(not_sig)), ha="left", va="top", fontsize=fontsize, color="gray",
        transform=ax.transAxes)
fig.savefig("../../../../for_winona/RNA-Seq/diff_expr/TF_human_v_mouse_scatter.w_sig_outline-no_UT.pdf", dpi="figure", bbox_inches="tight")


# ## Voom normalized counts

# In[203]:


fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

ax.scatter(orth_tf_expr_voom["mean_tpm_hESC"], 
           orth_tf_expr_voom["mean_tpm_mESC"],
           color=sns.color_palette("Set2")[2], alpha=0.75, s=10, 
           linewidths=0.5, edgecolors="white")
#ax.set_xscale("symlog")
#ax.set_yscale("symlog")
ax.plot([-4, 12], [-4, 12], "k", linestyle="dashed")
ax.set_xlim((-4, 12))
ax.set_ylim((-4, 12))

ax.set_xlabel("human TF log2(CPM) in hESC")
ax.set_ylabel("mouse TF log2(CPM) in mESC")

# annotate corr
no_nan = orth_tf_expr_voom[(~pd.isnull(orth_tf_expr_voom["mean_tpm_hESC"])) & 
                      (~pd.isnull(orth_tf_expr_voom["mean_tpm_mESC"]))]
r, p = spearmanr(no_nan["mean_tpm_hESC"], no_nan["mean_tpm_mESC"])
ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.90, "n = %s" % (len(no_nan)), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
fig.savefig("../../../../for_winona/RNA-Seq/diff_expr/TF_human_v_mouse_scatter.voom_cpm.full_model.pdf", dpi="figure", bbox_inches="tight")


# In[204]:


max(orth_tf_expr_voom['mean_tpm_hESC'])


# In[205]:


fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

sig = orth_tf_expr_voom[orth_tf_expr_voom["sig"] == "sig"]
not_sig = orth_tf_expr_voom[orth_tf_expr_voom["sig"] == "not sig"]

ax.scatter(sig["mean_tpm_hESC"], 
           sig["mean_tpm_mESC"],
           color=sns.color_palette("Set2")[2], alpha=0.75, s=10, 
           linewidths=0.5, edgecolors="white")

ax.scatter(not_sig["mean_tpm_hESC"], 
           not_sig["mean_tpm_mESC"],
           color="gray", alpha=0.9, s=10, 
           linewidths=0.5, edgecolors="white")

ax.plot([-4, 12], [-4, 12], "k", linestyle="dashed")
ax.set_xlim((-4, 12))
ax.set_ylim((-4, 12))

ax.set_xlabel("human TF log2(CPM) in hESC")
ax.set_ylabel("mouse TF log2(CPM) in mESC")

# annotate corr
no_nan = orth_tf_expr[(~pd.isnull(orth_tf_expr["mean_tpm_hESC"])) & 
                      (~pd.isnull(orth_tf_expr["mean_tpm_mESC"]))]
r, p = spearmanr(no_nan["mean_tpm_hESC"], no_nan["mean_tpm_mESC"])
ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.90, "# sig = %s" % (len(sig)), ha="left", va="top", fontsize=fontsize, 
        color=sns.color_palette("Set2")[2],
        transform=ax.transAxes)
ax.text(0.05, 0.83, "# not sig = %s" % (len(not_sig)), ha="left", va="top", fontsize=fontsize, color="gray",
        transform=ax.transAxes)
fig.savefig("../../../../for_winona/RNA-Seq/diff_expr/TF_human_v_mouse_scatter.w_sig_outline.voom_cpm.full_model.pdf", dpi="figure", bbox_inches="tight")


# ## Write files

# In[195]:


orth_tf_expr_f = "../../../../for_winona/RNA-Seq/diff_expr/orth_TF_expression.DESeq2.txt"
orth_tf_expr.to_csv(orth_tf_expr_f, sep="\t", index=False)


# In[196]:


hESC_TF_expr_f = "../../../../for_winona/RNA-Seq/diff_expr/hESC_TF_expression.DESeq2.txt"
hESC_TFs.to_csv(hESC_TF_expr_f, sep="\t", index=False)


# In[193]:


mESC_TFs = human_mouse_TFs[["gene_id_human", "gene_name_human", "gene_id_mouse", "gene_name_mouse", "mean_tpm_mouse"]].drop_duplicates()
len(mESC_TFs)


# In[197]:


mESC_TF_expr_f = "../../../../for_winona/RNA-Seq/diff_expr/mESC_TF_expression.DESeq2.txt"
mESC_TFs.to_csv(mESC_TF_expr_f, sep="\t", index=False)


# In[199]:


orth_tf_expr_f = "../../../../for_winona/RNA-Seq/diff_expr/orth_TF_expression.VOOM.txt"
orth_tf_expr_voom.to_csv(orth_tf_expr_f, sep="\t", index=False)

