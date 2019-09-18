
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import itertools
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys

from itertools import combinations 
from scipy.stats import boxcox
from scipy.stats import linregress
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from statsmodels.stats.anova import anova_lm

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

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


# In[4]:


QUANT_ALPHA = 0.05


# ## functions

# In[5]:


def calculate_gc(row, col):
    cs = row[col].count("C")
    gs = row[col].count("G")
    gc = (cs+gs)/len(row[col])
    return gc


# In[6]:


def calculate_cpg(row, col):
    cpgs = row[col].count("CG")
    cpg = cpgs/len(row[col])
    return cpg


# In[7]:


def sig_status(row):
    if row.padj_trans < 0.05:
        return "sig"
    else:
        return "not sig"


# In[8]:


def neg_odds(row):
    if row["sig_status"] == "sig hESC":
        return -row["hESC_odds"]
    elif row["sig_status"] == "sig mESC":
        return row["mESC_odds"]
    else:
        return np.nan


# In[9]:


def direction_match(row):
    if row.activ_or_repr == "activating":
        if row.beta_trans < 0 and row.corrected_l2fc < 0:
            return "match"
        elif row.beta_trans > 0 and row.corrected_l2fc > 0:
            return "match"
        else:
            return "no match"
    elif row.activ_or_repr == "repressing":
        if row.beta_trans < 0 and row.corrected_l2fc > 0:
            return "match"
        elif row.beta_trans > 0 and row.corrected_l2fc < 0:
            return "match"
        else:
            return "no match"
    else:
        return "unclear"


# ## variables

# In[10]:


human_motifs_f = "../../../data/04__mapped_motifs/human_motifs_filtered.txt.gz"
mouse_motifs_f = "../../../data/04__mapped_motifs/mouse_motifs_filtered.txt.gz"


# In[11]:


motif_info_dir = "../../../misc/01__motif_info"
motif_map_f = "%s/00__lambert_et_al_files/00__metadata/curated_motif_map.txt" % motif_info_dir
motif_info_f = "%s/00__lambert_et_al_files/00__metadata/motif_info.txt" % motif_info_dir


# In[12]:


sig_motifs_f = "../../../data/04__mapped_motifs/sig_motifs.txt"


# In[13]:


tss_map_f = "../../../data/01__design/01__mpra_list/mpra_tss.with_ids.UPDATED_WITH_DIV.txt"


# In[14]:


index_f = "../../../data/01__design/02__index/TWIST_pool4_v8_final.with_element_id.txt.gz"


# In[15]:


data_f = "../../../data/02__mpra/03__results/all_processed_results.txt"


# In[16]:


expr_dir = "../../../data/03__rna_seq/04__TF_expr"
orth_expr_f = "%s/orth_TF_expression.txt" % expr_dir
human_expr_f = "%s/hESC_TF_expression.txt" % expr_dir
mouse_expr_f = "%s/mESC_TF_expression.txt" % expr_dir


# In[17]:


orth_f = "../../../misc/00__ensembl_orthologs/ensembl96_human_mouse_orths.txt.gz"


# ## 1. import data

# In[18]:


index = pd.read_table(index_f, sep="\t")
index_elem = index[["element", "tile_type", "element_id", "name", "tile_number", "chrom", "strand", "actual_start", 
                    "actual_end", "dupe_info"]]
index_elem = index_elem.drop_duplicates()


# In[19]:


tss_map = pd.read_table(tss_map_f, sep="\t")
tss_map.head()


# In[20]:


human_motifs = pd.read_table(human_motifs_f, sep="\t")
human_motifs.head()


# In[21]:


mouse_motifs = pd.read_table(mouse_motifs_f, sep="\t")
mouse_motifs.head()


# In[22]:


motif_info = pd.read_table(motif_info_f, sep="\t")
motif_info.head()


# In[23]:


sig_motifs = pd.read_table(sig_motifs_f)
sig_motifs = sig_motifs[sig_motifs["padj"] < 0.05]
print(len(sig_motifs))
sig_motifs.head()


# In[24]:


data = pd.read_table(data_f)
data.head()


# In[25]:


orth_expr = pd.read_table(orth_expr_f, sep="\t")
orth_expr.head()


# In[26]:


human_expr = pd.read_table(human_expr_f, sep="\t")
human_expr.head()


# In[27]:


mouse_expr = pd.read_table(mouse_expr_f, sep="\t")
mouse_expr.head()


# In[28]:


orth = pd.read_table(orth_f, sep="\t")
orth.head()


# ## 2. merge data to build model

# In[29]:


index_elem = index_elem[index_elem["name"].str.contains("EVO")]
index_elem.head()


# In[30]:


index_elem["tss_id"] = index_elem["name"].str.split("__", expand=True)[1]
index_elem["tss_tile_num"] = index_elem["name"].str.split("__", expand=True)[2]
index_elem.sample(5)


# In[31]:


index_human = index_elem[index_elem["name"].str.contains("HUMAN")]
index_mouse = index_elem[index_elem["name"].str.contains("MOUSE")]
index_mouse.sample(5)


# In[32]:


print(len(data))
data_elem = data.merge(index_human[["element", "tss_id", "tss_tile_num"]], left_on=["hg19_id", "tss_tile_num"],
                       right_on=["tss_id", "tss_tile_num"])
data_elem = data_elem.merge(index_mouse[["element", "tss_id", "tss_tile_num"]], left_on=["mm9_id", "tss_tile_num"],
                            right_on=["tss_id", "tss_tile_num"], suffixes=("_human", "_mouse"))
data_elem.drop(["tss_id_human", "tss_id_mouse"], axis=1, inplace=True)
print(len(data))
data_elem.head()


# In[33]:


data_elem["gc_human"] = data_elem.apply(calculate_gc, col="element_human", axis=1)
data_elem["gc_mouse"] = data_elem.apply(calculate_gc, col="element_mouse", axis=1)
data_elem["cpg_human"] = data_elem.apply(calculate_cpg, col="element_human", axis=1)
data_elem["cpg_mouse"] = data_elem.apply(calculate_cpg, col="element_mouse", axis=1)
data_elem.sample(5)


# In[34]:


data_elem.columns


# In[35]:


data_human = data_elem[["hg19_id", "tss_tile_num", "logFC_trans_human", "gc_human", "cpg_human", "HUES64_padj_hg19", "trans_status_one"]]
data_mouse = data_elem[["mm9_id", "tss_tile_num", "logFC_trans_mouse", "gc_mouse", "cpg_mouse", "mESC_padj_mm9", "trans_status_one"]]
data_human.columns = ["tss_id", "tss_tile_num", "logFC_trans", "gc", "cpg", "padj", "trans_status"]
data_mouse.columns = ["tss_id", "tss_tile_num", "logFC_trans", "gc", "cpg", "padj", "trans_status"]
data_indiv = data_human.append(data_mouse).drop_duplicates()
print(len(data_indiv))
data_indiv.head()


# ## 3. build reduced model

# In[36]:


scaled_features = StandardScaler().fit_transform(data_indiv[["logFC_trans", "gc", "cpg"]])
data_norm = pd.DataFrame(scaled_features, index=data_indiv.index, columns=["logFC_trans", "gc", "cpg"])
data_norm["padj"] = data_indiv["padj"]
data_norm["tss_id"] = data_indiv["tss_id"]
data_norm["tss_tile_num"] = data_indiv["tss_tile_num"]
data_norm["trans_status"] = data_indiv["trans_status"]
data_norm.head()


# In[37]:


data_filt = data_norm[data_norm["padj"] < QUANT_ALPHA].drop_duplicates()
print(len(data_filt))
data_filt.head()


# In[38]:


mod = smf.ols(formula='logFC_trans ~ gc + cpg', 
              data=data_filt).fit()


# In[39]:


mod.summary()


# In[40]:


res = mod.resid

fig, ax = plt.subplots(figsize=(2.2, 2.2), ncols=1, nrows=1)
sm.qqplot(res, line='s', ax=ax)
ax.set_title("Normal QQ: trans effects model")
fig.savefig("avg_activ_qq.pdf", dpi="figure", bbox_inches="tight")


# In[41]:


reduced_llf = mod.llf
reduced_llf


# In[42]:


reduced_rsq = mod.rsquared
reduced_rsq


# ## 4. add motifs to model

# In[43]:


data_filt["tss_index"] = data_filt["tss_id"] + "__" + data_filt["tss_tile_num"]


# In[44]:


human_motifs["hg19_index"] = human_motifs["hg19_id"] + "__" + human_motifs["tss_tile_num"]
mouse_motifs["mm9_index"] = mouse_motifs["mm9_id"] + "__" + mouse_motifs["tss_tile_num"]


# In[45]:


uniq_motifs = list(set(list(human_motifs["#pattern name"].unique()) + list(mouse_motifs["#pattern name"].unique())))
len(uniq_motifs)


# In[46]:


def tss_motif(row):
    if row.human_motif:
        return True
    elif row.mouse_motif:
        return True
    else:
        return False


# In[47]:


motif_results = {}

for i, motif_id in enumerate(uniq_motifs):
    tmp = data_filt.copy()
    
    # determine whether motif is in human or mouse sequence
    human_motifs_sub = human_motifs[human_motifs["#pattern name"] == motif_id]["hg19_index"].unique()
    mouse_motifs_sub = mouse_motifs[mouse_motifs["#pattern name"] == motif_id]["mm9_index"].unique()
    tmp["human_motif"] = tmp["tss_index"].isin(human_motifs_sub)
    tmp["mouse_motif"] = tmp["tss_index"].isin(mouse_motifs_sub)
    tmp["tss_motif"] = tmp.apply(tss_motif, axis=1)
    n_w_motif = tmp["tss_motif"].sum()
    
    # make full model
    full_mod = smf.ols(formula='logFC_trans ~ gc + cpg + tss_motif', 
                       data=tmp).fit()
    full_llf = full_mod.llf
    full_rsq = full_mod.rsquared
    
#     # perform likelihood ratio test
#     lr, p = lrtest(reduced_llf, full_llf)
    
    # calculate additional variance explained
    rsq = full_rsq - reduced_rsq
    
    # record beta
    beta = list(full_mod.params)[1]
    
    # beta p
    beta_p = list(full_mod.pvalues)[1]
    
    print("(#%s) %s: n w/ motif: %s ... p: %s, rsquared: %s" % (i+1, motif_id, len(tmp), beta_p, rsq))
    motif_results[motif_id] = {"rsq": rsq, "beta": beta, "beta_p": beta_p, "n_w_motif": n_w_motif}


# In[48]:


motif_results = pd.DataFrame.from_dict(motif_results, orient="index").reset_index()
motif_results = motif_results[motif_results["n_w_motif"] >= 10]
print(len(motif_results))
motif_results.head()


# In[49]:


motif_results["padj"] = multicomp.multipletests(motif_results["beta_p"], method="fdr_bh")[1]
len(motif_results[motif_results["padj"] < 0.05])


# In[50]:


motif_results.sort_values(by="padj").head(10)


# ## 5. join w/ TF info

# In[51]:


motif_results_mrg = motif_results.merge(sig_motifs, on="index", suffixes=("_trans", "_activ"))
motif_results_mrg.sort_values(by="padj_trans").head()


# In[52]:


sig_results = motif_results_mrg[(motif_results_mrg["padj_trans"] < 0.05)]
sig_results["abs_beta"] = np.abs(sig_results["beta_trans"])
sig_results = sig_results.sort_values(by="abs_beta", ascending=False)


# In[53]:


pal = {"repressing": sns.color_palette("pastel")[3], "activating": sns.color_palette("pastel")[0]}


# In[54]:


full_pal = {}
for i, row in sig_results.iterrows():
    full_pal[row["HGNC symbol"]] = pal[row["activ_or_repr"]]


# In[55]:


sig_results_sub = sig_results.head(50)


# In[56]:


fig = plt.figure(figsize=(4.5, 8))

ax1 = plt.subplot2grid((1, 7), (0, 0), colspan=3)
ax2 = plt.subplot2grid((1, 7), (0, 3), colspan=3)
ax3 = plt.subplot2grid((1, 7), (0, 6), colspan=1)

yvals = []
symbs = []
c = 0
for i, row in sig_results_sub.iterrows():
    symb = row["HGNC symbol"]
    if symb not in symbs:
        yvals.append(c)
        symbs.append(symb)
        c += 1
    else:
        yvals.append(c)

sig_results_sub["yval"] = yvals
sns.barplot(y="HGNC symbol", x="beta_trans", data=sig_results_sub, palette=full_pal, ax=ax1)
ax1.set_ylabel("")
ax1.set_xlabel("effect size of motif disruption")

sns.barplot(y="HGNC symbol", x="rsq_activ", data=sig_results_sub, palette=full_pal, ax=ax2)
ax2.set_ylabel("")
ax2.tick_params(left=False, labelleft=False)
ax2.set_xlabel("additional variance explained")

melt = pd.melt(sig_results_sub, id_vars=["HGNC symbol", "yval"], value_vars=["no_CAGE_enr", "eRNA_enr",
                                                                                 "lncRNA_enr", "mRNA_enr"])
ax3.plot(melt["value"], melt["yval"], 'o', color="black")
ax3.set_xlim((-0.5, 3.5))
ax3.set_ylim((np.max(yvals)-0.5, -0.5))
ax3.tick_params(labelleft=False, labelbottom=False, bottom=False, left=False, top=True, labeltop=True)
ax3.xaxis.set_ticks([0, 1, 2, 3])
ax3.set_xticklabels(["no CAGE", "eRNA", "lncRNA", "mRNA"], rotation=60, ha="left", va="bottom")

plt.show()
fig.savefig("trans_motif_enrichment.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# ## 6. join with expression information

# In[57]:


orth_expr[orth_expr["gene_name_human"] == "POU5F1B"]


# In[58]:


orth_expr.head()


# In[59]:


trans_orth = motif_results_mrg.merge(orth_expr, left_on="HGNC symbol", right_on="gene_name_human")
len(trans_orth)


# In[60]:


trans_orth.head()


# In[61]:


# fisher's exact to see if trans are enriched in DE TFs
trans_ids = trans_orth[trans_orth["padj_trans"] < 0.05]["index"].unique()
no_trans_ids = trans_orth[trans_orth["padj_trans"] >= 0.05]["index"].unique()
DE_ids = trans_orth[trans_orth["sig"] == "sig"]["index"].unique()

trans_w_DE = len([x for x in trans_ids if x in DE_ids])
trans_wo_DE = len([x for x in trans_ids if x not in DE_ids])
no_trans_w_DE = len([x for x in no_trans_ids if x in DE_ids])
no_trans_wo_DE = len([x for x in no_trans_ids if x not in DE_ids])

# fisher's exact test
arr = np.zeros((2, 2))
arr[0, 0] = trans_w_DE
arr[0, 1] = trans_wo_DE
arr[1, 0] = no_trans_w_DE
arr[1, 1] = no_trans_wo_DE

odds, p = stats.fisher_exact(arr)
print(odds)
print(p)


# In[62]:


trans_orth_sig = trans_orth[trans_orth["padj_trans"] < 0.05]
trans_orth_sig["abs_beta"] = np.abs(trans_orth_sig["beta_trans"])
trans_orth_sig = trans_orth_sig.sort_values(by="abs_beta", ascending=False)
trans_orth_sig["corrected_l2fc"] = -trans_orth_sig["log2FoldChange"]
len(trans_orth_sig)


# In[63]:


trans_orth_sub = trans_orth_sig.head(50)


# In[64]:


fig = plt.figure(figsize=(4.5, 9))

ax1 = plt.subplot2grid((1, 7), (0, 0), colspan=3)
ax2 = plt.subplot2grid((1, 7), (0, 3), colspan=3)
ax3 = plt.subplot2grid((1, 7), (0, 6), colspan=1)

yvals = []
symbs = []
c = 0
for i, row in trans_orth_sub.iterrows():
    symb = row["HGNC symbol"]
    if symb not in symbs:
        yvals.append(c)
        symbs.append(symb)
        c += 1
    else:
        yvals.append(c)

trans_orth_sub["yval"] = yvals
sns.barplot(y="HGNC symbol", x="beta_trans", data=trans_orth_sub, palette=full_pal, ax=ax1)
ax1.set_ylabel("")
ax1.set_xlabel("odds ratio")

sns.barplot(y="HGNC symbol", x="corrected_l2fc", data=trans_orth_sub, palette=full_pal, ax=ax2)
ax2.set_ylabel("")
ax2.tick_params(left=False, labelleft=False)
ax2.set_xlabel("log2(mESC/hESC)")

melt = pd.melt(trans_orth_sub, id_vars=["HGNC symbol", "yval"], value_vars=["no_CAGE_enr", "eRNA_enr",
                                                                                 "lncRNA_enr", "mRNA_enr"])
ax3.plot(melt["value"], melt["yval"], 'o', color="black")
ax3.set_xlim((-0.5, 3.5))
ax3.set_ylim((np.max(yvals)-0.5, -0.5))
ax3.tick_params(labelleft=False, labelbottom=False, bottom=False, left=False, top=True, labeltop=True)
ax3.xaxis.set_ticks([0, 1, 2, 3])
ax3.set_xticklabels(["no CAGE", "eRNA", "lncRNA", "mRNA"], rotation=60, ha="left", va="bottom")

plt.show()
fig.savefig("trans_motif_enrichment.with_expr.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# In[65]:


# filter to those where direction matches
trans_orth_sig["direction_match"] = trans_orth_sig.apply(direction_match, axis=1)
trans_orth_sig.direction_match.value_counts()


# In[66]:


trans_orth_match = trans_orth_sig[trans_orth_sig["direction_match"] == "match"]


# In[67]:


fig = plt.figure(figsize=(4.5, 10))

ax1 = plt.subplot2grid((1, 7), (0, 0), colspan=3)
ax2 = plt.subplot2grid((1, 7), (0, 3), colspan=3)
ax3 = plt.subplot2grid((1, 7), (0, 6), colspan=1)

yvals = []
symbs = []
c = 0
for i, row in trans_orth_match.iterrows():
    symb = row["HGNC symbol"]
    if symb not in symbs:
        yvals.append(c)
        symbs.append(symb)
        c += 1
    else:
        yvals.append(c)

trans_orth_match["yval"] = yvals
sns.barplot(y="HGNC symbol", x="beta_trans", data=trans_orth_match, palette=full_pal, ax=ax1)
ax1.set_ylabel("")
ax1.set_xlabel("odds ratio")

sns.barplot(y="HGNC symbol", x="corrected_l2fc", data=trans_orth_match, palette=full_pal, ax=ax2)
ax2.set_ylabel("")
ax2.tick_params(left=False, labelleft=False)
ax2.set_xlabel("log2(mESC/hESC)")

melt = pd.melt(trans_orth_match, id_vars=["HGNC symbol", "yval"], value_vars=["no_CAGE_enr", "eRNA_enr",
                                                                                 "lncRNA_enr", "mRNA_enr"])
ax3.plot(melt["value"], melt["yval"], 'o', color="black")
ax3.set_xlim((-0.5, 3.5))
ax3.set_ylim((np.max(yvals)-0.5, -0.5))
ax3.tick_params(labelleft=False, labelbottom=False, bottom=False, left=False, top=True, labeltop=True)
ax3.xaxis.set_ticks([0, 1, 2, 3])
ax3.set_xticklabels(["no CAGE", "eRNA", "lncRNA", "mRNA"], rotation=60, ha="left", va="bottom")

plt.show()
fig.savefig("trans_motif_enrichment.with_expr.match_only.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# ## 7. join w/ % similarity information

# In[68]:


orth_sub = orth[["Gene name", "Mouse gene name", "%id. target Mouse gene identical to query gene"]]
orth_sub.columns = ["human_gene_name", "mouse_gene_name", "perc_similarity"]


# In[69]:


trans_orth = trans_orth.merge(orth_sub, left_on="HGNC symbol", right_on="human_gene_name").drop_duplicates()
print(len(trans_orth))
trans_orth.sample(5)


# In[70]:


trans_orth["corrected_l2fc"] = -trans_orth_sub["log2FoldChange"]
trans_orth["abs_l2fc"] = np.abs(trans_orth["log2FoldChange"])
trans_orth["sig_status"] = trans_orth.apply(sig_status, axis=1)
trans_orth.head()


# In[71]:


trans_orth.sig_status.value_counts()


# In[72]:


order = ["not sig", "sig"]
palette = {"not sig": "gray", "sig": sns.color_palette("Set2")[2]}


# In[73]:


fig = plt.figure(figsize=(1, 1.75))
ax = sns.boxplot(data=trans_orth, x="sig_status", y="abs_l2fc", palette=palette, order=order,
                 flierprops = dict(marker='o', markersize=5))
mimic_r_boxplot(ax)

ax.set_xticklabels(order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("| log2(mESC/hESC) |")

for i, label in enumerate(order):
    n = len(trans_orth[trans_orth["sig_status"] == label])
    ax.annotate(str(n), xy=(i, -0.6), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='top', 
                color=palette[label], size=fontsize)

ax.set_ylim((-2, 13.2))

# calc p-vals b/w dists
dist1 = np.asarray(trans_orth[trans_orth["sig_status"] == "sig"]["abs_l2fc"])
dist2 = np.asarray(trans_orth[trans_orth["sig_status"] != "sig"]["abs_l2fc"])

dist1 = dist1[~np.isnan(dist1)]
dist2 = dist2[~np.isnan(dist2)]

u, pval = stats.mannwhitneyu(dist1, dist2, alternative="two-sided", use_continuity=False)
print(pval)

annotate_pval(ax, 0.2, 0.8, 5, 0, 4.5, pval, fontsize)

plt.show()
fig.savefig("trans_v_l2fc_boxplot.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# In[74]:


fig = plt.figure(figsize=(1, 1.75))
ax = sns.boxplot(data=trans_orth, x="sig_status", y="perc_similarity", palette=palette, order=order,
                 flierprops = dict(marker='o', markersize=5))
mimic_r_boxplot(ax)

ax.set_xticklabels(order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("% sequence similarity")

for i, label in enumerate(order):
    n = len(trans_orth[trans_orth["sig_status"] == label])
    ax.annotate(str(n), xy=(i, 10), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='top', 
                color=palette[label], size=fontsize)

ax.set_ylim((0, 115))

# calc p-vals b/w dists
dist1 = np.asarray(trans_orth[trans_orth["sig_status"] == "sig"]["perc_similarity"])
dist2 = np.asarray(trans_orth[trans_orth["sig_status"] != "sig"]["perc_similarity"])

dist1 = dist1[~np.isnan(dist1)]
dist2 = dist2[~np.isnan(dist2)]

u, pval = stats.mannwhitneyu(dist1, dist2, alternative="two-sided", use_continuity=False)
print(pval)

annotate_pval(ax, 0.2, 0.8, 100, 0, 100, pval, fontsize)

plt.show()
fig.savefig("trans_v_similarity_boxplot.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# ## 8. look at strength of motifs associated w/ trans effects

# In[75]:


data_filt.head()


# In[76]:


human_motifs.columns


# In[77]:


trans_ids = data_filt[data_filt["trans_status"] == "significant trans effect"]["tss_index"].unique()
len(trans_ids)


# In[78]:


filt_ids = list(data_filt["tss_index"].unique())
len(filt_ids)


# In[79]:


len(sig_results)


# In[80]:


uniq_motifs_sig = list(sig_results["index"].unique())


# In[81]:


strength_results = pd.DataFrame()

for i, motif_id in enumerate(uniq_motifs_sig):
    
    # subset motif dfs to occurrences
    human_motifs_sub = human_motifs[human_motifs["#pattern name"] == motif_id]
    mouse_motifs_sub = mouse_motifs[mouse_motifs["#pattern name"] == motif_id]
    
    # subset motif dfs to data filt ids
    human_motifs_sub = human_motifs_sub[human_motifs_sub["hg19_index"].isin(filt_ids)]
    mouse_motifs_sub = mouse_motifs_sub[mouse_motifs_sub["mm9_index"].isin(filt_ids)]
    
    # subset into trans vs. no trans effects
    human_motifs_sub["trans_status"] = human_motifs_sub["hg19_index"].isin(trans_ids)
    mouse_motifs_sub["trans_status"] = mouse_motifs_sub["mm9_index"].isin(trans_ids)
    
    # get data we need
    human_sub = human_motifs_sub[["#pattern name", "score", "trans_status"]]
    mouse_sub = mouse_motifs_sub[["#pattern name", "score", "trans_status"]]
    human_sub.columns = ["index", "motif_score", "trans_status"]
    mouse_sub.columns = ["index", "motif_score", "trans_status"]
    
    # append to df
    strength_results = strength_results.append(human_sub).append(mouse_sub)
    print("(#%s) %s" % (i+1, motif_id))


# In[82]:


print(len(strength_results))
strength_results_mrg = strength_results.merge(sig_motifs, on="index")
strength_results_mrg.head()


# In[83]:


uniq_motifs_sig1 = uniq_motifs_sig[0:41]
uniq_motifs_sig2 = uniq_motifs_sig[41:82]
uniq_motifs_sig3 = uniq_motifs_sig[82:123]
uniq_motifs_sig4 = uniq_motifs_sig[123:]
print(len(uniq_motifs_sig1))
print(len(uniq_motifs_sig2))
print(len(uniq_motifs_sig3))
print(len(uniq_motifs_sig4))


# In[84]:


pal = {False: "gray", True: sns.color_palette("Set2")[2]}
hue_order = [False, True]


# In[85]:


fig = plt.figure(figsize=(15, 2))

sub = strength_results_mrg[strength_results_mrg["index"].isin(uniq_motifs_sig1)]
ax = sns.boxplot(data=sub, x="index", y="motif_score", hue="trans_status", hue_order=hue_order, palette=pal,
                 flierprops = dict(marker='o', markersize=5))
mimic_r_boxplot(ax)

ax.set_xlabel("")
ax.set_xticklabels(sub["index"], rotation=50, ha="right", va="top")
ax.set_ylabel("motif score")

