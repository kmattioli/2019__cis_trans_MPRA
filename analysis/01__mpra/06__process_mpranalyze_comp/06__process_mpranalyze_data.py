
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from scipy.stats import spearmanr

# import utils
sys.path.append("../../../utils")
from plotting_utils import *
from classify_utils import *

get_ipython().run_line_magic('matplotlib', 'inline')
# %config InlineBackend.figure_format = 'svg'
# mpl.rcParams['figure.autolayout'] = False


# In[2]:


sns.set(**PAPER_PRESET)
fontsize = PAPER_FONTSIZE


# In[3]:


np.random.seed(2019)


# In[4]:


QUANT_ALPHA = 0.05


# ## functions

# ## variables

# In[5]:


data_dir = "../../../data/02__mpra/02__activs"
alpha_f = "%s/alpha_per_elem.quantification.txt" % data_dir
human_vals_f = "%s/human_TSS_vals.both_tiles.txt" % data_dir
mouse_vals_f= "%s/mouse_TSS_vals.both_tiles.txt" % data_dir


# In[6]:


native_f = "%s/native_results.txt" % data_dir
HUES64_cis_f = "%s/HUES64_cis_results.txt" % data_dir
mESC_cis_f = "%s/mESC_cis_results.txt" % data_dir
human_trans_f = "%s/human_trans_results.txt" % data_dir
mouse_trans_f = "%s/mouse_trans_results.txt" % data_dir
cis_trans_int_f = "%s/cis_trans_interaction_results.txt" % data_dir


# In[7]:


tss_map_f = "../../../data/01__design/01__mpra_list/mpra_tss.with_ids.RECLASSIFIED_WITH_MAX.txt"


# In[8]:


#align_f = "../../../misc/00__tss_metadata/tss_map.seq_alignment.txt"


# In[9]:


#cage_data_f = "../../../misc/00__tss_metadata/All_TSS_and_enh.CAGE_grouped_exp.tissue_sp.txt.gz"


# ## 1. import data

# In[10]:


alpha = pd.read_table(alpha_f, sep="\t").reset_index()
alpha.head()


# In[11]:


len(alpha)


# In[12]:


human_vals = pd.read_table(human_vals_f)
mouse_vals = pd.read_table(mouse_vals_f)
human_vals.head()


# In[13]:


native = pd.read_table(native_f).reset_index()
native.columns = ["index", "stat_native", "pval_native", "fdr_native", "df.test_native", "df.dna_native", 
                  "df.rna.full_native", "df.rna.red_native", "logFC_native"]
native["index"] = native.apply(fix_ctrl_id, axis=1)
native.sample(5)


# In[14]:


HUES64_cis = pd.read_table(HUES64_cis_f).reset_index()
HUES64_cis.columns = ["index", "stat_cis_HUES64", "pval_cis_HUES64", "fdr_cis_HUES64", "df.test_cis_HUES64", 
                      "df.dna_cis_HUES64", "df.rna.full_cis_HUES64", "df.rna.red_cis_HUES64", "logFC_cis_HUES64"]
HUES64_cis["index"] = HUES64_cis.apply(fix_ctrl_id, axis=1)

mESC_cis = pd.read_table(mESC_cis_f).reset_index()
mESC_cis.columns = ["index", "stat_cis_mESC", "pval_cis_mESC", "fdr_cis_mESC", "df.test_cis_mESC", 
                    "df.dna_cis_mESC", "df.rna.full_cis_mESC", "df.rna.red_cis_mESC", "logFC_cis_mESC"]
mESC_cis["index"] = mESC_cis.apply(fix_ctrl_id, axis=1)


# In[15]:


human_trans = pd.read_table(human_trans_f).reset_index()
human_trans.columns = ["index", "stat_trans_human", "pval_trans_human", "fdr_trans_human", "df.test_trans_human", 
                       "df.dna_trans_human", "df.rna.full_trans_human", "df.rna.red_trans_human", "logFC_trans_human"]
human_trans["index"] = human_trans.apply(fix_ctrl_id, axis=1)

mouse_trans = pd.read_table(mouse_trans_f).reset_index()
mouse_trans.columns = ["index", "stat_trans_mouse", "pval_trans_mouse", "fdr_trans_mouse", "df.test_trans_mouse", 
                       "df.dna_trans_mouse", "df.rna.full_trans_mouse", "df.rna.red_trans_mouse", "logFC_trans_mouse"]
mouse_trans["index"] = mouse_trans.apply(fix_ctrl_id, axis=1)


# In[16]:


cis_trans_int = pd.read_table(cis_trans_int_f).reset_index()
cis_trans_int.columns = ["index", "stat_int", "pval_int", "fdr_int", "df.test_int", 
                         "df.dna_int", "df.rna.full_int", "df.rna.red_int", "logFC_int"]
cis_trans_int["index"] = cis_trans_int.apply(fix_ctrl_id, axis=1)


# In[17]:


tss_map = pd.read_table(tss_map_f)
tss_map.head()


# In[18]:


tss_map.minimal_biotype_hg19.value_counts()


# In[19]:


# align = pd.read_table(align_f, sep=",", index_col=0)
# align.head()


# In[20]:


# cage_data = pd.read_table(cage_data_f, sep="\t")
# cage_data.drop(["tissue_sp_3", "n_expr"], axis=1, inplace=True)
# cage_data.columns = ["cage_id_hg19", "av_cage_exp_hg19", "cage_tissue_sp_hg19", "cage_tss_type_hg19"]
# cage_data.head()


# ## 2. add biotype switch to TSS map

# In[21]:


tss_map[tss_map["hg19_id"] == "h.14"]


# In[22]:


tss_map.minimal_biotype_hg19.value_counts()


# In[23]:


# tss_map["biotype_switch_clean"] = tss_map.apply(biotype_switch_clean, axis=1)
# tss_map.biotype_switch_clean.value_counts()


# In[24]:


len(tss_map)


# In[25]:


tss_map["biotype_switch_minimal"] = tss_map.apply(biotype_switch_minimal, axis=1)
tss_map.biotype_switch_minimal.value_counts()


# ## 3. merge alphas with TSS map

# split up into tile1 and tile2

# In[26]:


human_vals_sub = human_vals[["element", "tss_id", "tss_tile_num"]]
human_vals_sub.columns = ["hg19_element", "hg19_id", "hg19_tile_num"]

mouse_vals_sub = mouse_vals[["element", "tss_id", "tss_tile_num"]]
mouse_vals_sub.columns = ["mm9_element", "mm9_id", "mm9_tile_num"]
mouse_vals_sub.sample(5)


# In[27]:


human_vals_tile1 = human_vals_sub[human_vals_sub["hg19_tile_num"] == "tile1"].drop_duplicates()
human_vals_tile2 = human_vals_sub[human_vals_sub["hg19_tile_num"] == "tile2"].drop_duplicates()
mouse_vals_tile1 = mouse_vals_sub[mouse_vals_sub["mm9_tile_num"] == "tile1"].drop_duplicates()
mouse_vals_tile2 = mouse_vals_sub[mouse_vals_sub["mm9_tile_num"] == "tile2"].drop_duplicates()

print(len(human_vals_tile1))
print(len(human_vals_tile2))
print(len(mouse_vals_tile1))
print(len(mouse_vals_tile2))


# In[28]:


# both_tile_ids = tss_map[(tss_map["n_tiles_hg19"] >= 2) & (tss_map["n_tiles_mm9"] >= 2)]
both_tile_ids = tss_map[(~pd.isnull(tss_map["n_tiles_hg19"]) & ~(pd.isnull(tss_map["n_tiles_mm9"])))]
len(both_tile_ids)


# In[29]:


tile1_ids = both_tile_ids[(both_tile_ids["tile_match"] == "tile1:tile1") | 
                          (both_tile_ids["tile_match"] == "tile1:tile2")][["hg19_id", "mm9_id"]].drop_duplicates()
len(tile1_ids)


# In[30]:


tile2_ids = both_tile_ids[(both_tile_ids["tile_match"] == "tile2:tile2")][["hg19_id", "mm9_id"]].drop_duplicates()
len(tile2_ids)


# In[31]:


tss_map_tile1 = tile1_ids.merge(tss_map, on=["hg19_id", "mm9_id"], how="left")
tss_map_tile1 = tss_map_tile1.merge(human_vals_tile1, on="hg19_id").merge(mouse_vals_tile1, on="mm9_id")
print(len(tss_map_tile1))


# In[32]:


tss_map_tile2 = tile2_ids.merge(tss_map, on=["hg19_id", "mm9_id"], how="left")
tss_map_tile2 = tss_map_tile2.merge(human_vals_tile2, on="hg19_id").merge(mouse_vals_tile2, on="mm9_id")
print(len(tss_map_tile2))


# In[33]:


tss_map_tile1 = tss_map_tile1.merge(alpha, 
                                    left_on="hg19_element", 
                                    right_on="index").merge(alpha,
                                                            left_on="mm9_element",
                                                            right_on="index",
                                                            suffixes=("_hg19", "_mm9"))
tss_map_tile1["tss_tile_num"] = "tile1"
tss_map_tile1.head()


# In[34]:


tss_map_tile2 = tss_map_tile2.merge(alpha, 
                                    left_on="hg19_element", 
                                    right_on="index").merge(alpha,
                                                            left_on="mm9_element",
                                                            right_on="index",
                                                            suffixes=("_hg19", "_mm9"))
tss_map_tile2["tss_tile_num"] = "tile2"
tss_map_tile2.head()


# In[35]:


tss_map = tss_map_tile1.append(tss_map_tile2)
tss_map.biotype_hg19.value_counts()


# In[36]:


data = tss_map[["hg19_id", "chr_tss_hg19", "start_tss_hg19", "biotype_hg19", 
                "minimal_biotype_hg19", "cage_id_hg19", "name_peak_hg19", "stem_exp_hg19", "max_cage_hg19", 
                "mm9_id", "chr_tss_mm9", 
                "start_tss_mm9", "biotype_mm9", "minimal_biotype_mm9", "cage_id_mm9", 
                "name_peak_mm9", "stem_exp_mm9", "max_cage_mm9", "tss_tile_num", "orig_species", 
                "biotype_switch_minimal", "HUES64_hg19", "mESC_hg19", "HUES64_mm9", "mESC_mm9", "HUES64_padj_hg19", 
                "mESC_padj_hg19", "HUES64_padj_mm9", "mESC_padj_mm9"]].drop_duplicates()
data.sample(5)


# In[37]:


len(data)


# ## 4. find appropriate FDR cutoffs for each model
# choose FDR cutoff as the one that calls < 10% of controls as significant

# In[38]:


native_ctrls = native[native["index"].str.contains("CONTROL")]
cis_HUES64_ctrls = HUES64_cis[HUES64_cis["index"].str.contains("CONTROL")]
cis_mESC_ctrls = mESC_cis[mESC_cis["index"].str.contains("CONTROL")]
trans_human_ctrls = human_trans[human_trans["index"].str.contains("CONTROL")]
trans_mouse_ctrls = mouse_trans[mouse_trans["index"].str.contains("CONTROL")]
cis_trans_int_ctrls = cis_trans_int[cis_trans_int["index"].str.contains("CONTROL")]


# In[39]:


print(len(native_ctrls))
print(len(cis_HUES64_ctrls))
print(len(cis_mESC_ctrls))
print(len(trans_human_ctrls))
print(len(trans_mouse_ctrls))
print(len(cis_trans_int_ctrls))


# make plots to show different #s of controls called as "significant" at alpha < 0.05 in each model

# In[40]:


n_sig_models = {}
for model, df, fdr in zip(["native effects", "HUES64 cis effects", "mESC cis effects", "human trans effects",
                           "mouse trans effects", "cis/trans interaction effects"],
                          [native_ctrls, cis_HUES64_ctrls, cis_mESC_ctrls, trans_human_ctrls, trans_mouse_ctrls,
                           cis_trans_int_ctrls],
                          ["fdr_native", "fdr_cis_HUES64", "fdr_cis_mESC", "fdr_trans_human", "fdr_trans_mouse",
                           "fdr_int"]):
    n_sig = len(df[df[fdr] < 0.05])
    n_sig_models[model] = [n_sig]
n_sig_models = pd.DataFrame.from_dict(n_sig_models, orient="index").reset_index()
n_sig_models.columns = ["model", "n_sig"]
n_sig_models


# In[41]:


order = ["native effects", "HUES64 cis effects", "mESC cis effects", "human trans effects", "mouse trans effects"]


# In[42]:


fig, ax = plt.subplots(figsize=(2.2, 1.5), nrows=1, ncols=1)

sns.barplot(data=n_sig_models, x="model", y="n_sig", color="darkgray", ax=ax, order=order)
ax.set_xlabel("")
ax.set_ylabel("number of controls significant\nat MPRAnalyze q-value < 0.05")
_ = ax.set_xticklabels(n_sig_models["model"], rotation=50, ha="right", va="top")

ax.set_title("MPRAnalyze q-value cut-off")
ax.set_ylim((0, 200))
fig.savefig("perc_ctrls_sig.05.pdf", dpi="figure", bbox_inches="tight")


# now always cut off at the 10th percentile of FDRs of controls within a given model

# In[43]:


NATIVE_THRESH = np.percentile(native_ctrls["fdr_native"], 5)
NATIVE_THRESH


# In[44]:


CIS_HUES64_THRESH = np.percentile(cis_HUES64_ctrls["fdr_cis_HUES64"], 5)
CIS_HUES64_THRESH


# In[45]:


CIS_MESC_THRESH = np.percentile(cis_mESC_ctrls["fdr_cis_mESC"], 5)
CIS_MESC_THRESH


# In[46]:


TRANS_HUMAN_THRESH = np.percentile(trans_human_ctrls["fdr_trans_human"], 5)
TRANS_HUMAN_THRESH


# In[47]:


TRANS_MOUSE_THRESH = np.percentile(trans_mouse_ctrls["fdr_trans_mouse"], 5)
TRANS_MOUSE_THRESH


# In[48]:


INT_THRESH = np.percentile(cis_trans_int_ctrls["fdr_int"], 5)
INT_THRESH


# now re-plot with threshold

# In[49]:


n_sig_models = {}
for model, df, fdr, thresh in zip(["native effects", "HUES64 cis effects", "mESC cis effects", "human trans effects",
                                   "mouse trans effects", "cis/trans interaction effects"],
                                  [native_ctrls, cis_HUES64_ctrls, cis_mESC_ctrls, trans_human_ctrls, 
                                   trans_mouse_ctrls, cis_trans_int_ctrls],
                                  ["fdr_native", "fdr_cis_HUES64", "fdr_cis_mESC", "fdr_trans_human", 
                                   "fdr_trans_mouse", "fdr_int"],
                                  [NATIVE_THRESH, CIS_HUES64_THRESH, CIS_MESC_THRESH, TRANS_HUMAN_THRESH,
                                   TRANS_MOUSE_THRESH, 0.05]):
    n_sig = len(df[df[fdr] < thresh])
    n_sig_models[model] = [n_sig]
n_sig_models = pd.DataFrame.from_dict(n_sig_models, orient="index").reset_index()
n_sig_models.columns = ["model", "n_sig"]
n_sig_models


# In[50]:


fig, ax = plt.subplots(figsize=(2.2, 1.5), nrows=1, ncols=1)

sns.barplot(data=n_sig_models, x="model", y="n_sig", color="darkgray", ax=ax, order=order)
ax.set_xlabel("")
ax.set_ylabel("number of controls significant\nat empirical FDR < 0.1")
_ = ax.set_xticklabels(n_sig_models["model"], rotation=50, ha="right", va="top")

ax.set_title("empirical FDR cut-off")
ax.set_ylim((0, 200))
fig.savefig("perc_ctrls_sig.thresh.pdf", dpi="figure", bbox_inches="tight")


# ## 5. plot controls vs. TSSs for each model

# control boxplots

# In[51]:


order = ["control", "TSS"]
pal = {"control": "gray", "TSS": "black"}

models = ["native", "HUES64 cis", "mESC cis", "human trans", "mouse trans", "cis/trans interaction"]
dfs = [native, HUES64_cis, mESC_cis, human_trans, mouse_trans, cis_trans_int]
logFCs = ["logFC_native", "logFC_cis_HUES64", "logFC_cis_mESC", "logFC_trans_human", "logFC_trans_mouse", "logFC_int"]
labels = ["native", "cis_HUES64", "cis_mESC", "trans_human", "trans_mouse", "cis_trans_int"]
ylims = [(-0.8, 6), (-0.8, 6), (-0.8, 6), (-0.8, 2.2), (-0.8, 2.2), (-0.8, 2.5)]
pxs = [(1.5, 1.4), (1.5, 1.4), (1.5, 1.4), (0.7, 0.7), (0.7, 0.7), (0.6, 0.5)]


for model, df, logFC, label, ylim, px in zip(models, dfs, logFCs, labels, ylims, pxs):
    df["is_ctrl"] = df.apply(is_ctrl, axis=1)
    df["abs_logFC"] = np.abs(df[logFC])
    
    fig = plt.figure(figsize=(1, 1.75))
    ax = sns.boxplot(data=df, x="is_ctrl", y="abs_logFC", flierprops = dict(marker='o', markersize=5), 
                     order=order, palette=pal)
    mimic_r_boxplot(ax)

    ax.set_xticklabels(["negative\ncontrols", "seqs"], rotation=50, ha='right', va='top')
    ax.set_xlabel("")
    ax.set_ylabel(r'$\vert$ %s effect size $\vert$' % model)

    for i, l in enumerate(order):
        n = len(df[df["is_ctrl"] == l])
        color = pal[l]
        ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color=color, size=fontsize)

    ax.set_ylim(ylim)

    # calc p-vals b/w dists
    dist1 = np.asarray(df[df["is_ctrl"] == "control"]["abs_logFC"])
    dist2 = np.asarray(df[df["is_ctrl"] != "control"]["abs_logFC"])

    dist1 = dist1[~np.isnan(dist1)]
    dist2 = dist2[~np.isnan(dist2)]

    u, pval = stats.mannwhitneyu(dist1, dist2, alternative="two-sided", use_continuity=False)
    print(pval)

    annotate_pval(ax, 0.2, 0.8, px[0], 0, px[1], pval, fontsize)
    plt.show()
    fig.savefig("%s_ctrl_effectsize_boxplot.pdf" % label, dpi="figure", bbox_inches="tight")
    plt.close()


# volcano plots

# In[52]:


threshs = [NATIVE_THRESH, CIS_HUES64_THRESH, CIS_MESC_THRESH, TRANS_HUMAN_THRESH, TRANS_MOUSE_THRESH, 0.05]
fdrs = ["fdr_native", "fdr_cis_HUES64", "fdr_cis_mESC", "fdr_trans_human", "fdr_trans_mouse", "fdr_int"]

for model, df, logFC, fdr, label, thresh in zip(models, dfs, logFCs, fdrs, labels, threshs):
    df["is_ctrl"] = df.apply(is_ctrl, axis=1)
    
    neg_ctrls = df[df["is_ctrl"] == "control"]
    tss = df[df["is_ctrl"] != "control"]
    
    fig, ax = plt.subplots(figsize=(1.75, 1), nrows=1, ncols=1)

    ax.scatter(tss[logFC], -np.log10(tss[fdr]), s=10, alpha=0.75, 
               color="black", linewidths=0.5, edgecolors="white")
    ax.scatter(neg_ctrls[logFC], -np.log10(neg_ctrls[fdr]), s=8, alpha=0.5, 
               color="gray", linewidths=0.5, edgecolors="white")

    plt.xlabel("%s effect size" % model)
    if model == "HUES64 cis":
        plt.xlabel("hESC cis effect size")
    plt.ylabel("-log10(q-value)")
    ax.axhline(y=-np.log10(thresh), color="black", linestyle="dashed")

    plt.show()
    fig.savefig("%s_volcano.ctrls_highlighted.pdf" % label, dpi="figure", bbox_inches="tight")
    plt.close()


# ## 6. split result file indeces

# In[53]:


native["hg19_id"] = native["index"].str.split("__", expand=True)[0]
native["biotype_hg19"] = native["index"].str.split("__", expand=True)[1]
native["mm9_id"] = native["index"].str.split("__", expand=True)[2]
native["biotype_mm9"] = native["index"].str.split("__", expand=True)[3]
native["tss_tile_num"] = native["index"].str.split("__", expand=True)[4]


# In[54]:


HUES64_cis["hg19_id"] = HUES64_cis["index"].str.split("__", expand=True)[0]
HUES64_cis["biotype_hg19"] = HUES64_cis["index"].str.split("__", expand=True)[1]
HUES64_cis["mm9_id"] = HUES64_cis["index"].str.split("__", expand=True)[2]
HUES64_cis["biotype_mm9"] = HUES64_cis["index"].str.split("__", expand=True)[3]
HUES64_cis["tss_tile_num"] = HUES64_cis["index"].str.split("__", expand=True)[4]


# In[55]:


mESC_cis["hg19_id"] = mESC_cis["index"].str.split("__", expand=True)[0]
mESC_cis["biotype_hg19"] = mESC_cis["index"].str.split("__", expand=True)[1]
mESC_cis["mm9_id"] = mESC_cis["index"].str.split("__", expand=True)[2]
mESC_cis["biotype_mm9"] = mESC_cis["index"].str.split("__", expand=True)[3]
mESC_cis["tss_tile_num"] = mESC_cis["index"].str.split("__", expand=True)[4]


# In[56]:


human_trans["hg19_id"] = human_trans["index"].str.split("__", expand=True)[0]
human_trans["biotype_hg19"] = human_trans["index"].str.split("__", expand=True)[1]
human_trans["mm9_id"] = human_trans["index"].str.split("__", expand=True)[2]
human_trans["biotype_mm9"] = human_trans["index"].str.split("__", expand=True)[3]
human_trans["tss_tile_num"] = human_trans["index"].str.split("__", expand=True)[4]


# In[57]:


mouse_trans["hg19_id"] = mouse_trans["index"].str.split("__", expand=True)[0]
mouse_trans["biotype_hg19"] = mouse_trans["index"].str.split("__", expand=True)[1]
mouse_trans["mm9_id"] = mouse_trans["index"].str.split("__", expand=True)[2]
mouse_trans["biotype_mm9"] = mouse_trans["index"].str.split("__", expand=True)[3]
mouse_trans["tss_tile_num"] = mouse_trans["index"].str.split("__", expand=True)[4]


# In[58]:


cis_trans_int["hg19_id"] = cis_trans_int["index"].str.split("__", expand=True)[0]
cis_trans_int["biotype_hg19"] = cis_trans_int["index"].str.split("__", expand=True)[1]
cis_trans_int["mm9_id"] = cis_trans_int["index"].str.split("__", expand=True)[2]
cis_trans_int["biotype_mm9"] = cis_trans_int["index"].str.split("__", expand=True)[3]
cis_trans_int["tss_tile_num"] = cis_trans_int["index"].str.split("__", expand=True)[4]


# ## 7. merge result files w/ activity data

# In[59]:


len(data)


# In[60]:


tmp = data.merge(native[["hg19_id", "mm9_id", "tss_tile_num", "logFC_native", "fdr_native"]], 
                 on=["hg19_id", "mm9_id", "tss_tile_num"], how="right")
len(tmp)


# In[61]:


tmp = tmp.merge(HUES64_cis[["hg19_id", "mm9_id", "tss_tile_num", "logFC_cis_HUES64", "fdr_cis_HUES64"]], 
                on=["hg19_id", "mm9_id", "tss_tile_num"], how="right")
len(tmp)


# In[62]:


tmp = tmp.merge(mESC_cis[["hg19_id", "mm9_id", "tss_tile_num", "logFC_cis_mESC", "fdr_cis_mESC"]], 
                on=["hg19_id", "mm9_id", "tss_tile_num"], how="right")
len(tmp)


# In[63]:


tmp = tmp.merge(human_trans[["hg19_id", "mm9_id", "tss_tile_num", "logFC_trans_human", "fdr_trans_human"]], 
                on=["hg19_id", "mm9_id", "tss_tile_num"], how="right")
len(tmp)


# In[64]:


tmp = tmp.merge(mouse_trans[["hg19_id", "mm9_id", "tss_tile_num", "logFC_trans_mouse", "fdr_trans_mouse"]], 
                on=["hg19_id", "mm9_id", "tss_tile_num"], how="right")
len(tmp)


# In[65]:


tmp = tmp.merge(cis_trans_int[["hg19_id", "mm9_id", "tss_tile_num", "logFC_int", "fdr_int"]], 
                on=["hg19_id", "mm9_id", "tss_tile_num"], how="right")
print(len(tmp))
tmp.sample(5)


# In[66]:


data = tmp.copy()


# In[67]:


data.columns


# ## 8. classify comparison effects as sig vs. not sig

# In[68]:


data["native_status"] = data.apply(comp_status, fdr_col="fdr_native", thresh=NATIVE_THRESH, txt="native", axis=1)
data["cis_HUES64_status"] = data.apply(comp_status, fdr_col="fdr_cis_HUES64", thresh=CIS_HUES64_THRESH, 
                                       txt="cis", axis=1)
data["cis_mESC_status"] = data.apply(comp_status, fdr_col="fdr_cis_mESC", thresh=CIS_MESC_THRESH, txt="cis", axis=1)
data["trans_human_status"] = data.apply(comp_status, fdr_col="fdr_trans_human", thresh=TRANS_HUMAN_THRESH,
                                        txt="trans", axis=1)
data["trans_mouse_status"] = data.apply(comp_status, fdr_col="fdr_trans_mouse", thresh=TRANS_MOUSE_THRESH,
                                        txt="trans", axis=1)
data["cis_trans_int_status"] = data.apply(comp_status, fdr_col="fdr_int", thresh=0.05, txt="cis/trans int.", axis=1)


# In[69]:


# data["native_status"] = data.apply(comp_status, fdr_col="fdr_native", thresh=0.05, txt="native", axis=1)
# data["cis_HUES64_status"] = data.apply(comp_status, fdr_col="fdr_cis_HUES64", thresh=0.05, 
#                                        txt="cis", axis=1)
# data["cis_mESC_status"] = data.apply(comp_status, fdr_col="fdr_cis_mESC", thresh=0.05, txt="cis", axis=1)
# data["trans_human_status"] = data.apply(comp_status, fdr_col="fdr_trans_human", thresh=0.05,
#                                         txt="trans", axis=1)
# data["trans_mouse_status"] = data.apply(comp_status, fdr_col="fdr_trans_mouse", thresh=0.05,
#                                         txt="trans", axis=1)
# data["cis_trans_int_status"] = data.apply(comp_status, fdr_col="fdr_int", thresh=0.05, txt="cis/trans int.", axis=1)


# In[70]:


data.sample(5)


# ## 9. compare l2fcs for elements called significant for each biotype & re-classify

# In[71]:


min_order = ["no CAGE activity", "eRNA", "lncRNA", "mRNA", "other"]
palette = sns.husl_palette(n_colors=len(min_order))


# In[72]:


l2fc_cols = ["logFC_native", "logFC_cis_HUES64", "logFC_cis_mESC", "logFC_trans_human", "logFC_trans_mouse", 
             "logFC_int"]
sig_cols = ["native_status", "cis_HUES64_status", "cis_mESC_status", "trans_human_status", "trans_mouse_status",
            "cis_trans_int_status"]
xlabels = ["native effect size", "cis effect size (HUES64)", "cis effect size (mESC)", "trans effect size (human)",
           "trans effect size (mouse)", "cis/trans interaction effect size"]
xs = [1, 0.9, 0.8, 0.4, 0.4, 0]
min_vals = []

for l2fc_col, sig_col, xlabel, x in zip(l2fc_cols, sig_cols, xlabels, xs):
    
    fig, ax = plt.subplots(figsize=(2, 1.5), nrows=1, ncols=1)
    filt = data[((data["HUES64_padj_hg19"] < QUANT_ALPHA) | (data["mESC_padj_mm9"] < QUANT_ALPHA))]
    ctrls = np.abs(data[data["hg19_id"].str.contains("ctrl.")][l2fc_col])
    
    sig_ctrls = data[(data["hg19_id"].str.contains("ctrl.")) & (data[sig_col].str.contains("significant"))][l2fc_col]
    min_val = np.min(np.abs(sig_ctrls))
    print(min_val)
    min_vals.append(min_val)
    
    sns.distplot(ctrls, hist=False, color="gray", label="neg. ctrls (n=%s)" % len(ctrls), ax=ax)
    
    sub = filt[~filt[sig_col].str.contains("no ")]
    for i, label in enumerate(min_order):
        
        vals = np.abs(sub[sub["minimal_biotype_hg19"] == label][l2fc_col])
        sns.distplot(vals, hist=False, color=palette[i], label="%s (n=%s)" % (label, len(vals)), ax=ax)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.axvline(x=x, linestyle="dashed", color="black")
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
    plt.show()
    fig.savefig("%s_effect_size_dist.pdf" % l2fc_col, dpi="figure", bbox_inches="tight")
    plt.close()


# In[73]:


data.native_status.value_counts()


# In[74]:


data["native_status"] = data.apply(comp_status_stringent, status_col="native_status", 
                                   l2fc_col="logFC_native", l2fc_thresh=min_vals[0], txt="native", axis=1)
data["cis_HUES64_status"] = data.apply(comp_status_stringent, status_col="cis_HUES64_status",  
                                       l2fc_col="logFC_cis_HUES64", l2fc_thresh=min_vals[1], txt="cis", axis=1)
data["cis_mESC_status"] = data.apply(comp_status_stringent, status_col="cis_mESC_status", 
                                     l2fc_col="logFC_cis_mESC", l2fc_thresh=min_vals[2], txt="cis", axis=1)
data["trans_human_status"] = data.apply(comp_status_stringent, status_col="trans_human_status", 
                                        l2fc_col="logFC_trans_human", l2fc_thresh=min_vals[3], txt="trans", axis=1)
data["trans_mouse_status"] = data.apply(comp_status_stringent, status_col="trans_mouse_status", 
                                        l2fc_col="logFC_trans_mouse", l2fc_thresh=min_vals[4], txt="trans", axis=1)
data["cis_trans_int_status"] = data.apply(comp_status_stringent, status_col="cis_trans_int_status", 
                                          l2fc_col="logFC_int", l2fc_thresh=0, txt="cis/trans int.", axis=1)


# In[75]:


data.native_status.value_counts()


# ## 9. classify effects as higher in mouse or human

# In[76]:


data["native_status_det"] = data.apply(comp_status_detail, status_col="native_status",
                                       logFC_col="logFC_native", txt="native", axis=1)
data["cis_HUES64_status_det"] = data.apply(comp_status_detail, status_col="cis_HUES64_status",
                                           logFC_col="logFC_cis_HUES64", txt="cis", axis=1)
data["cis_mESC_status_det"] = data.apply(comp_status_detail, status_col="cis_mESC_status",
                                         logFC_col="logFC_cis_mESC", txt="cis", axis=1)
data["trans_human_status_det"] = data.apply(comp_status_detail, status_col="trans_human_status",
                                            logFC_col="logFC_trans_human", txt="trans", axis=1)
data["trans_mouse_status_det"] = data.apply(comp_status_detail, status_col="trans_mouse_status",
                                            logFC_col="logFC_trans_mouse", txt="trans", axis=1)
data["cis_trans_int_status_det"] = data.apply(comp_status_detail, status_col="cis_trans_int_status", 
                                              logFC_col="logFC_int", txt="cis/trans int.", axis=1)


# In[77]:


data.sample(5)


# ## 10. classify cis & trans effects into one effect (since we measured in 2 contexts)

# In[78]:


data["cis_status_one"] = data.apply(comp_status_one, status_col1="cis_HUES64_status", 
                                    status_col2="cis_mESC_status", txt="cis", axis=1)
data["trans_status_one"] = data.apply(comp_status_one, status_col1="trans_human_status", 
                                      status_col2="trans_mouse_status", txt="trans", axis=1)


# In[79]:


data["cis_status_det_one"] = data.apply(comp_status_detail_one, status_col1="cis_HUES64_status", 
                                        status_col2="cis_mESC_status", logFC_col1="logFC_cis_HUES64", 
                                        logFC_col2="logFC_cis_mESC", txt="cis", axis=1)
data["trans_status_det_one"] = data.apply(comp_status_detail_one, status_col1="trans_human_status", 
                                          status_col2="trans_mouse_status", logFC_col1="logFC_trans_human", 
                                          logFC_col2="logFC_trans_mouse", txt="trans", axis=1)


# In[80]:


data["logFC_cis_one"] = data.apply(comp_logFC_one, status_col1="cis_HUES64_status", status_col2="cis_mESC_status",
                                   logFC_col1="logFC_cis_HUES64", logFC_col2="logFC_cis_mESC", axis=1)
data["logFC_trans_one"] = data.apply(comp_logFC_one, status_col1="trans_human_status", status_col2="trans_mouse_status",
                                     logFC_col1="logFC_trans_human", logFC_col2="logFC_trans_mouse", axis=1)


# ## 11. print numbers with each effect

# In[81]:


# remove ctrls
data = data[~data["hg19_id"].str.contains("ctrl")]
len(data)


# ## native

# In[82]:


data.native_status.value_counts()


# ## cis - HUES64

# In[83]:


data.cis_HUES64_status.value_counts()


# ## cis - mESC

# In[84]:


data.cis_mESC_status.value_counts()


# ## trans - human

# In[85]:


data.trans_human_status.value_counts()


# ## trans - mouse

# In[86]:


data.trans_mouse_status.value_counts()


# ## cis/trans interactions

# In[87]:


data.cis_trans_int_status.value_counts()


# In[88]:


np.max(np.abs(data[data["trans_mouse_status"].str.contains("significant")]["fdr_trans_mouse"]))


# ## 12. write files

# In[89]:


# rearrange columns for readability
data = data[['hg19_id', 'chr_tss_hg19', 'start_tss_hg19', 'biotype_hg19', 'cage_id_hg19', 'name_peak_hg19', 
             'minimal_biotype_hg19', 'stem_exp_hg19', 'mm9_id', 'chr_tss_mm9', 
             'start_tss_mm9', 'biotype_mm9', 'cage_id_mm9', 'name_peak_mm9', 
             'minimal_biotype_mm9', 'stem_exp_mm9', 'tss_tile_num', 'orig_species',  
             'biotype_switch_minimal', 'HUES64_hg19', 'mESC_hg19', 'HUES64_mm9', 'mESC_mm9', 'HUES64_padj_hg19', 
             'mESC_padj_hg19', 'HUES64_padj_mm9', 'mESC_padj_mm9', 'logFC_native', 'fdr_native', 'native_status', 
             'native_status_det', 'logFC_cis_HUES64', 'fdr_cis_HUES64', 'logFC_cis_mESC', 'fdr_cis_mESC',  
             'cis_HUES64_status', 'cis_mESC_status', 'cis_HUES64_status_det', 'cis_mESC_status_det', 'cis_status_one', 
             'cis_status_det_one', 'logFC_cis_one', 'logFC_trans_human', 'fdr_trans_human', 'logFC_trans_mouse', 
             'fdr_trans_mouse', 'trans_human_status', 'trans_mouse_status', 'trans_human_status_det', 
             'trans_mouse_status_det',  'trans_status_one', 'trans_status_det_one', 'logFC_trans_one',  'logFC_int', 
             'fdr_int', 'cis_trans_int_status', 'cis_trans_int_status_det']]


# In[90]:


len(data)


# In[91]:


data.head()


# In[92]:


data.to_csv("../../../data/02__mpra/03__results/all_processed_results.txt", sep="\t", index=False)

