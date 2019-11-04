
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

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


def consolidate_cage(row, biotype_col):
    if row[biotype_col] == "reclassified - CAGE peak":
        return "no CAGE activity"
    else:
        return row[biotype_col]


# In[5]:


def fix_cleaner_biotype(row, biotype_col):
    try:
        if row["name"] == "random_sequence":
            return "negative control"
        elif "samp" in row.element:
            return "positive control"
        else:
            return row[biotype_col]
    except:
        return row[biotype_col]


# In[6]:


def is_sig(row, col):
    if row[col] < 0.05:
        return "sig"
    else:
        return "not sig"


# In[7]:


def fix_cage_exp(row, col):
    if row[col] == "no cage activity":
        return 0
    else:
        return float(row[col])


# ## variables

# In[8]:


data_dir = "../../../data/02__mpra/02__activs"
alpha_f = "%s/alpha_per_elem.quantification.txt" % data_dir


# In[9]:


index_f = "../../../data/01__design/02__index/TWIST_pool4_v8_final.with_element_id.txt.gz"


# In[10]:


tss_map_f = "../../../data/01__design/01__mpra_list/mpra_tss.with_ids.RECLASSIFIED.txt"


# ## 1. import files

# In[11]:


alpha = pd.read_table(alpha_f, sep="\t")
alpha.reset_index(inplace=True)
alpha.head()


# In[12]:


index = pd.read_table(index_f, sep="\t")


# In[13]:


index_elem = index[["element", "tile_type", "element_id", "name", "tile_number", "chrom", "strand", "actual_start", 
                    "actual_end", "dupe_info"]]
index_elem = index_elem.drop_duplicates()


# In[14]:


tss_map = pd.read_table(tss_map_f, sep="\t")
tss_map.head()


# In[15]:


# tss_map["minimal_biotype_hg19"] = tss_map.apply(consolidate_cage, biotype_col="minimal_biotype_hg19", axis=1)
# tss_map["minimal_biotype_mm9"] = tss_map.apply(consolidate_cage, biotype_col="minimal_biotype_mm9", axis=1)


# ## 2. merge alphas w/ index

# In[16]:


pos_ctrls = alpha[alpha["index"].str.contains("__samp")]
pos_ctrls["HUES64_log"] = np.log10(pos_ctrls["HUES64"])
pos_ctrls["mESC_log"] = np.log10(pos_ctrls["mESC"])
len(pos_ctrls)


# In[17]:


alpha = alpha[~alpha["index"].str.contains("__samp")]
len(alpha)


# In[18]:


data = alpha.merge(index_elem, left_on="index", right_on="element", how="left")
data.drop("index", axis=1, inplace=True)
data.head()


# In[19]:


data["HUES64_log"] = np.log10(data["HUES64"])
data["mESC_log"] = np.log10(data["mESC"])
data.sample(5)


# ## 3. compare activities across biotypes + controls

# In[20]:


data["tss_id"] = data["name"].str.split("__", expand=True)[1]
data["species"] = data["name"].str.split("_", expand=True)[0]
data["tss_tile_num"] = data["name"].str.split("__", expand=True)[2]
data.sample(5)


# In[21]:


pos_ctrls.columns = ["element", "HUES64", "mESC", "HUES64_pval", "mESC_pval", "HUES64_padj", "mESC_padj", 
                     "HUES64_log", "mESC_log"]
pos_ctrls.head()


# In[22]:


human_df = data[(data["species"] == "HUMAN") | (data["name"] == "random_sequence")]
mouse_df = data[(data["species"] == "MOUSE") | (data["name"] == "random_sequence")]

human_df_w_ctrls = human_df.append(pos_ctrls)
mouse_df_w_ctrls = mouse_df.append(pos_ctrls)

human_df_w_ctrls = human_df_w_ctrls.merge(tss_map[["hg19_id", "biotype_hg19", "minimal_biotype_hg19", 
                                                   "stem_exp_hg19", "orig_species", "max_cage_hg19"]], 
                                          left_on="tss_id", right_on="hg19_id", how="left")
mouse_df_w_ctrls = mouse_df_w_ctrls.merge(tss_map[["mm9_id", "biotype_mm9", "minimal_biotype_mm9", "stem_exp_mm9", 
                                                   "orig_species", "max_cage_mm9"]], 
                                          left_on="tss_id", right_on="mm9_id", how="left")
mouse_df_w_ctrls.sample(5)


# In[23]:


human_df_w_ctrls["minimal_biotype_hg19"] = human_df_w_ctrls.apply(fix_cleaner_biotype, 
                                                                  biotype_col="minimal_biotype_hg19",
                                                                  axis=1)
human_df_w_ctrls.minimal_biotype_hg19.value_counts()


# In[24]:


mouse_df_w_ctrls["minimal_biotype_mm9"] = mouse_df_w_ctrls.apply(fix_cleaner_biotype, 
                                                                 biotype_col="minimal_biotype_mm9",
                                                                 axis=1)
mouse_df_w_ctrls.minimal_biotype_mm9.value_counts()


# In[25]:


min_ctrl_order = ["negative control", "no CAGE activity", "eRNA", 
                  "lncRNA", "mRNA", "positive control"]

min_human_ctrl_pal = {"negative control": "lightgray", "no CAGE activity": "gray", "reclassified - CAGE peak": "gray",
                      "eRNA": sns.color_palette("Set2")[1], "lncRNA": sns.color_palette("Set2")[1], 
                      "mRNA": sns.color_palette("Set2")[1], "positive control": "black"}

min_mouse_ctrl_pal = {"negative control": "lightgray", "no CAGE activity": "gray", "reclassified - CAGE peak": "gray",
                      "eRNA": sns.color_palette("Set2")[0], "lncRNA": sns.color_palette("Set2")[0], 
                      "mRNA": sns.color_palette("Set2")[0],"positive control": "black"}


# In[26]:


fig = plt.figure(figsize=(2.65, 1.5))
ax = sns.boxplot(data=human_df_w_ctrls, x="minimal_biotype_hg19", y="HUES64", flierprops = dict(marker='o', markersize=5),
                 order=min_ctrl_order, palette=min_human_ctrl_pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(min_ctrl_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_yscale("symlog")
ax.set_ylabel("MPRA activity in hESCs")

for i, label in enumerate(min_ctrl_order):
    n = len(human_df_w_ctrls[human_df_w_ctrls["minimal_biotype_hg19"] == label])
    color = min_human_ctrl_pal[label]
    ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-1, 60))
plt.show()
fig.savefig("better_neg_ctrl_boxplot.human.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# In[27]:


fig = plt.figure(figsize=(2.65, 1.5))
ax = sns.boxplot(data=mouse_df_w_ctrls, x="minimal_biotype_mm9", y="mESC", flierprops = dict(marker='o', markersize=5),
                 order=min_ctrl_order, palette=min_mouse_ctrl_pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(min_ctrl_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_yscale("symlog")
ax.set_ylabel("MPRA activity in mESCs")

for i, label in enumerate(min_ctrl_order):
    n = len(mouse_df_w_ctrls[mouse_df_w_ctrls["minimal_biotype_mm9"] == label])
    color = min_mouse_ctrl_pal[label]
    ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-1, 60))
plt.show()
fig.savefig("better_neg_ctrl_boxplot.mouse.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# ## 4. compare activities across tiles

# In[28]:


df = data[data["tss_tile_num"].isin(["tile1", "tile2"])]
human_df = df[df["species"] == "HUMAN"]
mouse_df = df[df["species"] == "MOUSE"]

human_df = human_df.merge(tss_map[["hg19_id", "minimal_biotype_hg19", "stem_exp_hg19", "orig_species"]], left_on="tss_id", 
                          right_on="hg19_id", how="right")
mouse_df = mouse_df.merge(tss_map[["mm9_id", "minimal_biotype_mm9", "stem_exp_mm9", "orig_species"]], left_on="tss_id", 
                          right_on="mm9_id", how="right")
mouse_df.sample(5)


# In[29]:


for df, species, colname, color in zip([human_df, mouse_df], ["hESCs", "mESCs"], ["HUES64", "mESC"], [sns.color_palette("Set2")[1], sns.color_palette("Set2")[0]]):
    fig = plt.figure(figsize=(1, 1))
    ax = sns.boxplot(data=df, x="tss_tile_num", y=colname, flierprops = dict(marker='o', markersize=5),
                     color=color)
    mimic_r_boxplot(ax)

    # calc p-vals b/w dists
    tile1_dist = np.asarray(df[df["tss_tile_num"] == "tile1"][colname])
    tile2_dist = np.asarray(df[df["tss_tile_num"] == "tile2"][colname])

    tile1_dist = tile1_dist[~np.isnan(tile1_dist)]
    tile2_dist = tile2_dist[~np.isnan(tile2_dist)]

    tile_u, tile_pval = stats.mannwhitneyu(tile1_dist, tile2_dist, alternative="two-sided", use_continuity=False)
    print(tile_pval)

    annotate_pval(ax, 0.2, 0.8, 5, 0, 5, tile_pval, fontsize)
    ax.set_yscale('symlog')
    ax.set_ylabel("MPRA activity\n(%s)" % species)
    ax.set_xlabel("")
    ax.set_title(species)
    ax.set_ylim((-0.1, 100))


# In[30]:


for df, species, colname, color, suffix in zip([human_df, mouse_df], ["hESCs", "mESCs"], ["HUES64", "mESC"], 
                                               [sns.color_palette("Set2")[1], sns.color_palette("Set2")[0]],
                                               ["hg19", "mm9"]):
    fig, axarr = plt.subplots(figsize=(4.25, 1.25), nrows=1, ncols=3, sharex=False, sharey=False)
    for i, biotype in enumerate(["eRNA", "lncRNA", "mRNA"]):
        ax = axarr[i]
        sub = df[df["minimal_biotype_%s" % suffix] == biotype]
        sub["%s_log" % colname] = np.log10(sub[colname] + 1)
        sub_tile1 = sub[sub["tss_tile_num"] == "tile1"][["%s_id" % suffix, "%s_log" % colname]]
        sub_tile2 = sub[sub["tss_tile_num"] == "tile2"][["%s_id" % suffix, "%s_log" % colname]]
        sub = sub_tile1.merge(sub_tile2, on="%s_id" % suffix, suffixes=("_tile1", "_tile2"))
        
        sns.regplot(data=sub, x="%s_log_tile1" % colname, y="%s_log_tile2" % colname, color=color, 
                    scatter_kws={"s": 15, "alpha": 0.75, "linewidth": 0.5, "edgecolor": "white"}, fit_reg=True, 
                    ax=ax)
        
        # annotate corr
        no_nan = sub[(~pd.isnull(sub["%s_log_tile1" % colname])) & (~pd.isnull(sub["%s_log_tile2" % colname]))]
        r, p = spearmanr(no_nan["%s_log_tile1" % colname], no_nan["%s_log_tile2" % colname])

        ax.text(0.95, 0.95, "r = {:.2f}".format(r), ha="right", va="top", fontsize=fontsize,
            transform=ax.transAxes)
        ax.text(0.95, 0.85, "n = %s" % (len(no_nan)), ha="right", va="top", fontsize=fontsize,
                transform=ax.transAxes)
        
        if i == 1:
            ax.set_xlabel("log10 tile 1 MPRA activity in %s" % species)
        else:
            ax.set_xlabel("")
        
        if i == 0:
            ax.set_ylabel("log10 tile 2 MPRA activity\nin %s" % species)
        else:
            ax.set_ylabel("")
        ax.set_title(biotype)
    
    plt.subplots_adjust(wspace=0.3)
    plt.show()
    fig.savefig("tile_corr.%s.pdf" % species, dpi="figure", bbox_inches="tight")
    plt.close()


# In[31]:


for df, species, colname, color, suffix in zip([human_df, mouse_df], ["hESCs", "mESCs"], ["HUES64", "mESC"], 
                                               [sns.color_palette("Set2")[1], sns.color_palette("Set2")[0]],
                                               ["hg19", "mm9"]):
    fig, axarr = plt.subplots(figsize=(3.25, 1.25), nrows=1, ncols=2, sharex=False, sharey=False)
    for i, tile_num in enumerate(["tile1", "tile2"]):
        ax = axarr[i]
        sub = df[df["tss_tile_num"] == tile_num]
        sns.boxplot(data=sub, x="minimal_biotype_%s" % suffix, y="mESC", 
                    flierprops = dict(marker='o', markersize=5),
                    order=["eRNA", "lncRNA", "mRNA"], color=color, ax=ax)
        mimic_r_boxplot(ax)
        
        ax.set_yscale("symlog")
        ax.set_xticklabels(["eRNA", "lncRNA", "mRNA"], rotation=50, ha='right', va='top')
        
        for j, label in enumerate(["eRNA", "lncRNA", "mRNA"]):
            n = len(sub[sub["minimal_biotype_%s" % suffix] == label])
            ax.annotate(str(n), xy=(j, -0.7), xycoords="data", xytext=(0, 0), 
                        textcoords="offset pixels", ha='center', va='bottom', 
                        color=color, size=fontsize)
            
        ax.set_title(tile_num)
        ax.set_xlabel("")
        ax.set_ylim((-1, 60))
        if i == 0:
            ax.set_ylabel("MPRA activity in %ss" % species)
        else:
            ax.set_ylabel("")
    
    plt.subplots_adjust(wspace=0.3)
    plt.show()
    fig.savefig("tile_biotype_boxplot.%s.pdf" % species, dpi="figure", bbox_inches="tight")
    plt.close()


# ## 5. correlate MPRA activities w/ endogenous activs

# In[32]:


human_tmp = human_df_w_ctrls
human_tmp.columns


# In[33]:


human_tmp["stem_exp_hg19_fixed"] = human_tmp.apply(fix_cage_exp, col="stem_exp_hg19", axis=1)
human_tmp.sample(5)


# In[34]:


for tile_num in ["tile1", "tile2"]:
    df = human_tmp[(human_tmp["tss_tile_num"] == tile_num) & 
                   (~human_tmp["minimal_biotype_hg19"].isin(["no CAGE activity"]))]
    
    fig, ax = plt.subplots(figsize=(1.5, 1.5), nrows=1, ncols=1)

    df["stem_exp_hg19_log"] = np.log10(df["stem_exp_hg19_fixed"] + 1)
    sub = df[~pd.isnull(df["HUES64_log"])]
    print(len(sub))

    sns.regplot(data=sub, x="stem_exp_hg19_log", y="HUES64_log", color=min_human_ctrl_pal["mRNA"], 
                scatter_kws={"s": 15, "alpha": 0.75, "linewidth": 0.5, "edgecolor": "white"}, fit_reg=True, ax=ax)

    # annotate corr
    no_nan = sub[(~pd.isnull(sub["stem_exp_hg19_log"])) & (~pd.isnull(sub["HUES64_log"]))]
    r, p = spearmanr(no_nan["stem_exp_hg19_log"], no_nan["HUES64_log"])

    ax.text(0.95, 0.15, "r = {:.2f}".format(r), ha="right", va="bottom", fontsize=fontsize,
        transform=ax.transAxes)
    ax.text(0.95, 0.05, "n = %s" % (len(no_nan)), ha="right", va="bottom", fontsize=fontsize,
            transform=ax.transAxes)

    ax.set_xlabel("log10(CAGE expression + 1)\n(hESCs)")
    ax.set_ylabel("log10(MPRA activity)\n(hESCs)")

    plt.show()
    #fig.savefig("cage_corr_human.all.%s.pdf" % tile_num, dpi="figure", bbox_inches="tight")
    plt.close()


# In[35]:


for tile_num in ["tile1", "tile2"]:
    df = human_tmp[(human_tmp["tss_tile_num"] == tile_num)]
    
    fig, ax = plt.subplots(figsize=(1.5, 1.5), nrows=1, ncols=1)

    sub = df[~pd.isnull(df["HUES64_log"])]
    sub["cage_log"] = np.log10(sub["max_cage_hg19"] + 1)
    print(len(sub))

    sns.regplot(data=sub, x="cage_log", y="HUES64_log", color=min_human_ctrl_pal["mRNA"], 
                scatter_kws={"s": 15, "alpha": 0.75, "linewidth": 0.5, "edgecolor": "white"}, fit_reg=True, ax=ax)

    # annotate corr
    no_nan = sub[(~pd.isnull(sub["cage_log"])) & (~pd.isnull(sub["HUES64_log"]))]
    r, p = spearmanr(no_nan["cage_log"], no_nan["HUES64_log"])

    ax.text(0.95, 0.15, "r = {:.2f}".format(r), ha="right", va="bottom", fontsize=fontsize,
        transform=ax.transAxes)
    ax.text(0.95, 0.05, "n = %s" % (len(no_nan)), ha="right", va="bottom", fontsize=fontsize,
            transform=ax.transAxes)

    ax.set_xlabel("log10(max CAGE counts + 1)")
    ax.set_ylabel("log10(MPRA activity)\n(hESCs)")

    plt.show()
    #fig.savefig("cage_corr_human.all.%s.pdf" % tile_num, dpi="figure", bbox_inches="tight")
    plt.close()


# In[36]:


mouse_tmp = mouse_df_w_ctrls
mouse_tmp["stem_exp_mm9_fixed"] = mouse_tmp.apply(fix_cage_exp, col="stem_exp_mm9", axis=1)
len(mouse_tmp)


# In[37]:


for tile_num in ["tile1", "tile2"]:
    df = mouse_tmp[(mouse_tmp["tss_tile_num"] == tile_num) & 
                   (~mouse_tmp["minimal_biotype_mm9"].isin(["no CAGE activity"]))]
    
    fig, ax = plt.subplots(figsize=(1.5, 1.5), nrows=1, ncols=1)

    df["stem_exp_mm9_log"] = np.log10(df["stem_exp_mm9_fixed"] + 1)
    sub = df[~pd.isnull(df["mESC_log"])]
    print(len(sub))

    sns.regplot(data=sub, x="stem_exp_mm9_log", y="mESC_log", color=min_mouse_ctrl_pal["mRNA"], 
                scatter_kws={"s": 15, "alpha": 0.75, "linewidth": 0.5, "edgecolor": "white"}, fit_reg=True, ax=ax)

    # annotate corr
    no_nan = sub[(~pd.isnull(sub["stem_exp_mm9_log"])) & (~pd.isnull(sub["mESC_log"]))]
    r, p = spearmanr(no_nan["stem_exp_mm9_log"], no_nan["mESC_log"])

    ax.text(0.95, 0.15, "r = {:.2f}".format(r), ha="right", va="bottom", fontsize=fontsize,
        transform=ax.transAxes)
    ax.text(0.95, 0.05, "n = %s" % (len(no_nan)), ha="right", va="bottom", fontsize=fontsize,
            transform=ax.transAxes)

    ax.set_xlabel("log10(CAGE expression + 1)\n(mESCs)")
    ax.set_ylabel("log10(MPRA activity)\n(mESCs)")

    plt.show()
#     fig.savefig("cage_corr_mouse.all.%s.pdf" % tile_num, dpi="figure", bbox_inches="tight")
    plt.close()


# In[38]:


for tile_num in ["tile1", "tile2"]:
    df = mouse_tmp[(mouse_tmp["tss_tile_num"] == tile_num)]
    
    fig, ax = plt.subplots(figsize=(1.5, 1.5), nrows=1, ncols=1)

    sub = df[~pd.isnull(df["mESC_log"])]
    sub["cage_log"] = np.log10(sub["max_cage_mm9"] + 1)
    print(len(sub))

    sns.regplot(data=sub, x="cage_log", y="mESC_log", color=min_mouse_ctrl_pal["mRNA"], 
                scatter_kws={"s": 15, "alpha": 0.75, "linewidth": 0.5, "edgecolor": "white"}, fit_reg=True, ax=ax)

    # annotate corr
    no_nan = sub[(~pd.isnull(sub["cage_log"])) & (~pd.isnull(sub["mESC_log"]))]
    r, p = spearmanr(no_nan["cage_log"], no_nan["mESC_log"])

    ax.text(0.95, 0.15, "r = {:.2f}".format(r), ha="right", va="bottom", fontsize=fontsize,
        transform=ax.transAxes)
    ax.text(0.95, 0.05, "n = %s" % (len(no_nan)), ha="right", va="bottom", fontsize=fontsize,
            transform=ax.transAxes)

    ax.set_xlabel("log10(max CAGE counts + 1)")
    ax.set_ylabel("log10(MPRA activity)\n(mESCs)")

    plt.show()
    #fig.savefig("cage_corr_mouse.all.%s.pdf" % tile_num, dpi="figure", bbox_inches="tight")
    plt.close()


# ## 6. how does endogenous CAGE expr compare between human and mouse

# In[39]:


human_tmp["species"] = "human"
mouse_tmp["species"] = "mouse"

human_tmp = human_tmp[["tss_id", "minimal_biotype_hg19", "stem_exp_hg19_fixed", "species", "max_cage_hg19"]]
human_tmp.columns = ["tss_id", "minimal_biotype", "stem_exp_fixed", "species", "max_cage"]
mouse_tmp = mouse_tmp[["tss_id", "minimal_biotype_mm9", "stem_exp_mm9_fixed", "species", "max_cage_mm9"]]
mouse_tmp.columns = ["tss_id", "minimal_biotype", "stem_exp_fixed", "species", "max_cage"]

tmp = human_tmp.append(mouse_tmp)
tmp["log_stem"] = np.log10(tmp["stem_exp_fixed"]+1)
tmp["log_max"] = np.log10(tmp["max_cage"]+1)
tmp.head(5)


# In[40]:


tmp_pal = {"human": sns.color_palette("Set2")[1], "mouse": sns.color_palette("Set2")[0]}


# In[41]:


fig = plt.figure(figsize=(2.5, 1.5))

ax = sns.boxplot(data=tmp, x="minimal_biotype", y="stem_exp_fixed", hue="species",
                 flierprops = dict(marker='o', markersize=5),
                 order=["eRNA", "lncRNA", "mRNA"], palette=tmp_pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(["eRNA", "lncRNA", "mRNA"], rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_yscale("symlog")
ax.set_ylabel("CAGE tpm\n(ESCs)")
plt.legend(loc=2, bbox_to_anchor=(1.05, 1))

ys = [1, 2, 22]
for i, label in enumerate(["eRNA", "lncRNA", "mRNA"]):
    sub = tmp[tmp["minimal_biotype"] == label]
    human = sub[sub["species"] == "human"]
    mouse = sub[sub["species"] == "mouse"]
    
    human_vals = np.asarray(human["stem_exp_fixed"])
    mouse_vals = np.asarray(mouse["stem_exp_fixed"])

    human_vals = human_vals[~np.isnan(human_vals)]
    mouse_vals = mouse_vals[~np.isnan(mouse_vals)]

    u, pval = stats.mannwhitneyu(human_vals, mouse_vals, alternative="two-sided", use_continuity=False)
    print(pval)
    
    if pval >= 0.05:
        annotate_pval(ax, i-0.1, i+0.1, ys[i], 0, ys[i], pval, fontsize)
    else:
        annotate_pval(ax, i-0.1, i+0.1, ys[i], 0, ys[i], pval, fontsize)
    
    n_human = len(human)
    n_mouse = len(mouse)

    ax.annotate(str(n_human), xy=(i-0.25, -1), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=tmp_pal["human"], size=fontsize)
    ax.annotate(str(n_mouse), xy=(i+0.25, -1), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=tmp_pal["mouse"], size=fontsize)

ax.set_ylim((-1.25, 1500))
fig.savefig("human_v_mouse_cage.min.pdf", dpi="figure", bbox_inches="tight")


# In[42]:


fig = plt.figure(figsize=(2.5, 1.5))

ax = sns.boxplot(data=tmp, x="minimal_biotype", y="max_cage", hue="species",
                 flierprops = dict(marker='o', markersize=5),
                 order=["eRNA", "lncRNA", "mRNA"], palette=tmp_pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(["eRNA", "lncRNA", "mRNA"], rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_yscale("symlog")
ax.set_ylabel("max CAGE reads")
plt.legend(loc=2, bbox_to_anchor=(1.05, 1))

ys = [200, 5000, 50000]
for i, label in enumerate(["eRNA", "lncRNA", "mRNA"]):
    sub = tmp[tmp["minimal_biotype"] == label]
    human = sub[sub["species"] == "human"]
    mouse = sub[sub["species"] == "mouse"]
    
    human_vals = np.asarray(human["max_cage"])
    mouse_vals = np.asarray(mouse["max_cage"])

    human_vals = human_vals[~np.isnan(human_vals)]
    mouse_vals = mouse_vals[~np.isnan(mouse_vals)]

    u, pval = stats.mannwhitneyu(human_vals, mouse_vals, alternative="two-sided", use_continuity=False)
    print(pval)
    
    if pval >= 0.05:
        annotate_pval(ax, i-0.1, i+0.1, ys[i], 0, ys[i], pval, fontsize)
    else:
        annotate_pval(ax, i-0.1, i+0.1, ys[i], 0, ys[i], pval, fontsize)
    
    n_human = len(human)
    n_mouse = len(mouse)

    ax.annotate(str(n_human), xy=(i-0.25, -1), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=tmp_pal["human"], size=fontsize)
    ax.annotate(str(n_mouse), xy=(i+0.25, -1), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=tmp_pal["mouse"], size=fontsize)

#ax.set_ylim((-1.25, 1500))
#fig.savefig("human_v_mouse_cage.min.pdf", dpi="figure", bbox_inches="tight")


# ## 7. write files

# In[43]:


human_df_filename = "%s/human_TSS_vals.both_tiles.txt" % data_dir
mouse_df_filename = "%s/mouse_TSS_vals.both_tiles.txt" % data_dir


# In[44]:


human_df.to_csv(human_df_filename, sep="\t", index=False)
mouse_df.to_csv(mouse_df_filename, sep="\t", index=False)

