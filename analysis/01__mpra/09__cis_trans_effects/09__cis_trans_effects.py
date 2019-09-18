
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
import sys

from itertools import combinations 
from scipy.integrate import cumtrapz
from scipy.stats import linregress
from scipy.stats import spearmanr
from scipy.stats import pearsonr

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


# ## variables

# In[5]:


data_f = "../../../data/02__mpra/03__results/all_processed_results.txt"


# ## 1. import data

# In[6]:


data = pd.read_table(data_f, sep="\t")
data.head()


# ## 2. filter data

# In[7]:


data = data[~pd.isnull(data["minimal_biotype_hg19"])]
len(data)


# In[8]:


data_filt = data[((data["HUES64_padj_hg19"] < QUANT_ALPHA) | (data["mESC_padj_mm9"] < QUANT_ALPHA))]
len(data_filt)


# In[9]:


data_filt_sp = data_filt.drop("orig_species", axis=1)
data_filt_sp.drop_duplicates(inplace=True)
len(data_filt_sp)


# In[10]:


data_filt_tile1 = data_filt[data_filt["tss_tile_num"] == "tile1"]
len(data_filt_tile1)


# In[11]:


data_filt_tile1_sp = data_filt_sp[data_filt_sp["tss_tile_num"] == "tile1"]
len(data_filt_tile1_sp)


# In[12]:


data_filt_tile2 = data_filt[data_filt["tss_tile_num"] == "tile2"]
len(data_filt_tile2)


# In[13]:


data_filt_tile2_sp = data_filt_sp[data_filt_sp["tss_tile_num"] == "tile2"]
len(data_filt_tile2_sp)


# ## 3. count of cis/trans interactions

# In[14]:


int_order = ["no cis/trans int. effect", "significant cis/trans int. effect"]
int_pal = {"no cis/trans int. effect": "gray", "significant cis/trans int. effect": "black"}


# In[15]:


dfs = [data_filt_sp, data_filt_tile1_sp, data_filt_tile2_sp]
titles = ["both tiles", "tile1 only", "tile2 only"]
labels = ["both_tiles", "tile1_only", "tile2_only"]


# In[16]:


for df, title, label in zip(dfs, titles, labels):
    
    fig, ax = plt.subplots(figsize=(0.75, 1.75), nrows=1, ncols=1)

    sns.countplot(data=df, x="cis_trans_int_status", palette=int_pal, order=int_order, linewidth=2, 
                  edgecolor=int_pal.values(), ax=ax)
    ax.set_xticklabels(["no cis/trans interaction", "cis/trans interaction"], va="top", ha="right", rotation=50)
    ax.set_xlabel("")
    ax.set_title(title)
    
    tot = 0
    for i, l in enumerate(int_order):
        n = len(df[df["cis_trans_int_status"] == l])
        tot += n
        ax.annotate(str(n), xy=(i, 2), xycoords="data", xytext=(0, 0), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color="white", size=fontsize)
    print("percent cis/trans sig: %s" % (n/tot))

    plt.show()
    fig.savefig("count_cistrans_status.%s.pdf" % label, dpi="figure", bbox_inches="tight")
    plt.close()


# ## 5. effect size differences across biotypes

# In[17]:


min_switch_order = ["CAGE turnover - eRNA", "CAGE turnover - lncRNA", "CAGE turnover - mRNA", 
                    "eRNA", "lncRNA", "mRNA"]


# In[18]:


for df, title, label in zip(dfs, titles, labels):
    df["abs_logFC_int"] = np.abs(df["logFC_int"])
    #df = df[df["native_status"] == "significant native effect"]

    fig = plt.figure(figsize=(2.5, 2))
    ax = sns.boxplot(data=df, x="biotype_switch_minimal", y="abs_logFC_int", 
                     flierprops = dict(marker='o', markersize=5), 
                     order=min_switch_order, color=sns.color_palette("Set2")[2])
    mimic_r_boxplot(ax)

    ax.set_xticklabels(min_switch_order, rotation=50, ha='right', va='top')
    ax.set_xlabel("")
    ax.set_ylabel("cis/trans interaction\neffect size")
    ax.set_title(title)

    for i, l in enumerate(min_switch_order):
        sub = df[df["biotype_switch_minimal"] == l]
        n = len(sub)
        print("%s median eff size: %s" % (l, sub["abs_logFC_int"].median()))
        color = sns.color_palette("Set2")[2]
        ax.annotate(str(n), xy=(i, -0.2), xycoords="data", xytext=(0, 0), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color=color, size=fontsize)

    ax.set_ylim((-0.3, 2))
    ax.axvline(x=2.5, linestyle="dashed", color="black")

    plt.show()
    fig.savefig("cistrans_minimal_biotype_switch_effectsize_boxplot.%s.pdf" % label, dpi="figure", bbox_inches="tight")
    plt.close()


# ## 6. percent sig across biotypes

# In[19]:


for df, title, label in zip(dfs, titles, labels):

    tots = df.groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
    sig = df[df["cis_trans_int_status"] != "no cis/trans int. effect"].groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
    clean_sig = tots.merge(sig, on="biotype_switch_minimal", how="left").fillna(0)
    clean_sig["percent_sig"] = (clean_sig["hg19_id_y"]/clean_sig["hg19_id_x"])*100
    
    fig = plt.figure(figsize=(2.5, 1.5))
    ax = sns.barplot(data=clean_sig, x="biotype_switch_minimal", y="percent_sig", 
                     order=min_switch_order, color=sns.color_palette("Set2")[2])

    ax.set_xticklabels(min_switch_order, rotation=50, ha='right', va='top')
    ax.set_xlabel("")
    ax.set_ylabel("% of seq. pairs with\ncis/trans interactions")
    ax.set_title(title)
    ax.axvline(x=2.5, linestyle="dashed", color="black")
    
    for i, l in enumerate(min_switch_order):
        sub = clean_sig[clean_sig["biotype_switch_minimal"] == l]
        print("%s perc sig: %s" % (l, sub["percent_sig"].iloc[0]))
        n = sub["hg19_id_x"].iloc[0]
        ax.annotate(str(n), xy=(i, 0.5), xycoords="data", xytext=(0, 0), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color="white", size=fontsize)
    
    plt.show()
    fig.savefig("perc_sig_cistrans_minimal_biotype_switch.%s.pdf" % label, dpi="figure", bbox_inches="tight")
    plt.close()


# ## 7. look generally at significant interactions

# In[20]:


for df, title, label in zip(dfs, titles, labels):
    
    # plot effect size agreement b/w the two cells
    fig, ax = plt.subplots(figsize=(1.75, 1.75), nrows=1, ncols=1)

    sig_int = df[df["cis_trans_int_status"] != "no cis/trans int. effect"]
    not_sig_int = df[df["cis_trans_int_status"] == "no cis/trans int. effect"]

    ax.scatter(not_sig_int["logFC_cis_HUES64"], not_sig_int["logFC_cis_mESC"], s=10, alpha=0.75, 
               color="gray", linewidths=0.5, edgecolors="white")
    ax.scatter(sig_int["logFC_cis_HUES64"], sig_int["logFC_cis_mESC"], s=10, alpha=1, 
               color=sns.color_palette("Set2")[3], linewidths=0.5, edgecolors="white")

    plt.xlabel("cis effect size in hESCs")
    plt.ylabel("cis effect size in mESCs")

    ax.axhline(y=0, color="black", linestyle="dashed")
    ax.axvline(x=0, color="black", linestyle="dashed")
    ax.set_xlim((-6, 6))
    ax.set_ylim((-6, 6))

    # annotate corr
    no_nan = df[(~pd.isnull(df["logFC_cis_HUES64"])) & (~pd.isnull(df["logFC_cis_mESC"]))]
    r, p = spearmanr(no_nan["logFC_cis_HUES64"], no_nan["logFC_cis_mESC"])
    ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
            transform=ax.transAxes)
    ax.text(0.05, 0.90, "n = %s" % (len(no_nan)), ha="left", va="top", fontsize=fontsize,
            transform=ax.transAxes)
    plt.show()
    #fig.savefig("cis_effect_bw_cells_scatter.sig_status_color.%s.pdf" % label, dpi="figure", bbox_inches="tight")
    plt.close()


# ## 8. look at highest cis/trans interactions

# In[21]:


sig_int = data_filt_tile1_sp[data_filt_tile1_sp["cis_trans_int_status"] != "no cis/trans int. effect"]
len(sig_int)


# In[22]:


sig_int_filt = sig_int[((sig_int["logFC_cis_HUES64"] < 0) & (sig_int["logFC_cis_mESC"] > 0)) |
                       ((sig_int["logFC_cis_HUES64"] > 0) & (sig_int["logFC_cis_mESC"] < 0))]
len(sig_int_filt)


# In[23]:


sub = sig_int_filt[["hg19_id", "mm9_id", "biotype_hg19", "biotype_mm9", "biotype_switch_minimal", "logFC_int", "logFC_cis_HUES64", "logFC_cis_mESC",
                    "HUES64_hg19", "mESC_hg19", "HUES64_mm9", "mESC_mm9"]]
sub


# In[24]:


pal = {"hg19": sns.color_palette("Set2")[1], "mm9": sns.color_palette("Set2")[0]}


# In[25]:


for row in sub.iterrows():
    samp = pd.DataFrame(row[1]).T
    melt = pd.melt(samp, id_vars=["hg19_id", "mm9_id", "biotype_hg19", "biotype_mm9", "biotype_switch_minimal", 
                                  "logFC_int", "logFC_cis_HUES64", "logFC_cis_mESC"])
    melt["cell"] = melt["variable"].str.split("_", expand=True)[0]
    melt["seq"] = melt["variable"].str.split("_", expand=True)[1]
    
    fig = plt.figure(figsize=(2, 2))
    ax = sns.barplot(data=melt, x="cell", hue="seq", y="value", palette=pal)
    ax.set_ylabel("MPRA activity")
    ax.set_xlabel("")
    ax.set_xticklabels(["hESCs", "mESCs"], rotation=50, ha='right', va='top')
    ax.set_title("human ID: %s, human biotype: %s\nmouse ID: %s, mouse biotype:%s" % (row[1].hg19_id, 
                                                                                      row[1].biotype_hg19, 
                                                                                      row[1].mm9_id,
                                                                                      row[1].biotype_mm9))
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
    plt.show()
    plt.close()
    fig.savefig("%s.%s.barplot.pdf" % (row[1].hg19_id, row[1].mm9_id), dpi="figure", bbox_inches="tight")


# ## 9. look at cis/trans when subsetting by native

# In[26]:


for df, title, label in zip(dfs, titles, labels):
    
    fig, axarr = plt.subplots(figsize=(5.8, 2), nrows=1, ncols=3, sharex=True, sharey=True)
    
    # all seqs
    ax = axarr[0]
    ax.scatter(df["logFC_cis_one"], df["logFC_trans_one"], s=10, alpha=0.75, 
               color="gray", linewidths=0.5, edgecolors="white")

    plt.xlabel("cis effect size")
    plt.ylabel("trans effect size")

    ax.axhline(y=0, color="black", linestyle="dashed")
    ax.axvline(x=0, color="black", linestyle="dashed")

    # annotate corr
    no_nan = df[(~pd.isnull(df["logFC_cis_one"])) & (~pd.isnull(df["logFC_trans_one"]))]
    r, p = spearmanr(no_nan["logFC_cis_one"], no_nan["logFC_trans_one"])
    ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
            transform=ax.transAxes)
    ax.text(0.05, 0.90, "n = %s" % (len(no_nan)), ha="left", va="top", fontsize=fontsize,
            transform=ax.transAxes)
    
    # native effects only
    sub = df[df["native_status"] == "significant native effect"]
    ax = axarr[1]
    ax.scatter(sub["logFC_cis_one"], sub["logFC_trans_one"], s=10, alpha=0.75, 
               color="gray", linewidths=0.5, edgecolors="white")

    plt.xlabel("cis effect size")
    plt.ylabel("trans effect size")

    ax.axhline(y=0, color="black", linestyle="dashed")
    ax.axvline(x=0, color="black", linestyle="dashed")

    # annotate corr
    no_nan = sub[(~pd.isnull(sub["logFC_cis_one"])) & (~pd.isnull(sub["logFC_trans_one"]))]
    r, p = spearmanr(no_nan["logFC_cis_one"], no_nan["logFC_trans_one"])
    ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
            transform=ax.transAxes)
    ax.text(0.05, 0.90, "n = %s" % (len(no_nan)), ha="left", va="top", fontsize=fontsize,
            transform=ax.transAxes)
    
    # no native effects
    sub = df[df["native_status"] == "no native effect"]
    ax = axarr[2]
    ax.scatter(sub["logFC_cis_one"], sub["logFC_trans_one"], s=10, alpha=0.75, 
               color="gray", linewidths=0.5, edgecolors="white")

    plt.xlabel("cis effect size")
    plt.ylabel("trans effect size")

    ax.axhline(y=0, color="black", linestyle="dashed")
    ax.axvline(x=0, color="black", linestyle="dashed")

    # annotate corr
    no_nan = sub[(~pd.isnull(sub["logFC_cis_one"])) & (~pd.isnull(sub["logFC_trans_one"]))]
    r, p = spearmanr(no_nan["logFC_cis_one"], no_nan["logFC_trans_one"])
    ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
            transform=ax.transAxes)
    ax.text(0.05, 0.90, "n = %s" % (len(no_nan)), ha="left", va="top", fontsize=fontsize,
            transform=ax.transAxes)
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    fig.savefig("cis_v_trans.%s.pdf" % label, dpi="figure", bbox_inches="tight")


# ## 10. look at invidiual directionality of cis/trans

# In[27]:


df.columns


# In[28]:


for df, title, label in zip(dfs, titles, labels):
    res = {}
    cis_trans = df[(df["cis_status_one"] == "significant cis effect") & 
                   (df["trans_status_one"] == "significant trans effect")]
    tots = len(cis_trans)
    print(tots)
    res["total"] = [tots]
    direc = cis_trans[((cis_trans["cis_status_det_one"].str.contains("higher in human") & 
                        cis_trans["trans_status_det_one"].str.contains("higher in human")) |
                       (cis_trans["cis_status_det_one"].str.contains("higher in mouse") &
                        cis_trans["trans_status_det_one"].str.contains("higher in mouse")))]    
    direc = len(direc)
    res["directional"] = [direc]

    comp = cis_trans[((cis_trans["cis_status_det_one"].str.contains("higher in human") & 
                        cis_trans["trans_status_det_one"].str.contains("higher in mouse")) |
                       (cis_trans["cis_status_det_one"].str.contains("higher in mouse") &
                        cis_trans["trans_status_det_one"].str.contains("higher in human")))]
    comp = len(comp)
    res["compensatory"] = [comp]
    res = pd.DataFrame.from_dict(res, orient="index").reset_index()
    res["perc"] = (res[0]/tots)*100
    
    order = ["directional", "compensatory"]
    fig, ax = plt.subplots(figsize=(1, 2), nrows=1, ncols=1)
    
    sns.barplot(data=res, x="index", y="perc", order=order, color=sns.color_palette("Set2")[2], ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("percent")
    ax.set_xticklabels(order, rotation=50, ha="right", va="top")
    fig.savefig("direc_v_comp.%s.pdf" % label, dpi="figure", bbox_inches="tight")


# In[29]:


for df, title, label in zip(dfs, titles, labels):

    cis_trans = df[(df["cis_status_one"] == "significant cis effect") & 
                   (df["trans_status_one"] == "significant trans effect")]
    tots = cis_trans.groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
    
    comp = cis_trans[((cis_trans["cis_status_det_one"].str.contains("higher in human") & 
                        cis_trans["trans_status_det_one"].str.contains("higher in mouse")) |
                       (cis_trans["cis_status_det_one"].str.contains("higher in mouse") &
                        cis_trans["trans_status_det_one"].str.contains("higher in human")))]
    sig = comp.groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
    clean_sig = tots.merge(sig, on="biotype_switch_minimal", how="left").fillna(0)
    clean_sig["percent_sig"] = (clean_sig["hg19_id_y"]/clean_sig["hg19_id_x"])*100
    
    fig = plt.figure(figsize=(2.5, 1.5))
    ax = sns.barplot(data=clean_sig, x="biotype_switch_minimal", y="percent_sig", 
                     order=min_switch_order, color=sns.color_palette("Set2")[2])

    ax.set_xticklabels(min_switch_order, rotation=50, ha='right', va='top')
    ax.set_xlabel("")
    ax.set_ylabel("% of seq. pairs with\ncompensatory cis/trans effects")
    ax.set_title(title)
    ax.axvline(x=2.5, linestyle="dashed", color="black")
    
    for i, l in enumerate(min_switch_order):
        sub = clean_sig[clean_sig["biotype_switch_minimal"] == l]
        print("%s perc sig: %s" % (l, sub["percent_sig"].iloc[0]))
        n = sub["hg19_id_x"].iloc[0]
        ax.annotate(str(n), xy=(i, 0.5), xycoords="data", xytext=(0, 0), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color="white", size=fontsize)
    
    plt.show()
    fig.savefig("perc_sig_compensatory_minimal_biotype_switch.%s.pdf" % label, dpi="figure", bbox_inches="tight")
    plt.close()

