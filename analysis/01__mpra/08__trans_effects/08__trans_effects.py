
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


# ## functions

# In[5]:


def trans_sig_status(row):
    if row.trans_human_status == "significant trans effect" and row.trans_mouse_status == "significant trans effect":
        return "sig_both"
    elif row.trans_human_status == "significant trans effect" and row.trans_mouse_status == "no trans effect":
        return "sig_human"
    elif row.trans_human_status == "no trans effect" and row.trans_mouse_status == "significant trans effect":
        return "sig_mouse"
    else:
        return "not_sig_both"


# ## variables

# In[6]:


data_f = "../../../data/02__mpra/03__results/all_processed_results.txt"


# ## 1. import data

# In[7]:


data = pd.read_table(data_f, sep="\t")
data.head()


# ## 2. filter data

# In[8]:


data = data[~pd.isnull(data["minimal_biotype_hg19"])]
len(data)


# In[9]:


data_filt = data[((data["HUES64_padj_hg19"] < QUANT_ALPHA) | (data["mESC_padj_mm9"] < QUANT_ALPHA))]
len(data_filt)


# In[10]:


data_filt_sp = data_filt.drop("orig_species", axis=1)
data_filt_sp.drop_duplicates(inplace=True)
len(data_filt_sp)


# In[11]:


data_filt_tile1 = data_filt[data_filt["tss_tile_num"] == "tile1"]
len(data_filt_tile1)


# In[12]:


data_filt_tile1_sp = data_filt_sp[data_filt_sp["tss_tile_num"] == "tile1"]
len(data_filt_tile1_sp)


# In[13]:


data_filt_tile2 = data_filt[data_filt["tss_tile_num"] == "tile2"]
len(data_filt_tile2)


# In[14]:


data_filt_tile2_sp = data_filt_sp[data_filt_sp["tss_tile_num"] == "tile2"]
len(data_filt_tile2_sp)


# ## 3. count of trans effects

# In[15]:


trans_order = ["no trans effect", "significant trans effect"]
trans_pal = {"no trans effect": "gray", "significant trans effect": "black"}

det_order = ["trans effect\n(higher in human)", "trans effect\n(higher in mouse)"]
complete_pal = {"trans effect\n(higher in human)": sns.color_palette("Set2")[1],
                "trans effect\n(higher in mouse)": sns.color_palette("Set2")[0]}
det_pal = {"trans effect\n(higher in human)": sns.light_palette(sns.color_palette("Set2")[1])[2],
           "trans effect\n(higher in mouse)": sns.light_palette(sns.color_palette("Set2")[0])[2]}


# In[16]:


dfs = [data_filt_sp, data_filt_tile1_sp, data_filt_tile2_sp]
titles = ["both tiles", "tile1 only", "tile2 only"]
labels = ["both_tiles", "tile1_only", "tile2_only"]


# In[17]:


for df, title, label in zip(dfs, titles, labels):
    
    fig, ax = plt.subplots(figsize=(0.75, 1.75), nrows=1, ncols=1)

    sns.countplot(data=df, x="trans_status_one", palette=trans_pal, order=trans_order, linewidth=2, 
                  edgecolor=trans_pal.values(), ax=ax)
    ax.set_xticklabels(["no trans effect", "trans effect"], va="top", ha="right", rotation=50)
    ax.set_xlabel("")
    ax.set_title(title)
    
    tot = 0
    for i, l in enumerate(trans_order):
        n = len(df[df["trans_status_one"] == l])
        tot += n
        ax.annotate(str(n), xy=(i, 2), xycoords="data", xytext=(0, 0), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color="white", size=fontsize)
    print("percent trans sig: %s" % (n/tot))

    plt.show()
    fig.savefig("count_trans_status.%s.pdf" % label, dpi="figure", bbox_inches="tight")
    plt.close()


# ## 4. plot trans effect sizes between human and mouse

# In[18]:


for df, title, label in zip(dfs, titles, labels):
    df["trans_sig_status"] = df.apply(trans_sig_status, axis=1)
    # plot effect size agreement b/w the two seqs
    fig, ax = plt.subplots(figsize=(1.75, 1.75), nrows=1, ncols=1)

    sig_human = df[df["trans_sig_status"] == "sig_human"]
    sig_mouse = df[df["trans_sig_status"] == "sig_mouse"]
    sig_both = df[df["trans_sig_status"] == "sig_both"]
    not_sig = df[df["trans_sig_status"] == "not_sig_both"]

    ax.scatter(not_sig["logFC_trans_human"], not_sig["logFC_trans_mouse"], s=10, alpha=0.75, 
               color="gray", linewidths=0.5, edgecolors="white")
    ax.scatter(sig_human["logFC_trans_human"], sig_human["logFC_trans_mouse"], s=10, alpha=0.75, 
               color=sns.color_palette("Set2")[1], linewidths=0.5, edgecolors="white")
    ax.scatter(sig_mouse["logFC_trans_human"], sig_mouse["logFC_trans_mouse"], s=10, alpha=0.75, 
               color=sns.color_palette("Set2")[0], linewidths=0.5, edgecolors="white")
    ax.scatter(sig_both["logFC_trans_human"], sig_both["logFC_trans_mouse"], s=12, alpha=1, 
               color="black", linewidths=0.5, edgecolors="white")

    plt.xlabel("human seq. trans effect size")
    plt.ylabel("mouse seq. trans effect size")

    ax.axhline(y=0, color="black", linestyle="dashed")
    ax.axvline(x=0, color="black", linestyle="dashed")
    ax.set_xlim((-3, 2))
    ax.set_ylim((-2, 2.5))

    # annotate corr
    no_nan = df[(~pd.isnull(df["logFC_trans_human"])) & (~pd.isnull(df["logFC_trans_mouse"]))]
    r, p = spearmanr(no_nan["logFC_trans_human"], no_nan["logFC_trans_mouse"])
    ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
            transform=ax.transAxes)
    ax.text(0.05, 0.90, "n = %s" % (len(no_nan)), ha="left", va="top", fontsize=fontsize,
            transform=ax.transAxes)
    fig.savefig("trans_effect_bw_seqs_scatter.sig_status_color.%s.pdf" % label, dpi="figure", bbox_inches="tight")


# ## 5. effect size differences across biotypes

# In[19]:


min_switch_order = ["CAGE turnover - eRNA", "CAGE turnover - lncRNA", "CAGE turnover - mRNA", 
                    "eRNA", "lncRNA", "mRNA"]


# In[20]:


for df, title, label in zip(dfs, titles, labels):
    df["abs_logFC_trans"] = np.abs(df["logFC_trans_one"])
    #df = df[df["native_status"] == "significant native effect"]

    fig = plt.figure(figsize=(2.5, 2))
    ax = sns.boxplot(data=df, x="biotype_switch_minimal", y="abs_logFC_trans", 
                     flierprops = dict(marker='o', markersize=5), 
                     order=min_switch_order, color=sns.color_palette("Set2")[2])
    mimic_r_boxplot(ax)

    ax.set_xticklabels(min_switch_order, rotation=50, ha='right', va='top')
    ax.set_xlabel("")
    ax.set_ylabel("trans effect size")
    ax.set_title(title)

    for i, l in enumerate(min_switch_order):
        sub = df[df["biotype_switch_minimal"] == l]
        n = len(sub)
        print("%s median eff size: %s" % (l, sub["abs_logFC_trans"].median()))
        color = sns.color_palette("Set2")[2]
        ax.annotate(str(n), xy=(i, -0.2), xycoords="data", xytext=(0, 0), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color=color, size=fontsize)

    ax.set_ylim((-0.3, 1.75))
    ax.axvline(x=2.5, linestyle="dashed", color="black")

    plt.show()
    fig.savefig("trans_minimal_biotype_switch_effectsize_boxplot.%s.pdf" % label, dpi="figure", bbox_inches="tight")
    plt.close()


# ## 6. percent sig across biotypes

# In[21]:


for df, title, label in zip(dfs, titles, labels):

    tots = df.groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
    sig = df[df["trans_status_one"] != "no trans effect"].groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
    clean_sig = tots.merge(sig, on="biotype_switch_minimal", how="left").fillna(0)
    clean_sig["percent_sig"] = (clean_sig["hg19_id_y"]/clean_sig["hg19_id_x"])*100
    
    fig = plt.figure(figsize=(2.5, 1.5))
    ax = sns.barplot(data=clean_sig, x="biotype_switch_minimal", y="percent_sig", 
                     order=min_switch_order, color=sns.color_palette("Set2")[2])

    ax.set_xticklabels(min_switch_order, rotation=50, ha='right', va='top')
    ax.set_xlabel("")
    ax.set_ylabel("% of seq. pairs with\ntrans effects")
    ax.set_title(title)
    ax.axvline(x=2.5, linestyle="dashed", color="black")
    
    for i, l in enumerate(min_switch_order):
        sub = clean_sig[clean_sig["biotype_switch_minimal"] == l]
        print("%s perc sig: %s" % (l, sub["percent_sig"].iloc[0]))
        n = sub["hg19_id_x"].iloc[0]
        ax.annotate(str(n), xy=(i, 2), xycoords="data", xytext=(0, 0), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color="white", size=fontsize)
    
    plt.show()
    fig.savefig("perc_sig_trans_minimal_biotype_switch.%s.pdf" % label, dpi="figure", bbox_inches="tight")
    plt.close()

