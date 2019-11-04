
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


def get_cage_fc(row):
    try:
        hg19_exp = float(row.stem_exp_hg19)+1
    except:
        hg19_exp = 1
    try:
        mm9_exp = float(row.stem_exp_mm9)+1
    except:
        mm9_exp = 1
    cage_fc = np.log2(mm9_exp/hg19_exp)
    return cage_fc


# In[6]:


def cage_status_det(row):
    if row.abs_cage_fc >= 1:
        if row.cage_fc > 0:
            return "higher in mouse"
        else:
            return "higher in human"
    else:
        return "not sig"


# ## variables

# In[7]:


data_f = "../../../data/02__mpra/03__results/all_processed_results.txt"


# ## 1. import data

# In[8]:


data = pd.read_table(data_f, sep="\t")
data.head()


# ## 2. filter data

# In[9]:


len(data)


# In[10]:


data = data[~pd.isnull(data["minimal_biotype_hg19"])]
len(data)


# In[11]:


data_filt = data[((data["HUES64_padj_hg19"] < QUANT_ALPHA) | (data["mESC_padj_mm9"] < QUANT_ALPHA))]
len(data_filt)


# In[12]:


data_filt_sp = data_filt.drop("orig_species", axis=1)
data_filt_sp.drop_duplicates(inplace=True)
len(data_filt_sp)


# In[13]:


data_filt_tile1 = data_filt[data_filt["tss_tile_num"] == "tile1"]
len(data_filt_tile1)


# In[14]:


data_filt_tile1_sp = data_filt_sp[data_filt_sp["tss_tile_num"] == "tile1"]
len(data_filt_tile1_sp)


# In[15]:


data_filt_tile2 = data_filt[data_filt["tss_tile_num"] == "tile2"]
len(data_filt_tile2)


# In[16]:


data_filt_tile2_sp = data_filt_sp[data_filt_sp["tss_tile_num"] == "tile2"]
len(data_filt_tile2_sp)


# ## 3. counts of native effects in general (across all biotypes)

# In[17]:


native_order = ["no native effect", "significant native effect"]
native_pal = {"no native effect": "gray", "significant native effect": "black"}

det_order = ["native effect\n(higher in human)", "native effect\n(higher in mouse)"]
complete_pal = {"native effect\n(higher in human)": sns.color_palette("Set2")[1],
                "native effect\n(higher in mouse)": sns.color_palette("Set2")[0]}
det_pal = {"native effect\n(higher in human)": sns.color_palette("Set2")[1],
           "native effect\n(higher in mouse)": sns.color_palette("Set2")[0]}


# In[18]:


dfs = [data_filt_sp, data_filt_tile1_sp, data_filt_tile2_sp]
titles = ["both tiles", "tile1 only", "tile2 only"]
labels = ["both_tiles", "tile1_only", "tile2_only"]


# In[19]:


for df, title, label in zip(dfs, titles, labels):
    
    fig, ax = plt.subplots(figsize=(0.75, 1.75), nrows=1, ncols=1)

    sns.countplot(data=df, x="native_status", palette=native_pal, order=native_order, linewidth=2, 
                  edgecolor=native_pal.values(), ax=ax)
    ax.set_xticklabels(["no native effect", "native effect"], va="top", ha="right", rotation=50)
    ax.set_xlabel("")
    ax.set_title(title)
    
    tot = 0
    for i, l in enumerate(native_order):
        n = len(df[df["native_status"] == l])
        tot += n
        ax.annotate(str(n), xy=(i, 2), xycoords="data", xytext=(0, 0), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color="white", size=fontsize)
    print("percent native sig: %s" % (n/tot))

    plt.show()
    fig.savefig("count_native_status.%s.pdf" % label, dpi="figure", bbox_inches="tight")
    plt.close()


# In[20]:


for df, title, label in zip(dfs, titles, labels):
    
    fig, ax = plt.subplots(figsize=(0.75, 1.75), nrows=1, ncols=1)

    sns.countplot(data=df, x="native_status_det", palette=complete_pal, order=det_order, ax=ax, linewidth=2,
                  edgecolor=det_pal.values())

    for i, l in enumerate(det_order):
        n = len(df[df["native_status_det"] == l])
        ax.annotate(str(n), xy=(i, 15), xycoords="data", xytext=(0, 0), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color="white", size=fontsize)

    ax.set_xticklabels(["higher in human", "higher in mouse"], va="top", ha="right", rotation=50)
    ax.set_xlabel("")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.savefig("count_native_status_detail.%s.pdf" % label, dpi="figure", bbox_inches="tight")


# ## 4. count of native status when looking only at CAGE

# In[21]:


det_order = ["higher in human", "higher in mouse"]
pal = {"higher in human": sns.color_palette("Set2")[1], "higher in mouse": sns.color_palette("Set2")[0]}

for df, title, label in zip(dfs, titles, labels):
    df["cage_fc"] = df.apply(get_cage_fc, axis=1)
    df["abs_cage_fc"] = np.abs(df["cage_fc"])
    df["cage_status_det"] = df.apply(cage_status_det, axis=1)
    sub = df[df["cage_status_det"] != "not sig"]
    
    fig, ax = plt.subplots(figsize=(0.75, 1.75), nrows=1, ncols=1)

    sns.countplot(data=sub, x="cage_status_det", palette=pal, order=det_order, linewidth=2,
                  edgecolor=pal.values(), ax=ax)

    for i, l in enumerate(det_order):
        n = len(df[df["cage_status_det"] == l])
        ax.annotate(str(n), xy=(i, 15), xycoords="data", xytext=(0, 0), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color="white", size=fontsize)

    ax.set_xticklabels(["higher in human", "higher in mouse"], va="top", ha="right", rotation=50)
    ax.set_xlabel("")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.savefig("count_cage_status_detail.%s.pdf" % label, dpi="figure", bbox_inches="tight")


# ## 5. effect sizes across biotypes

# In[22]:


min_switch_order = ["CAGE turnover - eRNA", "CAGE turnover - lncRNA", "CAGE turnover - mRNA", 
                    "eRNA", "lncRNA", "mRNA"]


# In[23]:


for df, title, label in zip(dfs, titles, labels):
    df["abs_logFC_native"] = np.abs(df["logFC_native"])

    fig = plt.figure(figsize=(2.5, 2))
    ax = sns.boxplot(data=df, x="biotype_switch_minimal", y="abs_logFC_native", 
                     flierprops = dict(marker='o', markersize=5), 
                     order=min_switch_order, color=sns.color_palette("Set2")[2])
    mimic_r_boxplot(ax)

    ax.set_xticklabels(min_switch_order, rotation=50, ha='right', va='top')
    ax.set_xlabel("")
    ax.set_ylabel("native effect size")
    ax.set_title(title)

    for i, l in enumerate(min_switch_order):
        sub = df[df["biotype_switch_minimal"] == l]
        n = len(sub)
        print("%s median eff size: %s" % (l, sub["abs_logFC_native"].median()))
        color = sns.color_palette("Set2")[2]
        ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color=color, size=fontsize)

    ax.set_ylim((-0.8, 4.5))
    ax.axvline(x=2.5, linestyle="dashed", color="black")

    plt.show()
    fig.savefig("native_minimal_biotype_switch_effectsize_boxplot.%s.pdf" % label, dpi="figure", bbox_inches="tight")
    plt.close()


# In[24]:


def turnover_status(row):
    if "CAGE turnover" in row["biotype_switch_minimal"]:
        return "CAGE turnover"
    else:
        return "none"
    
def turnover_biotype(row):
    if "CAGE turnover" in row["biotype_switch_minimal"]:
        return row["biotype_switch_minimal"].split(" - ")[1]
    else:
        return row["biotype_switch_minimal"]


# In[25]:


turnover_order = ["eRNA", "lncRNA", "mRNA"]
turnover_pal = {"CAGE turnover": "gray", "none": sns.color_palette("Set2")[2]}
hue_order = ["CAGE turnover", "none"]


# In[26]:


for df, title, label in zip(dfs, titles, labels):
    df["abs_logFC_native"] = np.abs(df["logFC_native"])
    df["turnover_status"] = df.apply(turnover_status, axis=1)
    df["turnover_biotype"] = df.apply(turnover_biotype, axis=1)
    
    fig = plt.figure(figsize=(2.5, 2))
    ax = sns.boxplot(data=df, x="turnover_biotype", y="abs_logFC_native", hue="turnover_status",
                     flierprops = dict(marker='o', markersize=5), 
                     order=turnover_order, hue_order=hue_order, palette=turnover_pal)
    mimic_r_boxplot(ax)

    ax.set_xticklabels(turnover_order, rotation=50, ha='right', va='top')
    ax.set_xlabel("")
    ax.set_ylabel("native effect size")
    ax.set_title(title)

    for i, l in enumerate(turnover_order):
        sub = df[df["turnover_biotype"] == l]
        dist1 = np.asarray(sub[sub["turnover_status"] == "CAGE turnover"]["abs_logFC_native"])
        dist2 = np.asarray(sub[sub["turnover_status"] == "none"]["abs_logFC_native"])
        
        dist1 = dist1[~np.isnan(dist1)]
        dist2 = dist2[~np.isnan(dist2)]
        
        u, pval = stats.mannwhitneyu(dist1, dist2, alternative="greater", use_continuity=False)
        
        if pval >= 0.05:
            annotate_pval(ax, i-0.1, i+0.1, 2.4, 0, 2.4, pval, fontsize)
        else:
            annotate_pval(ax, i-0.1, i+0.1, 2.4, 0, 2.2, pval, fontsize)
            
        ax.annotate(str(len(dist1)), xy=(i-0.25, -0.7), xycoords="data", xytext=(0, 0), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color="gray", size=fontsize)
        ax.annotate(str(len(dist2)), xy=(i+0.25, -0.7), xycoords="data", xytext=(0, 0), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color=sns.color_palette("Set2")[2], size=fontsize)

    ax.set_ylim((-0.8, 5))
    plt.legend(loc=2, bbox_to_anchor=(1.1, 1))
    plt.show()
#     fig.savefig("native_minimal_biotype_switch_effectsize_boxplot.%s.pdf" % label, dpi="figure", bbox_inches="tight")
    plt.close()


# ## 6. number sig across biotypes

# In[27]:


for df, title, label in zip(dfs, titles, labels):

    tots = df.groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
    sig = df[df["native_status"] != "no native effect"].groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
    clean_sig = tots.merge(sig, on="biotype_switch_minimal", how="left").fillna(0)
    clean_sig["percent_sig"] = (clean_sig["hg19_id_y"]/clean_sig["hg19_id_x"])*100
    
    fig = plt.figure(figsize=(2.5, 1.5))
    ax = sns.barplot(data=clean_sig, x="biotype_switch_minimal", y="percent_sig", 
                     order=min_switch_order, color=sns.color_palette("Set2")[2])

    ax.set_xticklabels(min_switch_order, rotation=50, ha='right', va='top')
    ax.set_xlabel("")
    ax.set_ylabel("% of seq. pairs with\nnative effects")
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
    fig.savefig("perc_sig_native_minimal_biotype_switch.%s.pdf" % label, dpi="figure", bbox_inches="tight")
    plt.close()


# ## 7. direc v. comp in native status

# In[28]:


for df, title, label in zip(dfs, titles, labels):
    res = {}
    native = df[df["native_status"] != "no native effect"]
    no_native = df[df["native_status"] == "no native effect"]
    
    native_cis_trans = native[(native["cis_status_one"] == "significant cis effect") & 
                              (native["trans_status_one"] == "significant trans effect")]
    no_native_cis_trans = no_native[(no_native["cis_status_one"] == "significant cis effect") & 
                                    (no_native["trans_status_one"] == "significant trans effect")]
    
    native_direc = native_cis_trans[((native_cis_trans["cis_status_det_one"].str.contains("higher in human") & 
                                      native_cis_trans["trans_status_det_one"].str.contains("higher in human")) |
                                     (native_cis_trans["cis_status_det_one"].str.contains("higher in mouse") &
                                      native_cis_trans["trans_status_det_one"].str.contains("higher in mouse")))]
    no_native_direc = no_native_cis_trans[((no_native_cis_trans["cis_status_det_one"].str.contains("higher in human") & 
                                      no_native_cis_trans["trans_status_det_one"].str.contains("higher in human")) |
                                     (no_native_cis_trans["cis_status_det_one"].str.contains("higher in mouse") &
                                      no_native_cis_trans["trans_status_det_one"].str.contains("higher in mouse")))]
    
    res["perc_native_direc"] = (len(native_direc)/len(native_cis_trans))*100
    res["perc_no_native_direc"] = (len(no_native_direc)/len(no_native_cis_trans)*100)
    res = pd.DataFrame.from_dict(res, orient="index").reset_index()
            
    order = ["perc_no_native_direc", "perc_native_direc"]
    pal = {"perc_no_native_direc": "gray", "perc_native_direc": sns.color_palette("Set2")[2]}
    fig, ax = plt.subplots(figsize=(1, 2), nrows=1, ncols=1)
    
    sns.barplot(data=res, x="index", y=0, order=order, palette=pal, ax=ax)
    ax.set_xticklabels(["no native effects", "native effects"], rotation=50, ha="right", va="top")
    ax.set_xlabel("")
    ax.set_ylabel("% of pairs with cis & trans effects\nshowing directional effects")
    plt.show()
    fig.savefig("native_cis_trans_breakup.%s.pdf" % label, dpi="figure", bbox_inches="tight")
    plt.close()

