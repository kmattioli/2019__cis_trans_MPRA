
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


def cis_trans_status(row):
    if row.cis_status_one == "significant cis effect":
        if row.trans_status_one == "significant trans effect":
            if "higher in human" in row.cis_status_det_one:
                if "higher in human" in row.trans_status_det_one:
                    return "cis/trans directional"
                else:
                    return "cis/trans compensatory"
            else:
                if "higher in human" in row.trans_status_det_one:
                    return "cis/trans compensatory"
                else:
                    return "cis/trans directional"
        else:
            return "cis effect only"
    else:
        if row.trans_status_one == "significant trans effect":
            return "trans effect only"
        else:
            return "no cis or trans effects"


# ## variables

# In[6]:


data_f = "../../../data/02__mpra/03__results/all_processed_results.txt"


# ## 1. import data

# In[7]:


data = pd.read_table(data_f, sep="\t")
data.head()


# In[8]:


data["cis_trans_status"] = data.apply(cis_trans_status, axis=1)


# ## 2. filter data

# In[9]:


data = data[~pd.isnull(data["minimal_biotype_hg19"])]
len(data)


# In[10]:


data_filt = data[((data["HUES64_padj_hg19"] < QUANT_ALPHA) | (data["mESC_padj_mm9"] < QUANT_ALPHA))]
len(data_filt)


# In[11]:


data_filt_sp = data_filt.drop("orig_species", axis=1)
data_filt_sp.drop_duplicates(inplace=True)
len(data_filt_sp)


# In[12]:


data_filt_sp.cis_trans_status.value_counts()


# In[13]:


# fisher's exact to see if cis/trans effects are enriched
cis_trans = len(data_filt_sp[data_filt_sp["cis_trans_status"].isin(["cis/trans directional", "cis/trans compensatory"])])
cis_no_trans = len(data_filt_sp[data_filt_sp["cis_trans_status"] == "cis effect only"])
trans_no_cis = len(data_filt_sp[data_filt_sp["cis_trans_status"] == "trans effect only"])
n_no_cis_trans = len(data_filt_sp[data_filt_sp["cis_trans_status"] == "no cis or trans effects"])

# fisher's exact test
arr = np.zeros((2, 2))
arr[0, 0] = cis_trans
arr[0, 1] = cis_no_trans
arr[1, 0] = trans_no_cis
arr[1, 1] = n_no_cis_trans
print(arr)

odds, p = stats.fisher_exact(arr)
print(odds)
print(p)


# In[14]:


stats.binom_test(95, 159)


# In[15]:


# data_filt_tile1 = data_filt[data_filt["tss_tile_num"] == "tile1"]
# len(data_filt_tile1)


# In[16]:


# data_filt_tile1_sp = data_filt_sp[data_filt_sp["tss_tile_num"] == "tile1"]
# len(data_filt_tile1_sp)


# In[17]:


# data_filt_tile2 = data_filt[data_filt["tss_tile_num"] == "tile2"]
# len(data_filt_tile2)


# In[18]:


# data_filt_tile2_sp = data_filt_sp[data_filt_sp["tss_tile_num"] == "tile2"]
# len(data_filt_tile2_sp)


# ## count of cis/trans/both

# In[19]:


len(data_filt_sp)


# In[20]:


len(data_filt_sp[(data_filt_sp["cis_status_one"] != "no cis effect") & (data_filt_sp["trans_status_one"] == "no trans effect")])


# In[21]:


len(data_filt_sp[(data_filt_sp["cis_status_one"] == "no cis effect") & (data_filt_sp["trans_status_one"] != "no trans effect")])


# In[22]:


len(data_filt_sp[(data_filt_sp["cis_status_one"] != "no cis effect") & (data_filt_sp["trans_status_one"] != "no trans effect")])


# ## 3. count of cis/trans interactions

# In[23]:


int_order = ["no cis/trans int. effect", "significant cis/trans int. effect"]
int_pal = {"no cis/trans int. effect": "gray", "significant cis/trans int. effect": "black"}


# In[24]:


# dfs = [data_filt_sp, data_filt_tile1_sp, data_filt_tile2_sp]
# titles = ["both tiles", "tile1 only", "tile2 only"]
# labels = ["both_tiles", "tile1_only", "tile2_only"]


# In[25]:


# for df, title, label in zip(dfs, titles, labels):
    
#     fig, ax = plt.subplots(figsize=(0.75, 1.75), nrows=1, ncols=1)

#     sns.countplot(data=df, x="cis_trans_int_status", palette=int_pal, order=int_order, linewidth=2, 
#                   edgecolor=int_pal.values(), ax=ax)
#     ax.set_xticklabels(["no cis/trans interaction", "cis/trans interaction"], va="top", ha="right", rotation=50)
#     ax.set_xlabel("")
#     ax.set_title(title)
    
#     tot = 0
#     colors = ["white", "black"]
#     for i, l in enumerate(int_order):
#         n = len(df[df["cis_trans_int_status"] == l])
#         tot += n
#         ax.annotate(str(n), xy=(i, 40), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color=colors[i], size=fontsize)
#     print("percent cis/trans sig: %s" % (n/tot))

#     plt.show()
#     fig.savefig("count_cistrans_status.%s.pdf" % label, dpi="figure", bbox_inches="tight")
#     plt.close()


# In[26]:


df = data_filt_sp


# In[27]:


fig, ax = plt.subplots(figsize=(0.75, 1.75), nrows=1, ncols=1)

sns.countplot(data=df, x="cis_trans_int_status", palette=int_pal, order=int_order, linewidth=2, 
              edgecolor=int_pal.values(), ax=ax)
ax.set_xticklabels(["no cis/trans interaction", "cis/trans interaction"], va="top", ha="right", rotation=50)
ax.set_xlabel("")

tot = 0
colors = ["white", "black"]
for i, l in enumerate(int_order):
    n = len(df[df["cis_trans_int_status"] == l])
    tot += n
    ax.annotate(str(n), xy=(i, 50), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=colors[i], size=fontsize)
print("percent cis/trans sig: %s" % (n/tot))

plt.show()
fig.savefig("count_cistrans_status.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# ## 5. effect size differences across biotypes

# In[28]:


min_switch_order = ["CAGE turnover - eRNA", "CAGE turnover - lncRNA", "CAGE turnover - mRNA", 
                    "eRNA", "lncRNA", "mRNA"]
min_switch_pal = {"CAGE turnover - eRNA": sns.color_palette("Set2")[2], 
                  "CAGE turnover - lncRNA": sns.color_palette("Set2")[2],
                  "CAGE turnover - mRNA": sns.color_palette("Set2")[2],
                  "eRNA": sns.color_palette("Set2")[7], 
                  "lncRNA": sns.color_palette("Set2")[7], 
                  "mRNA": sns.color_palette("Set2")[7]}


# In[29]:


# for df, title, label in zip(dfs, titles, labels):
#     df["abs_logFC_int"] = np.abs(df["logFC_int"])

#     fig = plt.figure(figsize=(2.5, 1.5))
#     ax = sns.boxplot(data=df, x="biotype_switch_minimal", y="abs_logFC_int", 
#                      flierprops = dict(marker='o', markersize=5), 
#                      order=min_switch_order, palette=min_switch_pal)
#     mimic_r_boxplot(ax)

#     ax.set_xticklabels(["eRNA", "lncRNA", "mRNA", "eRNA", "lncRNA", "mRNA"], rotation=50, ha='right', va='top')
#     ax.set_xlabel("")
#     ax.set_ylabel("cis/trans interaction\neffect size")
#     ax.set_title(title)

#     for i, l in enumerate(min_switch_order):
#         sub = df[df["biotype_switch_minimal"] == l]
#         n = len(sub)
#         print("%s median eff size: %s" % (l, sub["abs_logFC_int"].median()))
#         color = min_switch_pal[l]
#         ax.annotate(str(n), xy=(i, -0.3), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color=color, size=fontsize)
        
#     ### pvals ###
#     vals1 = np.asarray(df[df["biotype_switch_minimal"] == "CAGE turnover - eRNA"]["abs_logFC_int"])
#     vals2 = np.asarray(df[df["biotype_switch_minimal"] == "CAGE turnover - lncRNA"]["abs_logFC_int"])
#     vals3 = np.asarray(df[df["biotype_switch_minimal"] == "CAGE turnover - mRNA"]["abs_logFC_int"])
#     vals4 = np.asarray(df[df["biotype_switch_minimal"] == "eRNA"]["abs_logFC_int"])
#     vals5 = np.asarray(df[df["biotype_switch_minimal"] == "lncRNA"]["abs_logFC_int"])
#     vals6 = np.asarray(df[df["biotype_switch_minimal"] == "mRNA"]["abs_logFC_int"])
    
#     vals1 = vals1[~np.isnan(vals1)]
#     vals2 = vals2[~np.isnan(vals2)]
#     vals3 = vals3[~np.isnan(vals3)]
#     vals4 = vals4[~np.isnan(vals4)]
#     vals5 = vals5[~np.isnan(vals5)]
#     vals6 = vals6[~np.isnan(vals6)]
    
#     _, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)
#     _, pval13 = stats.mannwhitneyu(vals1, vals3, alternative="two-sided", use_continuity=False)
#     _, pval23 = stats.mannwhitneyu(vals2, vals3, alternative="two-sided", use_continuity=False)
#     _, pval45 = stats.mannwhitneyu(vals4, vals5, alternative="two-sided", use_continuity=False)
#     _, pval46 = stats.mannwhitneyu(vals4, vals6, alternative="two-sided", use_continuity=False)
#     _, pval56 = stats.mannwhitneyu(vals5, vals6, alternative="two-sided", use_continuity=False)
    
#     print(pval12)
#     print(pval13)
#     print(pval23)
#     print(pval45)
#     print(pval46)
#     print(pval56)
    
#     annotate_pval(ax, 0.2, 0.8, 1.25, 0, 1.25, pval12, fontsize-1)
#     annotate_pval(ax, 1.2, 1.8, 1.25, 0, 1.25, pval13, fontsize-1)
#     annotate_pval(ax, 0, 2, 1.75, 0, 1.75, pval23, fontsize-1)
#     annotate_pval(ax, 3.2, 3.8, 1.25, 0, 1.25, pval45, fontsize-1)
#     annotate_pval(ax, 4.2, 4.8, 1.25, 0, 1.25, pval56, fontsize-1)
#     annotate_pval(ax, 3, 5, 1.75, 0, 1.75, pval46, fontsize-1)

#     ax.set_ylim((-0.4, 2))
#     ax.axvline(x=2.5, linestyle="dashed", color="black")

#     plt.show()
#     fig.savefig("cistrans_minimal_biotype_switch_effectsize_boxplot.%s.pdf" % label, dpi="figure", bbox_inches="tight")
#     plt.close()


# In[30]:


df["abs_logFC_int"] = np.abs(df["logFC_int"])

fig = plt.figure(figsize=(2.5, 1.5))
ax = sns.boxplot(data=df, x="biotype_switch_minimal", y="abs_logFC_int", 
                 flierprops = dict(marker='o', markersize=5), 
                 order=min_switch_order, palette=min_switch_pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(["eRNA", "lncRNA", "mRNA", "eRNA", "lncRNA", "mRNA"], rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("cis/trans interaction\neffect size")

for i, l in enumerate(min_switch_order):
    sub = df[df["biotype_switch_minimal"] == l]
    n = len(sub)
    print("%s median eff size: %s" % (l, sub["abs_logFC_int"].median()))
    color = min_switch_pal[l]
    ax.annotate(str(n), xy=(i, -0.3), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

### pvals ###
vals1 = np.asarray(df[df["biotype_switch_minimal"] == "CAGE turnover - eRNA"]["abs_logFC_int"])
vals2 = np.asarray(df[df["biotype_switch_minimal"] == "CAGE turnover - lncRNA"]["abs_logFC_int"])
vals3 = np.asarray(df[df["biotype_switch_minimal"] == "CAGE turnover - mRNA"]["abs_logFC_int"])
vals4 = np.asarray(df[df["biotype_switch_minimal"] == "eRNA"]["abs_logFC_int"])
vals5 = np.asarray(df[df["biotype_switch_minimal"] == "lncRNA"]["abs_logFC_int"])
vals6 = np.asarray(df[df["biotype_switch_minimal"] == "mRNA"]["abs_logFC_int"])

vals1 = vals1[~np.isnan(vals1)]
vals2 = vals2[~np.isnan(vals2)]
vals3 = vals3[~np.isnan(vals3)]
vals4 = vals4[~np.isnan(vals4)]
vals5 = vals5[~np.isnan(vals5)]
vals6 = vals6[~np.isnan(vals6)]

_, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)
_, pval13 = stats.mannwhitneyu(vals1, vals3, alternative="two-sided", use_continuity=False)
_, pval23 = stats.mannwhitneyu(vals2, vals3, alternative="two-sided", use_continuity=False)
_, pval45 = stats.mannwhitneyu(vals4, vals5, alternative="two-sided", use_continuity=False)
_, pval46 = stats.mannwhitneyu(vals4, vals6, alternative="two-sided", use_continuity=False)
_, pval56 = stats.mannwhitneyu(vals5, vals6, alternative="two-sided", use_continuity=False)

print(pval12)
print(pval13)
print(pval23)
print(pval45)
print(pval46)
print(pval56)

annotate_pval(ax, 0.2, 0.8, 1.25, 0, 1.25, pval12, fontsize-1)
annotate_pval(ax, 1.2, 1.8, 1.25, 0, 1.25, pval13, fontsize-1)
annotate_pval(ax, 0, 2, 1.75, 0, 1.75, pval23, fontsize-1)
annotate_pval(ax, 3.2, 3.8, 1.25, 0, 1.25, pval45, fontsize-1)
annotate_pval(ax, 4.2, 4.8, 1.25, 0, 1.25, pval56, fontsize-1)
annotate_pval(ax, 3, 5, 1.75, 0, 1.75, pval46, fontsize-1)

ax.set_ylim((-0.4, 2))
ax.axvline(x=2.5, linestyle="dashed", color="black")

plt.show()
fig.savefig("cistrans_minimal_biotype_switch_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# In[31]:


def cage_status(row):
    if "CAGE turnover" in row.biotype_switch_minimal:
        return "turnover"
    else:
        return "conserved"


# In[32]:


def one_biotype(row):
    if row.minimal_biotype_hg19 == "no CAGE activity":
        return row.minimal_biotype_mm9
    elif row.biotype_switch_minimal == "biotype switch":
        return "biotype switch"
    else:
        return row.minimal_biotype_hg19


# In[33]:


pal = {"conserved": sns.color_palette("Set2")[7], "turnover": sns.color_palette("Set2")[2]}


# In[34]:


# for df, title, pltname in zip(dfs, titles, labels):
#     df["abs_logFC_int"] = np.abs(df["logFC_int"])
#     df["cage_status"] = df.apply(cage_status, axis=1)
#     df["one_biotype"] = df.apply(one_biotype, axis=1)
    
#     fig = plt.figure(figsize=(2.75, 1.5))

#     ax = sns.boxplot(data=df, x="one_biotype", y="abs_logFC_int", hue="cage_status",
#                      flierprops = dict(marker='o', markersize=5),
#                      order=["eRNA", "lncRNA", "mRNA"], hue_order=["turnover", "conserved"], palette=pal)
#     mimic_r_boxplot(ax)

#     ax.set_xticklabels(["eRNA", "lncRNA", "mRNA"], rotation=50, ha='right', va='top')
#     ax.set_xlabel("")
#     ax.set_ylabel("trans effect size")
#     plt.legend(loc=2, bbox_to_anchor=(1.05, 1))

#     ys = [0.75, 0.75, 0.75]
#     for i, label in enumerate(["eRNA", "lncRNA", "mRNA"]):
#         sub = df[df["one_biotype"] == label]
#         sub1 = sub[sub["cage_status"] == "turnover"]
#         sub2 = sub[sub["cage_status"] == "conserved"]

#         vals1 = np.asarray(sub1["abs_logFC_int"])
#         vals2 = np.asarray(sub2["abs_logFC_int"])

#         vals1 = vals1[~np.isnan(vals1)]
#         vals2 = vals2[~np.isnan(vals2)]

#         u, pval = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)
#         print(pval)

#         if pval >= 0.05:
#             annotate_pval(ax, i-0.1, i+0.1, ys[i], 0, ys[i], pval, fontsize-1)
#         else:
#             annotate_pval(ax, i-0.1, i+0.1, ys[i], 0, ys[i], pval, fontsize-1)

#         n1 = len(vals1)
#         n2 = len(vals2)

#         ax.annotate(str(n1), xy=(i-0.2, -0.3), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color=pal["turnover"], size=fontsize)
#         ax.annotate(str(n2), xy=(i+0.2, -0.3), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color=pal["conserved"], size=fontsize)

#     ax.set_ylim((-0.4, 2))
#     ax.set_title(title)
#     fig.savefig("cistrans_effect_biotype_sep_cage.%s.pdf" % pltname, dpi="figure", bbox_inches="tight")


# In[35]:


df["abs_logFC_int"] = np.abs(df["logFC_int"])
df["cage_status"] = df.apply(cage_status, axis=1)
df["one_biotype"] = df.apply(one_biotype, axis=1)

fig = plt.figure(figsize=(2.75, 1.5))

ax = sns.boxplot(data=df, x="one_biotype", y="abs_logFC_int", hue="cage_status",
                 flierprops = dict(marker='o', markersize=5),
                 order=["eRNA", "lncRNA", "mRNA"], hue_order=["turnover", "conserved"], palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(["eRNA", "lncRNA", "mRNA"], rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("trans effect size")
plt.legend(loc=2, bbox_to_anchor=(1.05, 1))

ys = [0.75, 0.75, 0.75]
for i, label in enumerate(["eRNA", "lncRNA", "mRNA"]):
    sub = df[df["one_biotype"] == label]
    sub1 = sub[sub["cage_status"] == "turnover"]
    sub2 = sub[sub["cage_status"] == "conserved"]

    vals1 = np.asarray(sub1["abs_logFC_int"])
    vals2 = np.asarray(sub2["abs_logFC_int"])

    vals1 = vals1[~np.isnan(vals1)]
    vals2 = vals2[~np.isnan(vals2)]

    u, pval = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)
    print(pval)

    if pval >= 0.05:
        annotate_pval(ax, i-0.1, i+0.1, ys[i], 0, ys[i], pval, fontsize-1)
    else:
        annotate_pval(ax, i-0.1, i+0.1, ys[i], 0, ys[i], pval, fontsize-1)

    n1 = len(vals1)
    n2 = len(vals2)

    ax.annotate(str(n1), xy=(i-0.2, -0.3), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=pal["turnover"], size=fontsize)
    ax.annotate(str(n2), xy=(i+0.2, -0.3), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=pal["conserved"], size=fontsize)

ax.set_ylim((-0.4, 2))
fig.savefig("cistrans_effect_biotype_sep_cage.pdf", dpi="figure", bbox_inches="tight")


# ## 6. percent sig across biotypes

# In[36]:


# for df, title, label in zip(dfs, titles, labels):

#     tots = df.groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
#     sig = df[df["cis_trans_int_status"] != "no cis/trans int. effect"].groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
#     clean_sig = tots.merge(sig, on="biotype_switch_minimal", how="left").fillna(0)
#     clean_sig["percent_sig"] = (clean_sig["hg19_id_y"]/clean_sig["hg19_id_x"])*100
    
#     fig = plt.figure(figsize=(2.5, 1.5))
#     ax = sns.barplot(data=clean_sig, x="biotype_switch_minimal", y="percent_sig", 
#                      order=min_switch_order, color=sns.color_palette("Set2")[2])

#     ax.set_xticklabels(["eRNA", "lncRNA", "mRNA", "eRNA", "lncRNA", "mRNA"], rotation=50, ha='right', va='top')
#     ax.set_xlabel("")
#     ax.set_ylabel("% of seq. pairs with\ncis/trans interactions")
#     ax.set_title(title)
#     ax.axvline(x=2.5, linestyle="dashed", color="black")
    
#     for i, l in enumerate(min_switch_order):
#         sub = clean_sig[clean_sig["biotype_switch_minimal"] == l]
#         print("%s perc sig: %s" % (l, sub["percent_sig"].iloc[0]))
#         n = sub["hg19_id_x"].iloc[0]
#         ax.annotate(str(n), xy=(i, 0.5), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color="white", size=fontsize)
    
#     plt.show()
#     fig.savefig("perc_sig_cistrans_minimal_biotype_switch.%s.pdf" % label, dpi="figure", bbox_inches="tight")
#     plt.close()


# In[37]:


tots = df.groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
sig = df[df["cis_trans_int_status"] != "no cis/trans int. effect"].groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
clean_sig = tots.merge(sig, on="biotype_switch_minimal", how="left").fillna(0)
clean_sig["percent_sig"] = (clean_sig["hg19_id_y"]/clean_sig["hg19_id_x"])*100

fig = plt.figure(figsize=(2.5, 1.5))
ax = sns.barplot(data=clean_sig, x="biotype_switch_minimal", y="percent_sig", 
                 order=min_switch_order, color=sns.color_palette("Set2")[2])

ax.set_xticklabels(["eRNA", "lncRNA", "mRNA", "eRNA", "lncRNA", "mRNA"], rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("% of seq. pairs with\ncis/trans interactions")
ax.axvline(x=2.5, linestyle="dashed", color="black")

for i, l in enumerate(min_switch_order):
    sub = clean_sig[clean_sig["biotype_switch_minimal"] == l]
    print("%s perc sig: %s" % (l, sub["percent_sig"].iloc[0]))
    n = sub["hg19_id_x"].iloc[0]
    ax.annotate(str(n), xy=(i, 0.5), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color="white", size=fontsize)

plt.show()
fig.savefig("perc_sig_cistrans_minimal_biotype_switch.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# ## 7. look generally at significant interactions

# In[38]:


# for df, title, label in zip(dfs, titles, labels):
    
#     # plot effect size agreement b/w the two cells
#     fig, ax = plt.subplots(figsize=(1.75, 1.75), nrows=1, ncols=1)

#     sig_int = df[df["cis_trans_int_status"] != "no cis/trans int. effect"]
#     not_sig_int = df[df["cis_trans_int_status"] == "no cis/trans int. effect"]

#     ax.scatter(not_sig_int["logFC_cis_HUES64"], not_sig_int["logFC_cis_mESC"], s=10, alpha=0.75, 
#                color="gray", linewidths=0.5, edgecolors="white")
#     ax.scatter(sig_int["logFC_cis_HUES64"], sig_int["logFC_cis_mESC"], s=10, alpha=1, 
#                color=sns.color_palette("Set2")[3], linewidths=0.5, edgecolors="white")

#     plt.xlabel("cis effect size in hESCs")
#     plt.ylabel("cis effect size in mESCs")

#     ax.axhline(y=0, color="black", linestyle="dashed")
#     ax.axvline(x=0, color="black", linestyle="dashed")
#     ax.set_xlim((-6, 6))
#     ax.set_ylim((-6, 6))

#     # annotate corr
#     no_nan = df[(~pd.isnull(df["logFC_cis_HUES64"])) & (~pd.isnull(df["logFC_cis_mESC"]))]
#     r, p = spearmanr(no_nan["logFC_cis_HUES64"], no_nan["logFC_cis_mESC"])
#     ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
#             transform=ax.transAxes)
#     ax.text(0.05, 0.90, "n = %s" % (len(no_nan)), ha="left", va="top", fontsize=fontsize,
#             transform=ax.transAxes)
#     plt.show()
#     fig.savefig("cis_effect_bw_cells_scatter.sig_status_color.%s.pdf" % label, dpi="figure", bbox_inches="tight")
#     plt.close()


# In[39]:


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
fig.savefig("cis_effect_bw_cells_scatter.sig_status_color.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# ## 8. look at highest cis/trans interactions

# In[40]:


sig_int = df[df["cis_trans_int_status"] != "no cis/trans int. effect"]
len(sig_int)


# In[41]:


# sig_int_filt = sig_int[((sig_int["logFC_cis_HUES64"] < 0) & (sig_int["logFC_cis_mESC"] > 0)) |
#                        ((sig_int["logFC_cis_HUES64"] > 0) & (sig_int["logFC_cis_mESC"] < 0))]
sig_int_filt = sig_int
len(sig_int_filt)


# In[42]:


sub = sig_int_filt[["hg19_id", "mm9_id", "biotype_hg19", "biotype_mm9", "biotype_switch_minimal", "logFC_int", "logFC_cis_HUES64", "logFC_cis_mESC",
                    "HUES64_hg19", "mESC_hg19", "HUES64_mm9", "mESC_mm9", "cis_trans_status"]]
sub


# In[43]:


pal = {"hg19": sns.color_palette("Set2")[1], "mm9": sns.color_palette("Set2")[0]}


# In[44]:


for row in sub.iterrows():
    samp = pd.DataFrame(row[1]).T
    melt = pd.melt(samp, id_vars=["hg19_id", "mm9_id", "biotype_hg19", "biotype_mm9", "biotype_switch_minimal", 
                                  "logFC_int", "logFC_cis_HUES64", "logFC_cis_mESC", "cis_trans_status"])
    melt["cell"] = melt["variable"].str.split("_", expand=True)[0]
    melt["seq"] = melt["variable"].str.split("_", expand=True)[1]
    
    fig = plt.figure(figsize=(1.5, 1.5))
    ax = sns.barplot(data=melt, x="cell", hue="seq", y="value", palette=pal)
    ax.set_ylabel("MPRA activity")
    ax.set_xlabel("")
    ax.set_xticklabels(["hESCs", "mESCs"], rotation=50, ha='right', va='top')
    ax.set_title("human ID: %s, human biotype: %s\nmouse ID: %s, mouse biotype:%s\nSTATUS:%s" % (row[1].hg19_id, 
                                                                                      row[1].biotype_hg19, 
                                                                                      row[1].mm9_id,
                                                                                      row[1].biotype_mm9,
                                                                                                 row[1].cis_trans_status))
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
    plt.show()
    plt.close()
#     fig.savefig("%s.%s.barplot.pdf" % (row[1].hg19_id, row[1].mm9_id), dpi="figure", bbox_inches="tight")


# examples:
# - compensatory: h.1433/m.1276, 
# - directional: h.321/m.207, h.554/m.440, h.1389/m.1247

# In[45]:


# for row in sub.head(3).iterrows():
#     arr = np.zeros((2, 2))
#     arr[0, 0] = row[1].HUES64_hg19
#     arr[0, 1] = row[1].HUES64_mm9
#     arr[1, 0] = row[1].mESC_hg19
#     arr[1, 1] = row[1].mESC_mm9
    
#     fig, ax = plt.subplots(figsize=(2, 2), nrows=1, ncols=1)
#     sns.heatmap(arr, annot=True, cmap="Greys", cbar=False, ax=ax, linewidths=30, vmin=0, vmax=20)
    
#     ax.plot([0, 0], [1, 1], color="black")
#     plt.show()


# ## 9. look at cis/trans when subsetting by native

# In[46]:


# for df, title, label in zip(dfs, titles, labels):
    
#     fig, axarr = plt.subplots(figsize=(5.8, 2), nrows=1, ncols=3, sharex=True, sharey=True)
    
#     # all seqs
#     ax = axarr[0]
#     ax.scatter(df["logFC_cis_one"], df["logFC_trans_one"], s=10, alpha=0.75, 
#                color="gray", linewidths=0.5, edgecolors="white")

#     plt.xlabel("cis effect size")
#     plt.ylabel("trans effect size")

#     ax.axhline(y=0, color="black", linestyle="dashed")
#     ax.axvline(x=0, color="black", linestyle="dashed")

#     # annotate corr
#     no_nan = df[(~pd.isnull(df["logFC_cis_one"])) & (~pd.isnull(df["logFC_trans_one"]))]
#     r, p = spearmanr(no_nan["logFC_cis_one"], no_nan["logFC_trans_one"])
#     ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
#             transform=ax.transAxes)
#     ax.text(0.05, 0.90, "n = %s" % (len(no_nan)), ha="left", va="top", fontsize=fontsize,
#             transform=ax.transAxes)
    
#     # native effects only
#     sub = df[df["native_status"] == "significant native effect"]
#     ax = axarr[1]
#     ax.scatter(sub["logFC_cis_one"], sub["logFC_trans_one"], s=10, alpha=0.75, 
#                color="gray", linewidths=0.5, edgecolors="white")

#     plt.xlabel("cis effect size")
#     plt.ylabel("trans effect size")

#     ax.axhline(y=0, color="black", linestyle="dashed")
#     ax.axvline(x=0, color="black", linestyle="dashed")

#     # annotate corr
#     no_nan = sub[(~pd.isnull(sub["logFC_cis_one"])) & (~pd.isnull(sub["logFC_trans_one"]))]
#     r, p = spearmanr(no_nan["logFC_cis_one"], no_nan["logFC_trans_one"])
#     ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
#             transform=ax.transAxes)
#     ax.text(0.05, 0.90, "n = %s" % (len(no_nan)), ha="left", va="top", fontsize=fontsize,
#             transform=ax.transAxes)
    
#     # no native effects
#     sub = df[df["native_status"] == "no native effect"]
#     ax = axarr[2]
#     ax.scatter(sub["logFC_cis_one"], sub["logFC_trans_one"], s=10, alpha=0.75, 
#                color="gray", linewidths=0.5, edgecolors="white")

#     plt.xlabel("cis effect size")
#     plt.ylabel("trans effect size")

#     ax.axhline(y=0, color="black", linestyle="dashed")
#     ax.axvline(x=0, color="black", linestyle="dashed")

#     # annotate corr
#     no_nan = sub[(~pd.isnull(sub["logFC_cis_one"])) & (~pd.isnull(sub["logFC_trans_one"]))]
#     r, p = spearmanr(no_nan["logFC_cis_one"], no_nan["logFC_trans_one"])
#     ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
#             transform=ax.transAxes)
#     ax.text(0.05, 0.90, "n = %s" % (len(no_nan)), ha="left", va="top", fontsize=fontsize,
#             transform=ax.transAxes)
    
#     ax.set_xlabel("")
#     ax.set_ylabel("")
    
# #     fig.savefig("cis_v_trans.%s.pdf" % label, dpi="figure", bbox_inches="tight")


# In[47]:


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

#     fig.savefig("cis_v_trans.%s.pdf" % label, dpi="figure", bbox_inches="tight")


# ## 10. look at invidiual directionality of cis/trans

# In[48]:


# for df, title, label in zip(dfs, titles, labels):
#     res = {}
#     cis_trans = df[(df["cis_status_one"] == "significant cis effect") & 
#                    (df["trans_status_one"] == "significant trans effect")]
#     tots = len(cis_trans)
#     print(tots)
#     res["total"] = [tots]
#     direc = cis_trans[((cis_trans["cis_status_det_one"].str.contains("higher in human") & 
#                         cis_trans["trans_status_det_one"].str.contains("higher in human")) |
#                        (cis_trans["cis_status_det_one"].str.contains("higher in mouse") &
#                         cis_trans["trans_status_det_one"].str.contains("higher in mouse")))]    
#     direc = len(direc)
#     res["directional"] = [direc]

#     comp = cis_trans[((cis_trans["cis_status_det_one"].str.contains("higher in human") & 
#                         cis_trans["trans_status_det_one"].str.contains("higher in mouse")) |
#                        (cis_trans["cis_status_det_one"].str.contains("higher in mouse") &
#                         cis_trans["trans_status_det_one"].str.contains("higher in human")))]
#     comp = len(comp)
#     res["compensatory"] = [comp]
#     res = pd.DataFrame.from_dict(res, orient="index").reset_index()
#     res["perc"] = (res[0]/tots)*100
#     res["tmp"] = "tmp"
    
#     fig, ax = plt.subplots(figsize=(0.5, 1.5), nrows=1, ncols=1)
#     sns.barplot(data=res[res["index"] == "total"], 
#                 x="tmp", y="perc", color=sns.color_palette("Set2")[7], ax=ax)
#     sns.barplot(data=res[res["index"] == "directional"], 
#                 x="tmp", y="perc", color=sns.color_palette("Set2")[2], ax=ax)
    
#     ax.set_xlabel("")
#     ax.set_ylabel("% of sequence pairs")
#     ax.set_xticklabels(["all pairs"], rotation=50, ha="right", va="top")
    
#     ax.annotate(str(tots), xy=(0, 5), xycoords="data", xytext=(0, 0), 
#                 textcoords="offset pixels", ha='center', va='bottom', 
#                 color="white", size=fontsize)
    
#     fig.savefig("direc_v_comp.%s.pdf" % label, dpi="figure", bbox_inches="tight")


# In[49]:


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
res["tmp"] = "tmp"
print(res)

fig, ax = plt.subplots(figsize=(0.5, 1.5), nrows=1, ncols=1)
sns.barplot(data=res[res["index"] == "total"], 
            x="tmp", y="perc", color=sns.color_palette("Set2")[7], ax=ax)
sns.barplot(data=res[res["index"] == "directional"], 
            x="tmp", y="perc", color=sns.color_palette("Set2")[2], ax=ax)

ax.set_xlabel("")
ax.set_ylabel("% of sequence pairs")
ax.set_xticklabels(["all pairs"], rotation=50, ha="right", va="top")

ax.annotate(str(tots), xy=(0, 5), xycoords="data", xytext=(0, 0), 
            textcoords="offset pixels", ha='center', va='bottom', 
            color="white", size=fontsize)

fig.savefig("direc_v_comp.pdf", dpi="figure", bbox_inches="tight")


# In[50]:


# for df, title, label in zip(dfs, titles, labels):

#     cis_trans = df[(df["cis_status_one"] == "significant cis effect") & 
#                    (df["trans_status_one"] == "significant trans effect")]
#     tots = cis_trans.groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
    
#     direc = cis_trans[((cis_trans["cis_status_det_one"].str.contains("higher in human") & 
#                         cis_trans["trans_status_det_one"].str.contains("higher in human")) |
#                        (cis_trans["cis_status_det_one"].str.contains("higher in mouse") &
#                         cis_trans["trans_status_det_one"].str.contains("higher in mouse")))]
#     sig = direc.groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
#     clean_sig = tots.merge(sig, on="biotype_switch_minimal", how="left").fillna(0)
#     clean_sig["percent_sig"] = (clean_sig["hg19_id_y"]/clean_sig["hg19_id_x"])*100
#     clean_sig["percent_tot"] = (clean_sig["hg19_id_x"]/clean_sig["hg19_id_x"])*100
    
#     fig = plt.figure(figsize=(2.5, 1.5))
#     ax = sns.barplot(data=clean_sig, x="biotype_switch_minimal", y="percent_tot", 
#                      order=min_switch_order, color=sns.color_palette("Set2")[7])
#     sns.barplot(data=clean_sig, x="biotype_switch_minimal", y="percent_sig", 
#                 order=min_switch_order, color=sns.color_palette("Set2")[2])

#     ax.set_xticklabels(["eRNA", "lncRNA", "mRNA", "eRNA", "lncRNA", "mRNA"], rotation=50, ha='right', va='top')
#     ax.set_xlabel("")
#     ax.set_ylabel("% of sequence pairs")
#     ax.set_title(title)
#     ax.axvline(x=2.5, linestyle="dashed", color="black")
    
#     for i, l in enumerate(min_switch_order):
#         sub = clean_sig[clean_sig["biotype_switch_minimal"] == l]
#         print("%s perc sig: %s" % (l, sub["percent_sig"].iloc[0]))
#         n = sub["hg19_id_x"].iloc[0]
#         ax.annotate(str(n), xy=(i, 5), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color="white", size=fontsize)
    
#     plt.show()
#     fig.savefig("perc_sig_compensatory_minimal_biotype_switch.%s.pdf" % label, dpi="figure", bbox_inches="tight")
#     plt.close()


# In[51]:


cis_trans = df[(df["cis_status_one"] == "significant cis effect") & 
               (df["trans_status_one"] == "significant trans effect")]
tots = cis_trans.groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()

direc = cis_trans[((cis_trans["cis_status_det_one"].str.contains("higher in human") & 
                    cis_trans["trans_status_det_one"].str.contains("higher in human")) |
                   (cis_trans["cis_status_det_one"].str.contains("higher in mouse") &
                    cis_trans["trans_status_det_one"].str.contains("higher in mouse")))]
sig = direc.groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
clean_sig = tots.merge(sig, on="biotype_switch_minimal", how="left").fillna(0)
clean_sig["percent_sig"] = (clean_sig["hg19_id_y"]/clean_sig["hg19_id_x"])*100
clean_sig["percent_tot"] = (clean_sig["hg19_id_x"]/clean_sig["hg19_id_x"])*100

fig = plt.figure(figsize=(2.5, 1.5))
ax = sns.barplot(data=clean_sig, x="biotype_switch_minimal", y="percent_tot", 
                 order=min_switch_order, color=sns.color_palette("Set2")[7])
sns.barplot(data=clean_sig, x="biotype_switch_minimal", y="percent_sig", 
            order=min_switch_order, color=sns.color_palette("Set2")[2])

ax.set_xticklabels(["eRNA", "lncRNA", "mRNA", "eRNA", "lncRNA", "mRNA"], rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("% directional effects")
ax.axvline(x=2.5, linestyle="dashed", color="black")

for i, l in enumerate(min_switch_order):
    sub = clean_sig[clean_sig["biotype_switch_minimal"] == l]
    print("%s perc sig: %s" % (l, sub["percent_sig"].iloc[0]))
    n = sub["hg19_id_x"].iloc[0]
    ax.annotate(str(n), xy=(i, 5), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color="white", size=fontsize)

plt.show()
fig.savefig("perc_sig_compensatory_minimal_biotype_switch.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# In[52]:


cis_trans_order = ["cis/trans compensatory", "cis/trans directional"]
cis_trans_pal = {"cis/trans compensatory": sns.color_palette("Set2")[7], 
                 "cis/trans directional": sns.color_palette("Set2")[2]}
df["abs_logFC_native"] = np.abs(df["logFC_native"])

fig = plt.figure(figsize=(1, 1.75))
ax = sns.boxplot(data=df, x="cis_trans_status", y="abs_logFC_native", 
                 flierprops = dict(marker='o', markersize=5), 
                 order=cis_trans_order, palette=cis_trans_pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(["compensatory", "directional"], rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel(r'$\vert$ native effect size $\vert$')

for i, l in enumerate(cis_trans_order):
    sub = df[df["cis_trans_status"] == l]
    n = len(sub)
    color = cis_trans_pal[l]
    ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)
    
sub1 = df[df["cis_trans_status"] == "cis/trans compensatory"]
sub2 = df[df["cis_trans_status"] == "cis/trans directional"]
    
vals1 = np.asarray(sub1["abs_logFC_native"])
vals2 = np.asarray(sub2["abs_logFC_native"])

vals1 = vals1[~np.isnan(vals1)]
vals2 = vals2[~np.isnan(vals2)]

u, pval = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)
annotate_pval(ax, 0, 1, 5, 0, 5, pval, fontsize-1)
ax.set_ylim((-0.8, 6))
fig.savefig("native_effect_comp_v_dir.pdf", dpi="figure", bbox_inches="tight")


# ## 11. plot some examples

# ### compensatory

# In[53]:


ex = df[df["hg19_id"] == "h.1433"]
ex = ex[["hg19_id", "mm9_id", "minimal_biotype_hg19", "minimal_biotype_mm9", "HUES64_hg19", "HUES64_mm9",
         "mESC_hg19", "mESC_mm9", "trans_human_status_det", "fdr_trans_human", "trans_mouse_status_det", 
         "fdr_trans_mouse", "cis_HUES64_status_det", "fdr_cis_HUES64", "cis_mESC_status_det", "fdr_cis_mESC",
         "logFC_trans_human", "logFC_trans_mouse", "logFC_cis_HUES64", "logFC_cis_mESC"]]
ex


# In[54]:


ex = pd.melt(ex, id_vars=["hg19_id", "mm9_id", "minimal_biotype_hg19", "minimal_biotype_mm9"])
ex = ex[ex["variable"].isin(["HUES64_hg19", "HUES64_mm9", "mESC_hg19", "mESC_mm9",
                             "fdr_cis_HUES64", "fdr_cis_mESC",
                             "fdr_trans_human", "fdr_trans_mouse", "logFC_cis_HUES64", "logFC_cis_mESC",
                             "logFC_trans_human", "logFC_trans_mouse"])]


# In[55]:


ex["cell"] = ex["variable"].str.split("_", expand=True)[0]
ex["seq"] = ex["variable"].str.split("_", expand=True)[1]
ex.head()


# In[56]:


order = ["HUES64", "mESC"]
hue_order = ["hg19", "mm9"]
pal = {"hg19": sns.color_palette("Set2")[1], "mm9": sns.color_palette("Set2")[0]}


# In[57]:


fig = plt.figure(figsize=(1.5, 1.5))

sub = ex[ex["cell"].isin(["HUES64", "mESC"])]
ax = sns.barplot(data=sub, x="cell", y="value", hue="seq", order=order, hue_order=hue_order, palette=pal)
ax.set_xticklabels(["hESCs", "mESCs"], rotation=50, va="top", ha="right")
ax.set_ylabel("MPRA activity")
ax.set_xlabel("")
ax.get_legend().remove()

ax.set_ylim((0, 14))
annotate_pval(ax, -0.25, 0.25, 9.5, 0, 9.5, ex[ex["variable"] == "fdr_cis_HUES64"]["value"].iloc[0], fontsize-1)
annotate_pval(ax, 0.75, 1.25, 8.25, 0, 8.25, ex[ex["variable"] == "fdr_cis_mESC"]["value"].iloc[0], fontsize-1)
annotate_pval(ax, -0.25, 0.75, 11.5, 0, 11.5, ex[ex["variable"] == "fdr_trans_human"]["value"].iloc[0], fontsize-1)
annotate_pval(ax, 0.25, 1.25, 12.75, 0, 12.75, ex[ex["variable"] == "fdr_trans_mouse"]["value"].iloc[0], fontsize-1)

fig.savefig("compensatory_example_barplot.pdf", dpi="figure", bbox_inches="tight")


# In[58]:


ex_sub = ex[ex["variable"].str.contains("logFC")]
ex_sub["sp"] = ex_sub["variable"].str.split("_", expand=True)[2]
ex_sub = ex_sub.sort_values(by=["seq", "sp"])
ex_sub


# In[59]:


def sp(row):
    if row.sp in ["HUES64", "human"]:
        return "human"
    else:
        return "mouse"
    
ex_sub["sp"] = ex_sub.apply(sp, axis=1)
ex_sub


# In[60]:


order = ["cis", "trans"]


# In[61]:


fig, axarr = plt.subplots(figsize=(1.5, 1.5), nrows=1, ncols=2, sharey=True)
human_ax = axarr[0]
mouse_ax = axarr[1]

sub = ex_sub[ex_sub["sp"] == "human"]
sns.barplot(data=sub, x="seq", y="value", ax=human_ax, color=sns.color_palette("Set2")[7])
human_ax.set_xticklabels(order, rotation=50, va="top", ha="right")
human_ax.set_ylabel("effect size")
human_ax.set_xlabel("")
human_ax.axhline(y=0, linestyle="dashed", color="black")

sub = ex_sub[ex_sub["sp"] == "mouse"]
sns.barplot(data=sub, x="seq", y="value", ax=mouse_ax, color=sns.color_palette("Set2")[7])
mouse_ax.set_xticklabels(order, rotation=50, va="top", ha="right")
mouse_ax.set_ylabel("")
mouse_ax.set_xlabel("")
mouse_ax.axhline(y=0, linestyle="dashed", color="black")
fig.savefig("compensatory_example_effectsize.pdf", dpi="figure", bbox_inches="tight")


# ### directional

# In[62]:


ex = df[df["hg19_id"] == "h.1389"]
ex = ex[["hg19_id", "mm9_id", "minimal_biotype_hg19", "minimal_biotype_mm9", "HUES64_hg19", "HUES64_mm9",
         "mESC_hg19", "mESC_mm9", "trans_human_status_det", "fdr_trans_human", "trans_mouse_status_det", 
         "fdr_trans_mouse", "cis_HUES64_status_det", "fdr_cis_HUES64", "cis_mESC_status_det", "fdr_cis_mESC",
         "logFC_trans_human", "logFC_trans_mouse", "logFC_cis_HUES64", "logFC_cis_mESC"]]
ex


# In[63]:


ex = pd.melt(ex, id_vars=["hg19_id", "mm9_id", "minimal_biotype_hg19", "minimal_biotype_mm9"])
ex = ex[ex["variable"].isin(["HUES64_hg19", "HUES64_mm9", "mESC_hg19", "mESC_mm9",
                             "fdr_cis_HUES64", "fdr_cis_mESC",
                             "fdr_trans_human", "fdr_trans_mouse", "logFC_cis_HUES64", "logFC_cis_mESC",
                             "logFC_trans_human", "logFC_trans_mouse"])]


# In[64]:


ex["cell"] = ex["variable"].str.split("_", expand=True)[0]
ex["seq"] = ex["variable"].str.split("_", expand=True)[1]
ex.head()


# In[65]:


order = ["HUES64", "mESC"]
hue_order = ["hg19", "mm9"]
pal = {"hg19": sns.color_palette("Set2")[1], "mm9": sns.color_palette("Set2")[0]}


# In[66]:


fig = plt.figure(figsize=(1.5, 1.5))

sub = ex[ex["cell"].isin(["HUES64", "mESC"])]
ax = sns.barplot(data=sub, x="cell", y="value", hue="seq", order=order, hue_order=hue_order, palette=pal)
ax.set_xticklabels(["hESCs", "mESCs"], rotation=50, va="top", ha="right")
ax.set_ylabel("MPRA activity")
ax.set_xlabel("")
ax.get_legend().remove()

ax.set_ylim((0, 7))
annotate_pval(ax, -0.25, 0.25, 5, 0, 5, ex[ex["variable"] == "fdr_cis_HUES64"]["value"].iloc[0], fontsize-1)
annotate_pval(ax, 0.75, 1.25, 2.25, 0, 2.25, ex[ex["variable"] == "fdr_cis_mESC"]["value"].iloc[0], fontsize-1)
annotate_pval(ax, -0.25, 0.75, 6.15, 0, 6.15, ex[ex["variable"] == "fdr_trans_human"]["value"].iloc[0], fontsize-1)
annotate_pval(ax, 0.25, 1.25, 3.25, 0, 3.25, ex[ex["variable"] == "fdr_trans_mouse"]["value"].iloc[0], fontsize-1)

fig.savefig("directional_example_barplot.pdf", dpi="figure", bbox_inches="tight")


# In[67]:


ex_sub = ex[ex["variable"].str.contains("logFC")]
ex_sub["sp"] = ex_sub["variable"].str.split("_", expand=True)[2]
ex_sub = ex_sub.sort_values(by=["seq", "sp"])
ex_sub["sp"] = ex_sub.apply(sp, axis=1)
ex_sub


# In[68]:


order = ["cis", "trans"]


# In[69]:


fig, axarr = plt.subplots(figsize=(1.5, 1.5), nrows=1, ncols=2, sharey=True)
human_ax = axarr[0]
mouse_ax = axarr[1]

sub = ex_sub[ex_sub["sp"] == "human"]
sns.barplot(data=sub, x="seq", y="value", ax=human_ax, color=sns.color_palette("Set2")[2])
human_ax.set_xticklabels(order, rotation=50, va="top", ha="right")
human_ax.set_ylabel("effect size")
human_ax.set_xlabel("")
human_ax.axhline(y=0, linestyle="dashed", color="black")

sub = ex_sub[ex_sub["sp"] == "mouse"]
sns.barplot(data=sub, x="seq", y="value", ax=mouse_ax, color=sns.color_palette("Set2")[2])
mouse_ax.set_xticklabels(order, rotation=50, va="top", ha="right")
mouse_ax.set_ylabel("")
mouse_ax.set_xlabel("")
mouse_ax.axhline(y=0, linestyle="dashed", color="black")
fig.savefig("directional_example_effectsize.pdf", dpi="figure", bbox_inches="tight")

