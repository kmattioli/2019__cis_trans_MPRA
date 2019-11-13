
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


def cis_sig_status(row):
    if row.cis_HUES64_status == "significant cis effect" and row.cis_mESC_status == "significant cis effect":
        return "sig_both"
    elif row.cis_HUES64_status == "significant cis effect" and row.cis_mESC_status == "no cis effect":
        return "sig_human"
    elif row.cis_HUES64_status == "no cis effect" and row.cis_mESC_status == "significant cis effect":
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


len(data)


# In[9]:


data = data[~pd.isnull(data["minimal_biotype_hg19"])]
len(data)


# In[10]:


data.cis_HUES64_status.value_counts()


# In[11]:


data_filt = data[((data["HUES64_padj_hg19"] < QUANT_ALPHA) | (data["mESC_padj_mm9"] < QUANT_ALPHA))]
len(data_filt)


# In[12]:


data_filt_sp = data_filt.drop("orig_species", axis=1)
data_filt_sp.drop_duplicates(inplace=True)
len(data_filt_sp)


# In[13]:


# data_filt_tile1 = data_filt[data_filt["tss_tile_num"] == "tile1"]
# len(data_filt_tile1)


# In[14]:


# data_filt_tile1_sp = data_filt_sp[data_filt_sp["tss_tile_num"] == "tile1"]
# len(data_filt_tile1_sp)


# In[15]:


# data_filt_tile2 = data_filt[data_filt["tss_tile_num"] == "tile2"]
# len(data_filt_tile2)


# In[16]:


# data_filt_tile2_sp = data_filt_sp[data_filt_sp["tss_tile_num"] == "tile2"]
# len(data_filt_tile2_sp)


# ## 3. counts of cis effects

# In[17]:


# dfs = [data_filt_sp, data_filt_tile1_sp, data_filt_tile2_sp]
# titles = ["both tiles", "tile1 only", "tile2 only"]
# labels = ["both_tiles", "tile1_only", "tile2_only"]


# In[18]:


cis_order = ["no cis effect", "significant cis effect"]
cis_pal = {"no cis effect": "gray", "significant cis effect": "black"}

det_order = ["cis effect\n(higher in human)", "cis effect\n(higher in mouse)"]
complete_pal = {"cis effect\n(higher in human)": sns.color_palette("Set2")[1],
                "cis effect\n(higher in mouse)": sns.color_palette("Set2")[0]}
det_pal = {"cis effect\n(higher in human)": sns.light_palette(sns.color_palette("Set2")[1])[2],
           "cis effect\n(higher in mouse)": sns.light_palette(sns.color_palette("Set2")[0])[2]}


# In[19]:


# for df, title, label in zip(dfs, titles, labels):
    
#     fig, ax = plt.subplots(figsize=(0.75, 1.75), nrows=1, ncols=1)

#     sns.countplot(data=df, x="cis_status_one", palette=cis_pal, order=cis_order, linewidth=2, 
#                   edgecolor=cis_pal.values(), ax=ax)
#     ax.set_xticklabels(["no cis effect", "cis effect"], va="top", ha="right", rotation=50)
#     ax.set_xlabel("")
#     ax.set_title(title)
    
#     tot = 0
#     for i, l in enumerate(cis_order):
#         n = len(df[df["cis_status_one"] == l])
#         tot += n
#         ax.annotate(str(n), xy=(i, 2), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color="white", size=fontsize)
#     print("percent cis sig: %s" % (n/tot))

#     plt.show()
#     fig.savefig("count_cis_status.%s.pdf" % label, dpi="figure", bbox_inches="tight")
#     plt.close()


# In[20]:


fig, ax = plt.subplots(figsize=(0.75, 1.75), nrows=1, ncols=1)

sns.countplot(data=data_filt_sp, x="cis_status_one", palette=cis_pal, order=cis_order, linewidth=2, 
              edgecolor=cis_pal.values(), ax=ax)
ax.set_xticklabels(["no cis effect", "cis effect"], va="top", ha="right", rotation=50)
ax.set_xlabel("")

tot = 0
for i, l in enumerate(cis_order):
    n = len(data_filt_sp[data_filt_sp["cis_status_one"] == l])
    tot += n
    ax.annotate(str(n), xy=(i, 2), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color="white", size=fontsize)
print("percent cis sig: %s" % (n/tot))

plt.show()
fig.savefig("count_cis_status.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# ## 4. plot cis effect sizes between HUES64 and mESC

# In[21]:


# for df, title, label in zip(dfs, titles, labels):
    
#     df["cis_sig_status"] = df.apply(cis_sig_status, axis=1)
    
#     # plot effect size agreement b/w the two cells
#     fig, ax = plt.subplots(figsize=(1.75, 1.75), nrows=1, ncols=1)

#     sig_human = df[df["cis_sig_status"] == "sig_human"]
#     sig_mouse = df[df["cis_sig_status"] == "sig_mouse"]
#     sig_both = df[df["cis_sig_status"] == "sig_both"]
#     not_sig = df[df["cis_sig_status"] == "not_sig_both"]

#     ax.scatter(sig_both["logFC_cis_HUES64"], sig_both["logFC_cis_mESC"], s=12, alpha=1, 
#                color="black", linewidths=0.5, edgecolors="white")
#     ax.scatter(sig_human["logFC_cis_HUES64"], sig_human["logFC_cis_mESC"], s=10, alpha=0.75, 
#                color=sns.color_palette("Set2")[1], linewidths=0.5, edgecolors="white")
#     ax.scatter(not_sig["logFC_cis_HUES64"], not_sig["logFC_cis_mESC"], s=10, alpha=0.75, 
#                color="gray", linewidths=0.5, edgecolors="white")
#     ax.scatter(sig_mouse["logFC_cis_HUES64"], sig_mouse["logFC_cis_mESC"], s=10, alpha=0.75, 
#                color=sns.color_palette("Set2")[0], linewidths=0.5, edgecolors="white")

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


# In[22]:


df = data_filt_sp


# In[23]:


df["cis_sig_status"] = df.apply(cis_sig_status, axis=1)

# plot effect size agreement b/w the two cells
fig, ax = plt.subplots(figsize=(1.75, 1.75), nrows=1, ncols=1)

sig_human = df[df["cis_sig_status"] == "sig_human"]
sig_mouse = df[df["cis_sig_status"] == "sig_mouse"]
sig_both = df[df["cis_sig_status"] == "sig_both"]
not_sig = df[df["cis_sig_status"] == "not_sig_both"]

ax.scatter(sig_both["logFC_cis_HUES64"], sig_both["logFC_cis_mESC"], s=12, alpha=1, 
           color="black", linewidths=0.5, edgecolors="white")
ax.scatter(sig_human["logFC_cis_HUES64"], sig_human["logFC_cis_mESC"], s=10, alpha=0.75, 
           color=sns.color_palette("Set2")[1], linewidths=0.5, edgecolors="white")
ax.scatter(not_sig["logFC_cis_HUES64"], not_sig["logFC_cis_mESC"], s=10, alpha=0.75, 
           color="gray", linewidths=0.5, edgecolors="white")
ax.scatter(sig_mouse["logFC_cis_HUES64"], sig_mouse["logFC_cis_mESC"], s=10, alpha=0.75, 
           color=sns.color_palette("Set2")[0], linewidths=0.5, edgecolors="white")

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


# ## 5. effect size differences across biotypes

# In[24]:


min_switch_order = ["CAGE turnover - eRNA", "CAGE turnover - lncRNA", "CAGE turnover - mRNA", 
                    "eRNA", "lncRNA", "mRNA"]
min_switch_pal = {"CAGE turnover - eRNA": sns.color_palette("Set2")[2], 
                  "CAGE turnover - lncRNA": sns.color_palette("Set2")[2],
                  "CAGE turnover - mRNA": sns.color_palette("Set2")[2],
                  "eRNA": sns.color_palette("Set2")[7], 
                  "lncRNA": sns.color_palette("Set2")[7], 
                  "mRNA": sns.color_palette("Set2")[7]}


# In[25]:


# for df, title, label in zip(dfs, titles, labels):
#     df["abs_logFC_cis"] = np.abs(df["logFC_cis_one"])
#     #df = df[df["native_status"] == "significant native effect"]

#     fig = plt.figure(figsize=(2.5, 1.5))
#     ax = sns.boxplot(data=df, x="biotype_switch_minimal", y="abs_logFC_cis", 
#                      flierprops = dict(marker='o', markersize=5), 
#                      order=min_switch_order, palette=min_switch_pal)
#     mimic_r_boxplot(ax)

#     ax.set_xticklabels(["eRNA", "lncRNA", "mRNA", "eRNA", "lncRNA", "mRNA"], rotation=50, ha='right', va='top')
#     ax.set_xlabel("")
#     ax.set_ylabel("cis effect size")
#     ax.set_title(title)

#     for i, l in enumerate(min_switch_order):
#         sub = df[df["biotype_switch_minimal"] == l]
#         n = len(sub)
#         print("%s median eff size: %s" % (l, sub["abs_logFC_cis"].median()))
#         color = min_switch_pal[l]
#         ax.annotate(str(n), xy=(i, -0.8), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color=color, size=fontsize)
        
#     ### pvals ###
#     vals1 = np.asarray(df[df["biotype_switch_minimal"] == "CAGE turnover - eRNA"]["abs_logFC_cis"])
#     vals2 = np.asarray(df[df["biotype_switch_minimal"] == "CAGE turnover - lncRNA"]["abs_logFC_cis"])
#     vals3 = np.asarray(df[df["biotype_switch_minimal"] == "CAGE turnover - mRNA"]["abs_logFC_cis"])
#     vals4 = np.asarray(df[df["biotype_switch_minimal"] == "eRNA"]["abs_logFC_cis"])
#     vals5 = np.asarray(df[df["biotype_switch_minimal"] == "lncRNA"]["abs_logFC_cis"])
#     vals6 = np.asarray(df[df["biotype_switch_minimal"] == "mRNA"]["abs_logFC_cis"])
    
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
    
#     annotate_pval(ax, 0.2, 0.8, 5, 0, 5, pval12, fontsize-1)
#     annotate_pval(ax, 1.2, 1.8, 5, 0, 5, pval13, fontsize-1)
#     annotate_pval(ax, 0, 2, 6, 0, 6, pval23, fontsize-1)
#     annotate_pval(ax, 3.2, 3.8, 5, 0, 5, pval45, fontsize-1)
#     annotate_pval(ax, 4.2, 4.8, 5, 0, 5, pval56, fontsize-1)
#     annotate_pval(ax, 3, 5, 6, 0, 6, pval46, fontsize-1)

#     ax.set_ylim((-0.9, 7))
#     ax.axvline(x=2.5, linestyle="dashed", color="black")

#     plt.show()
#     fig.savefig("cis_minimal_biotype_switch_effectsize_boxplot.%s.pdf" % label, dpi="figure", bbox_inches="tight")
#     plt.close()


# In[26]:


df["abs_logFC_cis"] = np.abs(df["logFC_cis_one"])
#df = df[df["native_status"] == "significant native effect"]

fig = plt.figure(figsize=(2.5, 1.75))
ax = sns.boxplot(data=df, x="biotype_switch_minimal", y="abs_logFC_cis", 
                 flierprops = dict(marker='o', markersize=5), 
                 order=min_switch_order, palette=min_switch_pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(["eRNA", "lncRNA", "mRNA", "eRNA", "lncRNA", "mRNA"], rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel(r"$\vert$ cis effect size $\vert$")

for i, l in enumerate(min_switch_order):
    sub = df[df["biotype_switch_minimal"] == l]
    n = len(sub)
    print("%s median eff size: %s" % (l, sub["abs_logFC_cis"].median()))
    color = min_switch_pal[l]
    ax.annotate(str(n), xy=(i, -0.8), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

### pvals ###
vals1 = np.asarray(df[df["biotype_switch_minimal"] == "CAGE turnover - eRNA"]["abs_logFC_cis"])
vals2 = np.asarray(df[df["biotype_switch_minimal"] == "CAGE turnover - lncRNA"]["abs_logFC_cis"])
vals3 = np.asarray(df[df["biotype_switch_minimal"] == "CAGE turnover - mRNA"]["abs_logFC_cis"])
vals4 = np.asarray(df[df["biotype_switch_minimal"] == "eRNA"]["abs_logFC_cis"])
vals5 = np.asarray(df[df["biotype_switch_minimal"] == "lncRNA"]["abs_logFC_cis"])
vals6 = np.asarray(df[df["biotype_switch_minimal"] == "mRNA"]["abs_logFC_cis"])

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

annotate_pval(ax, 0.2, 0.8, 5, 0, 5, pval12, fontsize-1)
annotate_pval(ax, 1.2, 1.8, 5, 0, 5, pval13, fontsize-1)
annotate_pval(ax, 0, 2, 6, 0, 6, pval23, fontsize-1)
annotate_pval(ax, 3.2, 3.8, 5, 0, 5, pval45, fontsize-1)
annotate_pval(ax, 4.2, 4.8, 5, 0, 5, pval56, fontsize-1)
annotate_pval(ax, 3, 5, 6, 0, 6, pval46, fontsize-1)

ax.set_ylim((-0.9, 7))
ax.axvline(x=2.5, linestyle="dashed", color="black")

plt.show()
fig.savefig("cis_minimal_biotype_switch_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# In[27]:


def cage_status(row):
    if "CAGE turnover" in row.biotype_switch_minimal:
        return "turnover"
    else:
        return "conserved"


# In[28]:


def one_biotype(row):
    if row.minimal_biotype_hg19 == "no CAGE activity":
        return row.minimal_biotype_mm9
    elif row.biotype_switch_minimal == "biotype switch":
        return "biotype switch"
    else:
        return row.minimal_biotype_hg19


# In[29]:


pal = {"conserved": sns.color_palette("Set2")[7], "turnover": sns.color_palette("Set2")[2]}


# In[30]:


# for df, title, pltname in zip(dfs, titles, labels):
#     df["abs_logFC_cis"] = np.abs(df["logFC_cis_one"])
#     df["cage_status"] = df.apply(cage_status, axis=1)
#     df["one_biotype"] = df.apply(one_biotype, axis=1)
    
#     fig = plt.figure(figsize=(2.75, 1.5))

#     ax = sns.boxplot(data=df, x="one_biotype", y="abs_logFC_cis", hue="cage_status",
#                      flierprops = dict(marker='o', markersize=5),
#                      order=["eRNA", "lncRNA", "mRNA"], hue_order=["turnover", "conserved"], palette=pal)
#     mimic_r_boxplot(ax)

#     ax.set_xticklabels(["eRNA", "lncRNA", "mRNA"], rotation=50, ha='right', va='top')
#     ax.set_xlabel("")
#     ax.set_ylabel("cis effect size")
#     plt.legend(loc=2, bbox_to_anchor=(1.05, 1))

#     ys = [2, 2, 2.5]
#     for i, label in enumerate(["eRNA", "lncRNA", "mRNA"]):
#         sub = df[df["one_biotype"] == label]
#         sub1 = sub[sub["cage_status"] == "turnover"]
#         sub2 = sub[sub["cage_status"] == "conserved"]

#         vals1 = np.asarray(sub1["abs_logFC_cis"])
#         vals2 = np.asarray(sub2["abs_logFC_cis"])

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

#         ax.annotate(str(n1), xy=(i-0.2, -0.8), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color=pal["turnover"], size=fontsize)
#         ax.annotate(str(n2), xy=(i+0.2, -0.8), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color=pal["conserved"], size=fontsize)

#     ax.set_ylim((-1, 6))
#     ax.set_title(title)
#     fig.savefig("cis_effect_biotype_sep_cage.%s.pdf" % pltname, dpi="figure", bbox_inches="tight")


# In[31]:


df["abs_logFC_cis"] = np.abs(df["logFC_cis_one"])
df["cage_status"] = df.apply(cage_status, axis=1)
df["one_biotype"] = df.apply(one_biotype, axis=1)

fig = plt.figure(figsize=(2.75, 1.75))

ax = sns.boxplot(data=df, x="one_biotype", y="abs_logFC_cis", hue="cage_status",
                 flierprops = dict(marker='o', markersize=5),
                 order=["eRNA", "lncRNA", "mRNA"], hue_order=["turnover", "conserved"], palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(["eRNA", "lncRNA", "mRNA"], rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel(r"$\vert$ cis effect size $\vert$")
plt.legend(loc=2, bbox_to_anchor=(1.05, 1))

ys = [2, 2, 2.5]
for i, label in enumerate(["eRNA", "lncRNA", "mRNA"]):
    sub = df[df["one_biotype"] == label]
    sub1 = sub[sub["cage_status"] == "turnover"]
    sub2 = sub[sub["cage_status"] == "conserved"]

    vals1 = np.asarray(sub1["abs_logFC_cis"])
    vals2 = np.asarray(sub2["abs_logFC_cis"])

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

    ax.annotate(str(n1), xy=(i-0.2, -0.8), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=pal["turnover"], size=fontsize)
    ax.annotate(str(n2), xy=(i+0.2, -0.8), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=pal["conserved"], size=fontsize)

ax.set_ylim((-1, 6))
fig.savefig("cis_effect_biotype_sep_cage.pdf", dpi="figure", bbox_inches="tight")


# In[32]:


pd.pivot_table(df, index="one_biotype", columns="cage_status", values="hg19_id", aggfunc="count")


# ## 6. percent sig across biotypes

# In[33]:


# for df, title, label in zip(dfs, titles, labels):

#     tots = df.groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
#     sig = df[df["cis_status_one"] != "no cis effect"].groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
#     clean_sig = tots.merge(sig, on="biotype_switch_minimal", how="left").fillna(0)
#     clean_sig["percent_sig"] = (clean_sig["hg19_id_y"]/clean_sig["hg19_id_x"])*100
    
#     fig = plt.figure(figsize=(2.5, 1.5))
#     ax = sns.barplot(data=clean_sig, x="biotype_switch_minimal", y="percent_sig", 
#                      order=min_switch_order, color=sns.color_palette("Set2")[2])

#     ax.set_xticklabels(["eRNA", "lncRNA", "mRNA", "eRNA", "lncRNA", "mRNA"], rotation=50, ha='right', va='top')
#     ax.set_xlabel("")
#     ax.set_ylabel("% of seq. pairs with\ncis effects")
#     ax.set_title(title)
#     ax.axvline(x=2.5, linestyle="dashed", color="black")
    
#     for i, l in enumerate(min_switch_order):
#         sub = clean_sig[clean_sig["biotype_switch_minimal"] == l]
#         print("%s perc sig: %s" % (l, sub["percent_sig"].iloc[0]))
#         n = sub["hg19_id_x"].iloc[0]
#         ax.annotate(str(n), xy=(i, 2), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color="white", size=fontsize)
    
#     plt.show()
#     fig.savefig("perc_sig_cis_minimal_biotype_switch.%s.pdf" % label, dpi="figure", bbox_inches="tight")
#     plt.close()


# In[34]:


tots = df.groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
sig = df[df["cis_status_one"] != "no cis effect"].groupby("biotype_switch_minimal")["hg19_id"].agg("count").reset_index()
clean_sig = tots.merge(sig, on="biotype_switch_minimal", how="left").fillna(0)
clean_sig["percent_sig"] = (clean_sig["hg19_id_y"]/clean_sig["hg19_id_x"])*100

fig = plt.figure(figsize=(2.5, 1.75))
ax = sns.barplot(data=clean_sig, x="biotype_switch_minimal", y="percent_sig", 
                 order=min_switch_order, color=sns.color_palette("Set2")[2])

ax.set_xticklabels(["eRNA", "lncRNA", "mRNA", "eRNA", "lncRNA", "mRNA"], rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("% of seq. pairs with\ncis effects")
ax.axvline(x=2.5, linestyle="dashed", color="black")

for i, l in enumerate(min_switch_order):
    sub = clean_sig[clean_sig["biotype_switch_minimal"] == l]
    print("%s perc sig: %s" % (l, sub["percent_sig"].iloc[0]))
    n = sub["hg19_id_x"].iloc[0]
    ax.annotate(str(n), xy=(i, 2), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color="white", size=fontsize)

plt.show()
fig.savefig("perc_sig_cis_minimal_biotype_switch.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# ## 7. plot example

# ### cis effect 

# In[35]:


ex = df[df["hg19_id"] == "h.1096"]
ex = ex[["hg19_id", "mm9_id", "minimal_biotype_hg19", "minimal_biotype_mm9", "HUES64_hg19", "HUES64_mm9",
         "mESC_hg19", "mESC_mm9", "cis_HUES64_status_det", "fdr_cis_HUES64", "cis_mESC_status_det", "fdr_cis_mESC"]]
ex


# In[36]:


ex = pd.melt(ex, id_vars=["hg19_id", "mm9_id", "minimal_biotype_hg19", "minimal_biotype_mm9"])
ex = ex[ex["variable"].isin(["HUES64_hg19", "HUES64_mm9", "mESC_hg19", "mESC_mm9", "fdr_cis_HUES64", "fdr_cis_mESC"])]


# In[37]:


ex["cell"] = ex["variable"].str.split("_", expand=True)[0]
ex["seq"] = ex["variable"].str.split("_", expand=True)[1]
ex.head()


# In[38]:


order = ["HUES64", "mESC"]
hue_order = ["hg19", "mm9"]
pal = {"hg19": sns.color_palette("Set2")[1], "mm9": sns.color_palette("Set2")[0]}


# In[39]:


fig = plt.figure(figsize=(1.2, 1.4))

sub = ex[ex["cell"].isin(["HUES64", "mESC"])]
ax = sns.barplot(data=sub, x="cell", y="value", hue="seq", order=order, hue_order=hue_order, palette=pal)
ax.set_xticklabels(["hESCs", "mESCs"], rotation=50, va="top", ha="right")
ax.set_ylabel("MPRA activity")
ax.set_xlabel("")
ax.get_legend().remove()

ax.set_ylim((0, 4.25))
annotate_pval(ax, -0.25, 0.25, 3.75, 0, 3.75, ex[ex["variable"] == "fdr_cis_HUES64"]["value"].iloc[0], fontsize-1)
annotate_pval(ax, 0.75, 1.25, 2.25, 0, 2.25, ex[ex["variable"] == "fdr_cis_mESC"]["value"].iloc[0], fontsize-1)

fig.savefig("cis_example_barplot.pdf", dpi="figure", bbox_inches="tight")


# In[40]:


np.median(np.abs(df[df["cis_status_one"] == "significant cis effect"]["logFC_cis_one"]))

