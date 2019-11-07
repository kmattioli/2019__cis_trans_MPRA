
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


# In[6]:


hg19_elems_f = "../../../misc/03__nearby_elems/hg19.num_elems_1Mb.bed"
mm9_elems_f = "../../../misc/03__nearby_elems/mm9.num_elems_1Mb.bed"


# In[7]:


hg19_tss_f = "../../../misc/03__nearby_elems/hg19.num_TSSs_1Mb.bed"
mm9_tss_f = "../../../misc/03__nearby_elems/mm9.num_TSSs_1Mb.bed"


# In[8]:


hg19_enh_f = "../../../misc/03__nearby_elems/hg19.num_enhs_1Mb.bed"
mm9_enh_f = "../../../misc/03__nearby_elems/mm9.num_enhs_1Mb.bed"


# ## 1. import data

# In[9]:


data = pd.read_table(data_f)
data.head()


# In[10]:


hg19_elems = pd.read_table(hg19_elems_f, sep="\t", header=None)
hg19_tss = pd.read_table(hg19_tss_f, sep="\t", header=None)
hg19_enh = pd.read_table(hg19_enh_f, sep="\t", header=None)
hg19_elems.columns = ["chr", "start", "end", "name", "n_elems_hg19"]
hg19_tss.columns = ["chr", "start", "end", "name", "n_tss_hg19"]
hg19_enh.columns = ["chr", "start", "end", "name", "n_enh_hg19"]
hg19_elems.head()


# In[11]:


mm9_elems = pd.read_table(mm9_elems_f, sep="\t", header=None)
mm9_tss = pd.read_table(mm9_tss_f, sep="\t", header=None)
mm9_enh = pd.read_table(mm9_enh_f, sep="\t", header=None)
mm9_elems.columns = ["chr", "start", "end", "name", "n_elems_mm9"]
mm9_tss.columns = ["chr", "start", "end", "name", "n_tss_mm9"]
mm9_enh.columns = ["chr", "start", "end", "name", "n_enh_mm9"]
mm9_elems.head()


# ## 2. join data w/ number of nearby elements

# In[12]:


hg19_elems["hg19_id"] = hg19_elems["name"].str.split("__", expand=True)[1]
hg19_elems["tss_tile_num"] = hg19_elems["name"].str.split("__", expand=True)[2].str.split(";", expand=True)[0]
hg19_tss["hg19_id"] = hg19_tss["name"].str.split("__", expand=True)[1]
hg19_tss["tss_tile_num"] = hg19_tss["name"].str.split("__", expand=True)[2].str.split(";", expand=True)[0]
hg19_enh["hg19_id"] = hg19_enh["name"].str.split("__", expand=True)[1]
hg19_enh["tss_tile_num"] = hg19_enh["name"].str.split("__", expand=True)[2].str.split(";", expand=True)[0]
hg19_elems.head()


# In[13]:


mm9_elems["mm9_id"] = mm9_elems["name"].str.split("__", expand=True)[1]
mm9_elems["tss_tile_num"] = mm9_elems["name"].str.split("__", expand=True)[2].str.split(";", expand=True)[0]
mm9_tss["mm9_id"] = mm9_tss["name"].str.split("__", expand=True)[1]
mm9_tss["tss_tile_num"] = mm9_tss["name"].str.split("__", expand=True)[2].str.split(";", expand=True)[0]
mm9_enh["mm9_id"] = mm9_enh["name"].str.split("__", expand=True)[1]
mm9_enh["tss_tile_num"] = mm9_enh["name"].str.split("__", expand=True)[2].str.split(";", expand=True)[0]
mm9_elems.head()


# In[14]:


len(data)


# In[15]:


data = data.merge(hg19_elems[["hg19_id", "tss_tile_num", "n_elems_hg19"]], on=["hg19_id", "tss_tile_num"], how="left")
data = data.merge(hg19_tss[["hg19_id", "tss_tile_num", "n_tss_hg19"]], on=["hg19_id", "tss_tile_num"], how="left")
data = data.merge(hg19_enh[["hg19_id", "tss_tile_num", "n_enh_hg19"]], on=["hg19_id", "tss_tile_num"], how="left")

data = data.merge(mm9_elems[["mm9_id", "tss_tile_num", "n_elems_mm9"]], on=["mm9_id", "tss_tile_num"], how="left")
data = data.merge(mm9_tss[["mm9_id", "tss_tile_num", "n_tss_mm9"]], on=["mm9_id", "tss_tile_num"], how="left")
data = data.merge(mm9_enh[["mm9_id", "tss_tile_num", "n_enh_mm9"]], on=["mm9_id", "tss_tile_num"], how="left")
print(len(data))
data.head()


# ## 3. look at overall numbers of nearby elems in human and mouse

# In[16]:


# remove the 1 seq that is on chr1_random in mouse
data = data[~pd.isnull(data["n_elems_mm9"])]


# In[17]:


fig = plt.figure(figsize=(1.5, 1))

ax = sns.distplot(data["n_elems_hg19"], color=sns.color_palette("Set2")[1], label="human", hist=False)
sns.distplot(data["n_elems_mm9"], color=sns.color_palette("Set2")[0], label="mouse", hist=False)

ax.set_xlabel("number of regulatory elements within 1 Mb")
ax.set_ylabel("density")
ax.get_legend().remove()

fig.savefig("n_elems_distplot.pdf", dpi="figure", bbox_inches="tight")


# In[18]:


fig = plt.figure(figsize=(1.5, 1))

ax = sns.distplot(data["n_tss_hg19"], color=sns.color_palette("Set2")[1], label="human", hist=False)
sns.distplot(data["n_tss_mm9"], color=sns.color_palette("Set2")[0], label="mouse", hist=False)

ax.set_xlabel("number of TSSs within 1 Mb")
ax.set_ylabel("density")
ax.get_legend().remove()

fig.savefig("n_tss_distplot.pdf", dpi="figure", bbox_inches="tight")


# In[19]:


fig = plt.figure(figsize=(1.5, 1))

ax = sns.distplot(data["n_enh_hg19"], color=sns.color_palette("Set2")[1], label="human", hist=False)
sns.distplot(data["n_enh_mm9"], color=sns.color_palette("Set2")[0], label="mouse", hist=False)

ax.set_xlabel("number of enhancers within 1 Mb")
ax.set_ylabel("density")
ax.get_legend().remove()

fig.savefig("n_enh_distplot.pdf", dpi="figure", bbox_inches="tight")


# In[20]:


data["mean_elems"] = data[["n_elems_hg19", "n_elems_mm9"]].mean(axis=1)
data["mean_tss"] = data[["n_tss_hg19", "n_tss_mm9"]].mean(axis=1)
data["mean_enh"] = data[["n_enh_hg19", "n_enh_mm9"]].mean(axis=1)


# ## 4. assign directional vs. compensatory status and filter

# In[21]:


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


# In[22]:


data["cis_trans_status"] = data.apply(cis_trans_status, axis=1)
data.cis_trans_status.value_counts()


# In[23]:


data = data[~pd.isnull(data["minimal_biotype_hg19"])]
len(data)


# In[24]:


data_filt = data[((data["HUES64_padj_hg19"] < QUANT_ALPHA) | (data["mESC_padj_mm9"] < QUANT_ALPHA))]
len(data_filt)


# In[25]:


data_filt_sp = data_filt.drop("orig_species", axis=1)
data_filt_sp.drop_duplicates(inplace=True)
len(data_filt_sp)


# In[26]:


# data_filt_tile1_sp = data_filt_sp[data_filt_sp["tss_tile_num"] == "tile1"]
# len(data_filt_tile1_sp)


# In[27]:


# data_filt_tile2 = data_filt[data_filt["tss_tile_num"] == "tile2"]
# len(data_filt_tile2)


# In[28]:


# data_filt_tile2_sp = data_filt_sp[data_filt_sp["tss_tile_num"] == "tile2"]
# len(data_filt_tile2_sp)


# In[29]:


# dfs = [data_filt_sp, data_filt_tile1_sp, data_filt_tile2_sp]
# titles = ["both tiles", "tile1 only", "tile2 only"]
# labels = ["both_tiles", "tile1_only", "tile2_only"]


# ## 5. look at reg elems vs. cis/trans status

# In[30]:


order = ["no cis or trans effects", "cis/trans compensatory", "cis effect only", "trans effect only",
         "cis/trans directional"]
min_order = ["cis/trans compensatory", "cis/trans directional"]
pal = {"no cis or trans effects": sns.color_palette("Set2")[7], "cis effect only": sns.color_palette("Set2")[2],
       "trans effect only": sns.color_palette("Set2")[2], "cis/trans directional": sns.color_palette("Set2")[2],
       "cis/trans compensatory": sns.color_palette("Set2")[7]}


# ### all REs

# In[31]:


# for df, title, label in zip(dfs, titles, labels):
    
#     fig = plt.figure(figsize=(1, 1.5))

#     ax = sns.boxplot(data=df, x="cis_trans_status", y="n_elems_hg19", order=min_order, 
#                      flierprops = dict(marker='o', markersize=5), palette=pal)
#     mimic_r_boxplot(ax)

#     ax.set_xticklabels(min_order, rotation=50, ha='right', va='top')
#     ax.set_xlabel("")
#     ax.set_ylabel("# total REs within 1 Mb\n(human)")
    
#     for i, l in enumerate(min_order):
#         sub = df[df["cis_trans_status"] == l]
#         n = len(sub)
#         print("%s median REs: %s" % (l, sub["n_elems_hg19"].median()))
#         color = pal[l]
#         ax.annotate(str(n), xy=(i, -120), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color=color, size=fontsize)
        
#     ### pvals ###
#     vals1 = np.asarray(df[df["cis_trans_status"] == "cis/trans compensatory"]["n_elems_hg19"])
#     vals2 = np.asarray(df[df["cis_trans_status"] == "cis/trans directional"]["n_elems_hg19"])
    
#     vals1 = vals1[~np.isnan(vals1)]
#     vals2 = vals2[~np.isnan(vals2)]
    
#     _, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)
    
#     annotate_pval(ax, 0.2, 0.8, 400, 0, 400, pval12, fontsize-1)
    
#     ax.set_ylim((-150, 1100))
#     ax.set_title(title)
#     fig.savefig("cis_trans_n_elems_hg19.%s.pdf" % label, dpi="figure", bbox_inches="tight")


# In[32]:


df = data_filt_sp


# In[33]:


fig = plt.figure(figsize=(1, 1.5))

ax = sns.boxplot(data=df, x="cis_trans_status", y="n_elems_hg19", order=min_order, 
                 flierprops = dict(marker='o', markersize=5), palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(min_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("# total REs within 1 Mb\n(human)")

for i, l in enumerate(min_order):
    sub = df[df["cis_trans_status"] == l]
    n = len(sub)
    print("%s median REs: %s" % (l, sub["n_elems_hg19"].median()))
    color = pal[l]
    ax.annotate(str(n), xy=(i, -120), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

### pvals ###
vals1 = np.asarray(df[df["cis_trans_status"] == "cis/trans compensatory"]["n_elems_hg19"])
vals2 = np.asarray(df[df["cis_trans_status"] == "cis/trans directional"]["n_elems_hg19"])

vals1 = vals1[~np.isnan(vals1)]
vals2 = vals2[~np.isnan(vals2)]

_, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)

annotate_pval(ax, 0.2, 0.8, 400, 0, 400, pval12, fontsize-1)

ax.set_ylim((-150, 1100))
fig.savefig("cis_trans_n_elems_hg19.pdf", dpi="figure", bbox_inches="tight")


# In[34]:


# for df, title, label in zip(dfs, titles, labels):
    
#     fig = plt.figure(figsize=(1, 1.5))

#     ax = sns.boxplot(data=df, x="cis_trans_status", y="n_elems_mm9", order=min_order, 
#                      flierprops = dict(marker='o', markersize=5), palette=pal)
#     mimic_r_boxplot(ax)

#     ax.set_xticklabels(min_order, rotation=50, ha='right', va='top')
#     ax.set_xlabel("")
#     ax.set_ylabel("# total REs within 1 Mb\n(mouse)")
    
#     for i, l in enumerate(min_order):
#         sub = df[df["cis_trans_status"] == l]
#         n = len(sub)
#         print("%s median REs: %s" % (l, sub["n_elems_mm9"].median()))
#         color = pal[l]
#         ax.annotate(str(n), xy=(i, -120), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color=color, size=fontsize)
        
#     ### pvals ###
#     vals1 = np.asarray(df[df["cis_trans_status"] == "cis/trans compensatory"]["n_elems_mm9"])
#     vals2 = np.asarray(df[df["cis_trans_status"] == "cis/trans directional"]["n_elems_mm9"])
    
#     vals1 = vals1[~np.isnan(vals1)]
#     vals2 = vals2[~np.isnan(vals2)]
    
#     _, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)
    
#     annotate_pval(ax, 0.2, 0.8, 400, 0, 400, pval12, fontsize-1)
    
#     ax.set_ylim((-150, 1100))
#     ax.set_title(title)
#     fig.savefig("cis_trans_n_elems_mm9.%s.pdf" % label, dpi="figure", bbox_inches="tight")


# In[35]:


fig = plt.figure(figsize=(1, 1.5))

ax = sns.boxplot(data=df, x="cis_trans_status", y="n_elems_mm9", order=min_order, 
                 flierprops = dict(marker='o', markersize=5), palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(min_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("# total REs within 1 Mb\n(mouse)")

for i, l in enumerate(min_order):
    sub = df[df["cis_trans_status"] == l]
    n = len(sub)
    print("%s median REs: %s" % (l, sub["n_elems_mm9"].median()))
    color = pal[l]
    ax.annotate(str(n), xy=(i, -120), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

### pvals ###
vals1 = np.asarray(df[df["cis_trans_status"] == "cis/trans compensatory"]["n_elems_mm9"])
vals2 = np.asarray(df[df["cis_trans_status"] == "cis/trans directional"]["n_elems_mm9"])

vals1 = vals1[~np.isnan(vals1)]
vals2 = vals2[~np.isnan(vals2)]

_, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)

annotate_pval(ax, 0.2, 0.8, 400, 0, 400, pval12, fontsize-1)

ax.set_ylim((-150, 1100))
fig.savefig("cis_trans_n_elems_mm9.pdf", dpi="figure", bbox_inches="tight")


# In[36]:


# for df, title, label in zip(dfs, titles, labels):
    
#     fig = plt.figure(figsize=(1, 1.5))

#     ax = sns.boxplot(data=df, x="cis_trans_status", y="mean_elems", order=min_order, 
#                      flierprops = dict(marker='o', markersize=5), palette=pal)
#     mimic_r_boxplot(ax)

#     ax.set_xticklabels(min_order, rotation=50, ha='right', va='top')
#     ax.set_xlabel("")
#     ax.set_ylabel("# total REs within 1 Mb\n(mean human & mouse)")
    
#     for i, l in enumerate(min_order):
#         sub = df[df["cis_trans_status"] == l]
#         n = len(sub)
#         print("%s median REs: %s" % (l, sub["mean_elems"].median()))
#         color = pal[l]
#         ax.annotate(str(n), xy=(i, -120), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color=color, size=fontsize)
        
#     ### pvals ###
#     vals1 = np.asarray(df[df["cis_trans_status"] == "cis/trans compensatory"]["mean_elems"])
#     vals2 = np.asarray(df[df["cis_trans_status"] == "cis/trans directional"]["mean_elems"])
    
#     vals1 = vals1[~np.isnan(vals1)]
#     vals2 = vals2[~np.isnan(vals2)]
    
#     _, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)
    
#     annotate_pval(ax, 0.2, 0.8, 400, 0, 400, pval12, fontsize-1)
    
#     ax.set_ylim((-150, 1100))
#     ax.set_title(title)
#     fig.savefig("cis_trans_n_elems_mean.%s.pdf" % label, dpi="figure", bbox_inches="tight")


# In[37]:


fig = plt.figure(figsize=(1, 1.5))

ax = sns.boxplot(data=df, x="cis_trans_status", y="mean_elems", order=min_order, 
                 flierprops = dict(marker='o', markersize=5), palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(min_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("# total REs within 1 Mb\n(mean human & mouse)")

for i, l in enumerate(min_order):
    sub = df[df["cis_trans_status"] == l]
    n = len(sub)
    print("%s median REs: %s" % (l, sub["mean_elems"].median()))
    color = pal[l]
    ax.annotate(str(n), xy=(i, -120), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

### pvals ###
vals1 = np.asarray(df[df["cis_trans_status"] == "cis/trans compensatory"]["mean_elems"])
vals2 = np.asarray(df[df["cis_trans_status"] == "cis/trans directional"]["mean_elems"])

vals1 = vals1[~np.isnan(vals1)]
vals2 = vals2[~np.isnan(vals2)]

_, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)

annotate_pval(ax, 0.2, 0.8, 400, 0, 400, pval12, fontsize-1)

ax.set_ylim((-150, 1100))
fig.savefig("cis_trans_n_elems_mean.pdf", dpi="figure", bbox_inches="tight")


# ### TSSs only

# In[38]:


# for df, title, label in zip(dfs, titles, labels):
    
#     fig = plt.figure(figsize=(1, 1.5))

#     ax = sns.boxplot(data=df, x="cis_trans_status", y="n_tss_hg19", order=min_order, 
#                      flierprops = dict(marker='o', markersize=5), palette=pal)
#     mimic_r_boxplot(ax)

#     ax.set_xticklabels(min_order, rotation=50, ha='right', va='top')
#     ax.set_xlabel("")
#     ax.set_ylabel("# TSSs within 1 Mb\n(human)")
    
#     for i, l in enumerate(min_order):
#         sub = df[df["cis_trans_status"] == l]
#         n = len(sub)
#         print("%s median REs: %s" % (l, sub["n_tss_hg19"].median()))
#         color = pal[l]
#         ax.annotate(str(n), xy=(i, -120), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color=color, size=fontsize)
        
#     ### pvals ###
#     vals1 = np.asarray(df[df["cis_trans_status"] == "cis/trans compensatory"]["n_tss_hg19"])
#     vals2 = np.asarray(df[df["cis_trans_status"] == "cis/trans directional"]["n_tss_hg19"])
    
#     vals1 = vals1[~np.isnan(vals1)]
#     vals2 = vals2[~np.isnan(vals2)]
    
#     _, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)
    
#     annotate_pval(ax, 0.2, 0.8, 400, 0, 400, pval12, fontsize-1)
    
#     ax.set_ylim((-150, 1100))
#     ax.set_title(title)
#     fig.savefig("cis_trans_n_tss_hg19.%s.pdf" % label, dpi="figure", bbox_inches="tight")


# In[39]:


fig = plt.figure(figsize=(1, 1.5))

ax = sns.boxplot(data=df, x="cis_trans_status", y="n_tss_hg19", order=min_order, 
                 flierprops = dict(marker='o', markersize=5), palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(min_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("# TSSs within 1 Mb\n(human)")

for i, l in enumerate(min_order):
    sub = df[df["cis_trans_status"] == l]
    n = len(sub)
    print("%s median REs: %s" % (l, sub["n_tss_hg19"].median()))
    color = pal[l]
    ax.annotate(str(n), xy=(i, -120), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

### pvals ###
vals1 = np.asarray(df[df["cis_trans_status"] == "cis/trans compensatory"]["n_tss_hg19"])
vals2 = np.asarray(df[df["cis_trans_status"] == "cis/trans directional"]["n_tss_hg19"])

vals1 = vals1[~np.isnan(vals1)]
vals2 = vals2[~np.isnan(vals2)]

_, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)

annotate_pval(ax, 0.2, 0.8, 400, 0, 400, pval12, fontsize-1)

ax.set_ylim((-150, 1100))
fig.savefig("cis_trans_n_tss_hg19.pdf", dpi="figure", bbox_inches="tight")


# In[40]:


# for df, title, label in zip(dfs, titles, labels):
    
#     fig = plt.figure(figsize=(1, 1.5))

#     ax = sns.boxplot(data=df, x="cis_trans_status", y="n_tss_mm9", order=min_order, 
#                      flierprops = dict(marker='o', markersize=5), palette=pal)
#     mimic_r_boxplot(ax)

#     ax.set_xticklabels(min_order, rotation=50, ha='right', va='top')
#     ax.set_xlabel("")
#     ax.set_ylabel("# TSSs within 1 Mb\n(mouse)")
    
#     for i, l in enumerate(min_order):
#         sub = df[df["cis_trans_status"] == l]
#         n = len(sub)
#         print("%s median REs: %s" % (l, sub["n_tss_mm9"].median()))
#         color = pal[l]
#         ax.annotate(str(n), xy=(i, -120), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color=color, size=fontsize)
        
#     ### pvals ###
#     vals1 = np.asarray(df[df["cis_trans_status"] == "cis/trans compensatory"]["n_tss_mm9"])
#     vals2 = np.asarray(df[df["cis_trans_status"] == "cis/trans directional"]["n_tss_mm9"])
    
#     vals1 = vals1[~np.isnan(vals1)]
#     vals2 = vals2[~np.isnan(vals2)]
    
#     _, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)
    
#     annotate_pval(ax, 0.2, 0.8, 400, 0, 400, pval12, fontsize-1)
    
#     ax.set_ylim((-150, 1200))
#     ax.set_title(title)
#     fig.savefig("cis_trans_n_tss_mm9.%s.pdf" % label, dpi="figure", bbox_inches="tight")


# In[41]:


fig = plt.figure(figsize=(1, 1.5))

ax = sns.boxplot(data=df, x="cis_trans_status", y="n_tss_mm9", order=min_order, 
                 flierprops = dict(marker='o', markersize=5), palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(min_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("# TSSs within 1 Mb\n(mouse)")

for i, l in enumerate(min_order):
    sub = df[df["cis_trans_status"] == l]
    n = len(sub)
    print("%s median REs: %s" % (l, sub["n_tss_mm9"].median()))
    color = pal[l]
    ax.annotate(str(n), xy=(i, -120), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

### pvals ###
vals1 = np.asarray(df[df["cis_trans_status"] == "cis/trans compensatory"]["n_tss_mm9"])
vals2 = np.asarray(df[df["cis_trans_status"] == "cis/trans directional"]["n_tss_mm9"])

vals1 = vals1[~np.isnan(vals1)]
vals2 = vals2[~np.isnan(vals2)]

_, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)

annotate_pval(ax, 0.2, 0.8, 400, 0, 400, pval12, fontsize-1)

ax.set_ylim((-150, 1200))
fig.savefig("cis_trans_n_tss_mm9.pdf", dpi="figure", bbox_inches="tight")


# In[42]:


# for df, title, label in zip(dfs, titles, labels):
    
#     fig = plt.figure(figsize=(1, 1.5))

#     ax = sns.boxplot(data=df, x="cis_trans_status", y="mean_tss", order=min_order, 
#                      flierprops = dict(marker='o', markersize=5), palette=pal)
#     mimic_r_boxplot(ax)

#     ax.set_xticklabels(min_order, rotation=50, ha='right', va='top')
#     ax.set_xlabel("")
#     ax.set_ylabel("# TSSs within 1 Mb\n(mean human & mouse)")
    
#     for i, l in enumerate(min_order):
#         sub = df[df["cis_trans_status"] == l]
#         n = len(sub)
#         print("%s median REs: %s" % (l, sub["mean_tss"].median()))
#         color = pal[l]
#         ax.annotate(str(n), xy=(i, -120), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color=color, size=fontsize)
        
#     ### pvals ###
#     vals1 = np.asarray(df[df["cis_trans_status"] == "cis/trans compensatory"]["mean_tss"])
#     vals2 = np.asarray(df[df["cis_trans_status"] == "cis/trans directional"]["mean_tss"])
    
#     vals1 = vals1[~np.isnan(vals1)]
#     vals2 = vals2[~np.isnan(vals2)]
    
#     _, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)
    
#     annotate_pval(ax, 0.2, 0.8, 400, 0, 400, pval12, fontsize-1)
    
#     ax.set_ylim((-150, 1100))
#     ax.set_title(title)
#     fig.savefig("cis_trans_n_tss_mean.%s.pdf" % label, dpi="figure", bbox_inches="tight")


# In[43]:


fig = plt.figure(figsize=(1, 1.5))

ax = sns.boxplot(data=df, x="cis_trans_status", y="mean_tss", order=min_order, 
                 flierprops = dict(marker='o', markersize=5), palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(min_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("# TSSs within 1 Mb\n(mean human & mouse)")

for i, l in enumerate(min_order):
    sub = df[df["cis_trans_status"] == l]
    n = len(sub)
    print("%s median REs: %s" % (l, sub["mean_tss"].median()))
    color = pal[l]
    ax.annotate(str(n), xy=(i, -120), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

### pvals ###
vals1 = np.asarray(df[df["cis_trans_status"] == "cis/trans compensatory"]["mean_tss"])
vals2 = np.asarray(df[df["cis_trans_status"] == "cis/trans directional"]["mean_tss"])

vals1 = vals1[~np.isnan(vals1)]
vals2 = vals2[~np.isnan(vals2)]

_, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)

annotate_pval(ax, 0.2, 0.8, 400, 0, 400, pval12, fontsize-1)

ax.set_ylim((-150, 1100))
fig.savefig("cis_trans_n_tss_mean.pdf", dpi="figure", bbox_inches="tight")


# ### enhancers only

# In[44]:


# for df, title, label in zip(dfs, titles, labels):
    
#     fig = plt.figure(figsize=(1, 1.5))

#     ax = sns.boxplot(data=df, x="cis_trans_status", y="n_enh_hg19", order=min_order, 
#                      flierprops = dict(marker='o', markersize=5), palette=pal)
#     mimic_r_boxplot(ax)

#     ax.set_xticklabels(min_order, rotation=50, ha='right', va='top')
#     ax.set_xlabel("")
#     ax.set_ylabel("# enhancers within 1 Mb\n(human)")
    
#     for i, l in enumerate(min_order):
#         sub = df[df["cis_trans_status"] == l]
#         n = len(sub)
#         print("%s median REs: %s" % (l, sub["n_enh_hg19"].median()))
#         color = pal[l]
#         ax.annotate(str(n), xy=(i, -20), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color=color, size=fontsize)
        
#     ### pvals ###
#     vals1 = np.asarray(df[df["cis_trans_status"] == "cis/trans compensatory"]["n_enh_hg19"])
#     vals2 = np.asarray(df[df["cis_trans_status"] == "cis/trans directional"]["n_enh_hg19"])
    
#     vals1 = vals1[~np.isnan(vals1)]
#     vals2 = vals2[~np.isnan(vals2)]
    
#     _, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)
    
#     annotate_pval(ax, 0.2, 0.8, 100, 0, 100, pval12, fontsize-1)
    
#     ax.set_ylim((-30, 150))
#     ax.set_title(title)
#     fig.savefig("cis_trans_n_enh_hg19.%s.pdf" % label, dpi="figure", bbox_inches="tight")


# In[45]:


fig = plt.figure(figsize=(1, 1.5))

ax = sns.boxplot(data=df, x="cis_trans_status", y="n_enh_hg19", order=min_order, 
                 flierprops = dict(marker='o', markersize=5), palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(min_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("# enhancers within 1 Mb\n(human)")

for i, l in enumerate(min_order):
    sub = df[df["cis_trans_status"] == l]
    n = len(sub)
    print("%s median REs: %s" % (l, sub["n_enh_hg19"].median()))
    color = pal[l]
    ax.annotate(str(n), xy=(i, -20), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

### pvals ###
vals1 = np.asarray(df[df["cis_trans_status"] == "cis/trans compensatory"]["n_enh_hg19"])
vals2 = np.asarray(df[df["cis_trans_status"] == "cis/trans directional"]["n_enh_hg19"])

vals1 = vals1[~np.isnan(vals1)]
vals2 = vals2[~np.isnan(vals2)]

_, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)

annotate_pval(ax, 0.2, 0.8, 100, 0, 100, pval12, fontsize-1)

ax.set_ylim((-30, 150))
fig.savefig("cis_trans_n_enh_hg19.pdf", dpi="figure", bbox_inches="tight")


# In[46]:


# for df, title, label in zip(dfs, titles, labels):
    
#     fig = plt.figure(figsize=(1, 1.5))

#     ax = sns.boxplot(data=df, x="cis_trans_status", y="n_enh_mm9", order=min_order, 
#                      flierprops = dict(marker='o', markersize=5), palette=pal)
#     mimic_r_boxplot(ax)

#     ax.set_xticklabels(min_order, rotation=50, ha='right', va='top')
#     ax.set_xlabel("")
#     ax.set_ylabel("# enhancers within 1 Mb\n(mouse)")
    
#     for i, l in enumerate(min_order):
#         sub = df[df["cis_trans_status"] == l]
#         n = len(sub)
#         print("%s median REs: %s" % (l, sub["n_tss_mm9"].median()))
#         color = pal[l]
#         ax.annotate(str(n), xy=(i, -20), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color=color, size=fontsize)
        
#     ### pvals ###
#     vals1 = np.asarray(df[df["cis_trans_status"] == "cis/trans compensatory"]["n_enh_mm9"])
#     vals2 = np.asarray(df[df["cis_trans_status"] == "cis/trans directional"]["n_enh_mm9"])
    
#     vals1 = vals1[~np.isnan(vals1)]
#     vals2 = vals2[~np.isnan(vals2)]
    
#     _, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)
    
#     annotate_pval(ax, 0.2, 0.8, 90, 0, 90, pval12, fontsize-1)
    
#     ax.set_ylim((-30, 150))
#     ax.set_title(title)
#     fig.savefig("cis_trans_n_enh_mm9.%s.pdf" % label, dpi="figure", bbox_inches="tight")


# In[47]:


fig = plt.figure(figsize=(1, 1.5))

ax = sns.boxplot(data=df, x="cis_trans_status", y="n_enh_mm9", order=min_order, 
                 flierprops = dict(marker='o', markersize=5), palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(min_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("# enhancers within 1 Mb\n(mouse)")

for i, l in enumerate(min_order):
    sub = df[df["cis_trans_status"] == l]
    n = len(sub)
    print("%s median REs: %s" % (l, sub["n_tss_mm9"].median()))
    color = pal[l]
    ax.annotate(str(n), xy=(i, -20), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

### pvals ###
vals1 = np.asarray(df[df["cis_trans_status"] == "cis/trans compensatory"]["n_enh_mm9"])
vals2 = np.asarray(df[df["cis_trans_status"] == "cis/trans directional"]["n_enh_mm9"])

vals1 = vals1[~np.isnan(vals1)]
vals2 = vals2[~np.isnan(vals2)]

_, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)

annotate_pval(ax, 0.2, 0.8, 90, 0, 90, pval12, fontsize-1)

ax.set_ylim((-30, 150))
fig.savefig("cis_trans_n_enh_mm9.pdf", dpi="figure", bbox_inches="tight")


# In[48]:


# for df, title, label in zip(dfs, titles, labels):
    
#     fig = plt.figure(figsize=(1, 1.5))

#     ax = sns.boxplot(data=df, x="cis_trans_status", y="mean_enh", order=min_order, 
#                      flierprops = dict(marker='o', markersize=5), palette=pal)
#     mimic_r_boxplot(ax)

#     ax.set_xticklabels(min_order, rotation=50, ha='right', va='top')
#     ax.set_xlabel("")
#     ax.set_ylabel("# enhancers within 1 Mb\n(mean human & mouse)")
    
#     for i, l in enumerate(min_order):
#         sub = df[df["cis_trans_status"] == l]
#         n = len(sub)
#         print("%s median REs: %s" % (l, sub["mean_enh"].median()))
#         color = pal[l]
#         ax.annotate(str(n), xy=(i, -20), xycoords="data", xytext=(0, 0), 
#                     textcoords="offset pixels", ha='center', va='bottom', 
#                     color=color, size=fontsize)
        
#     ### pvals ###
#     vals1 = np.asarray(df[df["cis_trans_status"] == "cis/trans compensatory"]["mean_enh"])
#     vals2 = np.asarray(df[df["cis_trans_status"] == "cis/trans directional"]["mean_enh"])
    
#     vals1 = vals1[~np.isnan(vals1)]
#     vals2 = vals2[~np.isnan(vals2)]
    
#     _, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)
    
#     annotate_pval(ax, 0.2, 0.8, 100, 0, 100, pval12, fontsize-1)
    
#     ax.set_ylim((-30, 150))
#     ax.set_title(title)
#     fig.savefig("cis_trans_n_enh_mean.%s.pdf" % label, dpi="figure", bbox_inches="tight")


# In[49]:


fig = plt.figure(figsize=(1, 1.5))

ax = sns.boxplot(data=df, x="cis_trans_status", y="mean_enh", order=min_order, 
                 flierprops = dict(marker='o', markersize=5), palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(min_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("# enhancers within 1 Mb\n(mean human & mouse)")

for i, l in enumerate(min_order):
    sub = df[df["cis_trans_status"] == l]
    n = len(sub)
    print("%s median REs: %s" % (l, sub["mean_enh"].median()))
    color = pal[l]
    ax.annotate(str(n), xy=(i, -20), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

### pvals ###
vals1 = np.asarray(df[df["cis_trans_status"] == "cis/trans compensatory"]["mean_enh"])
vals2 = np.asarray(df[df["cis_trans_status"] == "cis/trans directional"]["mean_enh"])

vals1 = vals1[~np.isnan(vals1)]
vals2 = vals2[~np.isnan(vals2)]

_, pval12 = stats.mannwhitneyu(vals1, vals2, alternative="two-sided", use_continuity=False)

annotate_pval(ax, 0.2, 0.8, 100, 0, 100, pval12, fontsize-1)

ax.set_ylim((-30, 150))
fig.savefig("cis_trans_n_enh_mean.pdf", dpi="figure", bbox_inches="tight")


# In[50]:


print("")
print("===")
print("COMPENSATORY")
sub = df[df["cis_trans_status"].str.contains("compensatory")]
int = sub[sub["cis_trans_int_status"].str.contains("significant")]
print((len(int))/(len(sub)))
print("")
print("DIRECTIONAL")
sub = df[df["cis_trans_status"].str.contains("directional")]
int = sub[sub["cis_trans_int_status"].str.contains("significant")]
print((len(int))/(len(sub)))
print("")
print("CIS ONLY")
sub = df[df["cis_trans_status"].str.contains("cis effect only")]
int = sub[sub["cis_trans_int_status"].str.contains("significant")]
print((len(int))/(len(sub)))
print("")
print("TRANS ONLY")
sub = df[df["cis_trans_status"].str.contains("trans effect only")]
int = sub[sub["cis_trans_int_status"].str.contains("significant")]
print((len(int))/(len(sub)))
print("")
print("NOTHING")
sub = df[df["cis_trans_status"].str.contains("no cis or trans")]
int = sub[sub["cis_trans_int_status"].str.contains("significant")]
print((len(int))/(len(sub)))
print("===")


# In[51]:


df[df["cis_trans_int_status"].str.contains("significant")]


# In[55]:


df[df["hg19_id"] == "h.1389"]

