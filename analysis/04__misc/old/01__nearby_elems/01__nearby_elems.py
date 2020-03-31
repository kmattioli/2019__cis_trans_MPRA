#!/usr/bin/env python
# coding: utf-8

# # 01__nearby_elems
# 
# in this notebook, i examine the relationship between cis/trans compensation and # of nearby regulatory elements

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


# In[9]:


hg19_elems_closest_f = "../../../misc/03__nearby_elems/hg19_evo.tile1_only.closest_hg19_evo.txt"
mm9_elems_closest_f = "../../../misc/03__nearby_elems/mm9_evo.tile1_only.closest_mm9_evo.txt"


# ## 1. import data

# In[10]:


data = pd.read_table(data_f)
data.head()


# In[11]:


hg19_elems = pd.read_table(hg19_elems_f, sep="\t", header=None)
hg19_tss = pd.read_table(hg19_tss_f, sep="\t", header=None)
hg19_enh = pd.read_table(hg19_enh_f, sep="\t", header=None)
hg19_elems.columns = ["chr", "start", "end", "name", "n_elems_hg19"]
hg19_tss.columns = ["chr", "start", "end", "name", "n_tss_hg19"]
hg19_enh.columns = ["chr", "start", "end", "name", "n_enh_hg19"]
hg19_elems.head()


# In[12]:


mm9_elems = pd.read_table(mm9_elems_f, sep="\t", header=None)
mm9_tss = pd.read_table(mm9_tss_f, sep="\t", header=None)
mm9_enh = pd.read_table(mm9_enh_f, sep="\t", header=None)
mm9_elems.columns = ["chr", "start", "end", "name", "n_elems_mm9"]
mm9_tss.columns = ["chr", "start", "end", "name", "n_tss_mm9"]
mm9_enh.columns = ["chr", "start", "end", "name", "n_enh_mm9"]
mm9_elems.head()


# In[13]:


hg19_elems_closest = pd.read_table(hg19_elems_closest_f, header=None, names=["tss_chr", "tss_start", "tss_end",
                                                                             "name", "score", "tss_strand", 
                                                                             "other_chr", "other_start", "other_end", 
                                                                             "other_id", "other_score",
                                                                             "other_strand", "distance"])
hg19_elems_closest = hg19_elems_closest[["name", "other_id", "distance"]].drop_duplicates()
hg19_elems_closest.head()


# In[14]:


mm9_elems_closest = pd.read_table(mm9_elems_closest_f, header=None, names=["tss_chr", "tss_start", "tss_end",
                                                                             "name", "score", "tss_strand", 
                                                                             "other_chr", "other_start", "other_end", 
                                                                             "other_id", "other_score",
                                                                             "other_strand", "distance"])
mm9_elems_closest = mm9_elems_closest[["name", "other_id", "distance"]].drop_duplicates()


# ## 2. join data w/ number of nearby elements

# In[15]:


hg19_elems["hg19_id"] = hg19_elems["name"].str.split("__", expand=True)[1]
hg19_elems["tss_tile_num"] = hg19_elems["name"].str.split("__", expand=True)[2].str.split(";", expand=True)[0]
hg19_tss["hg19_id"] = hg19_tss["name"].str.split("__", expand=True)[1]
hg19_tss["tss_tile_num"] = hg19_tss["name"].str.split("__", expand=True)[2].str.split(";", expand=True)[0]
hg19_enh["hg19_id"] = hg19_enh["name"].str.split("__", expand=True)[1]
hg19_enh["tss_tile_num"] = hg19_enh["name"].str.split("__", expand=True)[2].str.split(";", expand=True)[0]
hg19_elems.head()


# In[16]:


mm9_elems["mm9_id"] = mm9_elems["name"].str.split("__", expand=True)[1]
mm9_elems["tss_tile_num"] = mm9_elems["name"].str.split("__", expand=True)[2].str.split(";", expand=True)[0]
mm9_tss["mm9_id"] = mm9_tss["name"].str.split("__", expand=True)[1]
mm9_tss["tss_tile_num"] = mm9_tss["name"].str.split("__", expand=True)[2].str.split(";", expand=True)[0]
mm9_enh["mm9_id"] = mm9_enh["name"].str.split("__", expand=True)[1]
mm9_enh["tss_tile_num"] = mm9_enh["name"].str.split("__", expand=True)[2].str.split(";", expand=True)[0]
mm9_elems.head()


# In[17]:


len(data)


# In[18]:


data = data.merge(hg19_elems[["hg19_id", "tss_tile_num", "n_elems_hg19"]], on=["hg19_id", "tss_tile_num"], how="left")
data = data.merge(hg19_tss[["hg19_id", "tss_tile_num", "n_tss_hg19"]], on=["hg19_id", "tss_tile_num"], how="left")
data = data.merge(hg19_enh[["hg19_id", "tss_tile_num", "n_enh_hg19"]], on=["hg19_id", "tss_tile_num"], how="left")

data = data.merge(mm9_elems[["mm9_id", "tss_tile_num", "n_elems_mm9"]], on=["mm9_id", "tss_tile_num"], how="left")
data = data.merge(mm9_tss[["mm9_id", "tss_tile_num", "n_tss_mm9"]], on=["mm9_id", "tss_tile_num"], how="left")
data = data.merge(mm9_enh[["mm9_id", "tss_tile_num", "n_enh_mm9"]], on=["mm9_id", "tss_tile_num"], how="left")
print(len(data))
data.head()


# ## 3. look at overall numbers of nearby elems in human and mouse

# In[19]:


# remove the 1 seq that is on chr1_random in mouse
data = data[~pd.isnull(data["n_elems_mm9"])]


# In[20]:


fig = plt.figure(figsize=(1.5, 1))

ax = sns.distplot(data["n_elems_hg19"], color=sns.color_palette("Set2")[1], label="human", hist=False)
sns.distplot(data["n_elems_mm9"], color=sns.color_palette("Set2")[0], label="mouse", hist=False)

ax.set_xlabel("number of regulatory elements within 1 Mb")
ax.set_ylabel("density")
ax.get_legend().remove()

# fig.savefig("n_elems_distplot.pdf", dpi="figure", bbox_inches="tight")


# In[21]:


fig = plt.figure(figsize=(1.5, 1))

ax = sns.distplot(data["n_tss_hg19"], color=sns.color_palette("Set2")[1], label="human", hist=False)
sns.distplot(data["n_tss_mm9"], color=sns.color_palette("Set2")[0], label="mouse", hist=False)

ax.set_xlabel("number of TSSs within 1 Mb")
ax.set_ylabel("density")
ax.get_legend().remove()

# fig.savefig("n_tss_distplot.pdf", dpi="figure", bbox_inches="tight")


# In[22]:


fig = plt.figure(figsize=(1.5, 1))

ax = sns.distplot(data["n_enh_hg19"], color=sns.color_palette("Set2")[1], label="human", hist=False)
sns.distplot(data["n_enh_mm9"], color=sns.color_palette("Set2")[0], label="mouse", hist=False)

ax.set_xlabel("number of enhancers within 1 Mb")
ax.set_ylabel("density")
ax.get_legend().remove()

# fig.savefig("n_enh_distplot.pdf", dpi="figure", bbox_inches="tight")


# In[23]:


data["mean_elems"] = data[["n_elems_hg19", "n_elems_mm9"]].mean(axis=1)
data["mean_tss"] = data[["n_tss_hg19", "n_tss_mm9"]].mean(axis=1)
data["mean_enh"] = data[["n_enh_hg19", "n_enh_mm9"]].mean(axis=1)


# ## 4. assign directional vs. compensatory status and filter

# In[24]:


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


# In[25]:


data["cis_trans_status"] = data.apply(cis_trans_status, axis=1)
data.cis_trans_status.value_counts()


# In[26]:


data = data[~pd.isnull(data["minimal_biotype_hg19"])]
len(data)


# In[27]:


data_filt = data[((data["HUES64_padj_hg19"] < QUANT_ALPHA) | (data["mESC_padj_mm9"] < QUANT_ALPHA))]
len(data_filt)


# In[28]:


data_filt_sp = data_filt.drop("orig_species", axis=1)
data_filt_sp.drop_duplicates(inplace=True)
len(data_filt_sp)


# ## filter out elements that are super close together so it doesn't bias results

# In[108]:


hg19_elems_closest = hg19_elems_closest[hg19_elems_closest["name"] != hg19_elems_closest["other_id"]]
print(len(hg19_elems_closest))
hg19_elems_1mb = hg19_elems_closest[hg19_elems_closest["distance"].astype(int) <= 100000]
print(len(hg19_elems_1mb))
hg19_elems_1mb.head()


# In[109]:


mm9_elems_closest = mm9_elems_closest[mm9_elems_closest["name"] != mm9_elems_closest["other_id"]]
print(len(mm9_elems_closest))
mm9_elems_1mb = mm9_elems_closest[mm9_elems_closest["distance"].astype(int) <= 100000]
print(len(mm9_elems_1mb))
mm9_elems_1mb.head()


# In[110]:


# find those to filter out in human
hg19_filter_out = []
for i, row in hg19_elems_1mb.iterrows():
    name = row["name"]
    other_id = row["other_id"]
    if name in hg19_filter_out:
        continue
    else:
        hg19_filter_out.append(other_id)

hg19_filter_out = list(set(hg19_filter_out))
len(hg19_filter_out)


# In[111]:


# find those to filter out in human
mm9_filter_out = []
for i, row in mm9_elems_1mb.iterrows():
    name = row["name"]
    other_id = row["other_id"]
    if name in mm9_filter_out:
        continue
    else:
        mm9_filter_out.append(other_id)

mm9_filter_out = list(set(mm9_filter_out))
len(mm9_filter_out)


# In[112]:


hg19_filter_out = pd.DataFrame(data=hg19_filter_out)
hg19_filter_out.columns = ["name"]
hg19_filter_out["hg19_id"] = hg19_filter_out["name"].str.split("__", expand=True)[1]
hg19_filter_out.head()


# In[113]:


mm9_filter_out = pd.DataFrame(data=mm9_filter_out)
mm9_filter_out.columns = ["name"]
mm9_filter_out["mm9_id"] = mm9_filter_out["name"].str.split("__", expand=True)[1]
mm9_filter_out.head()


# ## 5. look at reg elems vs. cis/trans status

# In[114]:


order = ["no cis or trans effects", "cis/trans compensatory", "cis effect only", "trans effect only",
         "cis/trans directional"]
min_order = ["cis/trans compensatory", "cis/trans directional"]
pal = {"no cis or trans effects": sns.color_palette("Set2")[7], "cis effect only": sns.color_palette("Set2")[2],
       "trans effect only": sns.color_palette("Set2")[2], "cis/trans directional": sns.color_palette("Set2")[2],
       "cis/trans compensatory": sns.color_palette("Set2")[7]}


# ### all REs

# In[115]:


len(data_filt_sp)


# In[116]:


df = data_filt_sp[(~data_filt_sp["hg19_id"].isin(hg19_filter_out["hg19_id"])) & 
                  (~data_filt_sp["mm9_id"].isin(mm9_filter_out["mm9_id"]))]
len(df)


# In[117]:


fig = plt.figure(figsize=(1, 1.75))

ax = sns.boxplot(data=df, x="cis_trans_status", y="mean_elems", order=min_order, 
                 flierprops = dict(marker='o', markersize=5), palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(["compensatory", "directional"], rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("# total REs within 1 Mb")

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
fig.savefig("Fig6G.pdf", dpi="figure", bbox_inches="tight")


# ### TSSs only

# In[118]:


fig = plt.figure(figsize=(1, 1.75))

ax = sns.boxplot(data=df, x="cis_trans_status", y="mean_tss", order=min_order, 
                 flierprops = dict(marker='o', markersize=5), palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(["compensatory", "directional"], rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("# TSSs within 1 Mb")

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
fig.savefig("Fig6I.pdf", dpi="figure", bbox_inches="tight")


# ### enhancers only

# In[119]:


fig = plt.figure(figsize=(1, 1.75))

ax = sns.boxplot(data=df, x="cis_trans_status", y="mean_enh", order=min_order, 
                 flierprops = dict(marker='o', markersize=5), palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(["compensatory", "directional"], rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("# enhancers within 1 Mb")

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
fig.savefig("Fig6H.pdf", dpi="figure", bbox_inches="tight")


# In[ ]:




