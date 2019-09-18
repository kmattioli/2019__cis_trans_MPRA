
# coding: utf-8

# # 00__preprocess_counts
# 
# in this notebook, i aggregate barcode counts across technical replicates, and merge barcode counts from biological replicates into 1 dataframe. additionally, since hESC reps were sequenced at much higher depth than mESC reps, I downsample the mESC counts in order to accurately compare between the two, and make sure the inequality of sequencing depth isn't biasing anything. finally, i filter barcodes such that they have >= 5 counts in the DNA library, and i filter elements such that every element included has >= 3 barcodes represented at this filter.

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


# ## functions -- move to utils eventually

# In[4]:


def import_dna(counts_dir, dna_f):
    dna_dfs = []
    for i in range(len(dna_f)):
        f = dna_f[i]
        cols = ["barcode", "dna_%s" % (i+1)]
        tmp = pd.read_table("%s/%s" % (counts_dir, f), sep="\t")
        tmp.columns = cols
        dna_dfs.append(tmp)
    if len(dna_dfs) > 1:
        dna = reduce(lambda x, y: pd.merge(x, y, on = "barcode"), dna_dfs)
    else:
        dna = dna_dfs[0]
    return dna


# In[5]:


def import_rna(counts_dir, rna_f, dna):
    data = dna.copy()
    data_cols = list(dna.columns)
    for f in rna_f:
        rep = re.findall(r'\d+', f.split("_")[3])[0]
        tmp = pd.read_table("%s/%s" % (counts_dir, f), sep="\t")
        tmp.columns = ["barcode", "rna_%s" % rep]
        data_cols.append("rna_%s" % rep)
        data = data.merge(tmp, on="barcode", how="outer")
    return data, data_cols


# ## variables

# In[6]:


counts_dir = "../../../data/02__mpra/01__counts"


# In[7]:


barcode_dna_read_threshold = 5
barcode_rna_read_threshold = 0
n_barcodes_per_elem_threshold = 3


# ### DNA files

# In[8]:


dna_f = ["01__DNA_20180601_BARCODES.txt"]


# ### RNA files

# In[9]:


HUES64_rna_f = ["07__HUES64_rep6_lib1_BARCODES.txt",
                "07__HUES64_rep6_lib2_BARCODES.txt",
                "08__HUES64_rep7_lib1_BARCODES.txt",
                "08__HUES64_rep7_lib2_BARCODES.txt",
                "09__HUES64_rep8_lib1_BARCODES.txt",
                "09__HUES64_rep8_lib2_BARCODES.txt",
                "10__HUES64_rep9_lib1_BARCODES.txt",
                "10__HUES64_rep9_lib2_BARCODES.txt",
                "11__HUES64_rep10_lib1_BARCODES.txt",
                "11__HUES64_rep10_lib2_BARCODES.txt",
                "12__HUES64_rep11_lib1_BARCODES.txt",
                "12__HUES64_rep11_lib2_BARCODES.txt",
                "16__HUES64_rep12_lib1_BARCODES.txt",
                "16__HUES64_rep12_lib2_BARCODES.txt",
                "17__HUES64_rep13_lib1_BARCODES.txt",
                "17__HUES64_rep13_lib2_BARCODES.txt",
                "18__HUES64_rep14_lib1_BARCODES.txt",
                "18__HUES64_rep14_lib2_BARCODES.txt"]
HUES64_out_f = "HUES64__all_counts.txt"


# In[10]:


# re name once we get sample sheet from chiara if needed
mESC_rna_f = ["15__mESC_rep3_lib1_BARCODES.txt",
              "15__mESC_rep3_lib2_BARCODES.txt",
              "19__mESC_rep4_lib1_BARCODES.txt",
              "19__mESC_rep4_lib2_BARCODES.txt",
              "19__mESC_rep4_lib3_BARCODES.txt",
              "20__mESC_rep5_lib1_BARCODES.txt",
              "20__mESC_rep5_lib2_BARCODES.txt",
              "20__mESC_rep5_lib3_BARCODES.txt"]
mESC_out_f = "mESC__all_counts.txt"


# ### Index file

# In[11]:


index_f = "../../../data/01__design/02__index/TWIST_pool4_v8_final.txt.gz"


# ## 1. import index

# In[12]:


index = pd.read_table(index_f, sep="\t")


# In[13]:


index_elem = index[["element", "tile_type"]].drop_duplicates()


# ## 2. import dna

# In[14]:


dna = import_dna(counts_dir, dna_f)
dna.head()


# ## 3. import rna

# In[15]:


HUES64_data, HUES64_cols = import_rna(counts_dir, HUES64_rna_f, dna)
HUES64_data.head()


# In[16]:


mESC_data, mESC_cols = import_rna(counts_dir, mESC_rna_f, dna)
mESC_data.head()


# In[17]:


mESC_data.columns = ["barcode", "dna_1", "rna_3_x", "rna_3_y", "rna_4_x", "rna_4_y", "rna_4_z",
                     "rna_5_x", "rna_5_y", "rna_5_z"]
mESC_cols = list(mESC_data.columns)


# ## 4. look at correlation b/w technical replicates

# In[18]:


HUES64_reps = ["rna_6", "rna_7", "rna_8", "rna_9", "rna_10", "rna_11", "rna_12", "rna_13", "rna_14"]
for rep in HUES64_reps:
    cols = [x for x in HUES64_data.columns if rep in x]
    sub_df = HUES64_data[cols]
    sub_df[cols] = np.log10(HUES64_data[cols] + 1)
    
    # plot
    g = sns.jointplot(sub_df[cols[0]], sub_df[cols[1]], color="gray", size=2.2,
                      marginal_kws=dict(bins=15),
                      joint_kws=dict(s=5, rasterized=True))
    g.set_axis_labels("log10(%s (lib 1) + 1) counts" % rep, "log10(%s (lib 2) + 1) counts" % rep)
    #g.savefig("HUES64_%s_lib_corr_scatter.pdf" % (rep), dpi="figure", bbox_inches="tight")
    plt.show()


# In[19]:


mESC_reps = ["rna_3", "rna_4", "rna_5"]
for rep in mESC_reps:
    cols = [x for x in mESC_data.columns if rep in x]
    sub_df = mESC_data[cols]
    sub_df[cols] = np.log10(mESC_data[cols] + 1)
    
    # plot
    g = sns.jointplot(sub_df[cols[0]], sub_df[cols[1]], color="gray", size=2.2,
                      marginal_kws=dict(bins=15),
                      joint_kws=dict(s=5, rasterized=True))
    g.set_axis_labels("log10(%s (lib 1) + 1) counts" % rep, "log10(%s (lib 2) + 1) counts" % rep)
    #g.savefig("mESC_%s_lib_corr_scatter.pdf" % (rep), dpi="figure", bbox_inches="tight")
    plt.show()


# In[20]:


# heatmap incl all libraries
tmp1 = HUES64_data
tmp1_cols = ["barcode", "dna_1"]
tmp1_cols.extend(["HUES64_%s" % x for x in tmp1.columns if "barcode" not in x and "dna" not in x])
tmp1.columns = tmp1_cols

tmp2 = mESC_data
tmp2_cols = ["barcode", "dna_1"]
tmp2_cols.extend(["mESC_%s" % x for x in tmp2.columns if "barcode" not in x and "dna" not in x])
tmp2.columns = tmp2_cols

tmp = tmp1.merge(tmp2, on=["barcode", "dna_1"])
tmp = tmp.set_index("barcode")
tmp.drop("dna_1", axis=1, inplace=True)
tmp_cols = tmp.columns
tmp[tmp_cols] = np.log10(tmp[tmp_cols] + 1)
tmp_corr = tmp.corr(method="pearson")

cmap = sns.cubehelix_palette(as_cmap=True)
cg = sns.clustermap(tmp_corr, figsize=(9, 9), cmap=cmap, annot=True, vmin=0.6)
_ = plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.suptitle("pearson correlation of replicates\nlog10+1 counts of all barcodes")
#cg.savefig("rep_and_lib_corr_heatmap.pdf", dpi="figure", bbox_inches="tight")


# ## 5. sum technical replicates

# In[21]:


HUES64_data["HUES64_rna_6"] = HUES64_data[["HUES64_rna_6_x", "HUES64_rna_6_y"]].sum(axis=1)
HUES64_data["HUES64_rna_7"] = HUES64_data[["HUES64_rna_7_x", "HUES64_rna_7_y"]].sum(axis=1)
HUES64_data["HUES64_rna_8"] = HUES64_data[["HUES64_rna_8_x", "HUES64_rna_8_y"]].sum(axis=1)
HUES64_data["HUES64_rna_9"] = HUES64_data[["HUES64_rna_9_x", "HUES64_rna_9_y"]].sum(axis=1)
HUES64_data["HUES64_rna_10"] = HUES64_data[["HUES64_rna_10_x", "HUES64_rna_10_y"]].sum(axis=1)
HUES64_data["HUES64_rna_11"] = HUES64_data[["HUES64_rna_11_x", "HUES64_rna_11_y"]].sum(axis=1)
HUES64_data["HUES64_rna_12"] = HUES64_data[["HUES64_rna_12_x", "HUES64_rna_12_y"]].sum(axis=1)
HUES64_data["HUES64_rna_13"] = HUES64_data[["HUES64_rna_13_x", "HUES64_rna_13_y"]].sum(axis=1)
HUES64_data["HUES64_rna_14"] = HUES64_data[["HUES64_rna_14_x", "HUES64_rna_14_y"]].sum(axis=1)
mESC_data["mESC_rna_3"] = mESC_data[["mESC_rna_3_x", "mESC_rna_3_y"]].sum(axis=1)
mESC_data["mESC_rna_4"] = mESC_data[["mESC_rna_4_x", "mESC_rna_4_y", "mESC_rna_4_z"]].sum(axis=1)
mESC_data["mESC_rna_5"] = mESC_data[["mESC_rna_5_x", "mESC_rna_5_y", "mESC_rna_5_z"]].sum(axis=1)

old_cols = [x for x in HUES64_data.columns if "_x" in x or "_y" in x]
HUES64_data.drop(old_cols, axis=1, inplace=True)
HUES64_cols = HUES64_data.columns

old_cols = [x for x in mESC_data.columns if "_x" in x or "_y" in x or "_z" in x]
mESC_data.drop(old_cols, axis=1, inplace=True)
mESC_cols = HUES64_data.columns

HUES64_data.head()


# sum up technical transfection replicates from a given passage (biological replicates)

# In[22]:


tmp = HUES64_data.copy()
tmp["HUES64_rep_1"] = tmp["HUES64_rna_6"] + tmp["HUES64_rna_7"] + tmp["HUES64_rna_8"]
tmp["HUES64_rep_2"] = tmp["HUES64_rna_9"] + tmp["HUES64_rna_10"] + tmp["HUES64_rna_11"]
tmp["HUES64_rep_3"] = tmp["HUES64_rna_12"] + tmp["HUES64_rna_13"] + tmp["HUES64_rna_14"]

tmp[["HUES64_rep_1", "HUES64_rep_2", "HUES64_rep_3"]] = np.log10(tmp[["HUES64_rep_1", "HUES64_rep_2", "HUES64_rep_3"]] + 1)
    
# plot
g = sns.jointplot(tmp["HUES64_rep_1"], tmp["HUES64_rep_2"], color="gray", size=2.2, 
                  marginal_kws=dict(bins=15),
                  joint_kws=dict(s=5, rasterized=True))
g.set_axis_labels("log10(rep 1 counts + 1)", "log10(rep 2 counts + 1)")

r, p = spearmanr(tmp["HUES64_rep_1"], tmp["HUES64_rep_2"])
g.ax_joint.annotate( "r = {:.2f}".format(r), ha="left", xy=(0.1, .90), xycoords=g.ax_joint.transAxes, 
                    fontsize=fontsize)
    
#g.savefig("HUES64_barcode_counts_scatter.rep1_v_rep2.pdf", dpi="figure", bbox_inches="tight")


# In[23]:


# plot
g = sns.jointplot(tmp["HUES64_rep_1"], tmp["HUES64_rep_3"], color="gray", size=2.2, 
                  marginal_kws=dict(bins=15),
                  joint_kws=dict(s=5, rasterized=True))
g.set_axis_labels("log10(rep 1 counts + 1)", "log10(rep 3 counts + 1)")

r, p = spearmanr(tmp["HUES64_rep_1"], tmp["HUES64_rep_3"])
g.ax_joint.annotate( "r = {:.2f}".format(r), ha="left", xy=(0.1, .90), xycoords=g.ax_joint.transAxes, 
                    fontsize=fontsize)
    
#g.savefig("HUES64_barcode_counts_scatter.rep1_v_rep3.pdf", dpi="figure", bbox_inches="tight")


# In[24]:


tmp = mESC_data.copy()
tmp["mESC_rep_1"] = tmp["mESC_rna_3"]
tmp["mESC_rep_2"] = tmp["mESC_rna_4"]
tmp["mESC_rep_3"] = tmp["mESC_rna_5"]

tmp[["mESC_rep_1", "mESC_rep_2", "mESC_rep_3"]] = np.log10(tmp[["mESC_rep_1", "mESC_rep_2", "mESC_rep_3"]] + 1)
    
# plot
g = sns.jointplot(tmp["mESC_rep_1"], tmp["mESC_rep_2"], color="gray", size=2.2, 
                  marginal_kws=dict(bins=15),
                  joint_kws=dict(s=5, rasterized=True))
g.set_axis_labels("log10(rep 1 counts + 1)", "log10(rep 2 counts + 1)")

r, p = spearmanr(tmp["mESC_rep_1"], tmp["mESC_rep_2"])
g.ax_joint.annotate( "r = {:.2f}".format(r), ha="left", xy=(0.1, .90), xycoords=g.ax_joint.transAxes, 
                    fontsize=fontsize)
    
#g.savefig("mESC_barcode_counts_scatter.rep1_v_rep2.pdf", dpi="figure", bbox_inches="tight")


# In[25]:


# plot
g = sns.jointplot(tmp["mESC_rep_1"], tmp["mESC_rep_3"], color="gray", size=2.2, 
                  marginal_kws=dict(bins=15),
                  joint_kws=dict(s=5, rasterized=True))
g.set_axis_labels("log10(rep 1 counts + 1)", "log10(rep 3 counts + 1)")

r, p = spearmanr(tmp["mESC_rep_1"], tmp["mESC_rep_3"])
g.ax_joint.annotate( "r = {:.2f}".format(r), ha="left", xy=(0.1, .90), xycoords=g.ax_joint.transAxes, 
                    fontsize=fontsize)
    
#g.savefig("mESC_barcode_counts_scatter.rep1_v_rep3.pdf", dpi="figure", bbox_inches="tight")


# In[26]:


HUES64_data["rep_1"] = HUES64_data["HUES64_rna_6"] + HUES64_data["HUES64_rna_7"] + HUES64_data["HUES64_rna_8"]
HUES64_data["rep_2"] = HUES64_data["HUES64_rna_9"] + HUES64_data["HUES64_rna_10"] + HUES64_data["HUES64_rna_11"]
HUES64_data["rep_3"] = HUES64_data["HUES64_rna_12"] + HUES64_data["HUES64_rna_13"] + HUES64_data["HUES64_rna_14"]
HUES64_data = HUES64_data[["barcode", "dna_1", "rep_1", "rep_2", "rep_3"]]
HUES64_cols = list(HUES64_data.columns)
HUES64_data.head()


# In[27]:


mESC_data["rep_1"] = mESC_data["mESC_rna_3"]
mESC_data["rep_2"] = mESC_data["mESC_rna_4"]
mESC_data["rep_3"] = mESC_data["mESC_rna_5"]
mESC_data = mESC_data[["barcode", "dna_1", "rep_1", "rep_2", "rep_3"]]
mESC_cols = list(mESC_data.columns)
mESC_data.head()


# ## 6. downsample HUES64 barcodes

# In[28]:


HUES64_data[["rep_1", "rep_2", "rep_3"]].sum()


# In[29]:


mESC_data[["rep_1", "rep_2", "rep_3"]].sum()


# since HUES64 was sequenced ~3x as deeply, let's downsample these to make it match mESC

# In[30]:


# # calculate weights per replicate
# HUES64_tmp = HUES64_data.copy()
# HUES64_tmp["rep_1_p"] = HUES64_tmp["rep_1"]/HUES64_tmp["rep_1"].sum()
# HUES64_tmp["rep_2_p"] = HUES64_tmp["rep_2"]/HUES64_tmp["rep_2"].sum()
# HUES64_tmp["rep_3_p"] = HUES64_tmp["rep_3"]/HUES64_tmp["rep_3"].sum()
# HUES64_tmp.fillna(0, inplace=True)
# HUES64_tmp.head()


# In[31]:


# rep_1_samp_barcs = np.random.choice(list(HUES64_tmp["barcode"]), p=list(HUES64_tmp["rep_1_p"]), size=102000000)
# rep_1_samp_barcs[0:5]


# In[32]:


# rep_1_samp_barcs_uniq, rep_1_samp_counts = np.unique(rep_1_samp_barcs, return_counts=True)
# del rep_1_samp_barcs


# In[33]:


# rep_1_downsamp = pd.DataFrame({"barcode": rep_1_samp_barcs_uniq, "rep_1": rep_1_samp_counts})
# del rep_1_samp_barcs_uniq
# del rep_1_samp_counts
# rep_1_downsamp.head()


# In[34]:


# rep_2_samp_barcs = np.random.choice(list(HUES64_tmp["barcode"]), p=list(HUES64_tmp["rep_2_p"]), size=102000000)
# rep_2_samp_barcs[0:5]


# In[35]:


# rep_2_samp_barcs_uniq, rep_2_samp_counts = np.unique(rep_2_samp_barcs, return_counts=True)
# del rep_2_samp_barcs


# In[36]:


# rep_2_downsamp = pd.DataFrame({"barcode": rep_2_samp_barcs_uniq, "rep_2": rep_2_samp_counts})
# del rep_2_samp_barcs_uniq
# del rep_2_samp_counts
# rep_2_downsamp.head()


# In[37]:


# rep_3_samp_barcs = np.random.choice(list(HUES64_tmp["barcode"]), p=list(HUES64_tmp["rep_3_p"]), size=102000000)
# rep_3_samp_barcs[0:5]


# In[38]:


# rep_3_samp_barcs_uniq, rep_3_samp_counts = np.unique(rep_3_samp_barcs, return_counts=True)
# del rep_3_samp_barcs


# In[39]:


# rep_3_downsamp = pd.DataFrame({"barcode": rep_3_samp_barcs_uniq, "rep_3": rep_3_samp_counts})
# del rep_3_samp_barcs_uniq
# del rep_3_samp_counts
# rep_3_downsamp.head()


# In[40]:


# HUES64_tmp = HUES64_tmp.merge(rep_1_downsamp, on="barcode", how="left").merge(rep_2_downsamp, on="barcode", how="left").merge(rep_3_downsamp, on="barcode", how="left")
# HUES64_tmp.fillna(0, inplace=True)
# HUES64_data = HUES64_tmp[["barcode", "dna_1", "rep_1_y", "rep_2_y", "rep_3_y"]]
# HUES64_data.columns = ["barcode", "dna_1", "rep_1", "rep_2", "rep_3"]
# HUES64_data.head()


# ## 7. filter barcodes

# In[41]:


HUES64_data = HUES64_data.fillna(0)
mESC_data = mESC_data.fillna(0)


# In[42]:


HUES64_data_filt = HUES64_data[HUES64_data["dna_1"] >= barcode_dna_read_threshold]
mESC_data_filt = mESC_data[mESC_data["dna_1"] >= barcode_dna_read_threshold]


# In[43]:


HUES64_reps = [x for x in HUES64_data_filt.columns if "rep_" in x]
mESC_reps = [x for x in mESC_data_filt.columns if "rep_" in x]


# In[44]:


HUES64_data_filt[HUES64_reps] = HUES64_data_filt[HUES64_data_filt > barcode_rna_read_threshold][HUES64_reps]
HUES64_data_filt.head()


# In[45]:


mESC_data_filt[mESC_reps] = mESC_data_filt[mESC_data_filt > barcode_rna_read_threshold][mESC_reps]
mESC_data_filt.head()


# In[46]:


all_names = ["HUES64", "mESC"]

all_dfs = [HUES64_data_filt, mESC_data_filt]

all_cols = [HUES64_cols, mESC_cols]

print("FILTERING RESULTS:")
for n, df, cs in zip(all_names, all_dfs, all_cols):
    index_len = len(index)
        
    dna_barc_len = len(df)
    dna_barc_perc = (float(dna_barc_len)/index_len)*100
    
    print("%s: from %s barcodes to %s at DNA level (%s%%)" % (n, index_len, dna_barc_len, dna_barc_perc))
    
    reps = [x for x in cs if "rep_" in x]
    
    for r in reps:
        rep = r.split("_")[1]
        
        rna_barc_len = sum(~pd.isnull(df[r]))
        rna_barc_perc = (float(rna_barc_len)/index_len)*100
        
        print("\trep %s: %s barcodes at RNA level (%s%%)" % (rep, rna_barc_len, rna_barc_perc))
    print("")


# ## 8. filter elements

# In[47]:


HUES64_data_filt = HUES64_data_filt.merge(index, on="barcode", how="inner")
mESC_data_filt = mESC_data_filt.merge(index, on="barcode", how="inner")


# In[48]:


HUES64_barcodes_per_elem = HUES64_data_filt.groupby(["unique_name", "tile_type"])["barcode"].agg("count").reset_index()
HUES64_barcodes_per_elem_neg = HUES64_barcodes_per_elem[HUES64_barcodes_per_elem["tile_type"].isin(["RANDOM", "SCRAMBLED"])]
HUES64_barcodes_per_elem_no_neg = HUES64_barcodes_per_elem[~HUES64_barcodes_per_elem["tile_type"].isin(["RANDOM", "SCRAMBLED"])]

HUES64_barcodes_per_elem_no_neg_filt = HUES64_barcodes_per_elem_no_neg[HUES64_barcodes_per_elem_no_neg["barcode"] >= n_barcodes_per_elem_threshold]
HUES64_total_elems_rep = len(HUES64_barcodes_per_elem_no_neg)
HUES64_total_elems_filt_rep = len(HUES64_barcodes_per_elem_no_neg_filt)


# In[49]:


mESC_barcodes_per_elem = mESC_data_filt.groupby(["unique_name", "tile_type"])["barcode"].agg("count").reset_index()
mESC_barcodes_per_elem_neg = mESC_barcodes_per_elem[mESC_barcodes_per_elem["tile_type"].isin(["RANDOM", "SCRAMBLED"])]
mESC_barcodes_per_elem_no_neg = mESC_barcodes_per_elem[~mESC_barcodes_per_elem["tile_type"].isin(["RANDOM", "SCRAMBLED"])]

mESC_barcodes_per_elem_no_neg_filt = mESC_barcodes_per_elem_no_neg[mESC_barcodes_per_elem_no_neg["barcode"] >= n_barcodes_per_elem_threshold]
mESC_total_elems_rep = len(mESC_barcodes_per_elem_no_neg)
mESC_total_elems_filt_rep = len(mESC_barcodes_per_elem_no_neg_filt)


# In[50]:


print("ELEMENT FILTERING RESULTS:")
print("HUES64: filtered %s elements to %s represented at >= %s barcodes (%s%%)" % (HUES64_total_elems_rep, HUES64_total_elems_filt_rep,
                                                                                   n_barcodes_per_elem_threshold,
                                                                                   float(HUES64_total_elems_filt_rep)/HUES64_total_elems_rep*100))


# In[51]:


print("HUES64: filtered %s elements to %s represented at >= %s barcodes (%s%%)" % (mESC_total_elems_rep, mESC_total_elems_filt_rep,
                                                                                   n_barcodes_per_elem_threshold,
                                                                                   float(mESC_total_elems_filt_rep)/mESC_total_elems_rep*100))


# In[52]:


HUES64_good_elems = list(HUES64_barcodes_per_elem_no_neg_filt["unique_name"]) + list(HUES64_barcodes_per_elem_neg["unique_name"])
mESC_good_elems = list(mESC_barcodes_per_elem_no_neg_filt["unique_name"]) + list(mESC_barcodes_per_elem_neg["unique_name"])


# In[53]:


HUES64_data_filt = HUES64_data_filt[HUES64_data_filt["unique_name"].isin(HUES64_good_elems)]
mESC_data_filt = mESC_data_filt[mESC_data_filt["unique_name"].isin(mESC_good_elems)]


# ## 9. heatmap comparing barcode counts [biological replicates only]

# In[54]:


HUES64_cols = ["barcode"]
mESC_cols = ["barcode"]


# In[55]:


HUES64_cols.extend(["HUES64_%s" % x for x in HUES64_reps])
mESC_cols.extend(["mESC_%s" % x for x in mESC_reps])
HUES64_cols


# In[56]:


HUES64_counts = HUES64_data_filt.copy()
mESC_counts = mESC_data_filt.copy()
mESC_counts.head()


# In[57]:


HUES64_counts = HUES64_counts[["barcode", "rep_1", "rep_2", "rep_3"]]
mESC_counts = mESC_counts[["barcode", "rep_1", "rep_2", "rep_3"]]
HUES64_counts.head()


# In[58]:


HUES64_counts.columns = HUES64_cols
mESC_counts.columns = mESC_cols
HUES64_cols


# In[59]:


all_samples = HUES64_counts.merge(mESC_counts, on="barcode", how="outer")
all_samples.drop("barcode", axis=1, inplace=True)
cols = [x for x in HUES64_cols if x != "barcode"]
all_samples[cols] = np.log10(all_samples[cols]+1)
all_samples_corr = all_samples.corr(method="pearson")


# In[60]:


cmap = sns.cubehelix_palette(as_cmap=True)
cg = sns.clustermap(all_samples_corr, figsize=(2.5,2.5), cmap=cmap, annot=True)
_ = plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.suptitle("pearson correlation of replicates\nlog10+1 counts of barcodes with >5 DNA counts")
plt.subplots_adjust(top=0.8)
cg.savefig("biological_rep_corr_heatmap_filt.pearson.pdf", dpi="figure", transparent=True, bbox_inches="tight")


# ## 10. write final files

# In[62]:


HUES64_counts = HUES64_data_filt[["barcode", "dna_1", "rep_1", "rep_2", "rep_3"]]
mESC_counts = mESC_data_filt[["barcode", "dna_1", "rep_1", "rep_2", "rep_3"]]


# In[63]:


HUES64_counts.to_csv("%s/%s" % (counts_dir, HUES64_out_f), sep="\t", header=True, index=False)


# In[64]:


mESC_counts.to_csv("%s/%s" % (counts_dir, mESC_out_f), sep="\t", header=True, index=False)

