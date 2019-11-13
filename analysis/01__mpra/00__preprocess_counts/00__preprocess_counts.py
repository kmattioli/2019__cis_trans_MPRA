
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
        rep = re.findall(r'\d+', f.split("__")[1])[0]
        tmp = pd.read_table("%s/%s" % (counts_dir, f), sep="\t")
        tmp.columns = ["barcode", "rna_%s" % rep]
        data_cols.append("rna_%s" % rep)
        data = data.merge(tmp, on="barcode", how="outer")
    return data, data_cols


# ## variables

# In[6]:


# CHANGE THIS BACK TO COUNTS DIR
#counts_dir = "../../../data/02__mpra/01__counts"
counts_dir = "../../../GEO_submission"


# In[7]:


barcode_dna_read_threshold = 5
barcode_rna_read_threshold = 0
n_barcodes_per_elem_threshold = 3


# ### DNA files

# In[8]:


dna_f = ["DNA__rep1.BARCODES.txt"]


# ### RNA files

# In[9]:


HUES64_rna_f = ["HUES64__rep1__tfxn1.BARCODES.txt", "HUES64__rep1__tfxn2.BARCODES.txt", 
                "HUES64__rep1__tfxn3.BARCODES.txt", "HUES64__rep2__tfxn1.BARCODES.txt",
                "HUES64__rep2__tfxn2.BARCODES.txt", "HUES64__rep2__tfxn3.BARCODES.txt",
                "HUES64__rep3__tfxn1.BARCODES.txt", "HUES64__rep3__tfxn2.BARCODES.txt",
                "HUES64__rep3__tfxn3.BARCODES.txt"]
HUES64_out_f = "HUES64__all_counts.txt"


# In[10]:


# re name once we get sample sheet from chiara if needed
mESC_rna_f = ["mESC__rep1__tfxn1.BARCODES.txt", "mESC__rep2__tfxn1.BARCODES.txt", "mESC__rep3__tfxn1.BARCODES.txt"]
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


HUES64_data.columns = ["barcode", "dna_1", "HUES64_rep1_tfxn1", "HUES64_rep1_tfxn2", "HUES64_rep1_tfxn3",
                       "HUES64_rep2_tfxn1", "HUES64_rep2_tfxn2", "HUES64_rep2_tfxn3", "HUES64_rep3_tfxn1",
                       "HUES64_rep3_tfxn2", "HUES64_rep3_tfxn3"]


# In[18]:


mESC_data.columns = ["barcode", "dna_1", "mESC_rep1_tfxn1", "mESC_rep2_tfxn1", "mESC_rep3_tfxn1"]


# ## 4. heatmap showing replicate corrs and clustering

# In[19]:


# heatmap incl all libraries
tmp = HUES64_data.merge(mESC_data, on=["barcode", "dna_1"])
tmp = tmp.set_index("barcode")
tmp.drop("dna_1", axis=1, inplace=True)
tmp_cols = tmp.columns
tmp[tmp_cols] = np.log10(tmp[tmp_cols] + 1)
tmp_corr = tmp.corr(method="pearson")

cmap = sns.cubehelix_palette(as_cmap=True)
cg = sns.clustermap(tmp_corr, figsize=(5, 5), cmap=cmap, annot=True, vmin=0.7)
_ = plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.suptitle("pearson correlation of replicates\nlog10+1 counts of all barcodes")
#cg.savefig("rep_and_lib_corr_heatmap.pdf", dpi="figure", bbox_inches="tight")


# ## 5. sum technical replicates

# In[20]:


HUES64_data["HUES64_rep1"] = HUES64_data[["HUES64_rep1_tfxn1", "HUES64_rep1_tfxn2", "HUES64_rep1_tfxn3"]].sum(axis=1)
HUES64_data["HUES64_rep2"] = HUES64_data[["HUES64_rep2_tfxn1", "HUES64_rep2_tfxn2", "HUES64_rep2_tfxn3"]].sum(axis=1)
HUES64_data["HUES64_rep3"] = HUES64_data[["HUES64_rep3_tfxn1", "HUES64_rep3_tfxn2", "HUES64_rep3_tfxn3"]].sum(axis=1)

old_cols = [x for x in HUES64_data.columns if "_tfxn" in x]
HUES64_data.drop(old_cols, axis=1, inplace=True)
HUES64_data.columns = ["barcode", "dna_1", "rep_1", "rep_2", "rep_3"]
HUES64_data.head()


# In[21]:


mESC_data.columns = ["barcode", "dna_1", "rep_1", "rep_2", "rep_3"]
mESC_data.head()


# In[22]:


HUES64_data[["rep_1", "rep_2", "rep_3"]].sum(axis=0)


# In[23]:


mESC_data[["rep_1", "rep_2", "rep_3"]].sum(axis=0)


# ## 6. filter barcodes

# In[24]:


HUES64_data = HUES64_data.fillna(0)
mESC_data = mESC_data.fillna(0)


# In[25]:


HUES64_data_filt = HUES64_data[HUES64_data["dna_1"] >= barcode_dna_read_threshold]
mESC_data_filt = mESC_data[mESC_data["dna_1"] >= barcode_dna_read_threshold]


# In[26]:


HUES64_reps = [x for x in HUES64_data_filt.columns if "rep_" in x]
mESC_reps = [x for x in mESC_data_filt.columns if "rep_" in x]


# In[27]:


HUES64_data_filt[HUES64_reps] = HUES64_data_filt[HUES64_data_filt > barcode_rna_read_threshold][HUES64_reps]
HUES64_data_filt.head()


# In[28]:


mESC_data_filt[mESC_reps] = mESC_data_filt[mESC_data_filt > barcode_rna_read_threshold][mESC_reps]
mESC_data_filt.head()


# In[32]:


all_names = ["HUES64", "mESC"]

all_dfs = [HUES64_data_filt, mESC_data_filt]

all_cols = [HUES64_data.columns, mESC_data.columns]


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


# ## 7. filter elements

# In[33]:


HUES64_data_filt = HUES64_data_filt.merge(index, on="barcode", how="inner")
mESC_data_filt = mESC_data_filt.merge(index, on="barcode", how="inner")


# In[34]:


HUES64_barcodes_per_elem = HUES64_data_filt.groupby(["unique_name", "tile_type"])["barcode"].agg("count").reset_index()
HUES64_barcodes_per_elem_neg = HUES64_barcodes_per_elem[HUES64_barcodes_per_elem["tile_type"].isin(["RANDOM", "SCRAMBLED"])]
HUES64_barcodes_per_elem_no_neg = HUES64_barcodes_per_elem[~HUES64_barcodes_per_elem["tile_type"].isin(["RANDOM", "SCRAMBLED"])]

HUES64_barcodes_per_elem_no_neg_filt = HUES64_barcodes_per_elem_no_neg[HUES64_barcodes_per_elem_no_neg["barcode"] >= n_barcodes_per_elem_threshold]
HUES64_total_elems_rep = len(HUES64_barcodes_per_elem_no_neg)
HUES64_total_elems_filt_rep = len(HUES64_barcodes_per_elem_no_neg_filt)


# In[35]:


mESC_barcodes_per_elem = mESC_data_filt.groupby(["unique_name", "tile_type"])["barcode"].agg("count").reset_index()
mESC_barcodes_per_elem_neg = mESC_barcodes_per_elem[mESC_barcodes_per_elem["tile_type"].isin(["RANDOM", "SCRAMBLED"])]
mESC_barcodes_per_elem_no_neg = mESC_barcodes_per_elem[~mESC_barcodes_per_elem["tile_type"].isin(["RANDOM", "SCRAMBLED"])]

mESC_barcodes_per_elem_no_neg_filt = mESC_barcodes_per_elem_no_neg[mESC_barcodes_per_elem_no_neg["barcode"] >= n_barcodes_per_elem_threshold]
mESC_total_elems_rep = len(mESC_barcodes_per_elem_no_neg)
mESC_total_elems_filt_rep = len(mESC_barcodes_per_elem_no_neg_filt)


# In[36]:


print("ELEMENT FILTERING RESULTS:")
print("HUES64: filtered %s elements to %s represented at >= %s barcodes (%s%%)" % (HUES64_total_elems_rep, HUES64_total_elems_filt_rep,
                                                                                   n_barcodes_per_elem_threshold,
                                                                                   float(HUES64_total_elems_filt_rep)/HUES64_total_elems_rep*100))


# In[37]:


print("HUES64: filtered %s elements to %s represented at >= %s barcodes (%s%%)" % (mESC_total_elems_rep, mESC_total_elems_filt_rep,
                                                                                   n_barcodes_per_elem_threshold,
                                                                                   float(mESC_total_elems_filt_rep)/mESC_total_elems_rep*100))


# In[38]:


HUES64_good_elems = list(HUES64_barcodes_per_elem_no_neg_filt["unique_name"]) + list(HUES64_barcodes_per_elem_neg["unique_name"])
mESC_good_elems = list(mESC_barcodes_per_elem_no_neg_filt["unique_name"]) + list(mESC_barcodes_per_elem_neg["unique_name"])


# In[39]:


HUES64_data_filt = HUES64_data_filt[HUES64_data_filt["unique_name"].isin(HUES64_good_elems)]
mESC_data_filt = mESC_data_filt[mESC_data_filt["unique_name"].isin(mESC_good_elems)]


# ## 8. heatmap comparing barcode counts [biological replicates only]

# In[40]:


HUES64_cols = ["barcode"]
mESC_cols = ["barcode"]


# In[41]:


HUES64_cols.extend(["HUES64_%s" % x for x in HUES64_reps])
mESC_cols.extend(["mESC_%s" % x for x in mESC_reps])
HUES64_cols


# In[42]:


HUES64_counts = HUES64_data_filt.copy()
mESC_counts = mESC_data_filt.copy()
mESC_counts.head()


# In[43]:


HUES64_counts = HUES64_counts[["barcode", "rep_1", "rep_2", "rep_3"]]
mESC_counts = mESC_counts[["barcode", "rep_1", "rep_2", "rep_3"]]
HUES64_counts.head()


# In[44]:


HUES64_counts.columns = HUES64_cols
mESC_counts.columns = mESC_cols
HUES64_cols


# In[45]:


all_samples = HUES64_counts.merge(mESC_counts, on="barcode", how="outer")
all_samples.drop("barcode", axis=1, inplace=True)
cols = [x for x in HUES64_cols if x != "barcode"]
all_samples[cols] = np.log10(all_samples[cols]+1)
all_samples_corr = all_samples.corr(method="pearson")


# In[46]:


cmap = sns.cubehelix_palette(as_cmap=True)
cg = sns.clustermap(all_samples_corr, figsize=(2.5,2.5), cmap=cmap, annot=True)
_ = plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.suptitle("pearson correlation of replicates\nlog10+1 counts of barcodes with >5 DNA counts")
plt.subplots_adjust(top=0.8)
cg.savefig("biological_rep_corr_heatmap_filt.pearson.pdf", dpi="figure", transparent=True, bbox_inches="tight")


# ## 10. write final files

# In[47]:


HUES64_counts = HUES64_data_filt[["barcode", "dna_1", "rep_1", "rep_2", "rep_3"]]
mESC_counts = mESC_data_filt[["barcode", "dna_1", "rep_1", "rep_2", "rep_3"]]


# In[48]:


HUES64_counts.to_csv("%s/%s" % (counts_dir, HUES64_out_f), sep="\t", header=True, index=False)


# In[49]:


mESC_counts.to_csv("%s/%s" % (counts_dir, mESC_out_f), sep="\t", header=True, index=False)

