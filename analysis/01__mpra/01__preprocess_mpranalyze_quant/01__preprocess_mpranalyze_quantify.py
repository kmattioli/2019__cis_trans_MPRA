
# coding: utf-8

# # 01__preprocess_mpranalyze_quantify
# 
# in this notebook, i take the tidy-formatted counts and re-shape them to be in the input format needed to run MPRAnalyze. i also add some additional controls to serve as negative controls in the cell-type comparison model. counterintuitively, these negative controls are sampled from our positive controls: the null hypothesis is that their activities should not be too different between hESCs and mESCs, since it's the CMV promoter. there are 4 "tiles" of the CMV promoter, and i sample 13 barcodes from each tile 100 times, to create a total of 400 "negative" controls (...from our positive controls). [note: negative controls in the quantification model are just random sequences, expected to have no activity].

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


def get_barc_id(row):
    str_split = row.tile_id.split(".")
    return str_split[-1]


# In[5]:


def dna_status(row, barc_thresh, perc_barc_thresh, ctrl_elems):
    samp_cols = [x for x in row.index if "samp:" in x]
    vals = row[samp_cols]
    if row.element not in ctrl_elems:
        tot_barcs = len(vals)
        n_barcs_above_thresh = len([x for x in vals if x >= barc_thresh])
        perc_barcs_above_thresh = n_barcs_above_thresh / tot_barcs
        if perc_barcs_above_thresh >= perc_barc_thresh:
            return "good"
        else:
            return "bad"
    else:
        return "good"


# In[6]:


def get_ctrl_status(row):
    if pd.isnull(row.tile_type):
        return False
    elif row.tile_type == "RANDOM":
        return True
    else:
        return False


# ## variables

# In[7]:


counts_dir = "../../../data/02__mpra/01__counts"


# In[8]:


HUES64_data_f = "%s/HUES64__all_counts.txt" % counts_dir


# In[9]:


mESC_data_f = "%s/mESC__all_counts.txt" % counts_dir


# In[10]:


index_f = "../../../data/01__design/02__index/TWIST_pool4_v8_final.txt.gz"


# ## 1. import data

# In[11]:


HUES64_data = pd.read_table(HUES64_data_f, sep="\t")
HUES64_data.head()


# In[12]:


mESC_data = pd.read_table(mESC_data_f, sep="\t")
mESC_data.head()


# In[13]:


index = pd.read_table(index_f, sep="\t")
index_elem = index[["element", "tile_type"]].drop_duplicates()


# ## 2. merge data w/ index

# In[14]:


HUES64_data.columns = ["barcode", "dna_1", "HUES64_rep1", "HUES64_rep2", "HUES64_rep3"]
mESC_data.columns = ["barcode", "dna_1", "mESC_rep1", "mESC_rep2", "mESC_rep3"]


# In[15]:


all_counts = HUES64_data.merge(mESC_data, on=["barcode", "dna_1"])
all_counts.head()


# In[16]:


df = all_counts.merge(index[["barcode", "element", "tile_type", "tile_id"]], on="barcode")
df["barc_id"] = df.apply(get_barc_id, axis=1).astype(int)
df.head()


# ## 3. create separate dfs for dna & rna counts

# In[17]:


dna_counts = df[["element", "barcode", "tile_type", "barc_id", "dna_1"]]
rna_counts = df[["element", "barcode", "tile_type", "barc_id", "HUES64_rep1", "HUES64_rep2", "HUES64_rep3",
                 "mESC_rep1", "mESC_rep2", "mESC_rep3"]]
rna_counts.head()


# In[18]:


dna_counts = dna_counts.sort_values(by=["element", "barc_id"])
rna_counts = rna_counts.sort_values(by=["element", "barc_id"])


# ## 4. sum up all dna & rna counts, across all elements, for library depth correction

# In[19]:


dna_counts_elem = dna_counts.groupby(["element", "tile_type"])["dna_1"].agg("sum").reset_index()
dna_counts_elem = dna_counts_elem[["element", "dna_1"]]
dna_counts_elem.set_index("element", inplace=True)
dna_counts_elem.head()


# In[20]:


rna_counts_elem = rna_counts.groupby(["element", "tile_type"])["HUES64_rep1", "HUES64_rep2", "HUES64_rep3", "mESC_rep1", "mESC_rep2", "mESC_rep3"].agg("sum").reset_index()
rna_counts_elem = rna_counts_elem[["element", "HUES64_rep1", "HUES64_rep2", "HUES64_rep3", "mESC_rep1", "mESC_rep2", "mESC_rep3"]]
rna_counts_elem.set_index("element", inplace=True)
rna_counts_elem.head()


# ## 5. create annotation file for library depth correction

# In[21]:


# col annotations for depth estimation
dna_depth_anns = {"dna_1": {"sample": "1", "condition": "dna"}}
rna_depth_anns = {"HUES64_rep1": {"sample": "1", "condition": "HUES64"}, 
                  "HUES64_rep2": {"sample": "2", "condition": "HUES64"},
                  "HUES64_rep3": {"sample": "3", "condition": "HUES64"},
                  "mESC_rep1": {"sample": "1", "condition": "mESC"},
                  "mESC_rep2": {"sample": "2", "condition": "mESC"},
                  "mESC_rep3": {"sample": "3", "condition": "mESC"}}

dna_depth_anns = pd.DataFrame.from_dict(dna_depth_anns).T
rna_depth_anns = pd.DataFrame.from_dict(rna_depth_anns).T
rna_depth_anns


# ## 6. write library depth correction files

# In[22]:


# write depth estimation files
mpranalyze_dir = "%s/mpranalyze_files" % counts_dir

dna_depth_anns.to_csv("%s/dna_col_ann.for_depth_estimation.mpranalyze.txt" % mpranalyze_dir, sep="\t")
rna_depth_anns.to_csv("%s/rna_col_ann.for_depth_estimation.mpranalyze.txt" % mpranalyze_dir, sep="\t")

dna_counts_elem.to_csv("%s/dna_counts.for_depth_estimation.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)
rna_counts_elem.to_csv("%s/rna_counts.for_depth_estimation.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)


# ## 7. to run MPRAnalyze, get data in pivot format (opposite of tidy format)

# In[23]:


# first filter to the TSSs we care about quantifying
tss_elems = list(index[index["name"].str.contains("EVO_TSS")]["element"].unique())
ctrl_elems = list(index[index["tile_type"] == "RANDOM"]["element"].unique())
pos_ctrl_elems = list(index[index["tile_type"] == "CONTROL"]["element"].unique())
print(len(tss_elems))
print(len(ctrl_elems))
print(len(pos_ctrl_elems))
good_elems = tss_elems + ctrl_elems
len(good_elems)


# In[24]:


dna_counts_filt = dna_counts[dna_counts["element"].isin(good_elems)]
rna_counts_filt = rna_counts[rna_counts["element"].isin(good_elems)]
len(rna_counts_filt)


# In[25]:


dna_counts_pos_ctrls = dna_counts[dna_counts["element"].isin(pos_ctrl_elems)]
rna_counts_pos_ctrls = rna_counts[rna_counts["element"].isin(pos_ctrl_elems)]
len(rna_counts_pos_ctrls)


# In[26]:


dna_counts_filt = dna_counts_filt.melt(id_vars=["element", "barcode", "tile_type", "barc_id"])
rna_counts_filt = rna_counts_filt.melt(id_vars=["element", "barcode", "tile_type", "barc_id"])
rna_counts_filt.head()


# In[27]:


dna_counts_pos_ctrls = dna_counts_pos_ctrls.melt(id_vars=["element", "barcode", "tile_type", "barc_id"])
rna_counts_pos_ctrls = rna_counts_pos_ctrls.melt(id_vars=["element", "barcode", "tile_type", "barc_id"])
rna_counts_pos_ctrls.head()


# In[28]:


dna_counts_filt["samp_id"] = "samp:" + dna_counts_filt["variable"] + "__barc:" + dna_counts_filt["barc_id"].astype(str)
rna_counts_filt["samp_id"] = "samp:" + rna_counts_filt["variable"] + "__barc:" + rna_counts_filt["barc_id"].astype(str)
rna_counts_filt.sample(5)


# In[29]:


dna_counts_pos_ctrls["samp_id"] = "samp:" + dna_counts_pos_ctrls["variable"] + "__barc:" + dna_counts_pos_ctrls["barc_id"].astype(str)
rna_counts_pos_ctrls["samp_id"] = "samp:" + rna_counts_pos_ctrls["variable"] + "__barc:" + rna_counts_pos_ctrls["barc_id"].astype(str)
rna_counts_pos_ctrls.sample(5)


# In[30]:


dna_counts_piv = dna_counts_filt.pivot(index="element", columns="samp_id", values="value").reset_index()
rna_counts_piv = rna_counts_filt.pivot(index="element", columns="samp_id", values="value").reset_index()
rna_counts_piv.head()


# In[31]:


dna_counts_pos_ctrl_piv = dna_counts_pos_ctrls.pivot(index="element", columns="samp_id", values="value").reset_index()
rna_counts_pos_ctrl_piv = rna_counts_pos_ctrls.pivot(index="element", columns="samp_id", values="value").reset_index()
rna_counts_pos_ctrl_piv.head()


# In[32]:


dna_counts_piv.fillna(0, inplace=True)
rna_counts_piv.fillna(0, inplace=True)
rna_counts_piv.head()


# In[33]:


dna_counts_pos_ctrl_piv.fillna(0, inplace=True)
rna_counts_pos_ctrl_piv.fillna(0, inplace=True)


# ## 8. filter: remove any elements that don't have >=50% of barcodes with DNA counts >= 10

# In[34]:


dna_counts_piv["dna_status"] = dna_counts_piv.apply(dna_status, barc_thresh=10, perc_barc_thresh=0.5, 
                                                    ctrl_elems=ctrl_elems, axis=1)
dna_counts_pos_ctrl_piv["dna_status"] = dna_counts_pos_ctrl_piv.apply(dna_status, barc_thresh=10, 
                                                                      perc_barc_thresh=0.5, 
                                                                      ctrl_elems=ctrl_elems, axis=1)
dna_counts_piv.dna_status.value_counts()


# In[35]:


good_dna_elems = list(dna_counts_piv[dna_counts_piv["dna_status"] == "good"]["element"])
good_pos_ctrl_dna_elems = list(dna_counts_pos_ctrl_piv[dna_counts_pos_ctrl_piv["dna_status"] == "good"]["element"])


# In[36]:


dna_counts_piv_filt = dna_counts_piv[dna_counts_piv["element"].isin(good_dna_elems)]
dna_counts_piv_filt.drop("dna_status", axis=1, inplace=True)
rna_counts_piv_filt = rna_counts_piv[rna_counts_piv["element"].isin(good_dna_elems)]
print(len(dna_counts_piv_filt))
print(len(rna_counts_piv_filt))


# In[37]:


dna_counts_pos_ctrl_piv_filt = dna_counts_pos_ctrl_piv[dna_counts_pos_ctrl_piv["element"].isin(good_pos_ctrl_dna_elems)]
dna_counts_pos_ctrl_piv_filt.drop("dna_status", axis=1, inplace=True)
rna_counts_pos_ctrl_piv_filt = rna_counts_pos_ctrl_piv[rna_counts_pos_ctrl_piv["element"].isin(good_pos_ctrl_dna_elems)]
print(len(dna_counts_pos_ctrl_piv_filt))
print(len(rna_counts_pos_ctrl_piv_filt))


# ## 9. add new negative controls -- which are sampled from positive controls -- for MPRAnalyze comparison b/w cell types

# In[38]:


barc_ids = list(range(1, 61))
n_samps = 100
elems = list(dna_counts_pos_ctrl_piv_filt["element"])
elems


# In[39]:


rep_map_cols = list(set([x.split("__")[0] for x in list(rna_counts_pos_ctrl_piv_filt.columns) if x != "element"]))
rep_map_cols


# In[40]:


neg_ctrl_dna_counts = pd.DataFrame()
neg_ctrl_rna_counts = pd.DataFrame()

for i, elem in enumerate(elems):
    elem_dna_data = dna_counts_pos_ctrl_piv_filt[dna_counts_pos_ctrl_piv_filt["element"] == elem]
    elem_rna_data = rna_counts_pos_ctrl_piv_filt[rna_counts_pos_ctrl_piv_filt["element"] == elem]
    
    for j in range(n_samps):
        barcs_sampled = np.random.choice(barc_ids, size=13)
        
        dna_cols_sampled = ["element"]
        dna_cols_sampled.extend(["samp:dna_1__barc:%s" % x for x in barcs_sampled])
        new_dna_cols = ["element"]
        new_dna_cols.extend(["samp:dna_1__barc:%s" % x for x in range(1, 14)])
        
        rna_cols_sampled = ["element"]
        new_rna_cols = ["element"]
        for rep in rep_map_cols:
            rna_cols_sampled.extend(["%s__barc:%s" % (rep, x) for x in barcs_sampled])
            new_rna_cols.extend(["%s__barc:%s" % (rep, x) for x in range(1, 14)])
        
        # subsample dataframe w/ columns we just defined
        elem_dna_data_sampled = elem_dna_data[dna_cols_sampled]
        elem_rna_data_sampled = elem_rna_data[rna_cols_sampled]   
        
        # rename columns
        elem_dna_data_sampled.columns = new_dna_cols
        elem_rna_data_sampled.columns = new_rna_cols
                
        # rename element with element + samp #
        elem_dna_data_sampled["element"] = elem + "__samp%s" % (j+1)
        elem_rna_data_sampled["element"] = elem + "__samp%s" % (j+1)
        
        # for error checking -- print out the barcode that should be barcode 1"
#         print("error checking for %s__samp%s: %s" % (elem, j+1, barcs_sampled[0]))

        
        # append
        neg_ctrl_dna_counts = neg_ctrl_dna_counts.append(elem_dna_data_sampled)
        neg_ctrl_rna_counts = neg_ctrl_rna_counts.append(elem_rna_data_sampled)


# ## 10. get negative control IDs [negative controls only for quantification]

# In[41]:


dna_counts_piv_filt = dna_counts_piv_filt.append(neg_ctrl_dna_counts)
rna_counts_piv_filt = rna_counts_piv_filt.append(neg_ctrl_rna_counts)
print(len(dna_counts_piv_filt))
print(len(rna_counts_piv_filt))


# In[42]:


ctrls = rna_counts_piv_filt[["element"]].merge(index_elem[["element", "tile_type"]], on="element", how="left")
print(len(ctrls))
ctrls.head()


# In[43]:


ctrls["ctrl_status"] = ctrls.apply(get_ctrl_status, axis=1)
ctrls.sample(5)


# ## 11. create overall annotation file

# In[44]:


dna_cols = [x for x in dna_counts_piv_filt.columns if "samp:" in x]
rna_cols = [x for x in rna_counts_piv_filt.columns if "samp:" in x]


# In[45]:


dna_col_ann = {}
rna_col_ann = {}
for cols, ann in zip([dna_cols, rna_cols], [dna_col_ann, rna_col_ann]):
    for col in cols:
        samp = col.split("__")[0].split("_")[-1]
        cond = col.split(":")[1].split("_")[0]
        barc = col.split(":")[-1]
        ann[col] = {"sample": samp, "condition": cond, "barcode": barc}

dna_col_ann = pd.DataFrame.from_dict(dna_col_ann, orient="index")
rna_col_ann = pd.DataFrame.from_dict(rna_col_ann, orient="index")
rna_col_ann.sample(5)


# ## 12. write final files [for quantification analysis]

# In[46]:


dna_counts_piv_filt.set_index("element", inplace=True)
rna_counts_piv_filt.set_index("element", inplace=True)


# In[47]:


# write final files
dna_col_ann.to_csv("%s/dna_col_ann.mpranalyze.for_quantification.txt" % mpranalyze_dir, sep="\t")
rna_col_ann.to_csv("%s/rna_col_ann.mpranalyze.for_quantification.txt" % mpranalyze_dir, sep="\t")

ctrls = ctrls[["element", "ctrl_status"]]
ctrls.to_csv("%s/ctrl_status.mpranalyze.for_quantification.txt" % mpranalyze_dir, sep="\t", index=False)

dna_counts_piv_filt.to_csv("%s/dna_counts.mpranalyze.for_quantification.txt" % mpranalyze_dir, sep="\t", index=True)
rna_counts_piv_filt.to_csv("%s/rna_counts.mpranalyze.for_quantification.txt" % mpranalyze_dir, sep="\t", index=True)

