
# coding: utf-8

# # 04__preprocess_mpranalyze_compare
# 
# in this notebook, i re-shape the counts data to run MPRAnalyze comparison mode. importantly, i also include the negative controls for comparison mode that I made in the previous notebook (01). 

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import itertools
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


def ctrl_status(row):
    if "CONTROL" in row.comp_id:
        return True
    else:
        return False


# ## variables

# In[5]:


mpranalyze_dir = "../../../data/02__mpra/01__counts/mpranalyze_files"


# In[6]:


dna_counts_f = "%s/dna_counts.mpranalyze.for_quantification.txt" % mpranalyze_dir
rna_counts_f = "%s/rna_counts.mpranalyze.for_quantification.txt" % mpranalyze_dir


# In[7]:


data_dir = "../../../data/02__mpra/02__activs"


# In[8]:


# human_max_f = "%s/human_TSS_vals.max_tile.txt" % data_dir
# mouse_max_f = "%s/mouse_TSS_vals.max_tile.txt" % data_dir


# In[9]:


tss_map_f = "../../../data/01__design/01__mpra_list/mpra_tss.with_ids.RECLASSIFIED_WITH_MAX.txt"


# In[10]:


dna_col_ann_f = "%s/dna_col_ann.mpranalyze.for_quantification.txt" % mpranalyze_dir
rna_col_ann_f = "%s/rna_col_ann.mpranalyze.for_quantification.txt" % mpranalyze_dir


# In[11]:


human_vals_f = "%s/human_TSS_vals.both_tiles.txt" % data_dir
mouse_vals_f = "%s/mouse_TSS_vals.both_tiles.txt" % data_dir


# ## 1. import data

# In[12]:


dna_counts = pd.read_table(dna_counts_f)
dna_counts.head()


# In[13]:


rna_counts = pd.read_table(rna_counts_f)
rna_counts.head()


# In[14]:


human_vals = pd.read_table(human_vals_f)
mouse_vals = pd.read_table(mouse_vals_f)
human_vals.head()


# In[15]:


tss_map = pd.read_table(tss_map_f, sep="\t")
tss_map.head()


# In[16]:


old_dna_col_ann = pd.read_table(dna_col_ann_f, index_col=0)
old_dna_col_ann.head()


# In[17]:


old_rna_col_ann = pd.read_table(rna_col_ann_f, index_col=0)
old_rna_col_ann.head()


# ## 2. remove any sequences in TSS map that we removed at initial MPRAnalyze (low counts)

# In[18]:


# filter out any elements we removed at initial steps (low dna counts)
human_vals = human_vals[human_vals["element"].isin(dna_counts["element"])]
mouse_vals = mouse_vals[mouse_vals["element"].isin(dna_counts["element"])]


# ## 3. get positive ctrl dna/rna counts

# In[19]:


dna_counts_ctrl = dna_counts[dna_counts["element"].str.contains("samp")]
print(len(dna_counts_ctrl))
rna_counts_ctrl = rna_counts[rna_counts["element"].str.contains("samp")]
print(len(rna_counts_ctrl))


# # first make files needed for seq. comparison (native and cis effects)

# ## 1. merge ortholog pairs w/ counts
# old:::: this time, always pair tile1 with tile1 and tile2 with tile2
# new:::: pair tile1 with tile1 unless maximum is tile2 in both species

# In[20]:


tss_max = tss_map[["hg19_id", "mm9_id", "tile_match"]]
tss_max.head()


# In[21]:


human_vals.head()


# In[22]:


dna_counts_human_all = human_vals[["element", "tss_id", "tss_tile_num"]].merge(dna_counts, on="element").drop_duplicates()
dna_counts_mouse_all = mouse_vals[["element", "tss_id", "tss_tile_num"]].merge(dna_counts, on="element").drop_duplicates()
dna_counts_human_all.head()


# In[23]:


print(len(dna_counts_human_all))
print(len(dna_counts_mouse_all))


# In[24]:


rna_counts_human_all = human_vals[["element", "tss_id", "tss_tile_num"]].merge(rna_counts, on="element").drop_duplicates()
rna_counts_mouse_all = mouse_vals[["element", "tss_id", "tss_tile_num"]].merge(rna_counts, on="element").drop_duplicates()
rna_counts_human_all.head()


# In[25]:


print(len(rna_counts_human_all))
print(len(rna_counts_mouse_all))


# ## 2. merge human/mouse counts into 1 dataframe
# 
# new: merge tile1 with tile1 unless maximum is tile2 in both species (in which case use tile2); only consider tiles where both are avail in both species

# In[26]:


dna_counts_human_tile1 = dna_counts_human_all[dna_counts_human_all["tss_tile_num"] == "tile1"]
dna_counts_human_tile2 = dna_counts_human_all[dna_counts_human_all["tss_tile_num"] == "tile2"]
print(len(dna_counts_human_tile1))
print(len(dna_counts_human_tile2))


# In[27]:


rna_counts_human_tile1 = rna_counts_human_all[rna_counts_human_all["tss_tile_num"] == "tile1"]
rna_counts_human_tile2 = rna_counts_human_all[rna_counts_human_all["tss_tile_num"] == "tile2"]
print(len(rna_counts_human_tile1))
print(len(rna_counts_human_tile2))


# In[28]:


dna_counts_mouse_tile1 = dna_counts_mouse_all[dna_counts_mouse_all["tss_tile_num"] == "tile1"]
dna_counts_mouse_tile2 = dna_counts_mouse_all[dna_counts_mouse_all["tss_tile_num"] == "tile2"]
print(len(dna_counts_mouse_tile1))
print(len(dna_counts_mouse_tile2))


# In[29]:


rna_counts_mouse_tile1 = rna_counts_mouse_all[rna_counts_mouse_all["tss_tile_num"] == "tile1"]
rna_counts_mouse_tile2 = rna_counts_mouse_all[rna_counts_mouse_all["tss_tile_num"] == "tile2"]
print(len(rna_counts_mouse_tile1))
print(len(rna_counts_mouse_tile2))


# In[30]:


#both_tile_ids = tss_map[(tss_map["n_tiles_hg19"] >= 2) & (tss_map["n_tiles_mm9"] >= 2)]
both_tile_ids = tss_map[(~pd.isnull(tss_map["n_tiles_hg19"]) & ~(pd.isnull(tss_map["n_tiles_mm9"])))]
len(both_tile_ids)


# In[31]:


tile1_ids = both_tile_ids[(both_tile_ids["tile_match"] == "tile1:tile1") | 
                          (both_tile_ids["tile_match"] == "tile1:tile2")][["hg19_id", "mm9_id"]].drop_duplicates()
len(tile1_ids)


# In[32]:


tile2_ids = both_tile_ids[(both_tile_ids["tile_match"] == "tile2:tile2")][["hg19_id", "mm9_id"]].drop_duplicates()
len(tile2_ids)


# In[33]:


tss_map_mpra_tile1 = tile1_ids.merge(tss_map, on=["hg19_id", "mm9_id"])
tss_map_mpra_tile1 = tss_map_mpra_tile1.merge(rna_counts_human_tile1, left_on="hg19_id", 
                                              right_on="tss_id").merge(rna_counts_mouse_tile1, left_on="mm9_id", 
                                                                       right_on="tss_id",
                                                                       suffixes=("___seq:human", "___seq:mouse"))
tss_map_mpra_tile1.drop_duplicates(inplace=True)
print(len(tss_map_mpra_tile1))
tss_map_mpra_tile1.head(5)


# In[34]:


tss_map_mpra_tile2 = tile2_ids.merge(tss_map, on=["hg19_id", "mm9_id"])
tss_map_mpra_tile2 = tss_map_mpra_tile2.merge(rna_counts_human_tile2, left_on="hg19_id", 
                                              right_on="tss_id").merge(rna_counts_mouse_tile2, left_on="mm9_id", 
                                                                       right_on="tss_id",
                                                                       suffixes=("___seq:human", "___seq:mouse"))
tss_map_mpra_tile2.drop_duplicates(inplace=True)
print(len(tss_map_mpra_tile2))
tss_map_mpra_tile2.head(5)


# In[35]:


tss_map_dna_tile1 = tile1_ids.merge(tss_map, on=["hg19_id", "mm9_id"])
tss_map_dna_tile1 = tss_map_dna_tile1.merge(dna_counts_human_tile1, left_on="hg19_id", 
                                              right_on="tss_id").merge(dna_counts_mouse_tile1, left_on="mm9_id", 
                                                                       right_on="tss_id",
                                                                       suffixes=("___seq:human", "___seq:mouse"))
tss_map_dna_tile1.drop_duplicates(inplace=True)
print(len(tss_map_dna_tile1))
tss_map_dna_tile1.head(5)


# In[36]:


tss_map_dna_tile2 = tile2_ids.merge(tss_map, on=["hg19_id", "mm9_id"])
tss_map_dna_tile2 = tss_map_dna_tile2.merge(dna_counts_human_tile2, left_on="hg19_id", 
                                              right_on="tss_id").merge(dna_counts_mouse_tile2, left_on="mm9_id", 
                                                                       right_on="tss_id",
                                                                       suffixes=("___seq:human", "___seq:mouse"))
tss_map_dna_tile2.drop_duplicates(inplace=True)
print(len(tss_map_dna_tile2))
tss_map_dna_tile2.head(5)


# old: merge tile1 with tile1 and tile2 with tile2 always

# In[37]:


# tss_map_mpra_tile1 = tss_map.merge(rna_counts_human_tile1, left_on="hg19_id", 
#                                    right_on="tss_id").merge(rna_counts_mouse_tile1, left_on="mm9_id", right_on="tss_id",
#                                                             suffixes=("___seq:human", "___seq:mouse"))
# tss_map_mpra_tile1.drop_duplicates(inplace=True)
# print(len(tss_map_mpra_tile1))
# tss_map_mpra_tile1.head(5)


# In[38]:


# tss_map_mpra_tile2 = tss_map.merge(rna_counts_human_tile2, left_on="hg19_id", 
#                                    right_on="tss_id").merge(rna_counts_mouse_tile2, left_on="mm9_id", right_on="tss_id",
#                                                             suffixes=("___seq:human", "___seq:mouse"))
# tss_map_mpra_tile2.drop_duplicates(inplace=True)
# print(len(tss_map_mpra_tile2))
# tss_map_mpra_tile2.head(5)


# In[39]:


# tss_map_dna_tile1 = tss_map.merge(dna_counts_human_tile1, left_on="hg19_id", 
#                                   right_on="tss_id").merge(dna_counts_mouse_tile1, left_on="mm9_id", right_on="tss_id",
#                                                            suffixes=("___seq:human", "___seq:mouse"))
# tss_map_dna_tile1.drop_duplicates(inplace=True)
# print(len(tss_map_dna_tile1))
# tss_map_dna_tile1.head(5)


# In[40]:


# tss_map_dna_tile2 = tss_map.merge(dna_counts_human_tile2, left_on="hg19_id", 
#                                   right_on="tss_id").merge(dna_counts_mouse_tile2, left_on="mm9_id", right_on="tss_id",
#                                                            suffixes=("___seq:human", "___seq:mouse"))
# tss_map_dna_tile2.drop_duplicates(inplace=True)
# print(len(tss_map_dna_tile2))
# tss_map_dna_tile2.head(5)


# ## 3. assign each pair an ID

# In[41]:


HUES64_rna_cols = [x for x in tss_map_mpra_tile1.columns if "samp:HUES64" in x]
mESC_rna_cols = [x for x in tss_map_mpra_tile1.columns if "samp:mESC" in x]
all_dna_cols = [x for x in tss_map_dna_tile1.columns if "samp:dna" in x]

human_cols = ["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9"]
human_cols.extend(HUES64_rna_cols)

mouse_cols = ["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9"]
mouse_cols.extend(mESC_rna_cols)

dna_cols = ["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9"]
dna_cols.extend(all_dna_cols)

tss_map_mpra_human_tile1 = tss_map_mpra_tile1[human_cols]
tss_map_mpra_mouse_tile1 = tss_map_mpra_tile1[mouse_cols]

tss_map_mpra_human_tile2 = tss_map_mpra_tile2[human_cols]
tss_map_mpra_mouse_tile2 = tss_map_mpra_tile2[mouse_cols]

tss_map_dna_tile1 = tss_map_dna_tile1[dna_cols]
tss_map_dna_tile2 = tss_map_dna_tile2[dna_cols]

tss_map_mpra_human_tile1.head()


# In[42]:


tss_map_mpra_human_tile1["tile_num"] = "tile1"
tss_map_mpra_mouse_tile1["tile_num"] = "tile1"
tss_map_mpra_human_tile2["tile_num"] = "tile2"
tss_map_mpra_mouse_tile2["tile_num"] = "tile2"
tss_map_dna_tile1["tile_num"] = "tile1"
tss_map_dna_tile2["tile_num"] = "tile2"


# In[43]:


# all tile 1s
tss_map_mpra_human_tile1["comp_id"] = tss_map_mpra_human_tile1["hg19_id"] + "__" + tss_map_mpra_human_tile1["biotype_hg19"] + "__" + tss_map_mpra_human_tile1["mm9_id"] + "__" + tss_map_mpra_human_tile1["biotype_mm9"] + "__" + tss_map_mpra_human_tile1["tile_num"] 
tss_map_mpra_mouse_tile1["comp_id"] = tss_map_mpra_mouse_tile1["hg19_id"] + "__" + tss_map_mpra_mouse_tile1["biotype_hg19"] + "__" + tss_map_mpra_mouse_tile1["mm9_id"] + "__" + tss_map_mpra_mouse_tile1["biotype_mm9"] + "__" + tss_map_mpra_mouse_tile1["tile_num"]
tss_map_dna_tile1["comp_id"] = tss_map_dna_tile1["hg19_id"] + "__" + tss_map_dna_tile1["biotype_hg19"] + "__" + tss_map_dna_tile1["mm9_id"] + "__" + tss_map_dna_tile1["biotype_mm9"] + "__" + tss_map_dna_tile1["tile_num"]

# all tile 2s
tss_map_mpra_human_tile2["comp_id"] = tss_map_mpra_human_tile2["hg19_id"] + "__" + tss_map_mpra_human_tile2["biotype_hg19"] + "__" + tss_map_mpra_human_tile2["mm9_id"] + "__" + tss_map_mpra_human_tile2["biotype_mm9"] + "__" + tss_map_mpra_human_tile2["tile_num"] 
tss_map_mpra_mouse_tile2["comp_id"] = tss_map_mpra_mouse_tile2["hg19_id"] + "__" + tss_map_mpra_mouse_tile2["biotype_hg19"] + "__" + tss_map_mpra_mouse_tile2["mm9_id"] + "__" + tss_map_mpra_mouse_tile2["biotype_mm9"] + "__" + tss_map_mpra_mouse_tile2["tile_num"]
tss_map_dna_tile2["comp_id"] = tss_map_dna_tile2["hg19_id"] + "__" + tss_map_dna_tile2["biotype_hg19"] + "__" + tss_map_dna_tile2["mm9_id"] + "__" + tss_map_dna_tile2["biotype_mm9"] + "__" + tss_map_dna_tile2["tile_num"]

# drop redundant tiles
tss_map_mpra_human_tile1.drop(["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9", "tile_num"], axis=1, inplace=True)
tss_map_mpra_mouse_tile1.drop(["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9", "tile_num"], axis=1, inplace=True)
tss_map_dna_tile1.drop(["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9", "tile_num"], axis=1, inplace=True)
tss_map_mpra_human_tile2.drop(["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9", "tile_num"], axis=1, inplace=True)
tss_map_mpra_mouse_tile2.drop(["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9", "tile_num"], axis=1, inplace=True)
tss_map_dna_tile2.drop(["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9", "tile_num"], axis=1, inplace=True)

human_cols = ["comp_id"]
human_cols.extend(HUES64_rna_cols)

mouse_cols = ["comp_id"]
mouse_cols.extend(mESC_rna_cols)

dna_cols = ["comp_id"]
dna_cols.extend(all_dna_cols)

tss_map_mpra_human_tile1 = tss_map_mpra_human_tile1[human_cols]
tss_map_mpra_human_tile2 = tss_map_mpra_human_tile2[human_cols]
tss_map_mpra_mouse_tile1 = tss_map_mpra_mouse_tile1[mouse_cols]
tss_map_mpra_mouse_tile2 = tss_map_mpra_mouse_tile2[mouse_cols]
tss_map_dna_tile1 = tss_map_dna_tile1[dna_cols]
tss_map_dna_tile2 = tss_map_dna_tile2[dna_cols]

tss_map_mpra_human_tile1.head()


# In[44]:


# append tile 1 and tile2
tss_map_mpra_human = tss_map_mpra_human_tile1.append(tss_map_mpra_human_tile2).drop_duplicates()
tss_map_mpra_mouse = tss_map_mpra_mouse_tile1.append(tss_map_mpra_mouse_tile2).drop_duplicates()
tss_map_dna = tss_map_dna_tile1.append(tss_map_dna_tile2).drop_duplicates()
print(len(tss_map_mpra_human))
print(len(tss_map_mpra_mouse))
print(len(tss_map_dna))
tss_map_mpra_human.sample(5)


# In[45]:


# merge human and mouse so both cols in 1 df
tss_map_mpra = tss_map_mpra_human.merge(tss_map_mpra_mouse, on="comp_id")
len(tss_map_mpra)


# In[46]:


# also add dataframe for native comparisons
native_cols = ["comp_id"]
native_human_cols = [x for x in tss_map_mpra.columns if "HUES64" in x and "human" in x]
native_mouse_cols = [x for x in tss_map_mpra.columns if "mESC" in x and "mouse" in x]
native_cols.extend(native_human_cols)
native_cols.extend(native_mouse_cols)
tss_map_mpra_native = tss_map_mpra[native_cols]
tss_map_mpra_native.head()


# In[47]:


# remove duplicates
tss_map_dna.drop_duplicates(inplace=True)
print(len(tss_map_dna))
print(len(tss_map_dna["comp_id"].unique()))

tss_map_mpra_human.drop_duplicates(inplace=True)
print(len(tss_map_mpra_human))
print(len(tss_map_mpra_human["comp_id"].unique()))

tss_map_mpra_mouse.drop_duplicates(inplace=True)
print(len(tss_map_mpra_mouse))
print(len(tss_map_mpra_mouse["comp_id"].unique()))

tss_map_mpra_native.drop_duplicates(inplace=True)
print(len(tss_map_mpra_native))
print(len(tss_map_mpra_native["comp_id"].unique()))


# ## 4. pair positive controls together to serve as negative controls
# for each down-sampled control element (there are 4), randomly choose 100 pairs to serve as human/mouse

# In[48]:


ctrl_ids = rna_counts_ctrl.element.unique()
ctrl_ids[0:5]


# In[49]:


ctrl_seqs = set([x.split("__")[0] for x in ctrl_ids])
samp_ids = set([x.split("__")[1] for x in ctrl_ids])


# In[50]:


all_samp_id_pairs = list(itertools.combinations(samp_ids, 2))
all_samp_id_pairs_str = ["%s__%s" % (x[0], x[1]) for x in all_samp_id_pairs]
all_samp_id_pairs_str[0:5]


# In[51]:


sampled_samp_id_pairs = np.random.choice(all_samp_id_pairs_str, size=100)
sampled_samp_id_pairs[0:5]


# In[52]:


neg_ctrls_dna = pd.DataFrame()
neg_ctrls_human = pd.DataFrame()
neg_ctrls_mouse = pd.DataFrame()
neg_ctrls_native = pd.DataFrame()

for i, seq in enumerate(ctrl_seqs):
    print("ctrl #: %s" % (i+1))
    
    for j, samp_id_pair in enumerate(sampled_samp_id_pairs):
        if j % 50 == 0:
            print("...samp pair #: %s" % (j+1))
            
        samp1 = samp_id_pair.split("__")[0] # arbitrarily call 'human' seq
        samp2 = samp_id_pair.split("__")[1] # arbitrarily call 'mouse' seq
        
        human_elem = "%s__%s" % (seq, samp1)
        mouse_elem = "%s__%s" % (seq, samp2)
        
        human_sub_dna = dna_counts_ctrl[dna_counts_ctrl["element"] == human_elem]
        mouse_sub_dna = dna_counts_ctrl[dna_counts_ctrl["element"] == mouse_elem]
        
        human_sub_rna = rna_counts_ctrl[rna_counts_ctrl["element"] == human_elem]
        mouse_sub_rna = rna_counts_ctrl[rna_counts_ctrl["element"] == mouse_elem]
        
        # re-name columns w/ species name
        human_dna_cols = ["element"]
        mouse_dna_cols = ["element"]
        human_rna_cols = ["element"]
        mouse_rna_cols = ["element"]
        
        human_dna_cols.extend(["%s___seq:human" % x for x in human_sub_dna.columns if x != "element"])
        mouse_dna_cols.extend(["%s___seq:mouse" % x for x in mouse_sub_dna.columns if x != "element"])
        
        human_rna_cols.extend(["%s___seq:human" % x for x in human_sub_rna.columns if x != "element"])
        mouse_rna_cols.extend(["%s___seq:mouse" % x for x in mouse_sub_rna.columns if x != "element"])
        
        human_sub_dna.columns = human_dna_cols
        mouse_sub_dna.columns = mouse_dna_cols
        human_sub_rna.columns = human_rna_cols
        mouse_sub_rna.columns = mouse_rna_cols
        
        # add comp_id to each df
        comp_id = "CONTROL:%s__SAMP_PAIR:%s" % ((i+1), (j+1))
        human_sub_dna["comp_id"] = comp_id
        mouse_sub_dna["comp_id"] = comp_id
        human_sub_rna["comp_id"] = comp_id
        mouse_sub_rna["comp_id"] = comp_id
        
        # merge each df into 1
        human_sub_dna.drop("element", axis=1, inplace=True)
        mouse_sub_dna.drop("element", axis=1, inplace=True)
        human_sub_rna.drop("element", axis=1, inplace=True)
        mouse_sub_rna.drop("element", axis=1, inplace=True)
        
        sub_dna = human_sub_dna.merge(mouse_sub_dna, on="comp_id")
        sub_rna = human_sub_rna.merge(mouse_sub_rna, on="comp_id")
        
        # subset rna appropriately into each negative control bucket
        sub_rna_human_cols = [x for x in sub_rna.columns if x == "comp_id" or "HUES64" in x]
        sub_rna_mouse_cols = [x for x in sub_rna.columns if x == "comp_id" or "mESC" in x]
        sub_rna_native_cols = [x for x in sub_rna.columns if x == "comp_id" or ("HUES64" in x and "human" in x) or ("mESC" in x and "mouse" in x)]
        
        sub_rna_human = sub_rna[sub_rna_human_cols]
        sub_rna_mouse = sub_rna[sub_rna_mouse_cols]
        sub_rna_native = sub_rna[sub_rna_native_cols]
        
        # append
        neg_ctrls_dna = neg_ctrls_dna.append(sub_dna)
        neg_ctrls_human = neg_ctrls_human.append(sub_rna_human)
        neg_ctrls_mouse = neg_ctrls_mouse.append(sub_rna_mouse)
        neg_ctrls_native = neg_ctrls_native.append(sub_rna_native)


# In[53]:


all_dna = tss_map_dna.append(neg_ctrls_dna)
all_dna.set_index("comp_id", inplace=True)
len(all_dna)


# In[54]:


all_rna_human = tss_map_mpra_human.append(neg_ctrls_human)
all_rna_human.set_index("comp_id", inplace=True)
len(all_rna_human)


# In[55]:


all_rna_mouse = tss_map_mpra_mouse.append(neg_ctrls_mouse)
all_rna_mouse.set_index("comp_id", inplace=True)
len(all_rna_mouse)


# In[56]:


all_rna_native = tss_map_mpra_native.append(neg_ctrls_native)
all_rna_native.set_index("comp_id", inplace=True)
len(all_rna_native)


# In[57]:


# also make file w/ everything together to test interactions!
tmp_human = all_rna_human.reset_index()
tmp_mouse = all_rna_mouse.reset_index()
all_rna = tmp_human.merge(tmp_mouse, on="comp_id")
all_cols = all_rna.columns
all_rna.set_index("comp_id", inplace=True)
len(all_rna)


# ## 5. make annotation files

# In[58]:


dna_col_ann = {}
human_col_ann = {}
mouse_col_ann = {}
native_col_ann = {}
all_col_ann = {}

for cols, ann in zip([all_dna_cols, human_cols, mouse_cols, native_cols, all_cols], 
                     [dna_col_ann, human_col_ann, mouse_col_ann, native_col_ann, all_col_ann]):
    for col in cols:
        if col == "comp_id":
            continue
        cond = col.split(":")[1].split("_")[0]
        barc = col.split(":")[2].split("_")[0]
        seq = col.split(":")[-1]
        ann[col] = {"condition": cond, "barcode": barc, "seq": seq}

dna_col_ann = pd.DataFrame.from_dict(dna_col_ann, orient="index")
human_col_ann = pd.DataFrame.from_dict(human_col_ann, orient="index")
mouse_col_ann = pd.DataFrame.from_dict(mouse_col_ann, orient="index")
native_col_ann = pd.DataFrame.from_dict(native_col_ann, orient="index")
all_col_ann = pd.DataFrame.from_dict(all_col_ann, orient="index")
native_col_ann.sample(5)


# In[59]:


# merge w/ older annotations: first reset index
human_col_ann.reset_index(inplace=True)
mouse_col_ann.reset_index(inplace=True)
native_col_ann.reset_index(inplace=True)
all_col_ann.reset_index(inplace=True)

human_col_ann["colname"] = human_col_ann["index"]
mouse_col_ann["colname"] = mouse_col_ann["index"]
native_col_ann["colname"] = native_col_ann["index"]
all_col_ann["colname"] = all_col_ann["index"]


# In[60]:


# reset index on old annots and turn barcode into str
old_rna_col_ann.reset_index(inplace=True)
old_rna_col_ann["barcode"] = old_rna_col_ann["barcode"].astype(str)


# In[61]:


# merge
human_col_ann.sample(5)


# In[62]:


all_col_ann.sample(5)


# In[63]:


# reset index
human_col_ann.set_index("colname", inplace=True)
mouse_col_ann.set_index("colname", inplace=True)
native_col_ann.set_index("colname", inplace=True)
all_col_ann.set_index("colname", inplace=True)


# In[64]:


del human_col_ann.index.name
del mouse_col_ann.index.name
del native_col_ann.index.name
del all_col_ann.index.name


# In[65]:


# human_col_ann.drop("index", axis=1, inplace=True)
# mouse_col_ann.drop("index", axis=1, inplace=True)
# native_col_ann.drop("index", axis=1, inplace=True)
# all_col_ann.drop("index", axis=1, inplace=True)


# In[66]:


all_col_ann.head()


# In[67]:


all_col_ann.tail()


# ## 6. make control ID files

# In[68]:


ctrls = all_rna.reset_index()[["comp_id", "samp:HUES64_rep1__barc:10___seq:human"]]
ctrls["ctrl_status"] = ctrls.apply(ctrl_status, axis=1)
ctrls.drop("samp:HUES64_rep1__barc:10___seq:human", axis=1, inplace=True)
ctrls.ctrl_status.value_counts()


# In[69]:


ctrls.head()


# ## 7. write seq comparison files

# In[70]:


dna_col_ann.to_csv("%s/dna_col_ann.all_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t")
human_col_ann.to_csv("%s/HUES64_col_ann.seq_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t")
mouse_col_ann.to_csv("%s/mESC_col_ann.seq_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t")
native_col_ann.to_csv("%s/native_col_ann.seq_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t")
all_col_ann.to_csv("%s/all_col_ann.seq_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t")

ctrls.to_csv("%s/ctrl_status.all_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=False)

all_dna.to_csv("%s/dna_counts.all_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)
all_rna_human.to_csv("%s/HUES64_rna_counts.seq_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)
all_rna_mouse.to_csv("%s/mESC_rna_counts.seq_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)
all_rna_native.to_csv("%s/native_rna_counts.seq_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)
all_rna.to_csv("%s/all_rna_counts.seq_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)


# # then make files for cell line comparisons (trans effects)

# ## 1. run trans effects separately for human seqs & mouse seqs, so subset counts dataframe

# In[71]:


human_columns = [x for x in all_rna.columns if "seq:human" in x]
mouse_columns = [x for x in all_rna.columns if "seq:mouse" in x]


# In[72]:


human_trans = all_rna[human_columns]
mouse_trans = all_rna[mouse_columns]


# In[73]:


print(len(human_trans))


# In[74]:


print(len(mouse_trans))


# ## 2. subset annotation dataframe

# In[75]:


tmp = all_col_ann
tmp.head()


# In[76]:


human_trans_col_ann = tmp[tmp["index"].isin(human_columns)].set_index("index")
del human_trans_col_ann.index.name
human_trans_col_ann.sample(5)


# In[77]:


mouse_trans_col_ann = tmp[tmp["index"].isin(mouse_columns)].set_index("index")
del mouse_trans_col_ann.index.name
mouse_trans_col_ann.sample(5)


# In[78]:


print(len(human_columns))
print(len(human_trans_col_ann))
print(len(mouse_columns))
print(len(mouse_trans_col_ann))


# ## 3. write cell comparison files

# In[79]:


human_trans_col_ann.to_csv("%s/human_col_ann.cell_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t")
mouse_trans_col_ann.to_csv("%s/mouse_col_ann.cell_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t")

human_trans.to_csv("%s/human_rna_counts.cell_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)
mouse_trans.to_csv("%s/mouse_rna_counts.cell_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)


# # down-sample cis and trans files to estimate cis & trans effects from separate replicates

# In[82]:


all_rna_human_rep1_cols = [x for x in all_rna_human.columns if "rep1" in x]
all_rna_human_rep1 = all_rna_human[all_rna_human_rep1_cols]
all_rna_human_rep1.columns


# In[83]:


all_rna_mouse_rep1_cols = [x for x in all_rna_mouse.columns if "rep1" in x]
all_rna_mouse_rep1 = all_rna_mouse[all_rna_mouse_rep1_cols]
all_rna_mouse_rep1.columns


# In[85]:


human_col_ann_rep1 = human_col_ann[human_col_ann["index"].str.contains("rep1")]
mouse_col_ann_rep1 = mouse_col_ann[mouse_col_ann["index"].str.contains("rep1")]
mouse_col_ann_rep1.sample(5)


# In[87]:


human_trans_rep2_cols = [x for x in human_trans.columns if "rep2" in x]
human_trans_rep2 = human_trans[human_trans_rep2_cols]

mouse_trans_rep2_cols = [x for x in mouse_trans.columns if "rep2" in x]
mouse_trans_rep2 = mouse_trans[mouse_trans_rep2_cols]


# In[89]:


human_trans_col_ann_rep2 = human_trans_col_ann.reset_index()
human_trans_col_ann_rep2 = human_trans_col_ann_rep2[human_trans_col_ann_rep2["index"].str.contains("rep2")]
human_trans_col_ann_rep2.set_index("index", inplace=True)
human_trans_col_ann_rep2.sample(5)


# In[90]:


mouse_trans_col_ann_rep2 = mouse_trans_col_ann.reset_index()
mouse_trans_col_ann_rep2 = mouse_trans_col_ann_rep2[mouse_trans_col_ann_rep2["index"].str.contains("rep2")]
mouse_trans_col_ann_rep2.set_index("index", inplace=True)
mouse_trans_col_ann_rep2.sample(5)


# In[91]:


human_col_ann_rep1.to_csv("%s/HUES64_col_ann.seq_comp.REP1_ONLY.mpranalyze.txt" % mpranalyze_dir, sep="\t")
mouse_col_ann_rep1.to_csv("%s/mESC_col_ann.seq_comp.REP1_ONLY.mpranalyze.txt" % mpranalyze_dir, sep="\t")

all_rna_human_rep1.to_csv("%s/HUES64_rna_counts.seq_comp.REP1_ONLY.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)
all_rna_mouse_rep1.to_csv("%s/mESC_rna_counts.seq_comp.REP1_ONLY.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)


# In[92]:


human_trans_col_ann_rep2.to_csv("%s/human_col_ann.cell_comp.REP2_ONLY.mpranalyze.txt" % mpranalyze_dir, sep="\t")
mouse_trans_col_ann_rep2.to_csv("%s/mouse_col_ann.cell_comp.REP2_ONLY.mpranalyze.txt" % mpranalyze_dir, sep="\t")

human_trans_rep2.to_csv("%s/human_rna_counts.cell_comp.REP2_ONLY.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)
mouse_trans_rep2.to_csv("%s/mouse_rna_counts.cell_comp.REP2_ONLY.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)

