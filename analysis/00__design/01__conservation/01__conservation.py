
# coding: utf-8

# # 01__conservation

# In[3]:


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


# In[4]:


sns.set(**PAPER_PRESET)
fontsize = PAPER_FONTSIZE


# In[5]:


np.random.seed(2019)


# ## functions

# In[6]:


def cleaner_biotype(row, biotype_col):
    if row[biotype_col] in ["protein_coding", "div_pc"]:
        return "mRNA"
    elif row[biotype_col] == "intergenic":
        return "lncRNA"
    elif row[biotype_col] in ["antisense", "div_lnc"]:
        return "lncRNA"
    elif row[biotype_col] == "enhancer":
        return "eRNA"
    elif row[biotype_col] == "no cage activity":
        return "no CAGE activity"
    else:
        return "other"


# In[7]:


def get_perc(row):
    if not pd.isnull(row["cage_id"]):
        return (row["cage_id"]/row["cage_id_x"])*100
    else:
        return (row["cage_id_y"]/row["cage_id_x"])*100


# ## variables

# In[8]:


human_list_f = "../../../data/01__design/00__genome_list/hg19.PEAK_STATUS.txt.gz"
mouse_list_f = "../../../data/01__design/00__genome_list/mm9.PEAK_STATUS.txt.gz"


# ## 1. import data

# In[9]:


human_list = pd.read_table(human_list_f)
human_list.head()


# In[10]:


mouse_list = pd.read_table(mouse_list_f)
mouse_list.head()


# ## 2. clean biotypes

# In[11]:


human_list["clean_biotype_hg19"] = human_list.apply(cleaner_biotype, biotype_col="biotype", axis=1)
human_list["clean_biotype_mm9"] = human_list.apply(cleaner_biotype, biotype_col="other_sp_biotype", axis=1)
human_list.sample(5)


# In[12]:


mouse_list["clean_biotype_mm9"] = mouse_list.apply(cleaner_biotype, biotype_col="biotype", axis=1)
mouse_list["clean_biotype_hg19"] = mouse_list.apply(cleaner_biotype, biotype_col="other_sp_biotype", axis=1)
mouse_list.sample(5)


# In[13]:


human_list.clean_biotype_hg19.value_counts()


# In[14]:


mouse_list.clean_biotype_mm9.value_counts()


# In[15]:


human_list[human_list["seq_ortholog"] == 0].sample(5)


# ## 3. find % sequence conservation

# In[16]:


human_tots = human_list.groupby("clean_biotype_hg19")["cage_id"].agg("count").reset_index()

human_tsss = human_list[(human_list["clean_biotype_hg19"] != "eRNA") & (human_list["seq_ortholog"] == 1)]
human_tss_seqs = human_tsss.groupby("clean_biotype_hg19")["cage_id"].agg("count").reset_index()

# make sure we pick enhancers that have both TSSs map
human_enhs = human_list[(human_list["clean_biotype_hg19"] == "eRNA") & (human_list["seq_ortholog"] >= 2)]
human_enh_seqs = human_enhs.groupby("clean_biotype_hg19")["cage_id"].agg("count").reset_index()

# merge
human_seq_perc = human_tots.merge(human_tss_seqs, 
                                  on="clean_biotype_hg19", 
                                  how="left").merge(human_enh_seqs, 
                                                    on="clean_biotype_hg19", 
                                                    how="left")
human_seq_perc["perc"] = human_seq_perc.apply(get_perc, axis=1)
human_seq_perc.head()


# In[17]:


mouse_tots = mouse_list.groupby("clean_biotype_mm9")["cage_id"].agg("count").reset_index()

mouse_tsss = mouse_list[(mouse_list["clean_biotype_mm9"] != "eRNA") & (mouse_list["seq_ortholog"] == 1)]
mouse_tss_seqs = mouse_tsss.groupby("clean_biotype_mm9")["cage_id"].agg("count").reset_index()

# make sure we pick enhancers that have both TSSs map
mouse_enhs = mouse_list[(mouse_list["clean_biotype_mm9"] == "eRNA") & (mouse_list["seq_ortholog"] >= 2)]
mouse_enh_seqs = mouse_enhs.groupby("clean_biotype_mm9")["cage_id"].agg("count").reset_index()

# merge
mouse_seq_perc = mouse_tots.merge(mouse_tss_seqs, 
                                  on="clean_biotype_mm9", 
                                  how="left").merge(mouse_enh_seqs, 
                                                    on="clean_biotype_mm9", 
                                                    how="left")
mouse_seq_perc["perc"] = mouse_seq_perc.apply(get_perc, axis=1)
mouse_seq_perc.head()


# ## 4. find % CAGE conservation

# In[18]:


human_tots = human_list.groupby("clean_biotype_hg19")["cage_id"].agg("count").reset_index()

human_tsss = human_list[(human_list["clean_biotype_hg19"] != "eRNA") & (human_list["cage_ortholog"] == 1)]
human_tss_seqs = human_tsss.groupby("clean_biotype_hg19")["cage_id"].agg("count").reset_index()

# make sure we pick enhancers that have both TSSs map
human_enhs = human_list[(human_list["clean_biotype_hg19"] == "eRNA") & (human_list["cage_ortholog"] >= 2)]
human_enh_seqs = human_enhs.groupby("clean_biotype_hg19")["cage_id"].agg("count").reset_index()

# merge
human_cage_perc = human_tots.merge(human_tss_seqs, 
                                  on="clean_biotype_hg19", 
                                  how="left").merge(human_enh_seqs, 
                                                    on="clean_biotype_hg19", 
                                                    how="left")
human_cage_perc["perc"] = human_cage_perc.apply(get_perc, axis=1)
human_cage_perc.head()


# In[19]:


mouse_tots = mouse_list.groupby("clean_biotype_mm9")["cage_id"].agg("count").reset_index()

mouse_tsss = mouse_list[(mouse_list["clean_biotype_mm9"] != "eRNA") & (mouse_list["cage_ortholog"] == 1)]
mouse_tss_seqs = mouse_tsss.groupby("clean_biotype_mm9")["cage_id"].agg("count").reset_index()

# make sure we pick enhancers that have both TSSs map
mouse_enhs = mouse_list[(mouse_list["clean_biotype_mm9"] == "eRNA") & (mouse_list["cage_ortholog"] >= 2)]
mouse_enh_seqs = mouse_enhs.groupby("clean_biotype_mm9")["cage_id"].agg("count").reset_index()

# merge
mouse_cage_perc = mouse_tots.merge(mouse_tss_seqs, 
                                  on="clean_biotype_mm9", 
                                  how="left").merge(mouse_enh_seqs, 
                                                    on="clean_biotype_mm9", 
                                                    how="left")
mouse_cage_perc["perc"] = mouse_cage_perc.apply(get_perc, axis=1)
mouse_cage_perc.head()


# ## 5. make plots

# In[20]:


order = ["eRNA", "lncRNA", "mRNA"]


# In[21]:


fig, axarr = plt.subplots(figsize=(1.4, 1.8), ncols=1, nrows=2, sharex=True, sharey=True)

ax = axarr[0]
sns.barplot(data=human_seq_perc, x="clean_biotype_hg19", y="perc",
            order=order, color=sns.color_palette("Set2")[1], ax=ax)
ax.set_xlabel("")
ax.set_ylabel("% sequence\northologs")
ax.set_ylim((0, 100))

ax = axarr[1]
sns.barplot(data=human_cage_perc, x="clean_biotype_hg19", y="perc",
            order=order, color=sns.color_palette("Set2")[1], ax=ax)
ax.set_xlabel("")
ax.set_ylabel("% activity\northologs")
ax.set_xticklabels(order, rotation=50, ha='right', va='top')
ax.set_ylim((0, 100))
fig.savefig("human_orth_percents.pdf", dpi="figure", bbox_inches="tight")


# In[22]:


fig, axarr = plt.subplots(figsize=(1.4, 1.8), ncols=1, nrows=2, sharex=True, sharey=True)

ax = axarr[0]
sns.barplot(data=mouse_seq_perc, x="clean_biotype_mm9", y="perc",
            order=order, color=sns.color_palette("Set2")[0], ax=ax)
ax.set_xlabel("")
ax.set_ylabel("% sequence\northologs")
ax.set_ylim((0, 100))

ax = axarr[1]
sns.barplot(data=mouse_cage_perc, x="clean_biotype_mm9", y="perc",
            order=order, color=sns.color_palette("Set2")[0], ax=ax)
ax.set_xlabel("")
ax.set_ylabel("% activity\northologs")
ax.set_xticklabels(order, rotation=50, ha='right', va='top')
ax.set_ylim((0, 100))
fig.savefig("mouse_orth_percents.pdf", dpi="figure", bbox_inches="tight")

