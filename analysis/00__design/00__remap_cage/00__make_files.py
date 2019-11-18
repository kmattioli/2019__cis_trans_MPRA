
# coding: utf-8

# # 00__make_files
# 
# in this notebook, i make the files necessary for finding CAGE reads that intersect our regions of interest (orthologous TSSs between human and mouse). final files are BED files with a 50 bp buffer surrounding the TSS (in both human and mouse).

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


# ## variables

# In[2]:


human_master_f = "../../../data/01__design/00__genome_list/hg19.master_list.txt.gz"
mouse_master_f = "../../../data/01__design/00__genome_list/mm9.master_list.txt.gz"


# ## 1. import data

# In[3]:


human_master = pd.read_table(human_master_f, sep="\t")
human_master.head()


# In[4]:


mouse_master = pd.read_table(mouse_master_f, sep="\t")
mouse_master.head()


# ## 2. filter to seq orths only

# In[5]:


human_master_filt = human_master[human_master["seq_orth"]]
len(human_master_filt)


# In[6]:


mouse_master_filt = mouse_master[mouse_master["seq_orth"]]
len(mouse_master_filt)


# ## 3. find TSS coords for human/mouse paired regions
# do it for both the "human" file (started from human) and the "mouse" file (started from mouse)

# In[7]:


human_bed_hg19 = human_master_filt[["chr_tss_hg19", "start_tss_hg19", "end_tss_hg19", "cage_id_hg19",
                                    "score_tss_hg19", "strand_tss_hg19"]].drop_duplicates()
print(len(human_bed_hg19))
human_bed_hg19.head()


# In[8]:


human_bed_mm9 = human_master_filt[["chr_tss_mm9", "start_tss_mm9", "end_tss_mm9", "cage_id_hg19",
                                   "score_tss_hg19", "strand_tss_mm9"]].drop_duplicates()
print(len(human_bed_mm9))
human_bed_mm9.head()


# In[9]:


human_bed_mm9[human_bed_mm9["cage_id_hg19"] == "chr1:203273760..203273784,-"]


# In[10]:


mouse_bed_mm9 = mouse_master_filt[["chr_tss_mm9", "start_tss_mm9", "end_tss_mm9", "cage_id_mm9",
                                    "score_tss_mm9", "strand_tss_mm9"]].drop_duplicates()
print(len(mouse_bed_mm9))
mouse_bed_mm9.head()


# In[11]:


mouse_bed_hg19 = mouse_master_filt[["chr_tss_hg19", "start_tss_hg19", "end_tss_hg19", "cage_id_mm9",
                                    "score_tss_mm9", "strand_tss_hg19"]].drop_duplicates()
print(len(mouse_bed_hg19))
mouse_bed_hg19.head()


# ## 4. group hg19/mm9 files together for bed intersect

# In[12]:


human_bed_hg19["cage_id"] = "HUMAN_CAGE_ID__" + human_bed_hg19["cage_id_hg19"]
mouse_bed_hg19["cage_id"] = "MOUSE_CAGE_ID__" + mouse_bed_hg19["cage_id_mm9"]
human_bed_hg19["score"] = "HUMAN_SCORE__" + human_bed_hg19["score_tss_hg19"].astype(str)
mouse_bed_hg19["score"] = "MOUSE_SCORE__" + mouse_bed_hg19["score_tss_mm9"].astype(str)
human_bed_hg19.head()


# In[13]:


human_bed_mm9["cage_id"] = "HUMAN_CAGE_ID__" + human_bed_mm9["cage_id_hg19"]
mouse_bed_mm9["cage_id"] = "MOUSE_CAGE_ID__" + mouse_bed_mm9["cage_id_mm9"]
human_bed_mm9["score"] = "HUMAN_SCORE__" + human_bed_hg19["score_tss_hg19"].astype(str)
mouse_bed_mm9["score"] = "MOUSE_SCORE__" + mouse_bed_hg19["score_tss_mm9"].astype(str)
human_bed_mm9.head()


# In[14]:


hg19_bed = human_bed_hg19[["chr_tss_hg19", "start_tss_hg19", "end_tss_hg19", "cage_id", "score", "strand_tss_hg19"]]
hg19_bed = hg19_bed.append(mouse_bed_hg19[["chr_tss_hg19", "start_tss_hg19", "end_tss_hg19", "cage_id", "score", "strand_tss_hg19"]])
hg19_bed.drop_duplicates(inplace=True)
print(len(hg19_bed))
hg19_bed.sample(5)


# In[15]:


mm9_bed = human_bed_mm9[["chr_tss_mm9", "start_tss_mm9", "end_tss_mm9", "cage_id", "score", "strand_tss_mm9"]]
mm9_bed = mm9_bed.append(mouse_bed_mm9[["chr_tss_mm9", "start_tss_mm9", "end_tss_mm9", "cage_id", "score", "strand_tss_mm9"]])
mm9_bed.drop_duplicates(inplace=True)
print(len(mm9_bed))
mm9_bed.sample(5)


# ## 5. add buffer of +/- 50 bp

# In[16]:


hg19_bed["start_tss_hg19"] = hg19_bed["start_tss_hg19"].astype(int) - 49
hg19_bed["end_tss_hg19"] = hg19_bed["end_tss_hg19"].astype(int) + 50
hg19_bed["score"] = 0
hg19_bed.head()


# In[17]:


mm9_bed["start_tss_mm9"] = mm9_bed["start_tss_mm9"].astype(int) - 49
mm9_bed["end_tss_mm9"] = mm9_bed["end_tss_mm9"].astype(int) + 50
mm9_bed["score"] = 0
mm9_bed.head()


# ## 6. write files

# In[18]:


hg19_bed.to_csv("../../../data/01__design/00__genome_list/hg19_master.50buff.bed", header=False, index=False, sep="\t")
mm9_bed.to_csv("../../../data/01__design/00__genome_list/mm9_master.50buff.bed", header=False, index=False, sep="\t")

