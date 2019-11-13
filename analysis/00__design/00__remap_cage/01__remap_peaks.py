
# coding: utf-8

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


# ## functions

# In[3]:


def cleaner_biotype(row, biotype_col):
    if row[biotype_col] in ["protein_coding", "div_pc", "mRNA"]:
        return "mRNA"
    elif row[biotype_col] == "intergenic":
        return "lncRNA"
    elif row[biotype_col] in ["antisense", "div_lnc", "divergent", "lncRNA"]:
        return "lncRNA"
    elif row[biotype_col] == "enhancer" or row[biotype_col] == "eRNA":
        return "eRNA"
    elif row[biotype_col] in ["no cage activity", "no CAGE activity"]:
        return "no CAGE activity"
    else:
        return "other"


# In[4]:


def reclassify_biotype(row, biotype_col1, biotype_col2, cage_col):
    if row["seq_orth"]:
        if row[biotype_col1] == "no CAGE activity":
            if row[cage_col] < 10:
                return "no CAGE activity"
            else:
                return row[biotype_col2]
        else:
            return row[biotype_col1]
    else:
        return "no CAGE activity"


# ## variables

# In[5]:


human_master_f = "../../../data/01__design/00__genome_list/hg19.master_list.txt.gz"
mouse_master_f = "../../../data/01__design/00__genome_list/mm9.master_list.txt.gz"


# In[6]:


human_cage_f = "../../../misc/02__cage_reads/hg19_max_counts_oligo.bed"
mouse_cage_f = "../../../misc/02__cage_reads/mm9_max_counts_oligo.bed"


# ## 1. import data

# In[7]:


human_master = pd.read_table(human_master_f, sep="\t")
human_master["start_tss_hg19"] = human_master["start_tss_hg19"].astype(str)
human_master["end_tss_hg19"] = human_master["end_tss_hg19"].astype(str)
human_master.head()


# In[8]:


mouse_master = pd.read_table(mouse_master_f, sep="\t")
mouse_master["start_tss_mm9"] = mouse_master["start_tss_mm9"].astype(str)
mouse_master["end_tss_mm9"] = mouse_master["end_tss_mm9"].astype(str)
mouse_master.head()


# In[9]:


human_cage = pd.read_table(human_cage_f, sep="\t", header=None)
human_cage.columns = ["chr", "start", "end", "name", "score", "strand", "max_cage_hg19"]
human_cage.head()


# In[10]:


mouse_cage = pd.read_table(mouse_cage_f, sep="\t", header=None)
mouse_cage.columns = ["chr", "start", "end", "name", "score", "strand", "max_cage_mm9"]
mouse_cage.head()


# In[11]:


human_cage[human_cage["name"].str.contains("chr8:55389317-55389516")]


# In[12]:


mouse_cage[mouse_cage["name"].str.contains("chr8:55389317-55389516")]


# In[13]:


human_master[human_master["cage_id_hg19"] == "chr8:55389317-55389516"][["start_tss_hg19", "end_tss_hg19", "strand_tss_hg19", "start_tss_mm9", "end_tss_mm9", "strand_tss_mm9"]]


# ## 2. parse cage count files

# In[14]:


human_cage["start"] = human_cage["start"] + 49
human_cage["end"] = human_cage["end"] - 50
human_cage["start"] = human_cage["start"].astype(str)
human_cage["end"] = human_cage["end"].astype(str)
human_cage["orig_species"] = human_cage["name"].str.split("__", expand=True)[0]
human_cage["cage_id"] = human_cage["name"].str.split("__", expand=True)[1]
human_cage.head()


# In[15]:


orig_human_cage_hg19 = human_cage[human_cage["orig_species"].str.contains("HUMAN")][["cage_id", "chr", 
                                                                                     "start", "end", "strand",
                                                                                     "max_cage_hg19"]].drop_duplicates()
orig_human_cage_hg19.columns = ["cage_id_hg19", "chr_tss_hg19", "start_tss_hg19", "end_tss_hg19", 
                                "strand_tss_hg19", "max_cage_hg19"]
orig_human_cage_hg19.head()


# In[16]:


orig_mouse_cage_hg19 = human_cage[human_cage["orig_species"].str.contains("MOUSE")][["cage_id", "chr", 
                                                                                     "start", "end", "strand",
                                                                                     "max_cage_hg19"]].drop_duplicates()
orig_mouse_cage_hg19.columns = ["cage_id_mm9", "chr_tss_hg19", "start_tss_hg19", "end_tss_hg19", 
                                "strand_tss_hg19", "max_cage_hg19"]
orig_mouse_cage_hg19.head()


# In[17]:


mouse_cage["start"] = mouse_cage["start"] + 49
mouse_cage["end"] = mouse_cage["end"] - 50
mouse_cage["start"] = mouse_cage["start"].astype(str)
mouse_cage["end"] = mouse_cage["end"].astype(str)
mouse_cage["orig_species"] = mouse_cage["name"].str.split("__", expand=True)[0]
mouse_cage["cage_id"] = mouse_cage["name"].str.split("__", expand=True)[1]
mouse_cage.head()


# In[18]:


orig_mouse_cage_mm9 = mouse_cage[mouse_cage["orig_species"].str.contains("MOUSE")][["cage_id", "chr", 
                                                                                    "start", "end", "strand",
                                                                                    "max_cage_mm9"]].drop_duplicates()
orig_mouse_cage_mm9.columns = ["cage_id_mm9", "chr_tss_mm9", "start_tss_mm9", "end_tss_mm9", 
                               "strand_tss_mm9", "max_cage_mm9"]
orig_mouse_cage_mm9.head()


# In[19]:


orig_human_cage_mm9 = mouse_cage[mouse_cage["orig_species"].str.contains("HUMAN")][["cage_id", "chr", 
                                                                                    "start", "end", "strand",
                                                                                    "max_cage_mm9"]].drop_duplicates()
orig_human_cage_mm9.columns = ["cage_id_hg19", "chr_tss_mm9", "start_tss_mm9", "end_tss_mm9", 
                               "strand_tss_mm9", "max_cage_mm9"]
orig_human_cage_mm9.head()


# In[20]:


orig_human_cage_hg19[orig_human_cage_hg19["cage_id_hg19"] == "chr10:104210390..104210444,+"]


# In[21]:


orig_human_cage_mm9[orig_human_cage_mm9["cage_id_hg19"] == "chr10:104210390..104210444,+"]


# ## 3. merge cage counts with master files

# In[22]:


orig_human = orig_human_cage_hg19.merge(orig_human_cage_mm9, on=["cage_id_hg19"], how="left")
print(len(orig_human))
orig_human.sample(5)


# In[23]:


orig_human[orig_human["cage_id_hg19"] == "chr10:104210390..104210444,+"]


# In[24]:


orig_mouse = orig_mouse_cage_hg19.merge(orig_mouse_cage_mm9, on=["cage_id_mm9"], how="left")
print(len(orig_mouse))
orig_mouse.sample(5)


# In[25]:


print(len(human_master))
human_master_cage = human_master.merge(orig_human, on=["cage_id_hg19", "chr_tss_hg19", "start_tss_hg19",
                                                       "end_tss_hg19", "strand_tss_hg19", "chr_tss_mm9",
                                                       "start_tss_mm9", "end_tss_mm9", "strand_tss_mm9"], how="left")
print(len(human_master))
human_master_cage.sample(5)


# In[26]:


human_master_cage[human_master_cage["cage_id_hg19"] == "chr10:104210390..104210444,+"]


# In[27]:


human_master[human_master["cage_id_hg19"] == "chr10:104210390..104210444,+"][["cage_id_hg19", "chr_tss_hg19", "start_tss_hg19",
                                                       "end_tss_hg19", "strand_tss_hg19", "chr_tss_mm9",
                                                       "start_tss_mm9", "end_tss_mm9", "strand_tss_mm9", "biotype_hg19", "biotype_mm9"]]


# In[28]:


print(len(mouse_master))
mouse_master_cage = mouse_master.merge(orig_mouse, on=["cage_id_mm9", "chr_tss_mm9", "start_tss_mm9",
                                                       "end_tss_mm9", "strand_tss_mm9", "chr_tss_hg19",
                                                       "start_tss_hg19", "end_tss_hg19", "strand_tss_hg19"], how="left")
print(len(mouse_master))
mouse_master_cage.sample(5)


# ## 4. fix biotypes + plot CAGE peaks

# In[29]:


human_master_cage["min_biotype_hg19"] = human_master_cage.apply(cleaner_biotype, biotype_col="biotype_hg19", axis=1)
human_master_cage["min_biotype_mm9"] = human_master_cage.apply(cleaner_biotype, biotype_col="biotype_mm9", axis=1)
mouse_master_cage["min_biotype_mm9"] = mouse_master_cage.apply(cleaner_biotype, biotype_col="biotype_mm9", axis=1)
mouse_master_cage["min_biotype_hg19"] = mouse_master_cage.apply(cleaner_biotype, biotype_col="biotype_hg19", axis=1)


# In[30]:


human_master_cage[human_master_cage["cage_id_hg19"] == "chr1:203273760..203273784,-"]


# In[31]:


human_seq = human_master_cage[human_master_cage["seq_orth"]]
len(human_seq)


# In[32]:


mouse_seq = mouse_master_cage[mouse_master_cage["seq_orth"]]
len(mouse_seq)


# In[33]:


order = ["eRNA", "lncRNA", "mRNA"]


# In[34]:


fig = plt.figure(figsize=(1.5, 1.25))

ax = sns.boxplot(data=human_seq, x="min_biotype_hg19", y="max_cage_hg19", order=order, 
                 color=sns.color_palette("Set2")[1], flierprops = dict(marker='o', markersize=5))
mimic_r_boxplot(ax)

ax.set_xticklabels(order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_yscale("symlog")
ax.set_ylabel("max human\nCAGE counts")

for i, label in enumerate(order):
    n = len(human_seq[human_seq["min_biotype_hg19"] == label])
    ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=sns.color_palette("Set2")[1], size=fontsize)

ax.set_ylim((-1, 100000000))
plt.show()
#fig.savefig("human_biotype_cage.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# In[35]:


fig = plt.figure(figsize=(1.5, 1.25))

ax = sns.boxplot(data=mouse_seq, x="min_biotype_mm9", y="max_cage_mm9", order=order, 
                 color=sns.color_palette("Set2")[0], flierprops = dict(marker='o', markersize=5))
mimic_r_boxplot(ax)

ax.set_xticklabels(order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_yscale("symlog")
ax.set_ylabel("max mouse\nCAGE counts")

for i, label in enumerate(order):
    n = len(mouse_seq[mouse_seq["min_biotype_mm9"] == label])
    ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=sns.color_palette("Set2")[0], size=fontsize)

ax.set_ylim((-1, 100000000))
plt.show()
#fig.savefig("mouse_biotype_cage.pdf", dpi="figure", bbox_inches="tight")
plt.close()


# In[36]:


fig = plt.figure(figsize=(1.5, 1.25))

ax = sns.kdeplot(np.log10(human_seq[human_seq["min_biotype_mm9"] == "no CAGE activity"]["max_cage_mm9"]+1),
                  color=sns.color_palette("Set2")[0], cumulative=True)

ax.set_ylabel("cumulative frequency")
ax.set_xlabel("log10(max mouse CAGE counts + 1)")
ax.set_title("human CAGE peaks with no\northologous mouse peak")
ax.get_legend().remove()

ax.axvline(x=np.log10(10+1), color="black", linestyle="dashed")

fig.savefig("mouse_no_peak_counts.pdf", dpi="figure", bbox_inches="tight")


# In[37]:


fig = plt.figure(figsize=(1.5, 1.25))

ax = sns.kdeplot(np.log10(mouse_seq[mouse_seq["min_biotype_hg19"] == "no CAGE activity"]["max_cage_hg19"]+1),
                  color=sns.color_palette("Set2")[1], cumulative=True)

ax.set_ylabel("cumulative frequency")
ax.set_xlabel("log10(max human CAGE counts + 1)")
ax.set_title("mouse CAGE peaks with no\northologous human peak")
ax.get_legend().remove()

ax.axvline(x=np.log10(10+1), color="black", linestyle="dashed")

fig.savefig("human_no_peak_counts.pdf", dpi="figure", bbox_inches="tight")


# ## 5. re-classify CAGE peaks
# anthing ≥ 10 reads in the other species is considered a CAGE peak

# In[38]:


human_master_cage["new_biotype_mm9"] = human_master_cage.apply(reclassify_biotype, biotype_col1="min_biotype_mm9",
                                                               biotype_col2="min_biotype_hg19", 
                                                               cage_col="max_cage_mm9", axis=1)
human_master_cage.sample(5)


# In[39]:


human_master_cage[human_master_cage["cage_id_hg19"] == "chr17:43325288..43325306,+"]


# In[40]:


mouse_master_cage["new_biotype_hg19"] = mouse_master_cage.apply(reclassify_biotype, biotype_col1="min_biotype_hg19",
                                                                biotype_col2="min_biotype_mm9", 
                                                                cage_col="max_cage_hg19", axis=1)
mouse_master_cage.sample(5)


# In[41]:


mouse_master_cage[mouse_master_cage["cage_id_mm9"] == "chr13:46058556-46059202"]


# In[42]:


human_seq = human_master_cage[human_master_cage["seq_orth"]]
human_seq.min_biotype_mm9.value_counts()


# In[43]:


human_seq.new_biotype_mm9.value_counts()


# In[44]:


mouse_seq = mouse_master_cage[mouse_master_cage["seq_orth"]]
mouse_seq.min_biotype_hg19.value_counts()


# In[45]:


mouse_seq.new_biotype_hg19.value_counts()


# In[46]:


human_master_cage.cage_orth.value_counts()


# In[47]:


human_master_cage["cage_orth"] = human_master_cage["new_biotype_mm9"] != "no CAGE activity"
mouse_master_cage["cage_orth"] = mouse_master_cage["new_biotype_hg19"] != "no CAGE activity"


# In[48]:


human_master_cage.cage_orth.value_counts()


# ## 6. make new peak files

# In[49]:


human_master_cage["seq_orth_bin"] = human_master_cage["seq_orth"].astype(int)
human_master_cage["cage_orth_bin"] = human_master_cage["cage_orth"].astype(int)
human_master_cage = human_master_cage.drop_duplicates()
human_master_cage.sample(5)


# In[50]:


mouse_master_cage["seq_orth_bin"] = mouse_master_cage["seq_orth"].astype(int)
mouse_master_cage["cage_orth_bin"] = mouse_master_cage["cage_orth"].astype(int)
mouse_master_cage = mouse_master_cage.drop_duplicates()
mouse_master_cage.sample(5)


# In[51]:


human_sub = human_master_cage[["cage_id_hg19", "strand_tss_hg19", "min_biotype_hg19", "seq_orth_bin", 
                               "cage_orth_bin", "new_biotype_mm9"]].drop_duplicates()
mouse_sub = mouse_master_cage[["cage_id_mm9", "strand_tss_mm9", "min_biotype_mm9", "seq_orth_bin", 
                               "cage_orth_bin", "new_biotype_hg19"]].drop_duplicates()


# In[52]:


human_seq_grp = human_sub.groupby(["cage_id_hg19", "min_biotype_hg19"])["seq_orth_bin"].agg("sum").reset_index()
mouse_seq_grp = mouse_sub.groupby(["cage_id_mm9", "min_biotype_mm9"])["seq_orth_bin"].agg("sum").reset_index()
human_seq_grp.head()


# In[53]:


human_cage_grp = human_sub.groupby(["cage_id_hg19", "min_biotype_hg19"])["cage_orth_bin"].agg("sum").reset_index()
mouse_cage_grp = mouse_sub.groupby(["cage_id_mm9", "min_biotype_mm9"])["cage_orth_bin"].agg("sum").reset_index()
human_cage_grp.head()


# In[54]:


human_grp = human_seq_grp.merge(human_cage_grp, on=["cage_id_hg19", "min_biotype_hg19"], how="left")
human_grp = human_grp.merge(human_sub[["cage_id_hg19", "new_biotype_mm9"]], on="cage_id_hg19", how="left")
human_grp.sample(5)


# In[55]:


mouse_grp = mouse_seq_grp.merge(mouse_cage_grp, on=["cage_id_mm9", "min_biotype_mm9"], how="left")
mouse_grp = mouse_grp.merge(mouse_sub[["cage_id_mm9", "new_biotype_hg19"]], on="cage_id_mm9", how="left")
mouse_grp.sample(5)


# ## 7. write files

# In[56]:


human_grp.columns = ["cage_id", "biotype", "seq_ortholog", "cage_ortholog", "other_sp_biotype"]
mouse_grp.columns = ["cage_id", "biotype", "seq_ortholog", "cage_ortholog", "other_sp_biotype"]


# In[57]:


human_master_f = '../../../data/01__design/00__genome_list/hg19.master_list.reclassified.txt.gz'
mouse_master_f = '../../../data/01__design/00__genome_list/mm9.master_list.reclassified.txt.gz'


# In[58]:


human_peak_f = '../../../data/01__design/00__genome_list/hg19.PEAK_STATUS.txt.gz'
mouse_peak_f = '../../../data/01__design/00__genome_list/mm9.PEAK_STATUS.txt.gz'


# In[59]:


human_grp.to_csv(human_peak_f, sep="\t", index=False, compression="gzip")
mouse_grp.to_csv(mouse_peak_f, sep="\t", index=False, compression="gzip")


# In[60]:


human_master_cage.to_csv(human_master_f, sep="\t", index=False, compression="gzip")
mouse_master_cage.to_csv(mouse_master_f, sep="\t", index=False, compression="gzip")


# ## 8. fix MPRA map file

# In[61]:


tss_map_f = "../../../data/01__design/01__mpra_list/mpra_tss.with_ids.txt"
tss_map = pd.read_table(tss_map_f, sep="\t")
tss_map.head()


# In[62]:


len(tss_map)


# In[63]:


orig_human = tss_map[tss_map["orig_species"] == "human"]
orig_human["start_tss_hg19"] = orig_human["start_tss_hg19"].astype(str)
orig_human["end_tss_hg19"] = orig_human["end_tss_hg19"].astype(str)
orig_human["start_tss_mm9"] = orig_human["start_tss_mm9"].astype(str)
orig_human["end_tss_mm9"] = orig_human["end_tss_mm9"].astype(str)
print(len(orig_human))
orig_human[orig_human["cage_id_hg19"] == "chr2:42217488-42217883"]


# In[64]:


orig_mouse = tss_map[tss_map["orig_species"] == "mouse"]
orig_mouse["start_tss_mm9"] = orig_mouse["start_tss_mm9"].astype(str)
orig_mouse["end_tss_mm9"] = orig_mouse["end_tss_mm9"].astype(str)
orig_mouse["start_tss_hg19"] = orig_mouse["start_tss_hg19"].astype(str)
orig_mouse["end_tss_hg19"] = orig_mouse["end_tss_hg19"].astype(str)
print(len(orig_mouse))


# In[65]:


human_master_cage_sub = human_master_cage[["cage_id_hg19", "start_tss_hg19", "end_tss_hg19", "strand_tss_hg19", 
                                           "min_biotype_hg19", "start_tss_mm9", "end_tss_mm9", "strand_tss_mm9", 
                                           "new_biotype_mm9", "max_cage_hg19", "max_cage_mm9"]].drop_duplicates()
human_master_cage_sub.columns = ["cage_id_hg19", "start_tss_hg19", "end_tss_hg19", "strand_tss_hg19", 
                                 "minimal_biotype_hg19", "start_tss_mm9", "end_tss_mm9", "strand_tss_mm9", 
                                 "minimal_biotype_mm9", "max_cage_hg19", "max_cage_mm9"]


# In[66]:


mouse_master_cage_sub = mouse_master_cage[["cage_id_mm9", "start_tss_mm9", "end_tss_mm9", "strand_tss_mm9", 
                                           "min_biotype_mm9", "start_tss_hg19", "end_tss_hg19", "strand_tss_hg19", 
                                           "new_biotype_hg19", "max_cage_mm9", "max_cage_hg19"]].drop_duplicates()
mouse_master_cage_sub.columns = ["cage_id_mm9", "start_tss_mm9", "end_tss_mm9", "strand_tss_mm9", 
                                 "minimal_biotype_mm9", "start_tss_hg19", "end_tss_hg19", "strand_tss_hg19", 
                                 "minimal_biotype_hg19", "max_cage_mm9", "max_cage_hg19"]


# In[67]:


mrg_human = orig_human.merge(human_master_cage_sub, on=["cage_id_hg19", "start_tss_hg19", 
                                                        "end_tss_hg19", "strand_tss_hg19",
                                                        "start_tss_mm9", "end_tss_mm9", "strand_tss_mm9"], 
                              how="left").drop_duplicates()
print(len(mrg_human))


# In[68]:


mrg_mouse = orig_mouse.merge(mouse_master_cage_sub, on=["cage_id_mm9", "start_tss_mm9", 
                                                        "end_tss_mm9", "strand_tss_mm9",
                                                        "start_tss_hg19", "end_tss_hg19", "strand_tss_hg19"], 
                              how="left").drop_duplicates()
print(len(mrg_mouse))


# In[69]:


new_map = mrg_human.append(mrg_mouse)
print(len(new_map))
new_map.sample(5)


# some biotypes are null because these seqs were not expressed in hESCs and thus did not make it in our "master" bed file with all enhancers & TSSs (43 total)
# 
# for these, assign the original biotype that was assigned upon pool design

# In[70]:


def fix_biotype(row, suff):
    if pd.isnull(row["minimal_biotype_%s" % suff]):
        return row["biotype_%s" % suff]
    else:
        return row["minimal_biotype_%s" % suff]

new_map["tmp_hg19"] = new_map.apply(fix_biotype, suff="hg19", axis=1)
new_map["tmp_mm9"] = new_map.apply(fix_biotype, suff="mm9", axis=1)


# In[71]:


new_map["minimal_biotype_hg19"] = new_map.apply(cleaner_biotype, biotype_col="tmp_hg19", axis=1)
new_map["minimal_biotype_mm9"] = new_map.apply(cleaner_biotype, biotype_col="tmp_mm9", axis=1)


# In[72]:


new_map[pd.isnull(new_map["minimal_biotype_hg19"])]


# In[73]:


new_map = new_map[["hg19_id", "mm9_id", "cage_id_hg19", "cage_id_mm9", "name_peak_hg19", "name_peak_mm9", 
                   "biotype_hg19", "biotype_mm9", "minimal_biotype_hg19", "minimal_biotype_mm9", "chr_tss_hg19", 
                   "start_tss_hg19", "end_tss_hg19", "strand_tss_hg19", "chr_tss_mm9", "start_tss_mm9", "end_tss_mm9", 
                   "strand_tss_mm9", "avg_exp_hg19", "avg_exp_mm9", "stem_exp_hg19", "stem_exp_mm9", "max_cage_hg19", 
                   "max_cage_mm9","orig_species", "har"]]
new_map.head()


# In[74]:


new_map[new_map["cage_id_hg19"] == "chr1:203273760..203273784,-"]


# In[75]:


print(len(new_map[pd.isnull(new_map["minimal_biotype_hg19"])]))
new_map[pd.isnull(new_map["minimal_biotype_hg19"])]


# In[76]:


tss_map_f = "../../../data/01__design/01__mpra_list/mpra_tss.with_ids.RECLASSIFIED.txt"
new_map.to_csv(tss_map_f, sep="\t", index=False)

