
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
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys

from itertools import combinations 
from scipy.stats import boxcox
from scipy.stats import linregress
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from statsmodels.stats.anova import anova_lm

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


human_motifs_f = "../../../data/04__mapped_motifs/human_motifs_filtered.txt.gz"
mouse_motifs_f = "../../../data/04__mapped_motifs/mouse_motifs_filtered.txt.gz"


# In[6]:


motif_info_dir = "../../../misc/01__motif_info"
motif_map_f = "%s/00__lambert_et_al_files/00__metadata/curated_motif_map.txt" % motif_info_dir
motif_info_f = "%s/00__lambert_et_al_files/00__metadata/motif_info.txt" % motif_info_dir


# In[7]:


sig_motifs_f = "../../../data/04__mapped_motifs/sig_motifs.txt"


# In[8]:


data_f = "../../../data/02__mpra/03__results/all_processed_results.txt"


# In[9]:


orth_expr_f = "../../../data/03__rna_seq/04__TF_expr/orth_TF_expression.txt"


# ## 1. import data

# In[75]:


# this file is already filtered to correct tile nums
human_motifs = pd.read_table(human_motifs_f, sep="\t")
human_motifs.head()


# In[76]:


# this file is already filtered to correct tile nums
mouse_motifs = pd.read_table(mouse_motifs_f, sep="\t")
mouse_motifs.head()


# In[77]:


motif_info = pd.read_table(motif_info_f, sep="\t")
motif_info.head()


# In[78]:


sig_motifs = pd.read_table(sig_motifs_f)
sig_motifs = sig_motifs[sig_motifs["padj"] < 0.05]
print(len(sig_motifs))
sig_motifs.head()


# In[79]:


sig_motifs_str = sig_motifs[sig_motifs["rsq"] >= 0.01]
print(len(sig_motifs_str))
sig_motifs_str.head()


# In[80]:


data = pd.read_table(data_f)
data.head()


# In[81]:


orth_expr = pd.read_table(orth_expr_f)
orth_expr.head()


# ## 2. motif files already filtered

# In[82]:


len(human_motifs["#pattern name"].unique())


# In[83]:


len(mouse_motifs["#pattern name"].unique())


# ## 3. plot motifs for one example

# In[84]:


human_motifs = human_motifs.merge(sig_motifs, left_on="#pattern name", right_on="index")
mouse_motifs = mouse_motifs.merge(sig_motifs, left_on="#pattern name", right_on="index")
human_motifs.head()


# In[85]:


human_motifs.columns


# In[86]:


human_motifs_grp = human_motifs.groupby(["hg19_id", "minimal_biotype_hg19"])["index"].agg("count").reset_index()
enh_grp = human_motifs_grp[human_motifs_grp["minimal_biotype_hg19"] == "eRNA"]
enh_grp.to_csv("enh_grp.tmp.txt", sep="\t", index=False)


# In[87]:


trans_filt = data[~pd.isnull(data["minimal_biotype_hg19"])]
trans_filt = trans_filt[((trans_filt["HUES64_padj_hg19"] < QUANT_ALPHA) | (trans_filt["mESC_padj_mm9"] < QUANT_ALPHA))]
trans_filt = trans_filt[trans_filt["trans_human_status"] == "significant trans effect"]
print(len(trans_filt))
trans_filt.head()


# In[88]:


tmp = human_motifs[human_motifs["HGNC symbol"] == "DMBX1"]
tmp_grp = human_motifs_grp[human_motifs_grp["hg19_id"].isin(tmp["hg19_id"])]
tmp_grp = tmp_grp[tmp_grp["hg19_id"].isin(trans_filt["hg19_id"])]
tmp_grp.sort_values(by="index").head(20)


# In[89]:


len(human_motifs)


# In[90]:


def genome_start(row):
    if row.strand_y == "+":
        genome_start = row.actual_start + row.start - 1
    else:
        genome_start = row.actual_end - row.stop
    return genome_start
        
def genome_end(row):
    if row.strand_y == "+":
        genome_end = row.actual_start + row.stop
    else:
        genome_end = row.actual_end - row.start + 1
    return genome_end


# In[91]:


human_motifs["genome_start"] = human_motifs.apply(genome_start, axis=1)
human_motifs["genome_end"] = human_motifs.apply(genome_end, axis=1)
mouse_motifs["genome_start"] = mouse_motifs.apply(genome_start, axis=1)
mouse_motifs["genome_end"] = mouse_motifs.apply(genome_end, axis=1)


# In[92]:


human_motifs_bed = human_motifs[["chrom", "genome_start", "genome_end", "HGNC symbol", "score", "strand_x"]]
mouse_motifs_bed = mouse_motifs[["chrom", "genome_start", "genome_end", "HGNC symbol", "score", "strand_x"]]


# In[93]:


# human_motifs_bed.to_csv("human_motifs.bed", sep="\t", header=False, index=False)
# mouse_motifs_bed.to_csv("mouse_motifs.bed", sep="\t", header=False, index=False)


# In[96]:


human_motifs_orth = human_motifs.merge(orth_expr, left_on="HGNC symbol", right_on="gene_name_human")
print(len(human_motifs_orth))
human_motifs_orth.head()


# In[97]:


mouse_motifs_orth = mouse_motifs.merge(orth_expr, left_on="HGNC symbol", right_on="gene_name_human")
print(len(mouse_motifs_orth))
mouse_motifs_orth.head()


# In[98]:


human_motifs_DE = human_motifs_orth[human_motifs_orth["sig"] == "sig"]
human_motifs_not_DE = human_motifs_orth[human_motifs_orth["sig"] != "sig"]
mouse_motifs_DE = mouse_motifs_orth[mouse_motifs_orth["sig"] == "sig"]
mouse_motifs_not_DE = mouse_motifs_orth[mouse_motifs_orth["sig"] != "sig"]


# In[102]:


trans_filt = data[~pd.isnull(data["minimal_biotype_hg19"])]
trans_filt = trans_filt[((trans_filt["HUES64_padj_hg19"] < QUANT_ALPHA) | (trans_filt["mESC_padj_mm9"] < QUANT_ALPHA))]
trans_filt = trans_filt[trans_filt["trans_mouse_status"] == "significant trans effect"]
print(len(trans_filt))
trans_filt.head()


# In[104]:


mouse_motifs_DE_grp = mouse_motifs_DE.groupby(["mm9_id", "minimal_biotype_mm9"])["score"].agg("count").reset_index()
mouse_motifs_not_DE_grp = mouse_motifs_not_DE.groupby(["mm9_id", "minimal_biotype_mm9"])["score"].agg("count").reset_index()

mouse_motifs_grp = mouse_motifs_DE_grp.merge(mouse_motifs_not_DE_grp, on=["mm9_id", "minimal_biotype_mm9"], how="outer",
                                             suffixes=("_DE", "_not"))
mouse_motifs_grp_filt = mouse_motifs_grp[mouse_motifs_grp["mm9_id"].isin(trans_filt["mm9_id"])]
mouse_motifs_grp_filt.sort_values(by="score_DE")


# In[56]:


human_motifs_DE_bed = human_motifs_DE[["chrom", "genome_start", "genome_end", "HGNC symbol", "score", "strand_x"]]
human_motifs_not_DE_bed = human_motifs_not_DE[["chrom", "genome_start", "genome_end", "HGNC symbol", "score", "strand_x"]]

mouse_motifs_DE_bed = mouse_motifs_DE[["chrom", "genome_start", "genome_end", "HGNC symbol", "score", "strand_x"]]
mouse_motifs_not_DE_bed = mouse_motifs_not_DE[["chrom", "genome_start", "genome_end", "HGNC symbol", "score", "strand_x"]]


# In[57]:


human_motifs_DE_bed.to_csv("human_motifs.DE.bed", sep="\t", header=False, index=False)
human_motifs_not_DE_bed.to_csv("human_motifs.notDE.bed", sep="\t", header=False, index=False)
mouse_motifs_DE_bed.to_csv("mouse_motifs.DE.bed", sep="\t", header=False, index=False)
mouse_motifs_not_DE_bed.to_csv("mouse_motifs.notDE.bed", sep="\t", header=False, index=False)


# In[46]:


def getOverlap(a, b):
    return max(a[0], b[0]) - min(a[1], b[1])


# In[22]:


import pybedtools


# In[19]:


hu_test = human_motifs[human_motifs["hg19_id"] == "h.986"].sort_values(by="start").drop_duplicates()
hu_test_motifs = hu_test["HGNC symbol"].unique()
hu_test_motifs


# In[20]:


sig_cis = ["ASCL2", "ASCL5", "ASCL4", "ZNF528", "ZNF263", "ZNF398", "ZNF132", "ZNF468", "ELF5", "ELF2", "ELF4",
           "ERF", "ETV1", "ETV3L", "ETV6", "ETV5", "MAZ"]
[x for x in hu_test_motifs if x in sig_cis]


# In[23]:


tot_bed = pd.DataFrame()
for motif in hu_test_motifs:
    sub = hu_test[hu_test["HGNC symbol"] == motif]
    bed = sub[["chrom", "start", "stop", "HGNC symbol", "score", "strand_x"]]
    bed.columns = ["chrom", "start", "end", "name", "score", "strand"]
    if len(bed) <= 1:
        bed = bed[["chrom", "start", "end", "name"]]
        tot_bed = tot_bed.append(bed)
    else:
        bed.to_csv("tmp.bed", sep="\t", header=False, index=False)
    b = pybedtools.BedTool("tmp.bed")
    m = b.merge()
    bed = m.to_dataframe()
    bed["name"] = motif
    tot_bed = tot_bed.append(bed)


# In[24]:


human_bed = tot_bed.sort_values(by="start")
human_bed


# In[25]:


mo_test = mouse_motifs[mouse_motifs["mm9_id"] == "m.869"]
mo_test_motifs = mo_test["HGNC symbol"].unique()
mo_test_motifs


# In[26]:


hu_motif_pos = zip(list(human_bed["start"]), list(human_bed["end"]))
mo_motif_pos = zip(list(mo_test["start"]), list(mo_test["stop"]))


# In[27]:


fig, axarr = plt.subplots(figsize=(7, 4), nrows=2, ncols=1)

# human ax
ax = axarr[0]

# plot motif locations
xs = list(range(0, 144))
prev_plotted = {}

# iterate through things plotted at each prev_y value
# if any overlaps, move
linewidth=5
for i, pos in enumerate(hu_motif_pos):
    #print("")
    #print("i: %s, pos: %s" % (i, pos))
    plotted = False
    if i == 0:
        #print("first motif, plotting at y=0")
        ax.plot([pos[0], pos[1]], [0, 0], color="darkgrey", linewidth=linewidth, solid_capstyle="butt")
        plotted = True
        prev_plotted[0] = [pos]
        continue
    for prev_y in sorted(prev_plotted.keys(), reverse=True):
        vals = prev_plotted[prev_y]
        overlaps = []
        for prev_pos in vals:
            overlaps.append(getOverlap(prev_pos, pos))
        if any(x < 0 for x in overlaps):
            #print("motif overlaps w/ %s, continuing" % (prev_y))
            continue
        else:
            if not plotted:
                #print("motif doesn't overlap anything at y=%s, plotting" % prev_y)
                ax.plot([pos[0], pos[1]], [prev_y, prev_y], color="darkgrey", linewidth=linewidth, 
                              solid_capstyle="butt")
                if prev_y not in prev_plotted:
                    prev_plotted[prev_y] = [pos]
                else:
                    new_vals = list(prev_plotted[prev_y])
                    new_vals.extend([pos])
                    prev_plotted[prev_y] = new_vals
                plotted = True
    if not plotted:
        prev_y -= 0.25
        #print("motif overlaps at all prev_y, plotting at %s" % prev_y)
        ax.plot([pos[0], pos[1]], [prev_y, prev_y], color="darkgrey", linewidth=linewidth, 
                      solid_capstyle="butt")
        if prev_y not in prev_plotted:
            prev_plotted[prev_y] = [pos]
        else:
            new_vals = list(prev_plotted[prev_y])
            new_vals.extend([pos])
            prev_plotted[prev_y] = new_vals
        plotted = True
#     print(prev_plotted)

min_y = np.min(list(prev_plotted.keys()))

# labels
ax.set_xlim((-0.5, 144))
ax.set_ylim((min_y - 0.25, 0.25))
ax.set_xlabel("nucleotide number")
ax.set_ylabel("")
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.axis("off")



# mouse ax
ax = axarr[1]

# plot motif locations
xs = list(range(0, 144))
prev_plotted = {}

# iterate through things plotted at each prev_y value
# if any overlaps, move
linewidth=5
for i, pos in enumerate(mo_motif_pos):
    #print("")
    #print("i: %s, pos: %s" % (i, pos))
    plotted = False
    if i == 0:
        #print("first motif, plotting at y=0")
        ax.plot([pos[0], pos[1]], [0, 0], color="darkgrey", linewidth=linewidth, solid_capstyle="butt")
        plotted = True
        prev_plotted[0] = [pos]
        continue
    for prev_y in sorted(prev_plotted.keys(), reverse=True):
        vals = prev_plotted[prev_y]
        overlaps = []
        for prev_pos in vals:
            overlaps.append(getOverlap(prev_pos, pos))
        if any(x < 0 for x in overlaps):
            #print("motif overlaps w/ %s, continuing" % (prev_y))
            continue
        else:
            if not plotted:
                #print("motif doesn't overlap anything at y=%s, plotting" % prev_y)
                ax.plot([pos[0], pos[1]], [prev_y, prev_y], color="darkgrey", linewidth=linewidth, 
                              solid_capstyle="butt")
                if prev_y not in prev_plotted:
                    prev_plotted[prev_y] = [pos]
                else:
                    new_vals = list(prev_plotted[prev_y])
                    new_vals.extend([pos])
                    prev_plotted[prev_y] = new_vals
                plotted = True
    if not plotted:
        prev_y -= 0.25
        #print("motif overlaps at all prev_y, plotting at %s" % prev_y)
        ax.plot([pos[0], pos[1]], [prev_y, prev_y], color="darkgrey", linewidth=linewidth, 
                      solid_capstyle="butt")
        if prev_y not in prev_plotted:
            prev_plotted[prev_y] = [pos]
        else:
            new_vals = list(prev_plotted[prev_y])
            new_vals.extend([pos])
            prev_plotted[prev_y] = new_vals
        plotted = True
#     print(prev_plotted)

min_y = np.min(list(prev_plotted.keys()))

# labels
ax.set_xlim((-0.5, 144))
ax.set_ylim((min_y - 0.25, 0.25))
ax.set_xlabel("nucleotide number")
ax.set_ylabel("")
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.axis("off")
    
plt.show()


# In[42]:


prev_plotted

