
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

from Bio import SeqIO
from Bio import pairwise2
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


# ## variables

# In[4]:


tss_map_f = "../../../data/01__design/01__mpra_list/mpra_tss.with_ids.RECLASSIFIED.txt"


# In[5]:


index_f = "../../../data/01__design/02__index/TWIST_pool4_v8_final.with_element_id.txt.gz"


# In[6]:


data_f = "../../../data/02__mpra/03__results/all_processed_results.txt"


# ## 1. import data

# In[7]:


tss_map = pd.read_table(tss_map_f, sep="\t")
tss_map.head()


# In[8]:


index = pd.read_table(index_f)
index.head()


# In[9]:


len(index)


# In[10]:


data = pd.read_table(data_f)
data.head()


# ## 2. make bed file for every non-conserved TSS
# bed file with TSS region and lifted-over non-TSS region

# In[11]:


dedupe = tss_map.drop("orig_species", axis=1).drop_duplicates()
dedupe = dedupe[["chr_tss_hg19", "start_tss_hg19", "end_tss_hg19", "strand_tss_hg19", "cage_id_hg19", "hg19_id",
                 "chr_tss_mm9", "start_tss_mm9", "end_tss_mm9", "strand_tss_mm9", "cage_id_mm9", "mm9_id",
                 "minimal_biotype_hg19", "minimal_biotype_mm9"]].drop_duplicates()
len(dedupe)


# In[12]:


## join w/ tile info
tiles = data[["hg19_id", "mm9_id", "tss_tile_num"]].drop_duplicates()
dedupe = dedupe.merge(tiles, on=["hg19_id", "mm9_id"])
len(dedupe)


# In[13]:


non_cons = dedupe[(dedupe["minimal_biotype_hg19"] == "no CAGE activity") | 
                  (dedupe["minimal_biotype_mm9"] == "no CAGE activity")]
print(len(non_cons))
non_cons.head()


# In[14]:


def get_tile_start_coord(row, suffix):
    if row["tss_tile_num"] == "tile1":
        if row["strand_tss_%s" % suffix] == "+":
            tile_start = row["start_tss_%s" % suffix] - 114
        else:
            tile_start = row["start_tss_%s" % suffix] - 30
    else:
        if row["strand_tss_%s" % suffix] == "+":
            tile_start = row["end_tss_%s" % suffix] - 228
        else:
            tile_start = row["end_tss_%s" % suffix] + 84
    return tile_start

def get_tile_end_coord(row, suffix):
    if row["tss_tile_num"] == "tile1":
        if row["strand_tss_%s" % suffix] == "+":
            tile_end = row["start_tss_%s" % suffix] + 30
        else:
            tile_end = row["start_tss_%s" % suffix] + 114
    else:
        if row["strand_tss_%s" % suffix] == "+":
            tile_end = row["end_tss_%s" % suffix] - 84
        else:
            tile_end = row["end_tss_%s" % suffix] + 228
    return tile_end


# In[15]:


non_cons["start_tile_hg19"] = non_cons.apply(get_tile_start_coord, suffix="hg19", axis=1)
non_cons["end_tile_hg19"] = non_cons.apply(get_tile_end_coord, suffix="hg19", axis=1)
non_cons["start_tile_mm9"] = non_cons.apply(get_tile_start_coord, suffix="mm9", axis=1)
non_cons["end_tile_mm9"] = non_cons.apply(get_tile_end_coord, suffix="mm9", axis=1)
non_cons.head()


# In[16]:


non_cons_hu = non_cons[non_cons["minimal_biotype_hg19"] == "no CAGE activity"]
len(non_cons_hu)
non_cons_hu.sample(5)


# In[17]:


non_cons_hu_bed = non_cons_hu[["chr_tss_hg19", "start_tile_hg19", "end_tile_hg19", "hg19_id", 
                               "start_tss_hg19", "strand_tss_hg19"]]
non_cons_hu_bed = non_cons_hu_bed.sort_values(by=["chr_tss_hg19", "start_tile_hg19"])
non_cons_hu_bed.head()


# In[18]:


non_cons_mo = non_cons[non_cons["minimal_biotype_mm9"] == "no CAGE activity"]
len(non_cons_mo)


# In[19]:


non_cons_mo_bed = non_cons_mo[["chr_tss_mm9", "start_tile_mm9", "end_tile_mm9", "mm9_id", 
                               "start_tss_mm9", "strand_tss_mm9"]]
non_cons_mo_bed = non_cons_mo_bed.sort_values(by=["chr_tss_mm9", "start_tile_mm9"])
non_cons_mo_bed.head()


# In[20]:


# write files to perform intersections
non_cons_hu_bed.to_csv("non_cons_hu.bed", sep="\t", index=False, header=False)
non_cons_mo_bed.to_csv("non_cons_mo.bed", sep="\t", index=False, header=False)


# ## 3. find closest TSS to every non-conserved region

# In[21]:


# ran this outside of the notebook:
# sort -k1,1 -k2,2n non_cons_mo.bed | bedtools closest -s -d -a - -b ../../../data/01__design/00__genome_list/mm9.cage_peak_phase1and2combined_coord.TSS.sorted.bed > non_cons_mo.closest_TSS.bed
# sort -k1,1 -k2,2n non_cons_hu.bed | bedtools closest -s -d -a - -b ../../../data/01__design/00__genome_list/hg19.cage_peak_phase1and2combined_coord.TSS.sorted.bed > non_cons_hu.closest_TSS.bed


# In[22]:


non_cons_hu_closest = pd.read_table("non_cons_hu.closest_TSS.bed", header=None)
non_cons_hu_closest.columns = ["chr_tss_hg19", "start_tile_hg19", "end_tile_hg19", "hg19_id", "start_tss_hg19",
                               "strand_tss_hg19", "chr_closest_hg19", "start_closest_hg19", "end_closest_hg19",
                               "cage_id_closest_hg19", "score_closest_hg19", "strand_closest_hg19", 
                               "distance_closest"]
non_cons_hu_closest.head()


# In[23]:


non_cons_mo_closest = pd.read_table("non_cons_mo.closest_TSS.bed", header=None)
non_cons_mo_closest.columns = ["chr_tss_mm9", "start_tile_mm9", "end_tile_mm9", "mm9_id", "start_tss_mm9",
                               "strand_tss_mm9", "chr_closest_mm9", "start_closest_mm9", "end_closest_mm9",
                               "cage_id_closest_mm9", "score_closest_mm9", "strand_closest_mm9", 
                               "distance_closest"]
non_cons_mo_closest.head()


# ## 4. create bed file w/ human and mouse regions to make fasta

# In[24]:


hu_closest = non_cons_hu_closest[["chr_tss_hg19", "start_closest_hg19", "end_closest_hg19", "hg19_id", 
                                  "distance_closest", "strand_tss_hg19"]]
hu_closest.columns = ["chr_tss_hg19", "start_tss_hg19", "end_tss_hg19", "hg19_id", "dist", "strand_tss_hg19"]
hu_closest["tss_tile_num"] = "tile1"
hu_closest.head()


# In[25]:


hu_closest["start_tile_hg19"] = hu_closest.apply(get_tile_start_coord, suffix="hg19", axis=1)
hu_closest["end_tile_hg19"] = hu_closest.apply(get_tile_end_coord, suffix="hg19", axis=1)
hu_closest_bed = hu_closest[["chr_tss_hg19", "start_tile_hg19", "end_tile_hg19", "hg19_id", 
                             "dist", "strand_tss_hg19"]]
hu_closest_bed.head()


# In[26]:


mo_closest = non_cons_mo_closest[["chr_tss_mm9", "start_closest_mm9", "end_closest_mm9", "mm9_id", 
                                  "distance_closest", "strand_tss_mm9"]]
mo_closest.columns = ["chr_tss_mm9", "start_tss_mm9", "end_tss_mm9", "mm9_id", "dist", "strand_tss_mm9"]
mo_closest["tss_tile_num"] = "tile1"
mo_closest.head()


# In[27]:


mo_closest["start_tile_mm9"] = mo_closest.apply(get_tile_start_coord, suffix="mm9", axis=1)
mo_closest["end_tile_mm9"] = mo_closest.apply(get_tile_end_coord, suffix="mm9", axis=1)
mo_closest_bed = mo_closest[["chr_tss_mm9", "start_tile_mm9", "end_tile_mm9", "mm9_id", 
                             "dist", "strand_tss_mm9"]]
mo_closest_bed.head()


# ### append orig tiles, non-cons tiles, and closest tiles

# In[28]:


non_cons_hu_bed["name"] = non_cons_hu_bed["hg19_id"] + "__non_conserved"
non_cons_hu_bed.columns = ["chr", "start", "end", "id", "score", "strand", "name"]
non_cons_hu_bed = non_cons_hu_bed[["chr", "start", "end", "name", "score", "strand"]]
print(len(non_cons_hu_bed))
non_cons_hu_bed.head()


# In[29]:


cons_hu_bed = non_cons[non_cons["minimal_biotype_hg19"] != "no CAGE activity"][["chr_tss_hg19", "start_tile_hg19",
                                                                                "end_tile_hg19", "hg19_id",
                                                                                "start_tss_hg19", "strand_tss_hg19"]]
cons_hu_bed["name"] = cons_hu_bed["hg19_id"] + "__CAGE"
cons_hu_bed.columns = ["chr", "start", "end", "id", "score", "strand", "name"]
cons_hu_bed = cons_hu_bed[["chr", "start", "end", "name", "score", "strand"]]
print(len(cons_hu_bed))
cons_hu_bed.head()


# In[30]:


hu_closest_bed["name"] = hu_closest_bed["hg19_id"] + "__closest"
hu_closest_bed.columns = ["chr", "start", "end", "id", "score", "strand", "name"]
hu_closest_bed = hu_closest_bed[["chr", "start", "end", "name", "score", "strand"]]
print(len(hu_closest_bed))
hu_closest_bed.head()


# In[31]:


hu_all = cons_hu_bed.append(non_cons_hu_bed).append(hu_closest_bed)
hu_all.to_csv("hu_all.bed", sep="\t", index=False, header=False)


# In[32]:


non_cons_mo_bed["name"] = non_cons_mo_bed["mm9_id"] + "__non_conserved"
non_cons_mo_bed.columns = ["chr", "start", "end", "id", "score", "strand", "name"]
non_cons_mo_bed = non_cons_mo_bed[["chr", "start", "end", "name", "score", "strand"]]
print(len(non_cons_mo_bed))
non_cons_mo_bed.head()


# In[33]:


cons_mo_bed = non_cons[non_cons["minimal_biotype_mm9"] != "no CAGE activity"][["chr_tss_mm9", "start_tile_mm9",
                                                                                "end_tile_mm9", "mm9_id",
                                                                                "start_tss_mm9", "strand_tss_mm9"]]
cons_mo_bed["name"] = cons_mo_bed["mm9_id"] + "__CAGE"
cons_mo_bed.columns = ["chr", "start", "end", "id", "score", "strand", "name"]
cons_mo_bed = cons_mo_bed[["chr", "start", "end", "name", "score", "strand"]]
print(len(cons_mo_bed))
cons_mo_bed.head()


# In[34]:


mo_closest_bed["name"] = mo_closest_bed["mm9_id"] + "__closest"
mo_closest_bed.columns = ["chr", "start", "end", "id", "score", "strand", "name"]
mo_closest_bed = mo_closest_bed[["chr", "start", "end", "name", "score", "strand"]].drop_duplicates(subset=["name"])
print(len(mo_closest_bed))
mo_closest_bed.head()


# In[35]:


mo_all = cons_mo_bed.append(non_cons_mo_bed).append(mo_closest_bed)
mo_all.to_csv("mo_all.bed", sep="\t", index=False, header=False)


# ## 5. extract sequences

# In[36]:


# ran outside script:
# bedtools getfasta -name -s -fi /n/rinn_data2/users/kaia/assemblies/human/hg19/hg19.fa -bed hu_all.bed > hu_all.fasta
# bedtools getfasta -name -s -fi /n/rinn_data2/users/kaia/assemblies/mouse/mm9/mm9.fa -bed mo_all.bed > mo_all.fasta


# In[37]:


with open('hu_all.fasta') as fasta_file:  # Will close handle cleanly
    identifiers = []
    seqs = []
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
        identifiers.append(seq_record.id)
        seqs.append(str(seq_record.seq).upper())


# In[38]:


human_seqs = pd.DataFrame.from_dict({"id": identifiers, "seq": seqs})
human_seqs.head()


# In[39]:


with open('mo_all.fasta') as fasta_file:  # Will close handle cleanly
    identifiers = []
    seqs = []
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
        identifiers.append(seq_record.id)
        seqs.append(str(seq_record.seq).upper())


# In[40]:


mouse_seqs = pd.DataFrame.from_dict({"id": identifiers, "seq": seqs})
mouse_seqs.head()


# ## 6. join w/ paired info

# In[41]:


human_seqs["hg19_id"] = human_seqs["id"].str.split("__", expand=True)[0]
human_seqs["type"] = human_seqs["id"].str.split("__", expand=True)[1].str.split("::", expand=True)[0]
human_seqs.head()


# In[42]:


mouse_seqs["mm9_id"] = mouse_seqs["id"].str.split("__", expand=True)[0]
mouse_seqs["type"] = mouse_seqs["id"].str.split("__", expand=True)[1].str.split("::", expand=True)[0]
mouse_seqs.head()


# In[43]:


non_cons_hu = non_cons_hu.merge(mouse_seqs[mouse_seqs["type"] == "CAGE"][["mm9_id", "seq"]],
                                on="mm9_id")
non_cons_hu = non_cons_hu.merge(human_seqs[human_seqs["type"] == "non_conserved"][["hg19_id", "seq"]],
                                on="hg19_id")
non_cons_hu = non_cons_hu.merge(human_seqs[human_seqs["type"] == "closest"][["hg19_id", "seq"]],
                                on="hg19_id")
print(len(non_cons_hu))
non_cons_hu.head()


# In[44]:


non_cons_hu_sub = non_cons_hu[["hg19_id", "mm9_id", "minimal_biotype_hg19", "minimal_biotype_mm9",
                               "seq_x", "seq_y", "seq"]]
non_cons_hu_sub.columns = ["hg19_id", "mm9_id", "minimal_biotype_hg19", "minimal_biotype_mm9",
                           "orig_mouse_seq", "orig_noncons_human_seq", "closest_human_tss_seq"]
non_cons_hu_sub.head()


# In[45]:


non_cons_mo = non_cons_mo.merge(human_seqs[human_seqs["type"] == "CAGE"][["hg19_id", "seq"]],
                                on="hg19_id")
non_cons_mo = non_cons_mo.merge(mouse_seqs[mouse_seqs["type"] == "non_conserved"][["mm9_id", "seq"]],
                                on="mm9_id")
non_cons_mo = non_cons_mo.merge(mouse_seqs[mouse_seqs["type"] == "closest"][["mm9_id", "seq"]],
                                on="mm9_id")
print(len(non_cons_mo))
non_cons_mo.head()


# In[46]:


non_cons_mo_sub = non_cons_mo[["hg19_id", "mm9_id", "minimal_biotype_hg19", "minimal_biotype_mm9",
                               "seq_x", "seq_y", "seq"]]
non_cons_mo_sub.columns = ["hg19_id", "mm9_id", "minimal_biotype_hg19", "minimal_biotype_mm9",
                           "orig_human_seq", "orig_noncons_mouse_seq", "closest_mouse_tss_seq"]
non_cons_mo_sub.head()


# ## 7. do pairwise seq alignments

# In[47]:


orig_scores = []
orig_alignment_strings = []
closest_scores = []
closest_alignment_strings = []
for tup in non_cons_hu_sub.iterrows():
    idx = tup[0]
    if idx % 20 == 0:
        print(idx)
    row = tup[1]
    orig_seq = row.orig_mouse_seq
    noncons_seq = row.orig_noncons_human_seq
    closest_seq = row.closest_human_tss_seq
    
    orig_alignments = pairwise2.align.localms(orig_seq, noncons_seq, 2, -1, -1, -0.1, one_alignment_only=True)
    for a in orig_alignments:
        orig_scores.append(a[2])
        orig_alignment_strings.append(pairwise2.format_alignment(*a))
        
    closest_alignments = pairwise2.align.localms(orig_seq, closest_seq, 2, -1, -1, -0.1, one_alignment_only=True)
    for a in closest_alignments:
        closest_scores.append(a[2])
        closest_alignment_strings.append(pairwise2.format_alignment(*a))


# In[48]:


non_cons_hu_sub["orig_align_score"] = orig_scores
non_cons_hu_sub["orig_align_str"] = orig_alignment_strings
non_cons_hu_sub["closest_align_score"] = closest_scores
non_cons_hu_sub["closest_align_str"] = closest_alignment_strings
non_cons_hu_sub.head()


# In[49]:


orig_scores = []
orig_alignment_strings = []
closest_scores = []
closest_alignment_strings = []
for tup in non_cons_mo_sub.iterrows():
    idx = tup[0]
    if idx % 20 == 0:
        print(idx)
    row = tup[1]
    orig_seq = row.orig_human_seq
    noncons_seq = row.orig_noncons_mouse_seq
    closest_seq = row.closest_mouse_tss_seq
    
    orig_alignments = pairwise2.align.localms(orig_seq, noncons_seq, 2, -1, -1, -0.1, one_alignment_only=True)
    for a in orig_alignments:
        orig_scores.append(a[2])
        orig_alignment_strings.append(pairwise2.format_alignment(*a))
        
    closest_alignments = pairwise2.align.localms(orig_seq, closest_seq, 2, -1, -1, -0.1, one_alignment_only=True)
    for a in closest_alignments:
        closest_scores.append(a[2])
        closest_alignment_strings.append(pairwise2.format_alignment(*a))


# In[50]:


non_cons_mo_sub["orig_align_score"] = orig_scores
non_cons_mo_sub["orig_align_str"] = orig_alignment_strings
non_cons_mo_sub["closest_align_score"] = closest_scores
non_cons_mo_sub["closest_align_str"] = closest_alignment_strings
non_cons_mo_sub.head()


# ## 8. make some plots

# In[51]:


align_hu = non_cons_hu_sub[["hg19_id", "mm9_id", "minimal_biotype_hg19", "minimal_biotype_mm9",
                            "orig_align_score", "closest_align_score", "orig_align_str", "closest_align_str"]]
align_mo = non_cons_mo_sub[["hg19_id", "mm9_id", "minimal_biotype_hg19", "minimal_biotype_mm9",
                            "orig_align_score", "closest_align_score", "orig_align_str", "closest_align_str"]]
align = align_hu.append(align_mo)
align.sample(5)


# In[52]:


fig = plt.figure(figsize=(2, 2))

ax = sns.scatterplot(data=align, x="orig_align_score", y="closest_align_score", alpha=0.9, 
                     color=sns.color_palette("Set2")[2])
ax.set_xlim((70, 300))
ax.set_ylim((70, 300))
ax.plot([70, 300], [70, 300], linestyle="dashed", color="black")
ax.set_xlabel("pairwise alignment score\nwith liftover region (no TSS)")
ax.set_ylabel("pairwise alignment score\nwith closest TSS to liftover region")
fig.savefig("pairwise_score_scatter.pdf", dpi="figure", bbox_inches="tight")


# In[53]:


len(align)


# In[54]:


len(align[align["orig_align_score"] > align["closest_align_score"]])


# In[55]:


len(align[align["orig_align_score"] < align["closest_align_score"]])


# In[56]:


align[align["orig_align_score"] < align["closest_align_score"]]


# In[57]:


align[align["hg19_id"] == "h.1534"]


# In[59]:


mo_closest[mo_closest["mm9_id"] == "m.1346"]


# In[60]:


test = "TGTTTTGGGCTATACTGCCCCTGAGGCCCAGGGTCAAAGGCCACTGGGGAACTGCTGTCGTTCCCATCTCCACGTTAGGGCAGTTATAAAAGAGAACAAGGAAGCTCCCACAGGAAGAAAGCCGCCTGGCCTTGTTCCTATGTCT"


# In[62]:


hu_seq = non_cons_mo_sub[non_cons_mo_sub["mm9_id"] == "m.1346"]["orig_human_seq"].iloc[0]
hu_seq


# In[63]:


pairwise2.align.localms(hu_seq, test, 2, -1, -1, -0.1, one_alignment_only=True)


# In[64]:


neg_test = "AGACATAGGAACAAGGCCAGGCGGCTTTCTTCCTGTGGGAGCTTCCTTGTTCTCTTTTATAACTGCCCTAACGTGGAGATGGGAACGACAGCAGTTCCCCAGTGGCCTTTGACCCTGGGCCTCAGGGGCAGTATAGCCCAAAACA"


# In[65]:


pairwise2.align.localms(hu_seq, neg_test, 2, -1, -1, -0.1, one_alignment_only=True)

