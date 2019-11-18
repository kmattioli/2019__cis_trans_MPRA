
# coding: utf-8

# # 01__motif_model
# 
# in this notebook, i find motifs that are significantly associated w/ mean MPRA activity using linear models

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

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

# import utils
sys.path.append("../../../utils")
from plotting_utils import *
from classify_utils import *

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


# ## functions

# In[5]:


def calculate_gc(row):
    cs = row["index"].count("C")
    gs = row["index"].count("G")
    gc = (cs+gs)/len(row["index"])
    return gc


# In[6]:


def calculate_cpg(row):
    cpgs = row["index"].count("CG")
    cpg = cpgs/len(row["index"])
    return cpg


# In[7]:


def lrtest(llmin, llmax):
    lr = 2 * (llmax - llmin)
    p = stats.chisqprob(lr, 1) # llmax has 1 dof more than llmin
    return lr, p


# In[8]:


def activ_or_repress(row):
    if row.beta > 0:
        return "activating"
    elif row.beta < 0:
        return "repressing"


# ## variables

# In[9]:


data_f = "../../../data/02__mpra/02__activs/alpha_per_elem.quantification.txt"


# In[10]:


index_f = "../../../data/01__design/02__index/TWIST_pool4_v8_final.with_element_id.txt.gz"


# In[11]:


tss_map_f = "../../../data/01__design/01__mpra_list/mpra_tss.with_ids.RECLASSIFIED_WITH_MAX.txt"


# In[12]:


motif_dir = "../../../data/04__mapped_motifs/elem_fimo_out"
motifs_f = "%s/fimo.txt.gz" % motif_dir


# In[13]:


elem_map_f = "../../../data/04__mapped_motifs/fastas/elem_map.txt"


# In[14]:


motif_info_dir = "../../../misc/01__motif_info"
motif_map_f = "%s/00__lambert_et_al_files/00__metadata/curated_motif_map.txt" % motif_info_dir
motif_info_f = "%s/00__lambert_et_al_files/00__metadata/motif_info.txt" % motif_info_dir


# In[15]:


hESC_tf_f = "../../../data/03__rna_seq/04__TF_expr/hESC_TF_expression.txt"
mESC_tf_f = "../../../data/03__rna_seq/04__TF_expr/mESC_TF_expression.txt"


# ## 1. import data

# In[16]:


data = pd.read_table(data_f).reset_index()
data.head()


# In[17]:


index = pd.read_table(index_f, sep="\t")
index_elem = index[["element", "tile_type", "element_id", "name", "tile_number", "chrom", "strand", "actual_start", 
                    "actual_end", "dupe_info"]]
index_elem = index_elem.drop_duplicates()


# In[18]:


tss_map = pd.read_table(tss_map_f, sep="\t")
tss_map.head()


# In[19]:


motifs = pd.read_table(motifs_f, sep="\t")
motifs.head()


# In[20]:


elem_map = pd.read_table(elem_map_f, sep="\t")
elem_map.head()


# In[21]:


motif_map = pd.read_table(motif_map_f, sep="\t")
motif_map.head()


# In[22]:


motif_info = pd.read_table(motif_info_f, sep="\t")
motif_info.head()


# In[23]:


hESC_tf = pd.read_table(hESC_tf_f)
mESC_tf = pd.read_table(mESC_tf_f)
hESC_tf.head()


# ## 2. calculate GC and CpG content per element

# In[24]:


data["gc_content"] = data.apply(calculate_gc, axis=1)
data.sample(5)


# In[25]:


data["cpg_content"] = data.apply(calculate_cpg, axis=1)
data.sample(5)


# In[26]:


data["HUES64_log"] = np.log10(data["HUES64"]+1)
data["mESC_log"] = np.log10(data["mESC"]+1)
data.sample(5)


# In[27]:


data["avg_activ"] = data[["HUES64", "mESC"]].mean(axis=1)
data["avg_activ_log"] = np.log10(data["avg_activ"]+1)


# In[28]:


data["HUES64_box"] = boxcox(data["HUES64"])[0]
data["mESC_box"] = boxcox(data["mESC"])[0]
data["avg_activ_box"] = boxcox(data["avg_activ"])[0]
data.sample(5)


# In[29]:


data["short_elem"] = data["index"].str.split("__", expand=True)[0]
data.sample(5)


# ## 3. build reduced model (CG + CpG content)

# In[30]:


scaled_features = StandardScaler().fit_transform(data[["avg_activ_box", "gc_content", "cpg_content"]])
data_norm = pd.DataFrame(scaled_features, index=data.index, columns=["avg_activ_box", "gc_content", "cpg_content"])
data_norm["index"] = data['index']
data_norm["HUES64_padj"] = data["HUES64_padj"]
data_norm["mESC_padj"] = data["mESC_padj"]
data_norm["short_elem"] = data["short_elem"]
data_norm.head()


# In[31]:


data_filt = data_norm[((data_norm["HUES64_padj"] < QUANT_ALPHA) | (data_norm["mESC_padj"] < QUANT_ALPHA))]
len(data_filt)


# In[32]:


avg_mod = smf.ols(formula='avg_activ_box ~ gc_content + cpg_content', 
                   data=data_norm).fit()


# In[33]:


avg_mod.summary()


# In[34]:


res = avg_mod.resid

fig, ax = plt.subplots(figsize=(2.2, 2.2), ncols=1, nrows=1)
sm.qqplot(res, line='s', ax=ax)
ax.set_title("Normal QQ: average hESC/mESC model")
# fig.savefig("avg_activ_qq.pdf", dpi="figure", bbox_inches="tight")


# In[35]:


reduced_llf = avg_mod.llf
reduced_llf


# In[36]:


reduced_rsq = avg_mod.rsquared
reduced_rsq


# ## 4. add individual motifs to the model

# In[37]:


len(motifs)


# In[38]:


# only analyze the "best" motifs as determined by lambert et al
best_motifs = motif_info[~pd.isnull(motif_info["Best Motif(s)? (Figure 2A)"])]
len(best_motifs)


# In[39]:


# only analyze the TFs that are expressed in hESCs or mESCs
hESC_expr = hESC_tf[hESC_tf["mean_tpm"] > 1]
mESC_expr = mESC_tf[mESC_tf["mean_tpm_mouse"] > 1]
print(len(hESC_expr))
print(len(mESC_expr))


# In[40]:


best_motifs = best_motifs[((best_motifs["HGNC symbol"].isin(hESC_expr["gene_name"])) |
                           (best_motifs["HGNC symbol"].isin(mESC_expr["gene_name_human"])))]
len(best_motifs)


# In[41]:


best_motifs["short_id"] = best_motifs["CIS-BP ID"].str.split(".", expand=True)[0]
mapped_best_motifs = motifs[motifs["#pattern name"].isin(best_motifs["short_id"])]
len(mapped_best_motifs)


# In[42]:


uniq_motifs = list(best_motifs["short_id"].unique())
print(len(uniq_motifs))


# In[43]:


motif_results = {}

for i, motif_id in enumerate(uniq_motifs):
    tmp = data_norm.copy()

    elem_ids_w_motif = list(mapped_best_motifs[mapped_best_motifs["#pattern name"] == motif_id]["sequence name"].unique())
    elems_w_motif = list(elem_map[elem_map["elem_key"].isin(elem_ids_w_motif)]["elem"])
    
    tmp["has_motif"] = tmp["short_elem"].isin(elems_w_motif)
    tmp["has_motif"] = tmp["has_motif"].astype(int)
    
    # build model with motif included
    motif_mod = smf.ols(formula='avg_activ_box ~ gc_content + cpg_content + has_motif', 
                        data=tmp).fit()
    
    # perform likelihood ratio test
    motif_llf = motif_mod.llf
    lr, p = lrtest(reduced_llf, motif_llf)
    
    # calculate additional variance explained
    rsq = motif_mod.rsquared - reduced_rsq
    
    # record beta
    beta = list(motif_mod.params)[-1]
    
    print("(#%s) %s: n w/ motif: %s ... p: %s, rsquared: %s" % (i+1, motif_id, len(elems_w_motif), p, rsq))
    motif_results[motif_id] = {"lr_test": lr, "pval": p, "rsq": rsq, "beta": beta}


# In[44]:


motif_results = pd.DataFrame.from_dict(motif_results, orient="index").reset_index()
motif_results.head()


# In[45]:


motif_results["padj"] = multicomp.multipletests(motif_results["pval"], method="fdr_bh")[1]
len(motif_results[motif_results["padj"] < 0.05])


# ## 5. merge motifs w/ TF info

# In[46]:


all_motif_results = motif_results.merge(best_motifs[["short_id", "HGNC symbol"]], left_on="index", right_on="short_id")
all_motif_results.sample(5)


# In[47]:


all_motif_results["activ_or_repr"] = all_motif_results.apply(activ_or_repress, axis=1)
all_motif_results.sample(5)


# In[48]:


sig_motif_results = all_motif_results[all_motif_results["padj"] < 0.05]
len(sig_motif_results)


# In[49]:


sig_motif_results.sort_values(by="rsq", ascending=False).head()


# In[50]:


sig_motif_results.activ_or_repr.value_counts()


# ## 6. plot results

# In[51]:


over_1p = sig_motif_results[sig_motif_results["rsq"] >= 0.001].sort_values(by="rsq", ascending=False)
len(over_1p)


# In[52]:


over_1p.activ_or_repr.value_counts()


# In[53]:


sns.palplot(sns.color_palette("pastel"))


# In[54]:


pal = {"repressing": sns.color_palette("pastel")[3], "activating": sns.color_palette("pastel")[0]}


# In[55]:


full_pal = {}
for i, row in over_1p.iterrows():
    full_pal[row["HGNC symbol"]] = pal[row["activ_or_repr"]]


# ## 7. merge elements with metadata (tss_id, biotype)

# In[56]:


motifs_merged = mapped_best_motifs.merge(elem_map, left_on="sequence name", right_on="elem_key")
motifs_merged.head()


# In[57]:


motifs_merged = motifs_merged.merge(index_elem, left_on="elem", right_on="element")
motifs_merged.head()


# In[58]:


motifs_merged["tss_id"] = motifs_merged["name"].str.split("__", expand=True)[1]
motifs_merged["species"] = motifs_merged["name"].str.split("_", expand=True)[0]
motifs_merged["tss_tile_num"] = motifs_merged["name"].str.split("__", expand=True)[2]
motifs_merged.sample(5)


# In[59]:


human_df = motifs_merged[(motifs_merged["species"] == "HUMAN") | (motifs_merged["name"] == "random_sequence")]
mouse_df = motifs_merged[(motifs_merged["species"] == "MOUSE") | (motifs_merged["name"] == "random_sequence")]

human_df = human_df.merge(tss_map[["hg19_id", "biotype_hg19", "minimal_biotype_hg19", 
                                   "stem_exp_hg19", "orig_species", "mm9_id", "tile_match"]], 
                          left_on="tss_id", right_on="hg19_id", how="left")
mouse_df = mouse_df.merge(tss_map[["mm9_id", "biotype_mm9", "minimal_biotype_mm9", 
                                   "stem_exp_mm9", "orig_species", "hg19_id", "tile_match"]], 
                          left_on="tss_id", right_on="mm9_id", how="left")
mouse_df.sample(5)


# In[60]:


mouse_df.columns


# ## 8. find enrichment of motifs across biotypes

# In[61]:


both_tile_ids = tss_map[(~pd.isnull(tss_map["n_tiles_hg19"]) & ~(pd.isnull(tss_map["n_tiles_mm9"])))]
len(both_tile_ids)


# In[62]:


tile1_ids = both_tile_ids[(both_tile_ids["tile_match"] == "tile1:tile1") | 
                          (both_tile_ids["tile_match"] == "tile1:tile2")][["hg19_id", "mm9_id"]].drop_duplicates()
len(tile1_ids)


# In[63]:


tile2_ids = both_tile_ids[(both_tile_ids["tile_match"] == "tile2:tile2")][["hg19_id", "mm9_id"]].drop_duplicates()
len(tile2_ids)


# In[64]:


# limit dfs to tile1s where appropriate and tile2 where appropriate
human_tile1 = human_df.merge(tile1_ids, on=["hg19_id", "mm9_id"])
human_tile1 = human_tile1[human_tile1["tss_tile_num"] == "tile1"]
human_tile1 = human_tile1.drop(["orig_species", "mm9_id", "tile_match"], axis=1).drop_duplicates()
len(human_tile1)


# In[65]:


human_tile2 = human_df.merge(tile2_ids, on=["hg19_id", "mm9_id"])
human_tile2 = human_tile2[human_tile2["tss_tile_num"] == "tile2"]
human_tile2 = human_tile2.drop(["orig_species", "mm9_id", "tile_match"], axis=1).drop_duplicates()
len(human_tile2)


# In[66]:


mouse_tile1 = mouse_df.merge(tile1_ids, on=["mm9_id", "hg19_id"])
mouse_tile1 = mouse_tile1[mouse_tile1["tss_tile_num"] == "tile1"]
mouse_tile1 = mouse_tile1.drop(["orig_species", "hg19_id", "tile_match"], axis=1).drop_duplicates()
len(mouse_tile1)


# In[67]:


mouse_tile2 = mouse_df.merge(tile2_ids, on=["mm9_id", "hg19_id"])
mouse_tile2 = mouse_tile2[mouse_tile2["tss_tile_num"] == "tile2"]
mouse_tile2 = mouse_tile2.drop(["orig_species", "hg19_id", "tile_match"], axis=1).drop_duplicates()
len(mouse_tile2)


# In[68]:


print(len(human_tile1.hg19_id.unique()))
print(len(mouse_tile1.mm9_id.unique()))


# In[69]:


print(len(human_tile2.hg19_id.unique()))
print(len(mouse_tile2.mm9_id.unique()))


# In[70]:


human_df = human_tile1.append(human_tile2)
mouse_df = mouse_tile1.append(mouse_tile2)


# In[71]:


biotype_motif_res = {}

for i, row in sig_motif_results.iterrows():
    motif_id = row["index"]
    print("(#%s: %s)" % (i+1, motif_id))
    
    human_motifs_sub = human_df[(human_df["#pattern name"] == motif_id)]["hg19_id"].unique()
    mouse_motifs_sub = mouse_df[(mouse_df["#pattern name"] == motif_id)]["mm9_id"].unique()
    
    tmp = {}
    for biotype in ["no CAGE activity", "eRNA", "lncRNA", "mRNA"]:
        
        # group no CAGE + reclassified together here
        if biotype == "no CAGE activity":
            human_sub = tss_map[tss_map["minimal_biotype_hg19"].isin(["no CAGE activity", 
                                                                        "reclassified - CAGE peak"])]["hg19_id"].unique()
            mouse_sub = tss_map[tss_map["minimal_biotype_mm9"].isin(["no CAGE activity", 
                                                                       "reclassified - CAGE peak"])]["mm9_id"].unique()
            
            human_not_sub = tss_map[~tss_map["minimal_biotype_hg19"].isin(["no CAGE activity", 
                                                                             "reclassified - CAGE peak"])]["hg19_id"].unique()
            mouse_not_sub = tss_map[~tss_map["minimal_biotype_mm9"].isin(["no CAGE activity", 
                                                                            "reclassified - CAGE peak"])]["mm9_id"].unique()
        else:
            human_sub = tss_map[tss_map["minimal_biotype_hg19"] == biotype]["hg19_id"].unique()
            mouse_sub = tss_map[tss_map["minimal_biotype_mm9"] == biotype]["mm9_id"].unique()
            
            human_not_sub = tss_map[tss_map["minimal_biotype_hg19"] != biotype]["hg19_id"].unique()
            mouse_not_sub = tss_map[tss_map["minimal_biotype_mm9"] != biotype]["mm9_id"].unique()
    
        # count occurrences of biotypes w/ and w/o motifs
        n_human_biotype_w_motif = len([x for x in human_sub if x in human_motifs_sub])
        n_human_not_biotype_w_motif = len([x for x in human_not_sub if x in human_motifs_sub])
        n_mouse_biotype_w_motif = len([x for x in mouse_sub if x in mouse_motifs_sub])
        n_mouse_not_biotype_w_motif = len([x for x in mouse_not_sub if x in mouse_motifs_sub])

        n_human_biotype_wo_motif = len([x for x in human_sub if x not in human_motifs_sub])
        n_human_not_biotype_wo_motif = len([x for x in human_not_sub if x not in human_motifs_sub])
        n_mouse_biotype_wo_motif = len([x for x in mouse_sub if x not in mouse_motifs_sub])
        n_mouse_not_biotype_wo_motif = len([x for x in mouse_not_sub if x not in mouse_motifs_sub])

        # hypergeometric test - combined
        tot_biotype_w_motif = n_human_biotype_w_motif + n_mouse_biotype_w_motif
        tot_biotype = n_human_biotype_w_motif + n_human_biotype_wo_motif + n_mouse_biotype_w_motif + n_mouse_biotype_wo_motif
        tot_motif = tot_biotype_w_motif + n_human_not_biotype_w_motif + n_mouse_not_biotype_w_motif
        tot = tot_biotype + n_human_not_biotype_w_motif + n_human_not_biotype_wo_motif + n_mouse_not_biotype_w_motif + n_mouse_not_biotype_wo_motif

        both_pval = stats.hypergeom.sf(tot_biotype_w_motif-1, tot, tot_biotype, tot_motif)

        # note results
        if biotype == "no CAGE activity":
            s = "no_CAGE"
        else:
            s = biotype
        tmp["%s_pval" % s] = both_pval
        
    biotype_motif_res[motif_id] = tmp
    


# In[72]:


biotype_res = pd.DataFrame.from_dict(biotype_motif_res, orient="index").reset_index()
biotype_res.head()


# In[73]:


biotype_melt = pd.melt(biotype_res, id_vars="index")
biotype_melt.head()


# In[74]:


biotype_melt["padj"] = multicomp.multipletests(biotype_melt["value"], method="fdr_bh")[1]
len(biotype_melt[biotype_melt["padj"] < 0.05])


# In[75]:


biotype_melt.sample(5)


# In[76]:


def is_sig(row):
    if row["padj"] < 0.05:
        return 1
    else:
        return 0


# In[77]:


biotype_melt["sig"] = biotype_melt.apply(is_sig, axis=1)
biotype_melt.head()


# In[78]:


biotype_res = biotype_melt.pivot(index="index", columns="variable")["padj"]
biotype_res.head()


# In[79]:


def no_cage_vars(row):
    if row["no_CAGE_pval"] < 0.05:
        return 0
    else:
        return np.nan
    
def erna_vars(row):
    if row["eRNA_pval"] < 0.05:
        return 1
    else:
        return np.nan
    
def lncrna_vars(row):
    if row["lncRNA_pval"] < 0.05:
        return 2
    else:
        return np.nan
    
def mrna_vars(row):
    if row["mRNA_pval"] < 0.05:
        return 3
    else:
        return np.nan
    
biotype_res["no_CAGE_enr"] = biotype_res.apply(no_cage_vars, axis=1)
biotype_res["eRNA_enr"] = biotype_res.apply(erna_vars, axis=1)
biotype_res["lncRNA_enr"] = biotype_res.apply(lncrna_vars, axis=1)
biotype_res["mRNA_enr"] = biotype_res.apply(mrna_vars, axis=1)
biotype_res = biotype_res.reset_index()


# In[80]:


biotype_res.head()


# In[81]:


biotype_melt = pd.melt(biotype_res, id_vars="index", value_vars=["no_CAGE_enr", "eRNA_enr", "lncRNA_enr", "mRNA_enr"])
biotype_melt.head()


# In[82]:


all_tfs = over_1p["HGNC symbol"].unique()
print(len(all_tfs))
all_tfs[0:5]


# In[83]:


all_tfs1 = all_tfs[0:72]
all_tfs2 = all_tfs[72:144]
all_tfs3 = all_tfs[144:]
print(len(all_tfs1))
print(len(all_tfs2))
print(len(all_tfs3))


# In[84]:


for tfs, xlims, pt in zip([all_tfs1, all_tfs2, all_tfs3],
                          [(0, 0.055), (0, 0.01), (0, 0.002)],
                          ["1", "2", "3"]):
    sub = over_1p[over_1p["HGNC symbol"].isin(tfs)]
    
    fig = plt.figure(figsize=(8, 10))

    ax1 = plt.subplot2grid((1, 12), (0, 0), colspan=3)
    ax2 = plt.subplot2grid((1, 12), (0, 3), colspan=1, sharey=ax1)

    yvals = []
    symbs = []
    c = 0
    for i, row in sub.iterrows():
        symb = row["HGNC symbol"]
        if symb not in symbs:
            yvals.append(c)
            symbs.append(symb)
            c += 1
        else:
            yvals.append(c)

    sub["yval"] = yvals
    print(len(sub))
    sns.barplot(y="HGNC symbol", x="rsq", data=sub, palette=full_pal, ax=ax1)
    ax1.set_ylabel("")
    ax1.set_xlabel("additional variance explained")
    ax1.set_xlim(xlims)

    mrg = sub.merge(biotype_melt, on="index", how="left")
    mrg = mrg[["yval", "HGNC symbol", "variable", "value"]]
    ax2.plot(mrg["value"], mrg["yval"], 'o', color="black")
    ax2.set_xlim((-0.5, 3.5))
    ax2.set_ylim((np.max(yvals)-0.5, -0.5))
    ax2.tick_params(labelleft=False, labelbottom=False, bottom=False, left=False, top=True, labeltop=True)
    ax2.xaxis.set_ticks([0, 1, 2, 3])
    ax2.set_xticklabels(["no CAGE", "eRNA", "lncRNA", "mRNA"], rotation=60, ha="left", va="bottom")
    
    plt.show()
    fig.savefig("FigS9_%s.pdf" % (pt), dpi="figure", bbox_inches="tight")
    plt.close()


# ## 7. write file

# ### clean up file for supplement

# In[85]:


supp_file = all_motif_results[["HGNC symbol", "short_id", "rsq", "beta", "padj", "activ_or_repr"]]
supp_file = supp_file.sort_values(by="HGNC symbol")
supp_file.head()


# In[86]:


supp_file.columns = ["gene_name", "cisbp_motif_id", "var_explained", "beta", "padj", "activ_or_repr"]
supp_file.head()


# In[87]:


len(supp_file)


# In[88]:


len(supp_file.gene_name.unique())


# In[89]:


len(supp_file[supp_file["padj"] < 0.05])


# In[90]:


supp_file.to_csv("../../../data/04__mapped_motifs/SuppTable_Motifs.txt", sep="\t", index=False)


# ### make file for downstream analyses

# In[91]:


all_motif_results = all_motif_results.merge(biotype_res[["index", "no_CAGE_enr", "eRNA_enr", "lncRNA_enr",
                                                         "mRNA_enr"]], on="index")
all_motif_results.head()


# In[92]:


all_motif_results.to_csv("../../../data/04__mapped_motifs/sig_motifs.txt", sep="\t", index=False)

