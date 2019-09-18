
# coding: utf-8

# # notebook to reformat cisbp files to meme and other issues
# [all based on the files in the 2018 Lambert et al review "The Human Transcription Factors"]

# In[1]:


import numpy as np
import os
import pandas as pd
import sys


# ## variables

# In[2]:


tf_info_f = "00__metadata/TF_info.txt"
motif_info_f = "00__metadata/motif_info.txt"


# In[3]:


pwm_dir = "01__pwms"


# ## 1. import data

# In[4]:


tf_info = pd.read_table(tf_info_f, sep="\t")
tf_info.head()


# In[5]:


motif_info = pd.read_table(motif_info_f, sep="\t")
motif_info = motif_info[~pd.isnull(motif_info['CIS-BP ID'])]
motif_info.head()


# ## 2. read in motif files

# In[6]:


motifs = {}


# In[7]:


files = os.listdir(pwm_dir)
files = [f for f in files if "README" not in f]
print("n files: %s" % (len(files)))
for f in files:
    
    motif = f.split(".")[0]
    with open("%s/%s" % (pwm_dir, f)) as fp:
        for line in fp:
            if line.startswith("Pos"):
                continue
            info = line.split()
            if info[0] == "1":
                pwm = []

            info = line.split()
            
            # round the pwm info to 5 decimal points
            info = [round(float(x), 5) for x in info]
            
            pwm.append(info[1:])
    motifs[motif] = pwm


# In[8]:


list(motifs.keys())[0:5]


# In[9]:


motifs['HKR1']


# ## 3. map motifs to curated TFs

# In[10]:


curated_tfs = set(tf_info[tf_info["Is TF?"] == "Yes"]["Ensembl ID"])
len(curated_tfs)


# In[11]:


curated_motif_map = {}
curated_pwms = {}

for key in motifs:
    gene = motif_info[motif_info["CIS-BP ID"].str.contains(key)]["Ensembl ID"].iloc[0]
    gene_name = motif_info[motif_info["CIS-BP ID"].str.contains(key)]["HGNC symbol"].iloc[0]
    if gene in curated_tfs:
        pwm = motifs[key]
        
        # make sure the pwm sums to 1 in all rows!
        arr = np.asarray(pwm, dtype=np.float64)
        s = arr.sum(axis=1)
        if (s < 0.99).any() or (s > 1.01).any():
            print("bad motif: %s | len: %s | sums: %s" % (key, len(pwm), s))
        else:
            curated_pwms[key] = motifs[key]
            curated_motif_map[key] = {"gene_id": gene, "gene_name": gene_name}


# In[12]:


curated_motif_map = pd.DataFrame.from_dict(curated_motif_map, orient="index").reset_index()
curated_motif_map.head()


# In[13]:


len(curated_motif_map["gene_id"].unique())


# In[14]:


len(curated_motif_map)


# In[15]:


curated_motif_map_f = "00__metadata/curated_motif_map.txt"
curated_motif_map.to_csv(curated_motif_map_f, sep="\t", index=False)


# ## 4. convert to MEME format (for FIMO)

# In[16]:


out_f = "../01__meme_files/human_curated_tfs.txt"

with open(out_f, "w") as f:
    f.write("MEME version 4\n\n")
    f.write("ALPHABET= ACGT\n\n")
    f.write("strands: + -\n\n")
    f.write("Background letter frequencies (from uniform background):\nA 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n")

    # now write the motifs
    for key in curated_pwms:
        #print(key)

        # first find its name
        motif_name = curated_motif_map[curated_motif_map["index"] == key]["gene_name"].iloc[0]
        f.write("MOTIF %s %s\n\n" % (key, motif_name))

        pwm = curated_pwms[key]
        n_bp = len(pwm)
        f.write("letter-probability matrix: alength= 4 w= %s\n" % n_bp)
        for pos in pwm:
            f.write("  %s\t%s\t%s\t%s\n" % (round(float(pos[0]), 5), round(float(pos[1]), 5), 
                                            round(float(pos[2]), 5), round(float(pos[3]), 5)))
        f.write("\n")
f.close()

