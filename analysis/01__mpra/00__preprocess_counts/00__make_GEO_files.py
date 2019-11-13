
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


HUES64_rep1_tfxn1_fs = ["../../../data/02__mpra/01__counts/07__HUES64_rep6_lib1_BARCODES.txt",
                        "../../../data/02__mpra/01__counts/07__HUES64_rep6_lib2_BARCODES.txt"]
HUES64_rep1_tfxn2_fs = ["../../../data/02__mpra/01__counts/08__HUES64_rep7_lib1_BARCODES.txt",
                        "../../../data/02__mpra/01__counts/08__HUES64_rep7_lib2_BARCODES.txt"]
HUES64_rep1_tfxn3_fs = ["../../../data/02__mpra/01__counts/09__HUES64_rep8_lib1_BARCODES.txt",
                        "../../../data/02__mpra/01__counts/09__HUES64_rep8_lib2_BARCODES.txt"]


# In[3]:


HUES64_rep2_tfxn1_fs = ["../../../data/02__mpra/01__counts/10__HUES64_rep9_lib1_BARCODES.txt",
                        "../../../data/02__mpra/01__counts/10__HUES64_rep9_lib2_BARCODES.txt"]
HUES64_rep2_tfxn2_fs = ["../../../data/02__mpra/01__counts/11__HUES64_rep10_lib1_BARCODES.txt",
                        "../../../data/02__mpra/01__counts/11__HUES64_rep10_lib2_BARCODES.txt"]
HUES64_rep2_tfxn3_fs = ["../../../data/02__mpra/01__counts/12__HUES64_rep11_lib1_BARCODES.txt",
                        "../../../data/02__mpra/01__counts/12__HUES64_rep11_lib2_BARCODES.txt"]


# In[4]:


HUES64_rep3_tfxn1_fs = ["../../../data/02__mpra/01__counts/16__HUES64_rep12_lib1_BARCODES.txt",
                        "../../../data/02__mpra/01__counts/16__HUES64_rep12_lib2_BARCODES.txt"]
HUES64_rep3_tfxn2_fs = ["../../../data/02__mpra/01__counts/17__HUES64_rep13_lib1_BARCODES.txt",
                        "../../../data/02__mpra/01__counts/17__HUES64_rep13_lib2_BARCODES.txt"]
HUES64_rep3_tfxn3_fs = ["../../../data/02__mpra/01__counts/18__HUES64_rep14_lib1_BARCODES.txt",
                        "../../../data/02__mpra/01__counts/18__HUES64_rep14_lib2_BARCODES.txt"]


# In[5]:


mESC_rep1_tfxn1_fs = ["../../../data/02__mpra/01__counts/15__mESC_rep3_lib1_BARCODES.txt",
                      "../../../data/02__mpra/01__counts/15__mESC_rep3_lib2_BARCODES.txt"]


# In[6]:


mESC_rep2_tfxn1_fs = ["../../../data/02__mpra/01__counts/19__mESC_rep4_lib1_BARCODES.txt",
                      "../../../data/02__mpra/01__counts/19__mESC_rep4_lib2_BARCODES.txt",
                      "../../../data/02__mpra/01__counts/19__mESC_rep4_lib3_BARCODES.txt"]


# In[7]:


mESC_rep3_tfxn1_fs = ["../../../data/02__mpra/01__counts/20__mESC_rep5_lib1_BARCODES.txt",
                      "../../../data/02__mpra/01__counts/20__mESC_rep5_lib2_BARCODES.txt",
                      "../../../data/02__mpra/01__counts/20__mESC_rep5_lib3_BARCODES.txt"]


# ## 1. import, merge, sum

# ### HUES64 rep 1

# In[8]:


for i, f in enumerate(HUES64_rep1_tfxn1_fs):
    if i == 0:
        HUES64_rep1_tfxn1 = pd.read_table(f, sep="\t")
        print(len(HUES64_rep1_tfxn1))
    else:
        tmp = pd.read_table(f, sep="\t")
        print(len(tmp))
        HUES64_rep1_tfxn1 = HUES64_rep1_tfxn1.merge(tmp, on="barcode")
HUES64_rep1_tfxn1["count"] = HUES64_rep1_tfxn1[["count_x", "count_y"]].sum(axis=1)
HUES64_rep1_tfxn1.drop(["count_x", "count_y"], axis=1, inplace=True)
HUES64_rep1_tfxn1.head()


# In[9]:


for i, f in enumerate(HUES64_rep1_tfxn2_fs):
    if i == 0:
        HUES64_rep1_tfxn2 = pd.read_table(f, sep="\t")
        print(len(HUES64_rep1_tfxn2))
    else:
        tmp = pd.read_table(f, sep="\t")
        print(len(tmp))
        HUES64_rep1_tfxn2 = HUES64_rep1_tfxn2.merge(tmp, on="barcode")
HUES64_rep1_tfxn2["count"] = HUES64_rep1_tfxn2[["count_x", "count_y"]].sum(axis=1)
HUES64_rep1_tfxn2.drop(["count_x", "count_y"], axis=1, inplace=True)
HUES64_rep1_tfxn2.head()


# In[10]:


for i, f in enumerate(HUES64_rep1_tfxn3_fs):
    if i == 0:
        HUES64_rep1_tfxn3 = pd.read_table(f, sep="\t")
        print(len(HUES64_rep1_tfxn3))
    else:
        tmp = pd.read_table(f, sep="\t")
        print(len(tmp))
        HUES64_rep1_tfxn3 = HUES64_rep1_tfxn3.merge(tmp, on="barcode")
HUES64_rep1_tfxn3["count"] = HUES64_rep1_tfxn3[["count_x", "count_y"]].sum(axis=1)
HUES64_rep1_tfxn3.drop(["count_x", "count_y"], axis=1, inplace=True)
HUES64_rep1_tfxn3.head()


# ### HUES64 rep 2

# In[11]:


for i, f in enumerate(HUES64_rep2_tfxn1_fs):
    if i == 0:
        HUES64_rep2_tfxn1 = pd.read_table(f, sep="\t")
        print(len(HUES64_rep2_tfxn1))
    else:
        tmp = pd.read_table(f, sep="\t")
        print(len(tmp))
        HUES64_rep2_tfxn1 = HUES64_rep2_tfxn1.merge(tmp, on="barcode")
HUES64_rep2_tfxn1["count"] = HUES64_rep2_tfxn1[["count_x", "count_y"]].sum(axis=1)
HUES64_rep2_tfxn1.drop(["count_x", "count_y"], axis=1, inplace=True)
HUES64_rep2_tfxn1.head()


# In[12]:


for i, f in enumerate(HUES64_rep2_tfxn2_fs):
    if i == 0:
        HUES64_rep2_tfxn2 = pd.read_table(f, sep="\t")
        print(len(HUES64_rep2_tfxn2))
    else:
        tmp = pd.read_table(f, sep="\t")
        print(len(tmp))
        HUES64_rep2_tfxn2 = HUES64_rep2_tfxn2.merge(tmp, on="barcode")
HUES64_rep2_tfxn2["count"] = HUES64_rep2_tfxn2[["count_x", "count_y"]].sum(axis=1)
HUES64_rep2_tfxn2.drop(["count_x", "count_y"], axis=1, inplace=True)
HUES64_rep2_tfxn2.head()


# In[13]:


for i, f in enumerate(HUES64_rep2_tfxn3_fs):
    if i == 0:
        HUES64_rep2_tfxn3 = pd.read_table(f, sep="\t")
        print(len(HUES64_rep2_tfxn3))
    else:
        tmp = pd.read_table(f, sep="\t")
        print(len(tmp))
        HUES64_rep2_tfxn3 = HUES64_rep2_tfxn3.merge(tmp, on="barcode")
HUES64_rep2_tfxn3["count"] = HUES64_rep2_tfxn3[["count_x", "count_y"]].sum(axis=1)
HUES64_rep2_tfxn3.drop(["count_x", "count_y"], axis=1, inplace=True)
HUES64_rep2_tfxn3.head()


# ### HUES64 rep 3

# In[14]:


for i, f in enumerate(HUES64_rep3_tfxn1_fs):
    if i == 0:
        HUES64_rep3_tfxn1 = pd.read_table(f, sep="\t")
        print(len(HUES64_rep3_tfxn1))
    else:
        tmp = pd.read_table(f, sep="\t")
        print(len(tmp))
        HUES64_rep3_tfxn1 = HUES64_rep3_tfxn1.merge(tmp, on="barcode")
HUES64_rep3_tfxn1["count"] = HUES64_rep3_tfxn1[["count_x", "count_y"]].sum(axis=1)
HUES64_rep3_tfxn1.drop(["count_x", "count_y"], axis=1, inplace=True)
HUES64_rep3_tfxn1.head()


# In[15]:


for i, f in enumerate(HUES64_rep3_tfxn2_fs):
    if i == 0:
        HUES64_rep3_tfxn2 = pd.read_table(f, sep="\t")
        print(len(HUES64_rep3_tfxn2))
    else:
        tmp = pd.read_table(f, sep="\t")
        print(len(tmp))
        HUES64_rep3_tfxn2 = HUES64_rep3_tfxn2.merge(tmp, on="barcode")
HUES64_rep3_tfxn2["count"] = HUES64_rep3_tfxn2[["count_x", "count_y"]].sum(axis=1)
HUES64_rep3_tfxn2.drop(["count_x", "count_y"], axis=1, inplace=True)
HUES64_rep3_tfxn2.head()


# In[16]:


for i, f in enumerate(HUES64_rep3_tfxn3_fs):
    if i == 0:
        HUES64_rep3_tfxn3 = pd.read_table(f, sep="\t")
        print(len(HUES64_rep3_tfxn3))
    else:
        tmp = pd.read_table(f, sep="\t")
        print(len(tmp))
        HUES64_rep3_tfxn3 = HUES64_rep3_tfxn3.merge(tmp, on="barcode")
HUES64_rep3_tfxn3["count"] = HUES64_rep3_tfxn3[["count_x", "count_y"]].sum(axis=1)
HUES64_rep3_tfxn3.drop(["count_x", "count_y"], axis=1, inplace=True)
HUES64_rep3_tfxn3.head()


# ## mESC rep 1

# In[17]:


for i, f in enumerate(mESC_rep1_tfxn1_fs):
    if i == 0:
        mESC_rep1_tfxn1 = pd.read_table(f, sep="\t")
        print(len(mESC_rep1_tfxn1))
    else:
        tmp = pd.read_table(f, sep="\t")
        print(len(tmp))
        mESC_rep1_tfxn1 = mESC_rep1_tfxn1.merge(tmp, on="barcode")
mESC_rep1_tfxn1["count"] = mESC_rep1_tfxn1[["count_x", "count_y"]].sum(axis=1)
mESC_rep1_tfxn1.drop(["count_x", "count_y"], axis=1, inplace=True)
mESC_rep1_tfxn1.head()


# ## mESC rep 2

# In[18]:


for i, f in enumerate(mESC_rep2_tfxn1_fs):
    if i == 0:
        mESC_rep2_tfxn1 = pd.read_table(f, sep="\t")
        print(len(mESC_rep2_tfxn1))
    else:
        tmp = pd.read_table(f, sep="\t")
        print(len(tmp))
        mESC_rep2_tfxn1 = mESC_rep2_tfxn1.merge(tmp, on="barcode")
mESC_rep2_tfxn1["count"] = mESC_rep2_tfxn1[["count_x", "count_y", "count"]].sum(axis=1)
mESC_rep2_tfxn1.drop(["count_x", "count_y"], axis=1, inplace=True)
mESC_rep2_tfxn1.head()


# ## mESC rep 3

# In[19]:


for i, f in enumerate(mESC_rep3_tfxn1_fs):
    if i == 0:
        mESC_rep3_tfxn1 = pd.read_table(f, sep="\t")
        print(len(mESC_rep3_tfxn1))
    else:
        tmp = pd.read_table(f, sep="\t")
        print(len(tmp))
        mESC_rep3_tfxn1 = mESC_rep3_tfxn1.merge(tmp, on="barcode")
mESC_rep3_tfxn1["count"] = mESC_rep3_tfxn1[["count_x", "count_y", "count"]].sum(axis=1)
mESC_rep3_tfxn1.drop(["count_x", "count_y"], axis=1, inplace=True)
mESC_rep3_tfxn1.head()


# ## 2. write files

# In[20]:


HUES64_rep1_tfxn1.to_csv("../../../GEO_submission/MPRA__HUES64__rep1__tfxn1.BARCODES.txt", sep="\t", index=False)
HUES64_rep1_tfxn2.to_csv("../../../GEO_submission/MPRA__HUES64__rep1__tfxn2.BARCODES.txt", sep="\t", index=False)
HUES64_rep1_tfxn3.to_csv("../../../GEO_submission/MPRA__HUES64__rep1__tfxn3.BARCODES.txt", sep="\t", index=False)


# In[21]:


HUES64_rep2_tfxn1.to_csv("../../../GEO_submission/MPRA__HUES64__rep2__tfxn1.BARCODES.txt", sep="\t", index=False)
HUES64_rep2_tfxn2.to_csv("../../../GEO_submission/MPRA__HUES64__rep2__tfxn2.BARCODES.txt", sep="\t", index=False)
HUES64_rep2_tfxn3.to_csv("../../../GEO_submission/MPRA__HUES64__rep2__tfxn3.BARCODES.txt", sep="\t", index=False)


# In[22]:


HUES64_rep3_tfxn1.to_csv("../../../GEO_submission/MPRA__HUES64__rep3__tfxn1.BARCODES.txt", sep="\t", index=False)
HUES64_rep3_tfxn2.to_csv("../../../GEO_submission/MPRA__HUES64__rep3__tfxn2.BARCODES.txt", sep="\t", index=False)
HUES64_rep3_tfxn3.to_csv("../../../GEO_submission/MPRA__HUES64__rep3__tfxn3.BARCODES.txt", sep="\t", index=False)


# In[23]:


mESC_rep1_tfxn1.to_csv("../../../GEO_submission/MPRA__mESC__rep1__tfxn1.BARCODES.txt", sep="\t", index=False)


# In[24]:


mESC_rep2_tfxn1.to_csv("../../../GEO_submission/MPRA__mESC__rep2__tfxn1.BARCODES.txt", sep="\t", index=False)


# In[25]:


mESC_rep3_tfxn1.to_csv("../../../GEO_submission/MPRA__mESC__rep3__tfxn1.BARCODES.txt", sep="\t", index=False)

