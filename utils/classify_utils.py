
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[4]:


def fix_ctrl_id(row):
    if "CONTROL" in row["index"]:
        tile_num = row["index"].split("__")[0].split(":")[1]
        samp_num = row["index"].split("__")[1].split(":")[1]
        new_id = "ctrl.%s__CONTROL__samp.%s__CONTROL" % (tile_num, samp_num)
        return new_id
    else:
        return row["index"]


# In[2]:


def biotype_switch_clean(row):
    if row.cleaner_biotype_hg19 == row.cleaner_biotype_mm9:
        return row.cleaner_biotype_hg19
    elif row.cleaner_biotype_hg19 == "no CAGE activity":
        other = row.cleaner_biotype_mm9
        return "CAGE turnover - %s" % other
    elif row.cleaner_biotype_mm9 == "no CAGE activity":
        other = row.cleaner_biotype_hg19
        return "CAGE turnover - %s" % other
    elif "reclassified" in row.cleaner_biotype_hg19 or "reclassified" in row.cleaner_biotype_mm9:
        return "reclassified"
    else:
        return "biotype switch"


# In[3]:


def biotype_switch_minimal(row):
    if row.minimal_biotype_hg19 == row.minimal_biotype_mm9:
        return row.minimal_biotype_hg19
    elif row.minimal_biotype_hg19 == "no CAGE activity":
        other = row.minimal_biotype_mm9
        return "CAGE turnover - %s" % other
    elif row.minimal_biotype_mm9 == "no CAGE activity":
        other = row.minimal_biotype_hg19
        return "CAGE turnover - %s" % other
    elif "reclassified" in row.minimal_biotype_hg19 or "reclassified" in row.minimal_biotype_mm9:
        return "reclassified"
    else:
        return "biotype switch"


# In[5]:


def is_ctrl(row):
    if "CONTROL" in row["index"]:
        return "control"
    else:
        return "TSS"


# In[6]:


def comp_status(row, fdr_col, thresh, txt):
    if row[fdr_col] < thresh:
        return "significant %s effect" % txt
    else:
        return "no %s effect" % txt


# In[7]:


def comp_status_stringent(row, status_col, l2fc_col, l2fc_thresh, txt):
    if "significant" in row[status_col]:
        if np.abs(row[l2fc_col]) > l2fc_thresh:
            return "significant %s effect" % txt
        else:
            return "no %s effect" % txt
    else:
        return "no %s effect" % txt


# In[8]:


def comp_status_detail(row, status_col, logFC_col, txt):
    if "significant" in row[status_col]:
        if row[logFC_col] < 0:
            return "%s effect\n(higher in human)" % txt
        else:
            return "%s effect\n(higher in mouse)" % txt
    else:
        return "no %s effect" % txt


# In[9]:


def comp_status_one(row, status_col1, status_col2, txt):
    if "significant" in row[status_col1] or "significant" in row[status_col2]:
        return "significant %s effect" % txt
    else:
        return "no %s effect" % txt


# In[10]:


def comp_status_detail_one(row, status_col1, status_col2, logFC_col1, logFC_col2, txt):
    if "significant" in row[status_col1]:
        if "significant" in row[status_col2]:
            # 2 sig effects: check both
            if row[logFC_col1] < 0 and row[logFC_col2] < 0:
                return "%s effect\n(higher in human)" % txt
            elif row[logFC_col1] > 0 and row[logFC_col2] > 0:
                return "%s effect\n(higher in mouse)" % txt
            else:
                return "%s effect\n(direction interaction)" % txt
        else:
            if row[logFC_col1] < 0:
                return "%s effect\n(higher in human)" % txt
            else:
                return "%s effect\n(higher in mouse)" % txt
    else:
        if "significant" in row[status_col2]:
            if row[logFC_col2] < 0:
                return "%s effect\n(higher in human)" % txt
            else:
                return "%s effect\n(higher in mouse)" % txt
        else:
            # no sig effects
            return "no %s effect" % txt


# In[11]:


def signed_max(nums):
    abs_nums = np.abs(nums)
    max_idx = np.argmax(abs_nums)
    return nums[max_idx]


# In[12]:


def comp_logFC_one(row, status_col1, status_col2, logFC_col1, logFC_col2):
    if "significant" in row[status_col1]:
        if "significant" in row[status_col2]:
            # 2 sig trans effects: take max of both
            l2fcs = list(row[[logFC_col1, logFC_col2]])
            return signed_max(l2fcs)
        else:
            return row[logFC_col1]
    else:
        if "significant" in row[status_col2]:
            return row[logFC_col2]
        else:
            # no sig effects: take max of both
            l2fcs = list(row[[logFC_col1, logFC_col2]])
            return signed_max(l2fcs)

