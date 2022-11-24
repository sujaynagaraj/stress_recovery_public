import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

def shift_stress(x):
    if x == 1.0 or x==2.0:
        return 1
    elif x == 3.0 or x==4.0 or x==5.0 or x==6.0:
        return 0
    else:
        return x
    
def control_cat(x):
    if x>= 75.0:
        return 0
    elif 50.0<=x<75.0:
        return 1
    elif 25.0<=x<50.0:
        return 2
    elif x<25.0:
        return 3
    else:
        return 0

def already_computed(df):
    return df


def add_percentile(df, column):
    ##Compute percentile of individual total_stress
    df['hue_percentile'] = df[column].rank(pct = True)
    bins = [0.0, 0.25, 0.75, 1.0]
    labels=[0.25, 0.75, 1.0]
    df[column+"_percentile"] = pd.cut(df['hue_percentile'], bins, labels=labels, include_lowest=True).astype('float')
    
    return df

##Input: dataframe (global or individual) - which contains sam7, daily_control, daily_stressed
##Output: returns dataframe with added columns: shift_stress, control_cat, total_stress
#shift_stress: 1 or 2 on shift SAM stress
#control_cat: see def control_cat(x)
#total_stress: sum of shift_stress + control_cat + daily_stressed (already in dataframe)

def total_stress(df):

    #fill missing survey values with 0 before taking the sum
    df["daily_stressed"] =pd.to_numeric(df["daily_stressed"], errors ='coerce').fillna(0.0).astype('float')
    df["control_cat"] =pd.to_numeric(df["control_cat"], errors ='coerce').fillna(0.0).astype('float')
    df["shift_stress"] =pd.to_numeric(df["shift_stress"], errors ='coerce').fillna(0.0).astype('float')

    df["total_stress"] = df.apply(lambda row: row.daily_stressed/2 + row.shift_stress/2 + row.control_cat/3, axis=1)

    ##Compute percentile of individual total_stress
    df['hue_percentile'] = df["total_stress"].rank(pct = True)
    bins = [0.0, 0.25, 0.75, 1.0]
    labels=[0.25, 0.75, 1.0]
    df['total_stress_percentile'] = pd.cut(df['hue_percentile'], bins, labels=labels, include_lowest=True).astype('float')
    
    return df

def sleep_percentile(df):
    ##Compute percentile of individual total_stress
    df['percentile'] = df["score"].rank(pct = True)
    bins = [0.0, 0.25, 0.75, 1.0]
    labels=[0.25, 0.75, 1.0]
    df['sleep_percentile'] = pd.cut(df['percentile'], bins, labels=labels, include_lowest=True).astype('float')

    return df

def bad_sleep_personal(score):
    
    if score<0.75:
        return 1
    else:
        return 0

def bad_sleep_rolling(df, window_size):
    df = sleep_percentile(df)
    df['bad_sleep'] = df.apply (lambda row: bad_sleep_personal(row.sleep_percentile), axis=1)
    df["bad_sleep_rolling"] = df["bad_sleep"].rolling(window_size*2).sum()
    
    
    df['hue_percentile'] = df["bad_sleep_rolling"].rank(pct = True)
    bins = [0.0, 0.25, 0.75, 1.0]
    labels=[0.25, 0.75, 1.0]
    df['bad_sleep_rolling_percentile'] = pd.cut(df['hue_percentile'], bins, labels=labels, include_lowest=True).astype('float')
    return df

#unsure if this works
def add_column(df, col_name ,row_id, func):
    df[col_name] = df.apply (lambda row: func(row[row_id]), axis=1)
    return df

def total_stress_binarize(df, stress_cutoff, no_stress_cutoff):

    #fill missing survey values with 0 before taking the sum
    df["daily_stressed"] =pd.to_numeric(df["daily_stressed"], errors ='coerce').fillna(0.0).astype('float')
    df["control_cat"] =pd.to_numeric(df["control_cat"], errors ='coerce').fillna(0.0).astype('float')
    df["shift_stress"] =pd.to_numeric(df["shift_stress"], errors ='coerce').fillna(0.0).astype('float')

    df["total_stress"] = df.apply(lambda row: row.daily_stressed/2 + row.shift_stress/2 + row.control_cat/3, axis=1)

    ##Compute percentile of individual total_stress
    df['hue_percentile'] = df["total_stress"].rank(pct = True)
    bins = [0.0, 0.25, 0.75, 1.0]
    labels=[0.25, 0.75, 1.0]
    df['total_stress_percentile'] = df["total_stress"].rank(pct = True)
    
    df["total_stress"] = df.apply (lambda row: binarize(row.total_stress_percentile, stress_cutoff, no_stress_cutoff), axis=1)
    
    return df

def daily_control_binarize(df, stress_cutoff, no_stress_cutoff):
    df['daily_control_percentile'] = df["daily_control"].rank(pct = True)
    df["daily_control"] = df.apply (lambda row: binarize(row.daily_control_percentile, stress_cutoff, no_stress_cutoff), axis=1)
    
    return df

def binarize(x, stress_cutoff, no_stress_cutoff):
    if x>= stress_cutoff:
        return 1
    elif x<no_stress_cutoff:
        return 0
    else:
        return None

def binarize_reverse(x, stress_cutoff, no_stress_cutoff):
    if x>= stress_cutoff:
        return 0
    elif x<no_stress_cutoff:
        return 1
    else:
        return None

def control_cat_binarize(df, stress_cutoff, no_stress_cutoff):
    df["control_cat"] = df.apply (lambda row: binarize(row.daily_control, stress_cutoff, no_stress_cutoff), axis=1)
    
    return df

def ppe_binarize(df, stress_cutoff, no_stress_cutoff):
    df['ppe_percentile'] = df["ppe_6"].rank(pct = True)
    df["ppe_6"] = df.apply (lambda row: binarize(row.ppe_percentile, stress_cutoff, no_stress_cutoff), axis=1)
    
    return df

def pss4_binarize(df, stress_cutoff, no_stress_cutoff):
    df['pss4_percentile'] = df["pss4_score"].rank(pct = True)
    df["pss4_score"] = df.apply (lambda row: binarize(row.pss4_percentile, stress_cutoff, no_stress_cutoff), axis=1)
    
    return df

def hrv_binarize(df, stress_cutoff, no_stress_cutoff):
    df['hrv_percentile'] = df["log_hrv"].rank(pct = True)
    df["hrv_binary"] = df.apply (lambda row: binarize_reverse(row.hrv_percentile, stress_cutoff, no_stress_cutoff), axis=1)
    
    return df

def shift_binarize(df):
    df.loc[df["daily_shifts"] == 2.0, "daily_shifts"] = 1.0
    return df

def any_stress_helper(stress, shifts, shift_stress, hrv_binary):
    if stress == 1.0 or shifts ==1.0 or shift_stress==1.0 or hrv_binary == 1.0:
        return 1.0
    elif stress == 0.0 and shifts ==0.0 and shift_stress==0.0 and hrv_binary == 0.0:
        return 0.0
    else:
        return None
    
def any_stress(df):
    df["any_stress"] = df.apply (lambda row: any_stress_helper(row.daily_stressed, row.daily_shifts, row.shift_stress, row.hrv_binary), axis=1)
        
    return df

