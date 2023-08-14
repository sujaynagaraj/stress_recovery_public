import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import random

from functools import reduce

from src.definitions import *
from src.plotting import *
from pgmpy.base import DAG

#load saved sleep data from ~/SRfingers/data/processed/oura/
def load_sleep():
    sleep_df = pd.read_parquet("~/stress_recovery_public/data/processed/oura/sleep_concat.parquet")
    sleep_df['date']=sleep_df['summary_date'].astype(str) 

    return sleep_df

def load_once():
    once_merged = pd.read_parquet("~/stress_recovery_public/data/processed/survey/once_merged.parquet")
    return once_merged

#Load Daily
def load_daily():
    daily_merged = pd.read_parquet("~/stress_recovery_public/data/processed/survey/daily_merged.parquet")
    daily_merged['date']=daily_merged['date'].astype(str) 

    daily_merged["covid_shift_any"] = daily_merged.apply (lambda row: (row.daily_covid_shifts___1 or row.daily_covid_shifts___2 or row.daily_covid_shifts___3 ), axis=1)
    daily_merged["shift_stress"] = daily_merged.apply (lambda row: shift_stress(row.sam7), axis=1)
    daily_merged["control_cat"] = daily_merged.apply (lambda row: control_cat(row.daily_control), axis=1)
    
    return daily_merged

#Load Weekly
def load_weekly():
    weekly_merged = pd.read_parquet("~/stress_recovery_public/data/processed/survey/weekly_merged.parquet")
    weekly_merged['date']=weekly_merged['date'].astype(str)
    return weekly_merged 

#Load BiWeekly
def load_biweekly():
    biweekly_merged = pd.read_parquet("~/stress_recovery_public/data/processed/survey/biweekly_merged.parquet")
    biweekly_merged['date']=biweekly_merged['date'].astype(str) 
    return biweekly_merged

#Load Monthly
def load_monthly():
    monthly_merged = pd.read_parquet("~/stress_recovery_public/data/processed/survey/monthly_merged.parquet")
    monthly_merged['date']=monthly_merged['date'].astype(str) 
    return monthly_merged

#Load EBT
def load_ebt():
    ebt = pd.read_parquet("~/stress_recovery_public/data/processed/camcog/ebt.parquet")
    ebt['date']=ebt['date'].astype(str)
    return ebt

def load_garmin_daily():
    garmin_dailies = pd.read_parquet("~/stress_recovery_public/data/processed/garmin/garmin_dailies.parquet")
    garmin_dailies['date']=garmin_dailies['summary_date'].astype(str)   
    return garmin_dailies

def load_garmin_subset():
    sleep_df = load_sleep()
    daily_merged = load_daily()
    weekly_merged = load_weekly()
    biweekly_merged = load_biweekly()
    monthly_merged = load_monthly()
    ebt = load_ebt()
    garmin_dailies = load_garmin_daily()
    once_merged = load_once()

    #COMBINE ONCE SURVEYS WITH OURA SLEEP FIRST
    dfs = [garmin_dailies, once_merged]

    garmin_once = reduce(lambda left,right: pd.merge(left,right,how='left',on=["participant_id"]), dfs)

    #COMBINE ALL THE DFs
    dfs = [garmin_once, sleep_df, daily_merged, weekly_merged, biweekly_merged, monthly_merged, ebt]

    df_final = reduce(lambda left,right: pd.merge(left,right,how='left',on=["participant_id","date"]), dfs)

    df_final['date'] = df_final["date"].astype(str)
    df_final['date'] = pd.to_datetime(df_final["date"])

    #add log hrv
    df_final["log_hrv"] = df_final.apply (lambda row: np.log(row.rmssd), axis=1)

    df_final["zeroes"] = df_final.apply (lambda row: row.rmssd-row.rmssd, axis=1)

    #adding a few calculated rows
    #MOVE THESE TO PROCESSING NOTEBOOK
    df_final["phq9_score"] = df_final.apply (lambda row: row.phq9_1 + row.phq9_2 + row.phq9_3 + row.phq9_4 + row.phq9_5 + row.phq9_6 + row.phq9_7 +row.phq9_8 + row.phq9_9, axis=1)
    df_final["gad7_score"] = df_final.apply (lambda row: row.gad_1 + row.gad_2 + row.gad_3 + row.gad_4 + row.gad_5 + row.gad_6 + row.gad_7, axis=1)
    df_final["promis_sri_score"] = df_final.apply (lambda row: row.promis_sri1 + row.promis_sri2 + row.promis_sri3 + row.promis_sri4 + row.promis_sri5 + row.promis_sri6 + row.promis_sri7+row.promis_sri8, axis=1)
    df_final["pss4_score"] = df_final.apply (lambda row: row.pss4_1 + row.pss4_2 + row.pss4_3 + row.pss4_4, axis=1)
    df_final["promis_es_score"] = df_final.apply (lambda row: row.promis_es1 + row.promis_es2 + row.promis_es3 + row.promis_es4, axis=1)
    df_final["fss_score"] = df_final.apply (lambda row: row.fss_1 + row.fss_2 + row.fss_3 + row.fss_4 + row.fss_5 + row.fss_6 + row.fss_7 + row.fss_8 + row.fss_9, axis=1)
    df_final["ace_score"] = df_final.apply (lambda row: row.ace_1 + row.ace_2 + row.ace_3 + row.ace_4 + row.ace_5 + row.ace_6 + row.ace_7 + row.ace_8 + row.ace_9 + row.ace_10, axis=1)
    df_final["ptsd_score"] = df_final.apply (lambda row: row.pcl_1 + row.pcl_2 + row.pcl_3 + row.pcl_4 + row.pcl_5 + row.pcl_6 + row.pcl_7 + row.pcl_8 + row.pcl_9 + row.pcl_10 + row.pcl_11 + row.pcl_12 + row.pcl_13 + row.pcl_14 + row.pcl_15 + row.pcl_16 + row.pcl_17, axis=1)
    df_final["life_events_score"] = df_final.apply (lambda row: row.lec_01 + row.lec_02 + row.lec_03 + row.lec_04 + row.lec_05 + row.lec_06 + row.lec_07 + row.lec_08 + row.lec_09 + row.lec_10, axis=1)
    
    #add subject_id as an integer
    df_final['p_id'] = pd.factorize(df_final['participant_id'])[0]

    return df_final

def load_merge_all():
    sleep_df = load_sleep()
    daily_merged = load_daily()
    weekly_merged = load_weekly()
    biweekly_merged = load_biweekly()
    monthly_merged = load_monthly()
    ebt = load_ebt()
    garmin_dailies = load_garmin_daily()
    once_merged = load_once()

    #COMBINE ONCE SURVEYS WITH OURA SLEEP FIRST
    dfs = [sleep_df, once_merged]

    sleep_once = reduce(lambda left,right: pd.merge(left,right,how='outer',on=["participant_id"]), dfs)

    #COMBINE ALL THE DFs
    dfs = [sleep_once, daily_merged, weekly_merged, biweekly_merged, monthly_merged, ebt, garmin_dailies]

    df_final = reduce(lambda left,right: pd.merge(left,right,how='outer',on=["participant_id","date"]), dfs)

    df_final['date'] = df_final["date"].astype(str)
    df_final['date'] = pd.to_datetime(df_final["date"])

    #add log hrv
    df_final["log_hrv"] = df_final.apply (lambda row: np.log(row.rmssd), axis=1)

    df_final["zeroes"] = df_final.apply (lambda row: row.rmssd-row.rmssd, axis=1)

    #adding a few calculated rows
    #MOVE THESE TO PROCESSING NOTEBOOK
    df_final["phq9_score"] = df_final.apply (lambda row: row.phq9_1 + row.phq9_2 + row.phq9_3 + row.phq9_4 + row.phq9_5 + row.phq9_6 + row.phq9_7 +row.phq9_8 + row.phq9_9, axis=1)
    df_final["gad7_score"] = df_final.apply (lambda row: row.gad_1 + row.gad_2 + row.gad_3 + row.gad_4 + row.gad_5 + row.gad_6 + row.gad_7, axis=1)
    df_final["promis_sri_score"] = df_final.apply (lambda row: row.promis_sri1 + row.promis_sri2 + row.promis_sri3 + row.promis_sri4 + row.promis_sri5 + row.promis_sri6 + row.promis_sri7+row.promis_sri8, axis=1)
    df_final["pss4_score"] = df_final.apply (lambda row: row.pss4_1 + row.pss4_2 + row.pss4_3 + row.pss4_4, axis=1)
    df_final["promis_es_score"] = df_final.apply (lambda row: row.promis_es1 + row.promis_es2 + row.promis_es3 + row.promis_es4, axis=1)
    df_final["fss_score"] = df_final.apply (lambda row: row.fss_1 + row.fss_2 + row.fss_3 + row.fss_4 + row.fss_5 + row.fss_6 + row.fss_7 + row.fss_8 + row.fss_9, axis=1)
    df_final["ace_score"] = df_final.apply (lambda row: row.ace_1 + row.ace_2 + row.ace_3 + row.ace_4 + row.ace_5 + row.ace_6 + row.ace_7 + row.ace_8 + row.ace_9 + row.ace_10, axis=1)
    df_final["ptsd_score"] = df_final.apply (lambda row: row.pcl_1 + row.pcl_2 + row.pcl_3 + row.pcl_4 + row.pcl_5 + row.pcl_6 + row.pcl_7 + row.pcl_8 + row.pcl_9 + row.pcl_10 + row.pcl_11 + row.pcl_12 + row.pcl_13 + row.pcl_14 + row.pcl_15 + row.pcl_16 + row.pcl_17, axis=1)
    df_final["life_events_score"] = df_final.apply (lambda row: row.lec_01 + row.lec_02 + row.lec_03 + row.lec_04 + row.lec_05 + row.lec_06 + row.lec_07 + row.lec_08 + row.lec_09 + row.lec_10, axis=1)
    
    #add subject_id as an integer
    df_final['p_id'] = pd.factorize(df_final['participant_id'])[0]

    return df_final

def load_merge_all_nogarmin():
    sleep_df = load_sleep()
    daily_merged = load_daily()
    weekly_merged = load_weekly()
    biweekly_merged = load_biweekly()
    monthly_merged = load_monthly()
    ebt = load_ebt()
    #garmin_dailies = load_garmin_daily()
    once_merged = load_once()

    #COMBINE ONCE SURVEYS WITH OURA SLEEP FIRST
    dfs = [sleep_df, once_merged]

    sleep_once = reduce(lambda left,right: pd.merge(left,right,how='outer',on=["participant_id"]), dfs)

    #COMBINE ALL THE DFs
    dfs = [sleep_once, daily_merged, weekly_merged, biweekly_merged, monthly_merged, ebt]

    df_final = reduce(lambda left,right: pd.merge(left,right,how='outer',on=["participant_id","date"]), dfs)

    df_final['date'] = df_final["date"].astype(str)
    df_final['date'] = pd.to_datetime(df_final["date"])

    #add log hrv
    df_final["log_hrv"] = df_final.apply (lambda row: np.log(row.rmssd), axis=1)

    df_final["zeroes"] = df_final.apply (lambda row: row.rmssd-row.rmssd, axis=1)

    #adding a few calculated rows
    #MOVE THESE TO PROCESSING NOTEBOOK
    df_final["phq9_score"] = df_final.apply (lambda row: row.phq9_1 + row.phq9_2 + row.phq9_3 + row.phq9_4 + row.phq9_5 + row.phq9_6 + row.phq9_7 +row.phq9_8 + row.phq9_9, axis=1)
    df_final["gad7_score"] = df_final.apply (lambda row: row.gad_1 + row.gad_2 + row.gad_3 + row.gad_4 + row.gad_5 + row.gad_6 + row.gad_7, axis=1)
    df_final["promis_sri_score"] = df_final.apply (lambda row: row.promis_sri1 + row.promis_sri2 + row.promis_sri3 + row.promis_sri4 + row.promis_sri5 + row.promis_sri6 + row.promis_sri7+row.promis_sri8, axis=1)
    df_final["pss4_score"] = df_final.apply (lambda row: row.pss4_1 + row.pss4_2 + row.pss4_3 + row.pss4_4, axis=1)
    df_final["promis_es_score"] = df_final.apply (lambda row: row.promis_es1 + row.promis_es2 + row.promis_es3 + row.promis_es4, axis=1)
    df_final["fss_score"] = df_final.apply (lambda row: row.fss_1 + row.fss_2 + row.fss_3 + row.fss_4 + row.fss_5 + row.fss_6 + row.fss_7 + row.fss_8 + row.fss_9, axis=1)
    df_final["ace_score"] = df_final.apply (lambda row: row.ace_1 + row.ace_2 + row.ace_3 + row.ace_4 + row.ace_5 + row.ace_6 + row.ace_7 + row.ace_8 + row.ace_9 + row.ace_10, axis=1)
    df_final["ptsd_score"] = df_final.apply (lambda row: row.pcl_1 + row.pcl_2 + row.pcl_3 + row.pcl_4 + row.pcl_5 + row.pcl_6 + row.pcl_7 + row.pcl_8 + row.pcl_9 + row.pcl_10 + row.pcl_11 + row.pcl_12 + row.pcl_13 + row.pcl_14 + row.pcl_15 + row.pcl_16 + row.pcl_17, axis=1)
    df_final["life_events_score"] = df_final.apply (lambda row: row.lec_01 + row.lec_02 + row.lec_03 + row.lec_04 + row.lec_05 + row.lec_06 + row.lec_07 + row.lec_08 + row.lec_09 + row.lec_10, axis=1)
    
    #add subject_id as an integer
    df_final['p_id'] = pd.factorize(df_final['participant_id'])[0]

    return df_final

def load_sleep_daily():
    sleep_df = load_sleep()
    daily_merged = load_daily()

    dfs = [sleep_df, daily_merged]

    df_final = reduce(lambda left,right: pd.merge(left,right,how='left',on=["participant_id","date"]), dfs)

    df_final['date'] = df_final["date"].astype(str)
    df_final['date'] = pd.to_datetime(df_final["date"])

    #add log hrv
    df_final["log_hrv"] = df_final.apply (lambda row: np.log(row.rmssd), axis=1)

    df_final["zeroes"] = df_final.apply (lambda row: row.rmssd-row.rmssd, axis=1)

    return df_final

def load_once_complete():
    df_final = pd.read_parquet("~/stress_recovery_public/data/processed/survey/once_merged.parquet")

    df_final["ace_score"] = df_final.apply (lambda row: row.ace_1 + row.ace_2 + row.ace_3 + row.ace_4 + row.ace_5 + row.ace_6 + row.ace_7 + row.ace_8 + row.ace_9 + row.ace_10, axis=1)
    df_final["ptsd_score"] = df_final.apply (lambda row: row.pcl_1 + row.pcl_2 + row.pcl_3 + row.pcl_4 + row.pcl_5 + row.pcl_6 + row.pcl_7 + row.pcl_8 + row.pcl_9 + row.pcl_10 + row.pcl_11 + row.pcl_12 + row.pcl_13 + row.pcl_14 + row.pcl_15 + row.pcl_16 + row.pcl_17, axis=1)
    df_final["life_events_score"] = df_final.apply (lambda row: row.lec_01 + row.lec_02 + row.lec_03 + row.lec_04 + row.lec_05 + row.lec_06 + row.lec_07 + row.lec_08 + row.lec_09 + row.lec_10, axis=1)
    
    return df_final

#Input: individual participant dataframe
def missing_rows(df):
     #Create Empty NaN rows for missing dates
    df_copy = df.copy()

    df_copy = df_copy.set_index('date')
    df_copy.index = pd.to_datetime(df_copy.index)
    df_copy = df_copy.loc[~df_copy.index.duplicated(), :]

    df_copy = df_copy.resample('D').asfreq()
    return df_copy

#Input: individual participant dataframe, accepts frequency as input
def missing_rows_freq(df, freq='D'):
     #Create Empty NaN rows for missing dates
    df_copy = df.copy()

    df_copy = df_copy.set_index('date')
    df_copy.index = pd.to_datetime(df_copy.index)
    df_copy = df_copy.loc[~df_copy.index.duplicated(), :]

    df_copy = df_copy.resample(freq).asfreq()
    return df_copy

# apply the z-score method in Pandas using the .mean() and .std() methods
def z_score(df, columns):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in columns:
        if pd.isna(df_std[column].std()) or (df_std[column].std() == 0.0):
            df_std[column] = 0.0
        else:
            df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        
    return df_std

def downsample_rows(df, length):
    df = df.loc[:, ~df.columns.duplicated()]
    return df.rolling(length).mean().iloc[::length, :][1:]

#Forward Fill but maximum of N days ahead, leave rest blank
def ffill(df, days):
    return df.fillna(method='ffill', limit=days)

def lin_interp(df, feature):
    df[feature] = df[feature].interpolate(method="linear")
    return df

def lin_interp_multiple(df, features):
    for feature in features:
        df = lin_interp(df, feature)
    return df

def add_percentile(df, feature):
    ##Compute percentile of individual column
    df[feature+'_percentile'] = df[feature].rank(pct = True)
    
    return df

def add_percentile_multiple(df, features):
    for feature in features:
        df = add_percentile(df, feature)
    return df

#get random participant_id from dataframe
def rand_participant(df, n):
    participant_list = df.participant_id.unique().tolist()
    return random.sample(participant_list, n)[0]

def fetch_participant_df(df, p):
    return df.loc[df["participant_id"]==p]

def fetch_rand_participant(df):
    p = rand_participant(df,1)
    return fetch_participant_df(df, p)

def pad_stress(df, buffer, stress_label):
    df2 = df.copy().reset_index()

    i = buffer
    while i < len(df)-buffer:

        if df.iloc[i][stress_label] == 1.0:
            for j in range(1,buffer+1):
                df2.at[i-j, stress_label] = 1.0
                df2.at[i+j, stress_label] = 1.0
            i = i+buffer+1
        else:
            i = i + 1

    df2.set_index("date")
    return df2

def iid_windows(df, window, p_features, stress_label):
    i = 0
    df_empty_nostress = pd.DataFrame()
    df_empty_stress = pd.DataFrame()
    
    while i <= len(df)-1:
        growing_window = 1
        if i == len(df)-1:
            if df.iloc[i][stress_label] == 0.0:
                df_empty_nostress= df_empty_nostress.append(df.iloc[i][p_features], ignore_index=True)
            elif df.iloc[i][stress_label] == 1.0 or df.iloc[i][stress_label] == 2.0: #FOR DAILY SHIFT LABEL AS IT CAN be 2

                df_empty_stress= df_empty_stress.append(df.iloc[i][p_features], ignore_index=True)
            i=i+1
        else:
        
            while df.iloc[i][stress_label] == df.iloc[i+growing_window][stress_label] and df.iloc[i][stress_label] in [0.0,1.0, 2.0] and i+growing_window<len(df)-1:
                growing_window +=1
            if growing_window ==1:
                if df.iloc[i][stress_label] == 0.0:
                    df_empty_nostress= df_empty_nostress.append(df[i:i+growing_window][p_features], ignore_index=True)
                elif df.iloc[i][stress_label] == 1.0 or df.iloc[i][stress_label] == 2.0: #FOR DAILY SHIFT LABEL AS IT CAN be 2

                    df_empty_stress= df_empty_stress.append(df[i:i+growing_window][p_features], ignore_index=True)
                i=i+growing_window
            elif 2<=growing_window<=window:
                if df.iloc[i][stress_label] == 0.0:
                    df_empty_nostress= df_empty_nostress.append(df[i:i+growing_window][p_features].mean(axis=0).to_frame().T, ignore_index=True)
                elif df.iloc[i][stress_label] == 1.0 or df.iloc[i][stress_label] == 2.0:

                    df_empty_stress= df_empty_stress.append(df[i:i+growing_window][p_features].mean(axis=0).to_frame().T, ignore_index=True)
                i=i+growing_window
            elif growing_window>window:

                #OLD 
                #df_sample = df[i:i+growing_window].sample(n=window)
                #print(growing_window//2)
                df_sample = df[i:i+growing_window].sample(n=growing_window//2)

                if df.iloc[i][stress_label] == 0.0:
                    df_empty_nostress=df_empty_nostress.append(df_sample[p_features], ignore_index=True)
                elif df.iloc[i][stress_label] == 1.0 or df.iloc[i][stress_label] == 2.0:

                    df_empty_stress=df_empty_stress.append(df_sample[p_features], ignore_index=True)
                i=i+growing_window
        #print(growing_window)
    return df_empty_nostress, df_empty_stress


def add_weights(model,data):
    edges = list(model.edges())
    weighted_graph = DAG()
    for edge in edges:
        A,B = edge
        weight = data[A].corr(data[B])
        weighted_graph.add_edge(A,B, weight = weight)
    return weighted_graph