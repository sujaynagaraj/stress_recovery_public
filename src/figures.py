import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import random
from scipy.stats import ttest_ind
from statannot import add_stat_annotation
from sklearn import metrics
import sys

sys.path.insert(0,'..')
from src.definitions import *
from src.processing import *
from src.stats import *
from src.plotting import *

import networkx as nx
import missingno as msno

import pickle
import random
from pyvis import network as net
from pyvis.network import Network

from collections import Counter
import operator
import statistics

from tqdm import tqdm

from mpl_chord_diagram import chord_diagram
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA, SparsePCA
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE

evi_blue = '#34599E'
evi_green = '#4EB792'
evi_yellow = '#F4C15D'
evi_orange = '#EC8C46'
evi_purple = '#823266'
evi_gray = '#C5C5C5'



def rename_nodes(nodes):
    dict_rename = {"breath_average": "Breath", 
               "efficiency": "Efficiency", 
               "hr_average": "HR Average",
               "hr_lowest": "HR Lowest",
               "onset_latency": "Sleep Onset",
               "score": "Sleep Score",
               "temperature_delta": "Temp Change",
               'temperature_trend_deviation': "Temp Trend",
               "total": "Total Sleep",
               "log_hrv": "HRV",
               'daily_stressed': "Stress",
               'daily_control':"Control",
               'daily_shifts': "Shifts",
                'sam8': "SAM8",
                'shift_stress': "Shift Stress",
                'promis_sd5' :"Sleep Quality",
                'EBTBP': "EBT",
                'phq9_score': "PHQ9",
                'gad7_score': "GAD7",
                'promis_sri_score': "Impaired Sleep",
                'pss4_score' : "PSS4",
                'promis_es_score': "Emotional Support",
                'fss_score': "FSS",
                "phys":"Physiological",
                "sleep":"Sleep",
                "mental":"Mental Health"
              }
    return [dict_rename[node] for node in nodes]

def rename_nodes2(nodes):
    dict_rename = {"breath_average": "B", 
               "efficiency": "E", 
               "hr_average": "HR",
               "hr_lowest": "HRL",
               "onset_latency": "O",
               "score": "S",
               "temperature_delta": "TD",
               'temperature_trend_deviation': "TT",
               "total": "T",
               "log_hrv": "HRV",
               'daily_stressed': "DS",
               'daily_control':"DC",
               'daily_shifts': "SH",
                'sam8': "S8",
                'shift_stress': "SS",
                'promis_sd5' :"SQ",
                'EBTBP': "EBT",
                'phq9_score': "PHQ9",
                'gad7_score': "GAD7",
                'promis_sri_score': "SRI",
                'pss4_score' : "PSS4",
                'promis_es_score': "ES",
                'fss_score': "FSS",
                "phys":"Physiological",
                "sleep":"Sleep",
                "mental":"Mental Health"
              }
    return [dict_rename[node] for node in nodes]
    
def get_df(stress_definition):
    with open("/h/snagaraj/SRfingers/results/PC/reference/"+stress_definition+"/results_df.pkl", 'rb') as f:
        df  = pickle.load(f)
    return df

def get_participants(stress_definition):
    df =get_df(stress_definition)
    
    participants = []
    for p in df["participant_id"].unique():
        subset = df[df["participant_id"]==p]
        if len(subset[subset["label"]=="stress"].difference.unique())>1:
            participants.append(p)
    return participants

def df_unique_duration(df, threshold):
    counter = 0
    n_days = []
    fraction = []
    for participant in df.participant_id.unique():
        sub_df = fetch_participant_df(df, participant)
        #number of rows
        n_rows = len(sub_df)
        
        if n_rows >=threshold:
            counter +=1
            
        filled = missing_rows(sub_df)
        n_days.append(len(filled))
        fraction.append(n_rows/len(filled))
    print("\tNumber of participants w/ at least " + str(threshold) + " days:", counter)
    #Number of days
    avg_days = np.mean(df.groupby(["participant_id"])["date"].agg(['count']).values)
    std_days = np.std(df.groupby(["participant_id"])["date"].agg(['count']).values)
    print("\tAvg number of observed days: ", avg_days)
    print("\tStd of observed days: ", std_days)
    print("\tAvg number of total days (filled): ", np.mean(n_days))
    print("\tStd of total days (filled): ", np.std(n_days))
    print("\tAvg Fraction of days complete: ", np.mean(fraction))
    print("\tStd Fraction of days complete: ", np.std(fraction))



def survey_completeness(survey, freq):
    DATAPATH = "//h/snagaraj/Vaughan/datasets/stressrecov/"
    EXTENSION = "surveys_and_active_tasks/"
    
    dfs = {}
    for NAME in survey:
        df = pd.read_csv(DATAPATH+EXTENSION+NAME,compression='gzip')
        
        #get appropriate scheduledate column
        date_col = [col for col in df.columns if 'scheduledate' in col]
        #create new column with a clean date-time
        df["date"] = df.apply (lambda row: pd.to_datetime(row[date_col[0]]).date(), axis=1)
        
        dfs[NAME] = df
    
    results = {} #{survey_name: [% complete for each p]}
    for key in dfs.keys():
        df = dfs[key]
        results[key] = []
        for p in df.participant_id.unique():
            sub_df = fetch_participant_df(df, p)
            n_days = len(sub_df)
            
            filled = missing_rows_freq(sub_df, freq)
            n_filled = len(filled)
            fraction = n_days/n_filled
            
            results[key].append(fraction)
    
    for NAME in survey:
        vals = results[NAME]
        print(NAME)
        print("\tAvg fraction of days complete: ", np.mean(vals))
        print("\tStd fraction of days complete: ", np.std(vals))
        
def survey_completeness_once():
    DATAPATH = "//h/snagaraj/Vaughan/datasets/stressrecov/"
    EXTENSION = "surveys_and_active_tasks/"
    
    dfs = {}
    survey = [ 'participants.csv.gz',"demographics_survey.csv.gz","baseline_covid19_risk.csv.gz", "medical_history.csv.gz", "family_factors.csv.gz",
        "lifestyle_survey.csv.gz", "life_events_checklist_10item.csv.gz", "adverse_childhood_events_ace.csv.gz",
        "acceptabilitystudy_feedback.csv.gz", "ptsd_checklist_pclc.csv.gz", "personality_inventory_tipi.csv.gz"]

    for NAME in survey:
        df = pd.read_csv(DATAPATH+EXTENSION+NAME,compression='gzip').sort_values(by=['participant_id']).drop_duplicates(subset=["participant_id"], keep ="first")
        dfs[NAME] = df
    
    total_participants = len(dfs[survey[0]].participant_id.unique())
    
    results = {} #{survey_name: [% complete for each p]}
    for key in dfs.keys():
        df = dfs[key]
        results[key] = len(df.participant_id.unique()) / total_participants
    
    for NAME in survey:
        vals = results[NAME]
        print(NAME)
        print("\tNumber of participants completed: ", vals)


def daily_shifts_binarize(x):
    if x == 1.0 or x == 2.0:
        return 1.0
    elif x == 0.0:
        return 0.0
    else:
        return x
    
def recode_nan(x):
    if x == 1.0:
        return 1.0 #Stress
    elif x == 0.0:
        return -1.0 #No Stress
    else:
        return 0.0

def plot_stress_frequency(df, p):
    df1 = fetch_participant_df(df, p)

    #Add missing rows
    df1 = missing_rows(df1)

    df1 = hrv_binarize(df1, 0.75, 0.25)
    df1 = any_stress(df1)

    df1["daily_shifts_binary"] = df1.apply (lambda row: daily_shifts_binarize(row.daily_shifts), axis=1)
    df1["daily_stressed"] = df1.apply (lambda row: recode_nan(row.daily_stressed), axis=1)
    df1["daily_shifts"] = df1.apply (lambda row: recode_nan(row.daily_shifts), axis=1)
    df1["shift_stress"] = df1.apply (lambda row: recode_nan(row.shift_stress), axis=1)
    df1["hrv_binary"] = df1.apply (lambda row: recode_nan(row.hrv_binary), axis=1)
    #df1["any_stress"] = df1.apply (lambda row: recode_nan(row.any_stress), axis=1)
    
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(16,8))
    
    
    values = np.expand_dims(df1.daily_stressed.values, axis=0)
    #plt.figure(figsize = (20,1))
    sns.heatmap(values, cmap="vlag", cbar=False, ax = ax[0])
    ax[0].set(ylabel='Daily Stressed', yticklabels=[], xticklabels=[])

    values = np.expand_dims(df1.daily_shifts_binary.values, axis=0)
    #plt.figure(figsize = (20,1))
    sns.heatmap(values, cmap="vlag", cbar=False, ax = ax[1])
    ax[1].set(ylabel='Daily Shifts', yticklabels=[], xticklabels=[])
    
    values = np.expand_dims(df1.shift_stress.values, axis=0)
    #plt.figure(figsize = (20,1))
    sns.heatmap(values, cmap="vlag", cbar=False, ax = ax[2])
    ax[2].set(ylabel='Shift Stress', yticklabels=[], xticklabels=[])
    
    values = np.expand_dims(df1.hrv_binary.values, axis=0)
    #plt.figure(figsize = (20,1))
    im = sns.heatmap(values, cmap="vlag", cbar=False, ax = ax[3])
    ax[3].set(ylabel='HRV Binary', yticklabels=[], xticklabels=[])

    #values = np.expand_dims(df1.any_stress.values, axis=0)
    #plt.figure(figsize = (20,1))
    #im = sns.heatmap(values, cmap="vlag", cbar=False, ax = ax[3])
    #ax[4].set(ylabel='Any Stress', yticklabels=[], xticklabels=[])

    fig.suptitle(p)
    fig.supylabel("Stress Definition")
    fig.supxlabel('Time --->')
    #fig.colorbar(im, ax=ax.ravel().tolist())
    plt.tight_layout()


def plot_6_hrxhrv(df):
    
    participants=[]
    dfs = []
    while len(participants)!=6:
        p = rand_participant(merged_df, 1)
        df1 = fetch_participant_df(merged_df, p)
            #Add missing rows
        #df1 = missing_rows(df1)
        if len(df1.hr_average.dropna())>=10 and len(df1.log_hrv.dropna())>=10:
            participants.append(p)
            dfs.append(df1)

    

    fig, ax = plt.subplots(2, 3, sharex=True, figsize=(16,8))


    sns.regplot(x="hr_average", y="log_hrv", data=dfs[0], ax = ax[0,0])
    sns.regplot(x="hr_average", y="log_hrv", data=dfs[1], ax = ax[0,1])
    sns.regplot(x="hr_average", y="log_hrv", data=dfs[2], ax = ax[0,2])
    sns.regplot(x="hr_average", y="log_hrv", data=dfs[3], ax = ax[1,0])
    sns.regplot(x="hr_average", y="log_hrv", data=dfs[4], ax = ax[1,1])
    sns.regplot(x="hr_average", y="log_hrv", data=dfs[5], ax = ax[1,2])


    #ax[3].set(ylabel='HRV Binary', yticklabels=[], xticklabels=[])
    
    
    #fig.suptitle(p)
    fig.supylabel("Log HRV")
    fig.supxlabel('HR')
    plt.tight_layout()