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
from src.figures import *

import networkx as nx
from scipy.stats import ttest_ind

import pickle
import random
from pyvis import network as net
from pyvis.network import Network

from collections import Counter
import operator
import statistics

from tqdm import tqdm
from matplotlib.collections import PatchCollection

evi_blue = '#34599E'
evi_green = '#4EB792'
evi_yellow = '#F4C15D'
evi_orange = '#EC8C46'
evi_purple = '#823266'
evi_gray = '#C5C5C5'

def plot_p_val_hist(df, col_name, alpha):
    is_reject = 1*(df[col_name] < alpha).sum()    
    
    if is_reject>1:
        plt.figure()
        sns.histplot(data=df, x=col_name, binwidth = alpha).set_title("Number Rejected: " + str(is_reject) + " " + "Total Tests: " + str(len(df)))
        
        plt.figure()
        sns.histplot(data=df[df[col_name] < alpha], x="corr_values").set_title("Correlation Values of Rejected H0")
    else:
        plt.figure()
        sns.histplot(data=df, x=col_name, binwidth = alpha).set_title("Number Rejected: " + str(is_reject) + " " + "Total Tests: " + str(len(df)))

def plot_continuous(participant, features, hue, method=None):
    sns.set()
    sns.color_palette("rocket")
    plt.figure(figsize=(20, 2.5))
    new_df = df_final.loc[df_final["participant_id"]==participant]

    new_df = preprocess_df(new_df)

    ##Standardize continuous features
    #new_df=(new_df-new_df.mean())/new_df.std()
    
    rolling_features=[]
    for feature in features:
        new_df[feature + "_rolling"] = new_df[feature].rolling(5).mean()
        rolling_features.append(feature + "_rolling")
    
    if method == "percentile":
        new_df['hue_percentile'] = new_df[hue].rank(pct = True)
        bins = [0.0, 0.25, 0.75, 1.0]
        new_df['binned_percentile'] = pd.cut(new_df['hue_percentile'], bins)
        new_df['threshold'] =  new_df.apply (lambda row: 1 if row.hue_percentile >= threshold_perc else 0, axis=1)
    else:
        new_df['threshold'] =  new_df.apply (lambda row: 1 if row[hue] >= threshold_abs else 0, axis=1)
    for feature in rolling_features:
        if method == "percentile":
            p = sns.relplot(data=new_df, x=new_df.index, y=feature, hue="binned_percentile", palette="rocket_r",alpha=1, height=2, aspect=4.5, s=250)
        #p.set_ylabel(feature, fontsize = 20)
        else:
            p = sns.relplot(data=new_df, x=new_df.index, y=feature, hue=hue, alpha=1, height=2, aspect=4.5, s=250)



#function to take one participant, compute rolling MMD score on pairs of windows
def plot_rolling_MMD(df, p, features, window_size, func, condition, method, stress_definition, shift, windowed_method=False):
    
    if method == "percentile":
        condition = condition+"_percentile"

    ##Grab an individual
    df1 = fetch_participant_df(df, p)
    #Fill missing dates and interpolate missing value
    df1 = lin_interp_multiple(missing_rows(df1), features)
    
    #Z score normalize features
    df1 = z_score(df1, features)

    #ADD STRESS DEF
    if windowed_method == True:
        df1 = stress_definition(df1, window_size)
    else:
        df1 = stress_definition(df1)
        
    if condition not in df1.columns and method == "percentile":
        df1 = add_percentile(df1, condition[:-11])

    #In case values still missing for condition, force them to be 0
    df1[condition] =pd.to_numeric(df1[condition], errors ='coerce').fillna(0.0).astype('float')
    
    #In case values still missing for condition, force them to be 0
    df1[condition[:-11]] =pd.to_numeric(df1[condition[:-11]], errors ='coerce').fillna(0.0).astype('float')

    #Compute MMD
    df1 = rolling_MMD_multiple(df1, features, window_size, mmd_linear)
    
    #Compute MMD in stress domain
    df1 = rolling_MMD_single(df1, condition, window_size, mmd_linear)

    nrows = len(features)+3

    plt.figure(figsize=(20, len(features)*10))
    for n, feature in enumerate(features):
        # add a new subplot iteratively
        ax1 = plt.subplot(nrows, 1, n + 1)

        df1[feature + "_rolling"] = df1[feature].rolling(window=window_size*2, center=True).mean()
        # filter df and plot ticker on the new subplot axis
        if n == len(features)-1:
            sns.scatterplot(data=df1[window_size:len(df1)-window_size-2], x=df1[window_size:len(df1)-window_size-2].index, y=feature + "_rolling", hue=condition,alpha=1,  ax=ax1,s=250, legend=True).set_xlabel("")
        else:
            sns.scatterplot(data=df1[window_size:len(df1)-window_size-2], x=df1[window_size:len(df1)-window_size-2].index, y=feature + "_rolling", hue=condition,alpha=1,  ax=ax1,s=250, legend=False).set_xlabel("")

    plt.subplots_adjust(hspace=0.5)

    ax1 = plt.subplot(nrows, 1, len(features) + 1)
    sns.lineplot(data=df1[window_size:len(df1)-window_size-2], x=df1[window_size:len(df1)-window_size-2].index, y="average_MMD",  ax=ax1).set_xlabel("")

    test = corr(df1, "average_MMD", condition+"_MMD", shift)
    ax1 = plt.subplot(nrows, 1, len(features) + 2)
    if shift!=None:
        df1[condition+"_MMD_"+"_shift"+str(shift)] = df1[condition+"_MMD"].shift(shift)
        sns.lineplot(data=df1[window_size:len(df1)-window_size-2], x=df1[window_size:len(df1)-window_size-2].index, y=condition+"_MMD_"+"_shift"+str(shift), ax=ax1).set_xlabel("")
    else:
        sns.lineplot(data=df1[window_size:len(df1)-window_size-2], x=df1[window_size:len(df1)-window_size-2].index, y=condition+"_MMD", ax=ax1).set_xlabel("")
        
    ax1.set_title(str(test[0])+ " " + str(test[1]))

    #test = corr(df1, "average_MMD", condition, shift)
    #ax1 = plt.subplot(nrows, 1, len(features) + 3)
    #ax1 = sns.lineplot(data=df1[window_size:len(df1)-window_size-2], x=df1[window_size:len(df1)-window_size-2].index, y=condition).set_title(str(test[0])+ " " + str(test[1]))

    test = corr(df1, "average_MMD", condition[:-11], shift)
    ax1 = plt.subplot(nrows, 1, len(features) + 3)
    if shift!=None:
        df1[condition[:-11]+"_shift"+str(shift)] = df1[condition[:-11]].shift(shift)
        sns.lineplot(data=df1[window_size:len(df1)-window_size-2], x=df1[window_size:len(df1)-window_size-2].index, y=condition[:-11]+"_shift"+str(shift), ax=ax1).set_title(str(test[0])+ " " + str(test[1]))
    else:
        sns.lineplot(data=df1[window_size:len(df1)-window_size-2], x=df1[window_size:len(df1)-window_size-2].index, y=condition[:-11], ax=ax1).set_title(str(test[0])+ " " + str(test[1]))
        
    

def edge_distribution(model):
    #model is a dictionary of edge_lists, key is participant_id
    
    stress_lis = []
    participant_lis = []
    num_edges = []
    
    stress,no_stress = unpack_stress(model)
    stress_keys = list(stress.keys())
    no_stress_keys = list(no_stress.keys())
    
    keys = [value for value in stress_keys if value in no_stress_keys]  
    
    for key in keys:
        stress_lis.append("stress")
        participant_lis.append(key)
        num_edges.append(len(stress[key].edges()))

        stress_lis.append("no_stress")
        participant_lis.append(key)
        num_edges.append(len(no_stress[key].edges()))
        
    df = {"participant_id":participant_lis, "num_edges":num_edges, "stress_condition":stress_lis}
    
    df = pd.DataFrame.from_dict(df)
    
    cat1 = df[df['stress_condition']=='stress'].num_edges.values
    cat2 = df[df['stress_condition']=='no_stress'].num_edges.values

    t_stat, p_val = ttest_ind(cat1, cat2)
    
    print(p_val)
    
    sns.displot(df, x="num_edges", hue="stress_condition", kind="kde", common_norm=False)

    
def node_connectivity(model, features):
    #model is a dictionary of edge_lists, key is participant_id
    
    stress_lis = []
    participant_lis = []
    num_edges = []
    
    stress,no_stress = unpack_stress(model)
    
    stress_keys = list(stress.keys())
    no_stress_keys = list(no_stress.keys())
    
    keys = [value for value in stress_keys if value in no_stress_keys]  
    for key in keys:
        if len(stress[key].edges())!=0 and len(no_stress[key].edges())!=0:
            
            for feature in features:
                if feature not in stress[key].nodes():
                    stress[key].add_node(feature)
            for feature in features:
                if feature not in no_stress[key].nodes():
                    no_stress[key].add_node(feature)
            
            stress_lis.append("stress")
            participant_lis.append(key)
            num_edges.append(np.log(nx.average_node_connectivity(stress[key])))

            stress_lis.append("no_stress")
            participant_lis.append(key)
            num_edges.append(np.log(nx.average_node_connectivity(no_stress[key])))
        
    df = {"participant_id":participant_lis, "log_node_connectivity":num_edges, "stress_condition":stress_lis}
    
    df = pd.DataFrame.from_dict(df)
    
    cat1 = df[df['stress_condition']=='stress'].log_node_connectivity.values
    cat2 = df[df['stress_condition']=='no_stress'].log_node_connectivity.values

    t_stat, p_val = ttest_ind(cat1, cat2)
    
    print(p_val)
    
    sns.displot(df, x="log_node_connectivity", hue="stress_condition", kind="kde", common_norm=False)
    
def between_oura_connection(a,b):
    oura = ['breath_average','efficiency','hr_average','hr_lowest','onset_latency','score','temperature_delta','temperature_trend_deviation','total','log_hrv']
    if a in oura and b in oura:
        return True
    elif a in oura and b in oura:
        return True
    else:
        return False
    
def between_survey_connection(a,b):
    surveys = ['daily_stressed','daily_control','daily_shifts','sam8','shift_stress','promis_sd5','EBTBP','phq9_score','gad7_score','promis_sri_score','pss4_score','promis_es_score',"fss_score"]

    if a in surveys and b in surveys:
        return True
    elif a in surveys and b in surveys:
        return True
    else:
        return False

def between_modality_connection(a,b):
    oura = ['breath_average','efficiency','hr_average','hr_lowest','onset_latency','score','temperature_delta','temperature_trend_deviation','total','log_hrv']
    surveys = ['daily_stressed','daily_control','daily_shifts','sam8','shift_stress','promis_sd5','EBTBP','phq9_score','gad7_score','promis_sri_score','pss4_score','promis_es_score',"fss_score"]

    if a in oura and b in surveys:
        return True
    elif a in surveys and b in oura:
        return True
    else:
        return False


def between_modality_difference(a,b):
    a_count = 0
    b_count = 0
    
    for (edge_a,edge_b) in a.edges():
        if between_modality_connection(edge_a,edge_b):
            a_count +=1
    for (edge_a,edge_b) in b.edges():
        if between_modality_connection(edge_a,edge_b):
            b_count +=1

    return a_count - b_count

def between_oura_difference(a,b):
    a_count = 0
    b_count = 0
    
    for (edge_a,edge_b) in a.edges():
        if between_oura_connection(edge_a,edge_b):
            a_count +=1
    for (edge_a,edge_b) in b.edges():
        if between_oura_connection(edge_a,edge_b):
            b_count +=1

    return a_count - b_count

def between_survey_difference(a,b):
    a_count = 0
    b_count = 0
    
    for (edge_a,edge_b) in a.edges():
        if between_survey_connection(edge_a,edge_b):
            a_count +=1
    for (edge_a,edge_b) in b.edges():
        if between_survey_connection(edge_a,edge_b):
            b_count +=1

    return a_count - b_count

def between_phys_connection(a,b):
    phys = ['hr_lowest',"hr_average",'temperature_delta','log_hrv', "daily_stressed", "daily_control", "daily_shifts", "shift_stress", "pss4_score", "breath_average"]
    if a in phys and b in phys:
        return True
    elif a in phys and b in phys:
        return True
    else:
        return False

def between_sleep_connection(a,b):
    sleep = ["efficiency", "onset_latency", "score", "promis_sri_score", "fss_score", "promis_sd5", "total"]
    if a in sleep and b in sleep:
        return True
    elif a in sleep and b in sleep:
        return True
    else:
        return False
    
def between_mental_connection(a,b):
    mental = ["phq9_score", "gad7_score", "sam8", "EBTBP", "promis_es_score", "pss4_score"]
    if a in mental and b in mental:
        return True
    elif a in mental and b in mental:
        return True
    else:
        return False
    
def between_phys_sleep_connection(a,b):
    phys = ['hr_lowest',"hr_average",'temperature_delta','log_hrv', "daily_stressed", "daily_control", "daily_shifts", "shift_stress", "pss4_score", "breath_average"]
    sleep = ["efficiency", "onset_latency", "score", "promis_sri_score", "fss_score", "promis_sd5", "total"]

    if a in phys and b in sleep:
        return True
    elif a in sleep and b in phys:
        return True
    else:
        return False
    
def between_phys_mental_connection(a,b):
    phys = ['hr_lowest',"hr_average",'temperature_delta','log_hrv', "daily_stressed", "daily_control", "daily_shifts", "shift_stress", "pss4_score", "breath_average"]
    mental = ["phq9_score", "gad7_score", "sam8", "EBTBP", "promis_es_score", "pss4_score"]

    if a in phys and b in mental:
        return True
    elif a in mental and b in phys:
        return True
    else:
        return False
    
def between_sleep_mental_connection(a,b):
    sleep = ["efficiency", "onset_latency", "score", "promis_sri_score", "fss_score", "promis_sd5", "total"]
    mental = ["phq9_score", "gad7_score", "sam8", "EBTBP", "promis_es_score", "pss4_score"]

    if a in sleep and b in mental:
        return True
    elif a in mental and b in sleep:
        return True
    else:
        return False


def graph_comparison_helper(a,b,measure):
    if measure == "difference":
        value = np.log(nx.average_node_connectivity(a)) - np.log(nx.average_node_connectivity(b))
    elif measure == "int_union":
        value = compute_int_union(a,b)
    elif measure == "weight_diffs":
        value = intersection_weight_diff(a,b)
    elif measure == "edges_added":
        value = edges_added(a,b)
    elif measure == "edges_removed":
        value = edges_added(b,a)
    elif measure == "between_oura":
        value = between_oura_difference(a,b)
    elif measure == "between_survey":
        value = between_survey_difference(a,b)
    elif measure == "between_modality":
        value = between_modality_difference(a,b)
    elif measure == "between_phys":
        value = between_helper(a,b,between_phys_connection)
    elif measure == "between_sleep":
        value = between_helper(a,b,between_sleep_connection)
    elif measure == "between_mental":
        value = between_helper(a,b,between_mental_connection)
    elif measure == "between_phys_sleep":
        value = between_helper(a,b,between_phys_sleep_connection)
    elif measure == "between_phys_mental":
        value = between_helper(a,b,between_phys_mental_connection)
    elif measure == "between_sleep_mental":
        value = between_helper(a,b,between_sleep_mental_connection)
    return value

def between_helper(a,b,func):
    a_count = 0
    b_count = 0
    
    for (edge_a,edge_b) in a.edges():
        if func(edge_a,edge_b):
            a_count +=1
    for (edge_a,edge_b) in b.edges():
        if func(edge_a,edge_b):
            b_count +=1

    return a_count - b_count

def between_modality_edge_distribution(model):
    #model is a dictionary of edge_lists, key is participant_id
    
    stress_lis = []
    participant_lis = []
    num_edges = []
    
    stress,no_stress = unpack_stress(model)
    
    for key in stress:
        if key in no_stress.keys():
            stress_lis.append("stress")
            participant_lis.append(key)
            stress_count = 0
            for (a,b) in stress[key].edges():
                if between_modality_connection(a,b):
                    stress_count +=1
            num_edges.append(stress_count)
            
            stress_lis.append("no_stress")
            participant_lis.append(key)
            no_stress_count = 0
            for (a,b) in no_stress[key].edges():
                if between_modality_connection(a,b):
                    no_stress_count +=1
            num_edges.append(no_stress_count)
        
    df = {"participant_id":participant_lis, "between_modality_edges":num_edges, "stress_condition":stress_lis}
    
    df = pd.DataFrame.from_dict(df)
    
    cat1 = df[df['stress_condition']=='stress'].between_modality_edges.values
    cat2 = df[df['stress_condition']=='no_stress'].between_modality_edges.values

    t_stat, p_val = ttest_ind(cat1, cat2)
    
    print(p_val)
    
    sns.displot(df, x="between_modality_edges", hue="stress_condition", kind="kde",common_norm=False)
    
    
def edge_frequency_unnormalized(model):
    lis_tuples = []

    for key in model:
        for item in list(model[key].edges()):
            lis_tuples.append(item)
            
    frequency_list = Counter(lis_tuples)
    sorted_d = dict( sorted(frequency_list.items(), key=operator.itemgetter(1),reverse=True))
    
    return sorted_d

def top_k_edges(model, k):
    dict_d = edge_frequency_unnormalized(model)
    top_k = {}
    for key in sorted(dict_d, key=dict_d.get, reverse=True)[:k]:
        top_k[key] = dict_d[key]
    return top_k

def top_k_edges_unnordered(model, k):
    dict_d = edge_frequency_unordered(model)
    top_k = {}
    for key in sorted(dict_d, key=dict_d.get, reverse=True)[:k]:
        top_k[key] = dict_d[key]
    return top_k


def edge_frequency_unordered(model):
    lis_tuples = []

    for key in model:
        for item in list(model[key].edges()):
            a,b = item
            if (a,b) in lis_tuples:
                lis_tuples.append(item)
            elif (b,a) in lis_tuples:
                lis_tuples.append((b,a))
            else:
                lis_tuples.append(item)
            
    frequency_list = Counter(lis_tuples)
    sorted_d = dict( sorted(frequency_list.items(), key=operator.itemgetter(1),reverse=True))
    
    denom = len(model)
    for key,value in sorted_d.items():
        
        sorted_d[key] = value/denom
        
    return sorted_d

def edge_weights_unordered(model):
    lis_tuples = []
    
    for key in model:
        for item in list(model[key].edges()):
            a,b = item
            if (a,b) in lis_tuples:
                lis_tuples.append(item)
            elif (b,a) in lis_tuples:
                lis_tuples.append((b,a))
            else:
                lis_tuples.append(item)

    dict_d = {}
    for edge in lis_tuples:
        dict_d[edge] = []

    for key in model:
        for item in list(model[key].edges()):
            a,b = item
            if (a,b) in dict_d:
                dict_d[item].append(model[key][a][b]["weight"])
            elif (b,a) in dict_d:
                dict_d[(b,a)].append(model[key][a][b]["weight"])
        
    return dict_d

def plot_change(model, features, number):
    stress_lis = []
    participant_lis = []
    num_edges_stress = []
    num_edges_no_stress = []

    stress,no_stress = unpack_stress(model)

    stress_keys = list(stress.keys())
    no_stress_keys = list(no_stress.keys())

    keys = [value for value in stress_keys if value in no_stress_keys]  
    for key in keys:
        if len(stress[key].edges())!=0 and len(no_stress[key].edges())!=0:

            for feature in features:
                if feature not in stress[key].nodes():
                    stress[key].add_node(feature)
            for feature in features:
                if feature not in no_stress[key].nodes():
                    no_stress[key].add_node(feature)

            #stress_lis.append("stress")
            participant_lis.append(key)
            num_edges_stress.append(np.log(nx.average_node_connectivity(stress[key])))

            #stress_lis.append("no_stress")
            #participant_lis.append(key)
            num_edges_no_stress.append(np.log(nx.average_node_connectivity(no_stress[key])))

    df = {"participant_id":participant_lis, "stress":num_edges_stress, "no_stress":num_edges_no_stress,}

    df = pd.DataFrame.from_dict(df)
    
    df["decreasing"] = df.apply(lambda row: is_decreasing(row.no_stress, row.stress), axis=1)
    df["difference"] = df.apply(lambda row: np.abs(row.no_stress-row.stress), axis=1)

    plt.rcParams["figure.figsize"] = (3,10)

    # Reorder it following the values of the first value:
    ordered_df = df.sort_values(by='difference').tail(number)
    #my_range=range(1,len(df.index)+1)
    my_range = range(1,number+1)


    # The horizontal plot is made using the hline function
    plt.hlines(y=my_range, xmin=ordered_df['no_stress'], xmax=ordered_df['stress'], color='grey', alpha=1)
    plt.scatter(ordered_df['no_stress'], my_range, color='skyblue', alpha=1, label='no_stress')
    plt.scatter(ordered_df['stress'], my_range, color='green', alpha=1 , label='stress')
    plt.legend()

    # Add title and axis names
    plt.yticks(my_range, ordered_df['participant_id'])
    plt.title("Comparison of the NO STRESS and the STRESS", loc='left')
    plt.xlabel('Log Node Connectivity')
    plt.ylabel('Group')

    # Show the graph
    plt.show()

    
def is_decreasing(a,b):
    if b<=a:
        return True
    else:
        return False

def colour_list(arr):
    oura = ['breath_average','efficiency','hr_average','hr_lowest','onset_latency','score','temperature_delta','temperature_trend_deviation','total','log_hrv']
    garmin = ['steps','minHeartRateInBeatsPerMinute','maxHeartRateInBeatsPerMinute','averageHeartRateInBeatsPerMinute','restingHeartRateInBeatsPerMinute','averageStressLevel','maxStressLevel','stressDurationInSeconds','restStressDurationInSeconds','activityStressDurationInSeconds','lowStressDurationInSeconds','mediumStressDurationInSeconds','highStressDurationInSeconds']

    surveys = ["daily_stressed",'daily_control','daily_shifts','sam8','shift_stress','promis_sd5','EBTBP','phq9_score','gad7_score','promis_sri_score','pss4_score','promis_es_score',"fss_score"]
    population = ["p_id", "dem_age", "dem_gender", "height", "weight", "ace_score", "hx_any", "mhx_any", "ptsd_score", "life_events_score"]
    colour_lis =[]
    for elem in arr:
        if elem in oura:
            colour_lis.append(evi_blue)
        elif elem in garmin:
            colour_lis.append('#32a89c')
        elif elem in surveys:
            colour_lis.append(evi_purple)
        elif elem in population:
            colour_lis.append('#8351c9')
        else:
            colour_lis.append('#8351c9')
    return colour_lis

def colour_list_themes(arr):
  
    colour_lis =[]
    for elem in arr:
        if elem =="phys":
            colour_lis.append(evi_blue)
        elif elem =="sleep":
            colour_lis.append('#32a89c')
        elif elem =="mental":
            colour_lis.append(evi_purple)
        else:
            colour_lis.append('#8351c9')
    return colour_lis

def graph(dict_model, features):
    nodes = features
    #nodes = list(set([item for sublist in dict_model.edges() for item in sublist]))
    colour_lis = colour_list(nodes)

    net = Network('500px', '1000px', notebook=True, directed=True)
    net.add_nodes(nodes, color= colour_lis)
    
    for key in dict_model.edges():
        net.add_edge(key[0], key[1])
        
    return net


#Number of edges per person?

def edges_per_person(dict_model):
    dict_edges ={}
    for key in dict_model:
        dict_edges[key] = len(list(dict_model[key].edges()))
    return dict_edges

#Compute frequency of unique edges

def edge_frequency(model):
    lis_tuples = []

    for key in model:
        for item in list(model[key].edges()):
            lis_tuples.append(item)
            
    frequency_list = Counter(lis_tuples)
    sorted_d = dict( sorted(frequency_list.items(), key=operator.itemgetter(1),reverse=True))
    
    dict_nodes = {}
    for key in model:
        dict_nodes[key] = list(model[key].nodes())

    for key,value in sorted_d.items():
        denom = 0
        for participant in dict_nodes:
            if key[0] in dict_nodes[participant] and key[1] in dict_nodes[participant]:
                denom +=1
        sorted_d[key] = value/denom
        
    return sorted_d

def remove_duplicates(dict_freq, cutoff):
    to_pop=[]
    dict_copy = dict_freq
    for key, item in dict_freq.items():
        #if (key[1], key[0]) in dict_freq:
            #Remove "Duplicates" with smaller freq
            #if dict_freq[(key[1], key[0])] >= dict_freq[key]:
            #    to_pop.append(key)
        if dict_freq[key] < cutoff and key not in to_pop:
            to_pop.append(key)

    for item in to_pop:
        dict_copy.pop(item)
    
    return dict_copy

def weighted_graph(dict_freq, features):
    #nodes = list(set([item for sublist in dict_freq for item in sublist]))
    nodes = features
    colour_lis = colour_list(nodes)

    net = Network('500px', '1000px', notebook=True, directed=True)
    net.add_nodes(nodes, color= colour_lis)
    
    values = [value for key,value in dict_freq.items()]
    #mean = sum(values)/len(values)
    #std = statistics.stdev(values)
    for key,value in dict_freq.items():
        net.add_edge(key[0], key[1], weight=value, title=value,width = 10*value)
    net.toggle_physics(True)
    net.show_buttons()
    return net
    
def pipeline(dict_model, cutoff, features):
    
    return weighted_graph(remove_duplicates(edge_frequency(dict_model), cutoff),features)

#Compute a weighted graph with the weight being the frequency of a unique edge
#if freq(A,B) > freq (B,A), remove (B,A)

#For a given feature, what are the most frequent features coming from/to?


def unpack_stress(model):
    stress = {}
    no_stress = {}

    for key in model:
        for (a,b) in model[key]:
            if a == "stress":
                stress[key] = b
            else:
                no_stress[key] = b
    return stress, no_stress


def plot_edge_frequency_change(model, number):
    stress,no_stress = unpack_stress(model)


    edges = []
    no_stress_frequency = []
    stress_frequency = []
    no_stress_weights = []
    stress_weights = []

    no_stress_dict = edge_frequency_unordered(no_stress)
    stress_dict = edge_frequency_unordered(stress)
    no_stress_dict_weights = edge_weights_unordered(no_stress)
    stress_dict_weights = edge_weights_unordered(stress)

    for edge in no_stress_dict:
        a,b = edge
        if (a,b) in stress_dict or (b,a) in stress_dict:

            edges.append(edge)
            no_stress_frequency.append(no_stress_dict[edge])
            no_stress_weights.append(no_stress_dict_weights[edge])
            try:
                stress_frequency.append(stress_dict[edge])
                stress_weights.append(stress_dict_weights[edge])
            except:
                stress_frequency.append(stress_dict[(b,a)])
                stress_weights.append(stress_dict_weights[(b,a)])
    df = { "edge":edges, "no_stress":no_stress_frequency, "stress":stress_frequency, "no_stress_weights":no_stress_weights,
         "stress_weights":stress_weights}

    df = pd.DataFrame.from_dict(df)

    df["decreasing"] = df.apply(lambda row: is_decreasing(row.no_stress, row.stress), axis=1)
    df["difference"] = df.apply(lambda row: np.abs(row.no_stress-row.stress), axis=1)
    
    plt.rcParams["figure.figsize"] = (3,6)

    # Reorder it following the values of the first value:
    ordered_df = df.sort_values(by='no_stress').tail(number)
    #my_range=range(1,len(df.index)+1)
    my_range = range(1,number+1)


    # The horizontal plot is made using the hline function
    plt.hlines(y=my_range, xmin=ordered_df['no_stress'], xmax=ordered_df['stress'], color='grey', alpha=1)
    plt.scatter(ordered_df['no_stress'], my_range, color='skyblue', alpha=1, label='no_stress')
    plt.scatter(ordered_df['stress'], my_range, color='green', alpha=1 , label='stress')
    plt.legend()

    # Add title and axis names
    plt.yticks(my_range, ordered_df['edge'])
    plt.title("Comparison of the NO STRESS and the STRESS", loc='left')
    plt.xlabel('% of population observed')
    plt.ylabel('Group')

    # Show the graph
    plt.show()
    
    plt.rcParams["figure.figsize"] = (3,6)

    # Reorder it following the values of the first value:
    ordered_df = df.sort_values(by='difference').tail(number)
    #my_range=range(1,len(df.index)+1)
    my_range= range(1,number+1)

    # The horizontal plot is made using the hline function
    plt.hlines(y=my_range, xmin=ordered_df['no_stress'], xmax=ordered_df['stress'], color='grey', alpha=1)
    plt.scatter(ordered_df['no_stress'], my_range, color='skyblue', alpha=1, label='no_stress')
    plt.scatter(ordered_df['stress'], my_range, color='green', alpha=1 , label='stress')
    plt.legend()

    # Add title and axis names
    plt.yticks(my_range, ordered_df['edge'])
    plt.title("Comparison of the NO STRESS and the STRESS - ordered by biggest change", loc='left')
    plt.xlabel('% of population observed')
    plt.ylabel('Group')

    # Show the graph
    plt.show()
    
    plt.rcParams["figure.figsize"] = (3,6)

    # Reorder it following the values of the first value:
    ordered_df = df.sort_values(by='difference')
    ordered_df = ordered_df[ordered_df["decreasing"]==True].sort_values(by='difference').tail(10)
    #my_range=range(1,len(ordered_df.index)+1)
    my_range = range(1,11) 

    # The horizontal plot is made using the hline function
    plt.hlines(y=my_range, xmin=ordered_df['no_stress'], xmax=ordered_df['stress'], color='grey', alpha=1)
    plt.scatter(ordered_df['no_stress'], my_range, color='skyblue', alpha=1, label='no_stress')
    plt.scatter(ordered_df['stress'], my_range, color='green', alpha=1 , label='stress')
    plt.legend()

    # Add title and axis names
    plt.yticks(my_range, ordered_df['edge'])
    plt.title("Comparison of the NO STRESS and the STRESS - ordered by biggest change, decreasing only", loc='left')
    plt.xlabel('% of population observed')
    plt.ylabel('Group')

    # Show the graph
    plt.show()
    
    plt.rcParams["figure.figsize"] = (3,6)

    # Reorder it following the values of the first value:
    ordered_df = df.sort_values(by='difference')
    ordered_df = ordered_df[ordered_df["decreasing"]==False].sort_values(by='difference').tail(10)
    #my_range=range(1,len(ordered_df.index)+1)
    my_range = range(1,11) 

    # The horizontal plot is made using the hline function
    plt.hlines(y=my_range, xmin=ordered_df['no_stress'], xmax=ordered_df['stress'], color='grey', alpha=1)
    plt.scatter(ordered_df['no_stress'], my_range, color='skyblue', alpha=1, label='no_stress')
    plt.scatter(ordered_df['stress'], my_range, color='green', alpha=1 , label='stress')
    plt.legend()

    # Add title and axis names
    plt.yticks(my_range, ordered_df['edge'])
    plt.title("Comparison of the NO STRESS and the STRESS - ordered by biggest change, increasing only", loc='left')
    plt.xlabel('% of population observed')
    plt.ylabel('Group')

    # Show the graph
    plt.show()

def plot_edge_weight_distribution(model):
    stress,no_stress = unpack_stress(model)


    edges = []
    stress_label =[]
    weights = []

    no_stress_dict = edge_frequency_unordered(no_stress)
    stress_dict = edge_frequency_unordered(stress)

    no_stress_dict_weights = edge_weights_unordered(no_stress)
    stress_dict_weights = edge_weights_unordered(stress)

    for edge in no_stress_dict:
        a,b = edge
        if (a,b) in stress_dict or (b,a) in stress_dict:

            for item in no_stress_dict_weights[edge]:
                edges.append(edge)
                stress_label.append("no_stress")
                weights.append(item)

            if (a,b) in stress_dict:
                for item in stress_dict_weights[(a,b)]:
                    edges.append(edge)
                    stress_label.append("stress")
                    weights.append(item)
            else:
                for item in stress_dict_weights[(b,a)]:
                    edges.append(edge)
                    stress_label.append("stress")
                    weights.append(item)
    df = { "edge":edges, "stress_label":stress_label, "weight":weights}

    df = pd.DataFrame.from_dict(df)

    for item in df.edge.unique():
        sub_df = df[df["edge"]==item]
        if len(sub_df[sub_df["stress_label"]=="stress"]) >=10 and len(sub_df[sub_df["stress_label"]=="no_stress"])>=10:
            plt.figure()
            g = sns.displot(data=sub_df, x="weight", hue="stress_label", kind = "kde", common_norm = False).set(title = item)
            g.set_titles(item)
            plt.show()

def unpack_reference_random(model):
    stress = {}
    no_stress = {}

    for key in model:
        stress[key] = []
        no_stress[key] = []
        for (a,b) in model[key]:
            if a == "a":
                stress[key].append(b)
            else:
                no_stress[key].append(b)
    return stress, no_stress

def unpack_stress_random(model):
    stress = {}
    no_stress = {}

    for key in model:
        stress[key] = []
        no_stress[key] = []
        for (a,b) in model[key]:
            if a == "stress":
                stress[key].append(b)
            else:
                no_stress[key].append(b)
    return stress, no_stress

def union_edges(a,b):
    a_edges = a.edges()
    b_edges = b.edges()
    
    union = list(a_edges)
    for (x,y) in b_edges:
        if (x,y) not in a_edges and (y,x) not in a_edges:
            union.append((x,y))
            
    return union

def intersection_edges(a,b):
    a_edges = a.edges()
    b_edges = b.edges()
    
    intersection = []
    
    for (x,y) in a_edges:
        if (x,y) in b_edges or (y,x) in b_edges:
            intersection.append((x,y))
    return intersection
    
#given a pair of graphs compute the # Intersecting Edges / Union of Edges
#Needs to handle (a,b) (b,a) pairs - ie: directionless
def compute_int_union(a,b):
    intersection = intersection_edges(a,b)
    union = union_edges(a,b)
    
    return len(intersection)/len(union)
  
#a-b is the order
def intersection_weight_diff(a,b):
    intersection = intersection_edges(a,b)
    
    dict_weights = {}
    for (x,y) in intersection:
        if (x,y) in a.edges():
            a_weight = a[x][y]["weight"]
        elif (y,x) in a.edges():
            a_weight = a[y][x]["weight"]
        if (x,y) in b.edges():
            b_weight = b[x][y]["weight"]
        elif (y,x) in b.edges():
            b_weight = b[y][x]["weight"] 
            
        diff = a_weight - b_weight
        dict_weights[(x,y)] = diff
    
    return dict_weights

#given a pair of graphs return list of Edges added ON stress
#Edges that a has that b does not have
def edges_added(a,b):
    intersection = intersection_edges(a,b)
    
    edges = []
    for (x,y) in a.edges():
        if (x,y) not in b.edges() and (y,x) not in b.edges():
            edges.append((x,y))
            
    return edges

def compare_reference(dict_reference, dict_models, features, stress_definition):
    a,b = unpack_reference_random(dict_reference)
    stress, no_stress = unpack_stress_random(dict_models)

    stress_keys = list(stress.keys())
    no_stress_keys = list(no_stress.keys())

    df = {"participant_id":[], 'label':[]}

    measures = ["difference", "int_union", "weight_diffs", "edges_added", "edges_removed", "between_oura",
            "between_survey", "between_modality", "between_phys", "between_sleep", "between_mental", "between_phys_sleep",
            "between_phys_mental", "between_sleep_mental"]

    keys = [value for value in stress_keys if value in no_stress_keys]  
    for key in tqdm(keys):
        if len(stress[key])!=0 and len(no_stress[key])!=0:
            for x, y in zip(stress[key], no_stress[key]):
                for feature in features:
                    if feature not in x.nodes() and feature != stress_definition:
                        x.add_node(feature)
                    if feature not in y.nodes() and feature != stress_definition:
                        y.add_node(feature)

                df["participant_id"].append(key)
                
                for measure in measures:
                    if measure in df:
                        value = graph_comparison_helper(y,x,measure)
                        df[measure].append(value)
                    else:
                        df[measure] = []
                        value = graph_comparison_helper(y,x,measure)
                        df[measure].append(value)
                
                df["label"].append("stress")
        if len(a[key])!=0 and len(b[key])!=0:
            for x, y in zip(a[key], b[key]):
                for feature in features:
                    if feature not in x.nodes():
                        x.add_node(feature)
                    if feature not in y.nodes():
                        y.add_node(feature)

                df["participant_id"].append(key)
                
                for measure in measures:
                    if measure in df:
                        value = graph_comparison_helper(y,x,measure)
                        df[measure].append(value)
                    else:
                        df[measure] = []
                        value = graph_comparison_helper(y,x,measure)
                        df[measure].append(value)
                
                df["label"].append("reference")


    #df = {"participant_id":participant_lis, "between_oura": between_ouras, "between_survey": between_surveys,"between_modality": between_modalities, "difference":differences,"int_union":int_unions ,"weights":weight_diffs,"edges_added_stress":edges_added_stress, "edges_removed_stress":edges_removed_stress,"label":label}

    df = pd.DataFrame.from_dict(df)

    return df


#Given an individual's DF, plot the distribution of weights STRESS vs REFERENCE

def plot_weight_change(df):
    #get weights 
    #stress_weights = {(a,b):[weight, weight, weight]}
    
    stress_weights = {}
    reference_weights = {}
    
    stress_edges = []
    reference_edges = []
    
    subset_stress = df[df["label"]=="stress"]
    subset_reference = df[df["label"]=="reference"]
    
    for item in subset_stress.weights.values:
        for (a,b) in item.keys():
            if (a,b) not in stress_edges and (b,a) not in stress_edges:
                stress_edges.append((a,b))
    for item in subset_reference.weights.values:
        for (a,b) in item.keys():
            if (a,b) not in reference_edges and (b,a) not in reference_edges:
                reference_edges.append((a,b))
    
    edges_intersection = []
    
    for (a,b) in stress_edges:
        if (a,b) in reference_edges:
            edges_intersection.append((a,b))
        elif (b,a) in reference_edges:
            edges_intersection.append((a,b))
            
    for item in edges_intersection:
        stress_weights[item]= []
        reference_weights[item] = []
    
    for item in subset_stress.weights.values:
        for (a,b) in item.keys():
            if (a,b) in edges_intersection:
                stress_weights[(a,b)].append(item[(a,b)])
            elif (b,a) in edges_intersection:
                stress_weights[(b,a)].append(item[(a,b)])
    
    for item in subset_reference.weights.values:
        for (a,b) in item.keys():
            if (a,b) in edges_intersection:
                reference_weights[(a,b)].append(item[(a,b)])
            elif (b,a) in edges_intersection:
                reference_weights[(b,a)].append(item[(a,b)])
    
    edges = []
    means = []
    p_vals = []     
    for edge in edges_intersection:
        if len(stress_weights[edge]) == 100:
            weights = stress_weights[edge] + reference_weights[edge]
            labels_stress = ["stress" for i in range(len(stress_weights[edge]))]
            labels_reference = ["reference" for i in range(len(reference_weights[edge]))]
            labels = labels_stress + labels_reference
            plot_df = {"weights": weights, "labels":labels}
            plot_df = pd.DataFrame.from_dict(plot_df)
            print(edge)
            sns.displot(data=plot_df, x="weights", hue = "labels", kind="kde", common_norm=False)
            plt.show()
    
            cat1 = plot_df[plot_df['labels']=="stress"]["weights"].dropna().values
            cat2 = plot_df[plot_df['labels']=="reference"]["weights"].dropna().values

            t_stat, p_val = ttest_ind(cat1, cat2)
            edges.append(edge)
            means.append(np.mean(cat1))
            p_vals.append(p_val)

    
    
#Given an individual's DF, return intersecting edges, means, and p_vals

def return_weight_change(df):
    #get weights 
    #stress_weights = {(a,b):[weight, weight, weight]}
    
    stress_weights = {}
    reference_weights = {}
    
    stress_edges = []
    reference_edges = []
    
    subset_stress = df[df["label"]=="stress"]
    subset_reference = df[df["label"]=="reference"]
    
    for item in subset_stress.weights.values:
        for (a,b) in item.keys():
            if (a,b) not in stress_edges and (b,a) not in stress_edges:
                stress_edges.append((a,b))
    for item in subset_reference.weights.values:
        for (a,b) in item.keys():
            if (a,b) not in reference_edges and (b,a) not in reference_edges:
                reference_edges.append((a,b))
    
    edges_intersection = []
    
    for (a,b) in stress_edges:
        if (a,b) in reference_edges:
            edges_intersection.append((a,b))
        elif (b,a) in reference_edges:
            edges_intersection.append((a,b))
            
    for item in edges_intersection:
        stress_weights[item]= []
        reference_weights[item] = []
    
    for item in subset_stress.weights.values:
        for (a,b) in item.keys():
            if (a,b) in edges_intersection:
                stress_weights[(a,b)].append(item[(a,b)])
            elif (b,a) in edges_intersection:
                stress_weights[(b,a)].append(item[(a,b)])
    
    for item in subset_reference.weights.values:
        for (a,b) in item.keys():
            if (a,b) in edges_intersection:
                reference_weights[(a,b)].append(item[(a,b)])
            elif (b,a) in edges_intersection:
                reference_weights[(b,a)].append(item[(a,b)])
    
    edges = []
    means = []
    p_vals = []     
    for edge in edges_intersection:
        if len(stress_weights[edge]) == 100:
            weights = stress_weights[edge] + reference_weights[edge]
            labels_stress = ["stress" for i in range(len(stress_weights[edge]))]
            labels_reference = ["reference" for i in range(len(reference_weights[edge]))]
            labels = labels_stress + labels_reference
            plot_df = {"weights": weights, "labels":labels}
            plot_df = pd.DataFrame.from_dict(plot_df)
            cat1 = plot_df[plot_df['labels']=="stress"]["weights"].dropna().values
            cat2 = plot_df[plot_df['labels']=="reference"]["weights"].dropna().values

            t_stat, p_val = ttest_ind(cat1, cat2)
            edges.append(edge)
            means.append(np.mean(cat1))
            p_vals.append(p_val)
    return edges, means, p_vals


def plot_population_characteristics(p_vals_df, measures, stress_definition, plot=True):
    df_once = load_once_complete()

    stress_definitions = []
    measures_list = []
    features_list = []
    p_vals = []
    directions = []

    for measure in measures:
        print(measure)
        df = p_vals_df[p_vals_df["measure"]==measure]
        df = df[df["stress_definition"]==stress_definition]

        merged_df = df.merge(df_once, on="participant_id", how = "left")

        features = ["dem_age", "weight", "ace_score", "ptsd_score", "life_events_score"]


        #PLOT EFFECT OF CHANGE
        # for feature in features:

        #     cat1 = merged_df[merged_df['reject_fdr']==0][feature].dropna().values
        #     cat2 = merged_df[merged_df['reject_fdr']==1][feature].dropna().values

        #     t_stat, p_val = ttest_ind(cat1, cat2)

        #     if p_val<0.05:
        #         print("    effect of CHANGE vs NO CHANGE on " + feature +":", p_val)
        #         if plot == True:
        #             plt.figure()
        #             sns.displot(data=merged_df, x=feature, hue = "reject_fdr", kind="kde", common_norm=False)

        #PLOT GAIN VS LOSS
        for feature in features:
            sub_df = merged_df[merged_df["reject_fdr"]==1] 

            cat1 = sub_df[sub_df['direction']=="Gain"][feature].dropna().values
            cat2 = sub_df[sub_df['direction']=="Loss"][feature].dropna().values

            t_stat, p_val = ttest_ind(cat1, cat2)
            stress_definitions.append(stress_definition)
            measures_list.append(measure)
            features_list.append(feature)
            p_vals.append(p_val)
            if np.mean(cat1) > np.mean(cat2):
                directions.append("Gain Higher")
            else:
                directions.append("Loss Higher")

            if p_val<0.05:
                print("    effect of GAIN vs LOSS on " + feature +":", p_val)
                if plot == True:
                    plt.figure()
                    sns.displot(data=sub_df, x=feature, hue = "direction", kind="kde", common_norm=False)

        X =["direction"]
        Y = ["dem_gender", 'hx_none', 'hx_diabetes', 'hx_obesity', 'hx_heart',
            'hx_lung', 'hx_rheum', 'hx_kidney', 'hx_liver', 'hx_allergies',
            'hx_pain', 'hx_gi', 'hx_ent', 'hx_neuro', 'hx_other','hx_any',
            'mhx_none', 'mhx_mood', 'mhx_anxiety', 'mhx_psych', 'mhx_eating',
            'mhx_neurodev', 'mhx_sleep', 'mhx_suicide', 'mhx_learning',
            'mhx_other', 'mhx_any','alcohol', 'binge', 'smoke', 'exercise']


        from scipy.stats import chi2_contingency


        for x in X:
            for y in Y:
                if x == "reject_fdr":
                    con_table = pd.crosstab(index=merged_df['reject_fdr'], columns=merged_df[y]).values
                    stat, p, dof, expected = chi2_contingency(con_table)
                    if p<0.05:
                        print("    effect of CHANGE vs NO CHANGE on " + y +":", p)
                        if plot == True:
                            plt.figure()
                            (merged_df
                            .groupby(x)[y]
                            .value_counts(normalize=True)
                            .mul(100)
                            .rename('proportion')
                            .reset_index()
                            .pipe((sns.catplot,'data'), x=x,y='proportion',hue=y,kind='bar'))
                else:
                    sub_df = merged_df[merged_df["reject_fdr"]==1]
                    con_table = pd.crosstab(index=sub_df["direction"], columns=sub_df[y]).values
                    stat, p, dof, expected = chi2_contingency(con_table)

                    stress_definitions.append(stress_definition)
                    measures_list.append(measure)
                    features_list.append(y)
                    p_vals.append(p)
                    directions.append("Categorical")


                    if p<0.05:
                        print("    effect of GAIN vs LOSS on " + y +":", p)
                        if plot == True:
                            plt.figure()
                            (sub_df
                            .groupby(x)[y]
                            .value_counts(normalize=True)
                            .mul(100)
                            .rename('proportion')
                            .reset_index()
                            .pipe((sns.catplot,'data'), x=x,y='proportion',hue=y,kind='bar'))


    final_df = {"stress_definition": stress_definitions, "measure":measures_list, "feature":features_list, "p_val":p_vals, "direction":directions}
    final_df = pd.DataFrame.from_dict(final_df)

    concat = []
    for measure in measures:
        sub_df = final_df[(final_df["measure"]== measure)]

        sub_df = bonferroni_correct(sub_df, "p_val")
        sub_df = bh_correct(sub_df, "p_val")

        p_vals_df.loc[sub_df.index] = sub_df

        reject_fdr = sub_df[sub_df["reject_fdr"]==True]
        reject_bc = sub_df[sub_df["reject_bc"]==True]

        concat.append(sub_df)
        
    return pd.concat(concat)


def return_edge_frequency(df):
    #get weights 
    #stress_weights = {(a,b):[weight, weight, weight]}
    
    stress_weights = {}
    reference_weights = {}
    
    stress_edges = []
    reference_edges = []
    
    subset_stress = df[df["label"]=="stress"]
    subset_reference = df[df["label"]=="reference"]
    
    for item in subset_stress.weights.values:
        for (a,b) in item.keys():
            if (a,b) not in stress_edges and (b,a) not in stress_edges:
                stress_edges.append((a,b))
    for item in subset_reference.weights.values:
        for (a,b) in item.keys():
            if (a,b) not in reference_edges and (b,a) not in reference_edges:
                reference_edges.append((a,b))
    
    edges_intersection = []
    
    for (a,b) in stress_edges:
        if (a,b) in reference_edges:
            edges_intersection.append((a,b))
        elif (b,a) in reference_edges:
            edges_intersection.append((a,b))
            
    for item in edges_intersection:
        stress_weights[item]= []
        reference_weights[item] = []
    
    for item in subset_stress.weights.values:
        for (a,b) in item.keys():
            if (a,b) in edges_intersection:
                stress_weights[(a,b)].append(item[(a,b)])
            elif (b,a) in edges_intersection:
                stress_weights[(b,a)].append(item[(a,b)])
    
    for item in subset_reference.weights.values:
        for (a,b) in item.keys():
            if (a,b) in edges_intersection:
                reference_weights[(a,b)].append(item[(a,b)])
            elif (b,a) in edges_intersection:
                reference_weights[(b,a)].append(item[(a,b)])
    
    edges = []
    means = []
    p_vals = []     
    for edge in edges_intersection:
        if len(stress_weights[edge]) == 100:
            weights = stress_weights[edge] + reference_weights[edge]
            labels_stress = ["stress" for i in range(len(stress_weights[edge]))]
            labels_reference = ["reference" for i in range(len(reference_weights[edge]))]
            labels = labels_stress + labels_reference
            plot_df = {"weights": weights, "labels":labels}
            plot_df = pd.DataFrame.from_dict(plot_df)
            cat1 = plot_df[plot_df['labels']=="stress"]["weights"].dropna().values
            cat2 = plot_df[plot_df['labels']=="reference"]["weights"].dropna().values

            t_stat, p_val = ttest_ind(cat1, cat2)
            edges.append(edge)
            means.append(np.mean(cat1))
            p_vals.append(p_val)
    return edges, means, p_vals

#UNIQUE EDGES AT POPULATION LEVEL
def return_unique_edges(df):
    unique_edges = {}

    #GET ONLY STRESS DEFINITION COMPARISONS
    stress_only = df[df["label"]=="stress"]

    for item in stress_only.edges_added.values:
        for (a,b) in item:
            if (a,b) not in unique_edges and (b,a) not in unique_edges:
                unique_edges[(a,b)] = (0,0) #ADDED,REMOVED

    for item in stress_only.edges_removed.values:
        for (a,b) in item:
            if (a,b) not in unique_edges and (b,a) not in unique_edges:
                unique_edges[(a,b)] = (0,0)
    return unique_edges


def compute_edges(df,p,unique_edges_master):
    #GET AN INDIVIDUAL
    sub_df = df[df["participant_id"]==p]
    
    unique_edges = unique_edges_master.copy()
    
    for item in sub_df.edges_added.values:

        for (a,b) in item:
            if (a,b) in unique_edges:     
                unique_edges[(a,b)] = list(unique_edges[(a,b)])
                unique_edges[(a,b)][0] += 0.01
                unique_edges[(a,b)] = tuple(unique_edges[(a,b)])
            elif (b,a) in unique_edges:        
                unique_edges[(b,a)] = list(unique_edges[(b,a)])
                unique_edges[(b,a)][0] += 0.01
                unique_edges[(b,a)] = tuple(unique_edges[(b,a)])

    for item in sub_df.edges_removed.values:

        for (a,b) in item:
            if (a,b) in unique_edges:     
                unique_edges[(a,b)] = list(unique_edges[(a,b)])
                unique_edges[(a,b)][1] += 0.01
                unique_edges[(a,b)] = tuple(unique_edges[(a,b)])
            elif (b,a) in unique_edges:        
                unique_edges[(b,a)] = list(unique_edges[(b,a)])
                unique_edges[(b,a)][1] += 0.01
                unique_edges[(b,a)] = tuple(unique_edges[(b,a)])
    return unique_edges

def return_edge_summary(stress_definition):

    df = get_df(stress_definition)

    participants = get_participants(stress_definition)

    #GET ONLY STRESS DEFINITION COMPARISONS
    stress_only = df[df["label"]=="stress"]

    #GET UNIQUE EDGES AT POP LEVEL
    unique_edges_master = return_unique_edges(stress_only)


    participants_edges = []
    edges = []
    percent_added = []
    percent_removed = []

    for p in participants:
        unique_edges = compute_edges(stress_only,p,unique_edges_master)
        for edge in unique_edges:
            participants_edges.append(p)
            edges.append(edge)
            percent_added.append(unique_edges[edge][0])
            percent_removed.append(unique_edges[edge][1])

    final_dict = {"participant_id": participants_edges, "edge": edges, "percent_added": percent_added, 
                  "percent_removed":percent_removed}
    final_dict = pd.DataFrame.from_dict(final_dict)

    return final_dict   


def print_frequent_edges(df, cutoff):
    print("EDGES FREQUENTLY GAINED ON STRESS")
    for edge in df.edge.unique():
        edge_df = df[df["edge"]==edge]
        sub_df = edge_df[edge_df["percent_added"]>=cutoff]
        if len(sub_df)>=10:
            print(str(edge)+":  "+str(len(sub_df)))

    print("\nEDGES FREQUENTLY LOST ON STRESS")
    for edge in df.edge.unique():
        edge_df = df[df["edge"]==edge]
        sub_df = edge_df[edge_df["percent_removed"]>=cutoff]
        if len(sub_df)>=10:
            print(str(edge)+":  "+str(len(sub_df)))
