import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import random
from scipy.stats import ttest_ind, ks_2samp
from statannot import add_stat_annotation
from sklearn import metrics
import numpy_ext
from scipy.stats.stats import pearsonr

from src.definitions import *
from src.processing import *
from src.plotting import *
from tqdm import tqdm
import time

from statsmodels.sandbox.stats.multicomp import multipletests

from pgmpy.estimators import PC
from pgmpy.base import DAG
from pgmpy.independencies import Independencies

from CCIT import CCIT
from CCIT import DataGen

from pgmpy.models import BayesianNetwork
import networkx as nx
import pylab as plt
import multiprocessing

import scipy.stats as stats
import math
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    delta = np.array(delta)
    return delta.dot(delta.T)


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def t_test(X,Y):
    return ttest_ind(X, Y, equal_var=False).pvalue

def ks_test(X,Y, alternative = "two-sided"):
    return ks_2samp(X,Y, alternative).pvalue

def window_test(vals, size, func):
    if len(vals) == size*2:
        return func(vals[:size], vals[size:])
    else:
        print("Error dimension of input and window size do not match")

#Function to get a single feature MMD Scores from a single participants dataframe
def rolling_MMD_single(df, feature, window_size, mmd_type):
    
    window_MMD = lambda vals: mmd_type(vals[:window_size], vals[window_size:])
    
    df[feature + "_MMD"] = df[feature].rolling(window_size*2, center=True).apply(window_MMD)
    
    return df

#Function to get multiple features MMD scores from a single participant and return the average MMD scores

def rolling_MMD_multiple(df, features, window_size, mmd_type):
    for feature in features:
        df = rolling_MMD_single(df, feature, window_size, mmd_type)
    df['average_MMD'] = df[[feature+"_MMD" for feature in features]].mean(axis=1)
    return df

#Function to get multivariate MMD scores from a single participant's dataframe
#TODO

#Function to compute correlation between two columns of a df
#returns (pearson_corr, p_value)
def corr(df, feature1, feature2, shift=None):
    #Accounting for NAN values in the first feature1 due to sliding window
    
    if shift!=None and shift!=0:
        arr1 = df[feature1].values[np.invert(np.isnan(df[feature1].values))]
        arr2 = df[feature2].values[np.invert(np.isnan(df[feature1].values))]
        if shift <0:
            arr1 = arr1[:shift]
            arr2 = arr2[-shift:]
        else:
            arr1 = arr1[shift:]
            arr2 = arr2[:-shift]
    else:
        arr1 = df[feature1].values[np.invert(np.isnan(df[feature1].values))]
        arr2 = df[feature2].values[np.invert(np.isnan(df[feature1].values))]

    if len(arr1)<2 or len(arr2)<2:
        return (None, None)

    return pearsonr(arr1, arr2)

#Function to Bonferroni correct a DF of P_values

def bonferroni_correct(df, p_value_column):
    alpha = 0.05
    df["p_value_bc"] = df[p_value_column] * len(df)
    df["reject_bc"] = 1*(df[p_value_column] < alpha / len(df))
    return df

#Function to Benjamini-Hochberg correct a DF of P_values
def bh_correct(df, p_value_column):
    is_reject, corrected_pvals, _, _ = multipletests(df[p_value_column], alpha=0.1, method='fdr_bh')
    df["reject_fdr"] = 1*is_reject
    df["p_value_fdr"] = corrected_pvals
    return df



def cohort_MMD(features, window_size, condition, stress_definition, shifts, rolling_mmd_method, mmd_func, method, windowed_method=False):

    participant_list = []
    corr_values = []
    p_values = []
    measures = []
    conditions = []
    shift_values =[]

    df = load_sleep_daily()
    
    if method == "percentile":
        condition = condition+"_percentile"

    for p in tqdm(df.participant_id.unique().tolist()):

        ##Grab an individual
        df1 = fetch_participant_df(df, p)
        
        #Fill missing dates and interpolate missing value
        df1 = lin_interp_multiple(missing_rows(df1), features)
        
        #Add Percentiles for each feature
        df1 = add_percentile_multiple(df1, features)
        
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

        #Compute MMD in feature domain
        df1 = rolling_mmd_method(df1, features, window_size, mmd_func)
        
        #Compute MMD in stress domain
        df1 = rolling_MMD_single(df1, condition, window_size, mmd_func)

        ##CHECK TO SEE IF WE HAVE ENOUGH TOTAL_STRESS EVENTS AND LENGTH OF DF IS AT LEAST A CERTAIN SIZE?

        if len(df1)>=10 and not(np.all(df1[condition].values == df1[condition].values[0])):
            for shift in shifts:
                test = corr(df1, "average_MMD", condition+"_MMD", shift)
                corr_values.append(test[0])
                p_values.append(test[1])
                participant_list.append(p)
                measures.append(features[0])
                conditions.append(condition)
                shift_values.append(shift)

    dict_test = {"participant":participant_list, "corr_values":corr_values, "p_values":p_values, "features":measures, "conditions":conditions, "shift":shift_values}

    dict_test = pd.DataFrame.from_dict(dict_test)

    dict_dropped = dict_test.dropna()
    dict_dropped = bonferroni_correct(dict_dropped, "p_values")
    dict_dropped = bh_correct(dict_dropped, "p_values")

    dict_dropped.sort_values(["p_value_fdr"], ascending=True)
    
    return dict_dropped


def fill_column(features, window_size, condition, stress_definition, method, shift):
    if condition == "bad_sleep_rolling":
        windowed_method = True
    else:
        windowed_method = False
        
    individual_features = [[f] for f in features]

    for f in individual_features:
        if f == individual_features[0]:
            oura_total_stress = cohort_MMD(f, window_size, condition, stress_definition, shift,  rolling_MMD_multiple, mmd_linear, method)
        else:
            oura_total_stress = oura_total_stress.append(cohort_MMD(f, window_size, condition, stress_definition,shift, rolling_MMD_multiple, mmd_linear, method))

    candidate_feature = []
    candidate_stress = []
    number_rejected = []
    total_tested = []
    correlation_values = []
    participants = []

    for f in np.unique(oura_total_stress.features.values):
        for s in np.unique(oura_total_stress.conditions.values):
            df_sub = oura_total_stress[(oura_total_stress["features"] == f) & (oura_total_stress["conditions"] == s) ]
            df_rejected = df_sub[df_sub["reject_fdr"]==1]

            participants.append(df_rejected.participant.values)
            candidate_feature.append(f)
            candidate_stress.append(s)
            number_rejected.append(len(df_rejected))
            total_tested.append(len(df_sub))
            correlation_values.append(df_rejected.corr_values.values)

    df =  {"participants":participants,"candidate_feature":candidate_feature, "candidate_stress":candidate_stress, "number_rejected":number_rejected, "total_tested":total_tested, "correlation_values":correlation_values}

    df = pd.DataFrame.from_dict(df)
    return df

def CCIT_test(X, Y, Z, data, boolean=True, **kwargs):
    r"""
    Computes Pearson correlation coefficient and p-value for testing non-correlation.
    Should be used only on continuous data. In case when :math:`Z != \null` uses
    linear regression and computes pearson coefficient on residuals.

    Parameters
    ----------
    X: str
        The first variable for testing the independence condition X \u27C2 Y | Z

    Y: str
        The second variable for testing the independence condition X \u27C2 Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X \u27C2 Y | Z

    data: pandas.DataFrame
        The dataset in which to test the indepenedence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the pearson correlation coefficient and p_value
            of the test.

    Returns
    -------
    CI Test results: tuple or bool
        If boolean=True, returns True if p-value >= significance_level, else False. If
        boolean=False, returns a tuple of (Pearson's correlation Coefficient, p-value)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    [2] https://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
    """
    # Step 1: Test if the inputs are correct
    num_cores = multiprocessing.cpu_count()
    
    if not hasattr(Z, "__iter__"):
        raise ValueError(f"Variable Z. Expected type: iterable. Got type: {type(Z)}")
    else:
        Z = list(Z)

    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            f"Variable data. Expected type: pandas.DataFrame. Got type: {type(data)}"
        )

    # Step 2: If Z is empty compute a non-conditional test.
    if len(Z) == 0:

        p_value = CCIT.CCIT(np.expand_dims(data.loc[:, X].values, axis=1),np.expand_dims(data.loc[:, Y].values, axis=1), None, num_iter = 30, bootstrap = True, nthread = num_cores)    
    
    else:
        p_value = CCIT.CCIT(np.expand_dims(data.loc[:, X].values, axis=1),np.expand_dims(data.loc[:, Y].values, axis=1), data.loc[:, Z].values, num_iter = 30, bootstrap = True, nthread = num_cores)  

    if boolean:
        if p_value >= kwargs["significance_level"]:
            return True
        else:
            return False
    else:
        return p_value



def population_graph(df, features, method, stress_definition,downsample=0,missing_node=0, max_cond_vars=2, significance_level = 0.05):
    participant_list = df.participant_id.unique().tolist()

    list_dfs = []
    
    population_features = ["p_id", "dem_age", "dem_gender", "height", "weight", "ace_score", "hx_any", "mhx_any", "ptsd_score", "life_events_score"]

    features_pop = features + population_features

    if missing_node == 1:
        features_pop = features_pop + ["missingness"]
        features = features + ["missingness"]

    for p in tqdm(participant_list):
            ##Grab an individual
        df1 = fetch_participant_df(df, p)

        #Add stress definition
        if stress_definition != None:
            if stress_definition not in df1.columns:
                if stress_definition == "total_stress":
                    df1 = total_stress(df1)


        #Add missing rows
        df1 = missing_rows(df1)

        if missing_node == 1:
            #Add missingness node
            df1["missingness"] = df1.isnull().astype(bool).sum(axis=1)

        #interpolate missing values
        df1 = lin_interp_multiple(df1, features)

        #if anything still missing, fill with column mean
        df1[features] = df1[features].fillna(df1[features].mean())

        #downsample
        if downsample!=0:
            df1 = downsample_rows(df1,downsample)

        #Z score normalize features
        df1 = z_score(df1, features)

        list_dfs.append(df1)
    
    
    #Concatenate all rows
    final_df = pd.concat(list_dfs, axis = 0)

    #grab features we want plus p_id    
    data = final_df[features_pop]
    #Interpolate anything else missing and drop values
    c = PC(data.interpolate().dropna())
    if method == "CCIT":
        model = c.estimate(ci_test = CCIT_test, max_cond_vars=max_cond_vars, significance_level=significance_level)
    else:
        model = c.estimate(ci_test = "pearsonr", variant = "parallel", max_cond_vars=max_cond_vars, significance_level=significance_level)
    return model


def individual_graph(df, features, method, stress_definition, downsample=0,missing_node=0,max_cond_vars=2, significance_level = 0.05):
    participant_list = df.participant_id.unique().tolist()

    dict_models = {}
    if 'steps' in features:
        df = load_garmin_subset()
    

    participant_list = df.participant_id.unique().tolist()
    
    cutoff = 10

    for p in (participant_list):
        p_features = features.copy()
        df1 = fetch_participant_df(df, p)
        
        if len(df1)<cutoff:
            continue
        
        count_flag = 0
        for feature in features:
            if df1[feature].nunique() <2:
                p_features.remove(feature)
        

        #Add stress definition
        if stress_definition != None:
            if stress_definition not in df1.columns:
                if stress_definition == "total_stress":
                    df1 = total_stress(df1)

        #Add missing rows
        df1 = missing_rows(df1)

        if missing_node == 1:
            #Add missingness node
            df1["missingness"] = df1.isnull().astype(bool).sum(axis=1)
        
            p_features = p_features + ["missingness"]

        #Z score normalize features
        df1 = z_score(df1, p_features)

        #interpolate missing values
        df1 = lin_interp_multiple(df1, p_features)

        #if anything still missing, fill with column mean
        df1[p_features] = df1[p_features].fillna(df1[p_features].mean())

            #downsample
        if downsample!=0:
            cutoff = 5
            df1 = downsample_rows(df1,downsample)

        
        data = df1[p_features]
        data = data.interpolate().dropna()


        if len(data)>=cutoff:
            c = PC(data)
            if method == "CCIT":
                model = c.estimate(ci_test = CCIT_test, max_cond_vars=max_cond_vars, significance_level=significance_level)
            else:
                model = c.estimate(ci_test = "pearsonr", variant = "parallel", max_cond_vars=max_cond_vars, significance_level=significance_level)
            dict_models[p] = model
            print(model.edges())
            
    return dict_models
    
        
def individual_graph_stress(df, features, method, stress_window, stress_cutoff, no_stress_cutoff, buffer, missing_node=0,max_cond_vars=2, significance_level = 0.05):
    cutoff = 10

            #DROP THE STRESS WINDOW FROM FEATURES
    if stress_window in features:
        features.remove(stress_window)

    if stress_window == "total_stress":
        for i in ["daily_stressed", "daily_control", "shift_stress"]:
            if i in features:
                features.remove(i)

    #if stress_window == "shift_stress":
    #    if "daily_shifts" in features:
    #        features.remove("daily_shifts")

    if stress_window == "daily_control" or stress_window == "control_cat":
        if "daily_stressed" in features:
            features.remove("daily_stressed")

    if 'steps' in features:
        df = load_garmin_subset()

    participant_list = df.participant_id.unique().tolist()

    dict_models = {}
    if missing_node == 1:
            features = features + ["missingness"]

    for p in tqdm(participant_list):
            ##Grab an individual
        p_features = features.copy()
        df1 = fetch_participant_df(df, p)

        if len(df1)<cutoff:
            continue

        count_flag = 0
        for feature in features:
            if df1[feature].nunique() <2:
                p_features.remove(feature)

        #Add missing rows
        df1 = missing_rows(df1)

        if missing_node == 1:
            #Add missingness node
            df1["missingness"] = df1.isnull().astype(bool).sum(axis=1)


        #Z Score Normalize
        df1 = z_score(df1, p_features)

        #interpolate missing values
        df1 = lin_interp_multiple(df1, p_features)

        #if anything still missing, fill with column mean
        df1[p_features] = df1[p_features].fillna(df1[p_features].mean())
        
        #BACKFILL PPE and COVID EXPOSURE
        if stress_window == "covid_past_24" or stress_window == "ppe_6":
            df1[stress_window] = df1[stress_window].bfill(limit=7)

        #Pad Stress Windows with a buffer
        #df1 = pad_stress(df1, buffer, stress_window)


        if stress_window == "total_stress":
            df1 = total_stress_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "daily_control":
            df1 = daily_control_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "control_cat":
            df1 = control_cat_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "ppe_6":
            df1 = ppe_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "pss4_score":
            df1 = pss4_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "hrv_binary":
            df1 = hrv_binarize(df1, stress_cutoff, no_stress_cutoff)
            
        df_nostress, df_stress = iid_windows(df1, buffer, p_features, stress_window)
        dict_models[p] =[]

        if len(df_nostress)< len(df_stress):
            df_stress = df_stress.sample(len(df_nostress))
    
        elif len(df_nostress)> len(df_stress):
            df_nostress = df_nostress.sample(len(df_stress))

        if len(df_nostress)>=cutoff and len(df_stress)>=cutoff:
            data_nostress = df_nostress[p_features]
            data_nostress = data_nostress.interpolate().dropna()
            c = PC(data_nostress)
            if method == "CCIT":
                model = c.estimate(ci_test = CCIT_test, max_cond_vars=max_cond_vars, significance_level=significance_level)
            else:
                model = c.estimate(ci_test = "pearsonr", variant = "parallel", max_cond_vars=max_cond_vars, significance_level=significance_level)

            weighted_model = add_weights(model,df_nostress)
            dict_models[p].append(("no_stress",weighted_model))
            print("No Stress")
            print(model.edges())


            data_stress = df_stress[p_features]
            data_stress = data_stress.interpolate().dropna()
            c = PC(data_stress)
            if method == "CCIT":
                model = c.estimate(ci_test = CCIT_test, max_cond_vars=max_cond_vars, significance_level=significance_level)
            else:
                model = c.estimate(ci_test = "pearsonr", variant = "parallel", max_cond_vars=max_cond_vars, significance_level=significance_level)
            weighted_model = add_weights(model,df_stress)
            dict_models[p].append (("stress",weighted_model))
            print("Stress")
            print(model.edges())
                    
    return dict_models    


def individual_granger_graph(df, features, stress_definition, downsample=0,missing_node=0,max_cond_vars=2, significance_level = 0.05):
    dict_models = {}
    if 'steps' in features:
        df = load_garmin_subset()


    participant_list = df.participant_id.unique().tolist()

    cutoff = 25

    for p in tqdm(participant_list):
        p_features = features.copy()
        df1 = fetch_participant_df(df, p)

        if len(df1)<cutoff:
            continue

        count_flag = 0
        for feature in features:
            if df1[feature].nunique() <2:
                p_features.remove(feature)

        #Add stress definition
        if stress_definition != None:
            if stress_definition not in df1.columns:
                if stress_definition == "total_stress":
                    df1 = total_stress(df1)

        #Add missing rows
        df1 = missing_rows(df1)

        if missing_node == 1:
            #Add missingness node
            df1["missingness"] = df1.isnull().astype(bool).sum(axis=1)

            p_features = p_features + ["missingness"]

        #Z score normalize features
        df1 = z_score(df1, p_features)

        #interpolate missing values
        df1 = lin_interp_multiple(df1, p_features)

        #if anything still missing, fill with column mean
        df1[p_features] = df1[p_features].fillna(df1[p_features].mean())

            #downsample
        if downsample!=0:
            cutoff = 5
            df1 = downsample_rows(df1,downsample)


        data = df1[p_features]
        data = data.interpolate().dropna()


        if len(data)>=cutoff:
            
            #Get Non-Stationary signals
            not_stationary = []
            for column in data.columns:
                result = adfuller(data[column].values, autolag='BIC')
                p_val = result[1]
                if p_val>0.05:
                    not_stationary.append(column)
            
            #First order difference until column is sationary
            while len(not_stationary)!=0:
                for col in not_stationary:
                    data[col] = data[col].diff().bfill()
                    result = adfuller(data[col].values, autolag='BIC')
                    p_val = result[1]
                    if p_val<0.05:
                        not_stationary.remove(col)
                        
            try:
                model = VAR(data)
                model_fit = model.fit(maxlags = 2, ic = 'bic', trend = 'c')
            except:
                continue
            
            edges = []
            if model_fit.k_ar!=0:
                
                for column in data.columns:
                    for col in data.columns:
                        if column != col:
                            test1 = model_fit.test_causality(col, causing=[column], kind='f', signif=0.05)
                            if (test1.pvalue)<0.05:
                                #print(column +" causes "+col +": " + str(test1.pvalue))
                                edges.append((column, col))
                dict_models[p] = edges
                print(edges)
                
    return dict_models



def individual_graph_random(df, features, method, stress_window, stress_cutoff, no_stress_cutoff, bootstraps,buffer, missing_node=0,max_cond_vars=2, significance_level = 0.05):
    cutoff = 10

            #DROP THE STRESS WINDOW FROM FEATURES
    if stress_window in features:
        features.remove(stress_window)

    if stress_window == "total_stress":
        for i in ["daily_stressed", "daily_control", "shift_stress"]:
            if i in features:
                features.remove(i)

    #if stress_window == "shift_stress":
    #    if "daily_shifts" in features:
    #        features.remove("daily_shifts")

    if stress_window == "daily_control" or stress_window == "control_cat":
        if "daily_stressed" in features:
            features.remove("daily_stressed")

    if 'steps' in features:
        df = load_garmin_subset()

    participant_list = df.participant_id.unique().tolist()

    dict_models = {}
    if missing_node == 1:
            features = features + ["missingness"]

    for p in tqdm(participant_list):
            ##Grab an individual
        p_features = features.copy()
        df1 = fetch_participant_df(df, p)

        if len(df1)<cutoff:
            continue

        count_flag = 0
        for feature in features:
            if df1[feature].nunique() <2:
                p_features.remove(feature)

        #Add missing rows
        df1 = missing_rows(df1)

        if missing_node == 1:
            #Add missingness node
            df1["missingness"] = df1.isnull().astype(bool).sum(axis=1)


        #Z Score Normalize
        df1 = z_score(df1, p_features)

        #interpolate missing values
        df1 = lin_interp_multiple(df1, p_features)

        #if anything still missing, fill with column mean
        df1[p_features] = df1[p_features].fillna(df1[p_features].mean())
        
        #BACKFILL PPE and COVID EXPOSURE
        if stress_window == "covid_past_24" or stress_window == "ppe_6":
            df1[stress_window] = df1[stress_window].bfill(limit=7)

        #Pad Stress Windows with a buffer
        #df1 = pad_stress(df1, buffer, stress_window)


        if stress_window == "total_stress":
            df1 = total_stress_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "daily_control":
            df1 = daily_control_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "control_cat":
            df1 = control_cat_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "ppe_6":
            df1 = ppe_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "pss4_score":
            df1 = pss4_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "hrv_binary":
            df1 = hrv_binarize(df1, stress_cutoff, no_stress_cutoff)
            
        dict_models[p] =[]
    
        for i in range(bootstraps):
            
            df_nostress, df_stress = iid_windows(df1, buffer, p_features, stress_window)

            #COPY DATA AS WE WILL DROP ROWS AFTER SAMPLING
            df_copy = df1.copy()
            
            #GET NUMBER OF WINDOWS TO SAMPLE
            num_stress = len(df_stress)
            num_nostress = len(df_nostress)

            num_sample = min(num_stress, num_nostress)

            #SAMPLE STRESS AND REMOVE
            df_stress = df_copy.sample(num_sample)

            df_copy = df_copy.drop(df_stress.index)

            #SAMPLE NO STRESS FROM REMAINING ROWS
            df_nostress = df_copy.sample(num_sample)


            if len(df_nostress)>=cutoff and len(df_stress)>=cutoff:
                data_nostress = df_nostress[p_features]
                data_nostress = data_nostress.interpolate().dropna()
                c = PC(data_nostress)
                if method == "CCIT":
                    model = c.estimate(ci_test = CCIT_test, max_cond_vars=max_cond_vars, significance_level=significance_level)
                else:
                    model = c.estimate(ci_test = "pearsonr", variant = "parallel", max_cond_vars=max_cond_vars, significance_level=significance_level)

                weighted_model = add_weights(model,df_nostress)
                dict_models[p].append(("no_stress",weighted_model))
                #print("No Stress")
                #print(model.edges())

                data_stress = df_stress[p_features]
                data_stress = data_stress.interpolate().dropna()
                c = PC(data_stress)
                if method == "CCIT":
                    model = c.estimate(ci_test = CCIT_test, max_cond_vars=max_cond_vars, significance_level=significance_level)
                else:
                    model = c.estimate(ci_test = "pearsonr", variant = "parallel", max_cond_vars=max_cond_vars, significance_level=significance_level)
                weighted_model = add_weights(model,df_stress)
                dict_models[p].append (("stress",weighted_model))
                #print("Stress")
                #print(model.edges())
                    
    return dict_models 


def individual_graph_multiple(df, features, method, stress_window, stress_cutoff, no_stress_cutoff, bootstraps,buffer, missing_node=0,max_cond_vars=2, significance_level = 0.05):
    cutoff = 10

            #DROP THE STRESS WINDOW FROM FEATURES
    if stress_window in features:
        features.remove(stress_window)

    if stress_window == "total_stress":
        for i in ["daily_stressed", "daily_control", "shift_stress"]:
            if i in features:
                features.remove(i)

    #if stress_window == "shift_stress":
    #    if "daily_shifts" in features:
    #        features.remove("daily_shifts")

    if stress_window == "daily_control" or stress_window == "control_cat":
        if "daily_stressed" in features:
            features.remove("daily_stressed")

    if 'steps' in features:
        df = load_garmin_subset()

    participant_list = df.participant_id.unique().tolist()

    dict_models = {}
    if missing_node == 1:
            features = features + ["missingness"]

    for p in tqdm(participant_list):
            ##Grab an individual
        p_features = features.copy()
        df1 = fetch_participant_df(df, p)

        if len(df1)<cutoff:
            continue

        count_flag = 0
        for feature in features:
            if df1[feature].nunique() <2:
                p_features.remove(feature)

        #Add missing rows
        df1 = missing_rows(df1)

        if missing_node == 1:
            #Add missingness node
            df1["missingness"] = df1.isnull().astype(bool).sum(axis=1)


        #Z Score Normalize
        df1 = z_score(df1, p_features)

        #interpolate missing values
        df1 = lin_interp_multiple(df1, p_features)

        #if anything still missing, fill with column mean
        df1[p_features] = df1[p_features].fillna(df1[p_features].mean())
        
        #BACKFILL PPE and COVID EXPOSURE
        if stress_window == "covid_past_24" or stress_window == "ppe_6":
            df1[stress_window] = df1[stress_window].bfill(limit=7)

        #Pad Stress Windows with a buffer
        #df1 = pad_stress(df1, buffer, stress_window)


        if stress_window == "total_stress":
            df1 = total_stress_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "daily_control":
            df1 = daily_control_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "control_cat":
            df1 = control_cat_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "ppe_6":
            df1 = ppe_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "pss4_score":
            df1 = pss4_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "hrv_binary":
            df1 = hrv_binarize(df1, stress_cutoff, no_stress_cutoff)
            
        dict_models[p] =[]
    
        for i in range(bootstraps):
            
            df_nostress, df_stress = iid_windows(df1, buffer, p_features, stress_window)
            
            #GET NUMBER OF WINDOWS TO SAMPLE
            num_stress = len(df_stress)
            num_nostress = len(df_nostress)

            num_sample = min(num_stress, num_nostress)

            #SAMPLE STRESS 
            df_stress_sample = df_stress.sample(num_sample)

            #SAMPLE NO STRESS
            df_nostress_sample = df_nostress.sample(num_sample)


            if len(df_nostress_sample)>=cutoff and len(df_stress_sample)>=cutoff:
                data_nostress = df_nostress_sample[p_features]
                data_nostress = data_nostress.interpolate().dropna()
                c = PC(data_nostress)
                if method == "CCIT":
                    model = c.estimate(ci_test = CCIT_test, max_cond_vars=max_cond_vars, significance_level=significance_level)
                else:
                    model = c.estimate(ci_test = "pearsonr", variant = "parallel", max_cond_vars=max_cond_vars, significance_level=significance_level)

                weighted_model = add_weights(model,df_nostress_sample)
                dict_models[p].append(("no_stress",weighted_model))
                #print("No Stress")
                #print(model.edges())

                data_stress = df_stress_sample[p_features]
                data_stress = data_stress.interpolate().dropna()
                c = PC(data_stress)
                if method == "CCIT":
                    model = c.estimate(ci_test = CCIT_test, max_cond_vars=max_cond_vars, significance_level=significance_level)
                else:
                    model = c.estimate(ci_test = "pearsonr", variant = "parallel", max_cond_vars=max_cond_vars, significance_level=significance_level)
                weighted_model = add_weights(model,df_stress_sample)
                dict_models[p].append (("stress",weighted_model))
                #print("Stress")
                #print(model.edges())
                    
    return dict_models 


def individual_graph_reference(df, features, method, stress_window, stress_cutoff, no_stress_cutoff, bootstraps,buffer, missing_node=0,max_cond_vars=2, significance_level = 0.05):
    cutoff = 10

            #DROP THE STRESS WINDOW FROM FEATURES
    if stress_window in features:
        features.remove(stress_window)

    if stress_window == "total_stress":
        for i in ["daily_stressed", "daily_control", "shift_stress"]:
            if i in features:
                features.remove(i)
    
    #if stress_window == "any_stress":
    #    for i in ["daily_stressed", "daily_shifts", "shift_stress"]:
    #        if i in features:
    #            features.remove(i)

    #if stress_window == "shift_stress":
    #    if "daily_shifts" in features:
    #        features.remove("daily_shifts")

    if stress_window == "daily_control" or stress_window == "control_cat":
        if "daily_stressed" in features:
            features.remove("daily_stressed")

    if 'steps' in features:
        df = load_garmin_subset()

    participant_list = df.participant_id.unique().tolist()

    dict_models = {}
    dict_reference = {}
    dict_features = {}

    if missing_node == 1:
        features = features + ["missingness"]

    for p in tqdm(participant_list):
            ##Grab an individual
        p_features = features.copy()
        df1 = fetch_participant_df(df, p)

        if len(df1)<cutoff:
            continue

        count_flag = 0
        for feature in features:
            if df1[feature].nunique() <2:
                p_features.remove(feature)

        #Add missing rows
        df1 = missing_rows(df1)

        if missing_node == 1:
            #Add missingness node
            df1["missingness"] = df1.isnull().astype(bool).sum(axis=1)


        #Z Score Normalize
        df1 = z_score(df1, p_features)

        #interpolate missing values
        df1 = lin_interp_multiple(df1, p_features)

        #if anything still missing, fill with column mean
        df1[p_features] = df1[p_features].fillna(df1[p_features].mean())
        
        #BACKFILL PPE and COVID EXPOSURE
        if stress_window == "covid_past_24" or stress_window == "ppe_6":
            df1[stress_window] = df1[stress_window].bfill(limit=7)

        #Pad Stress Windows with a buffer
        #df1 = pad_stress(df1, buffer, stress_window)


        if stress_window == "total_stress":
            df1 = total_stress_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "daily_control":
            df1 = daily_control_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "control_cat":
            df1 = control_cat_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "ppe_6":
            df1 = ppe_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "pss4_score":
            df1 = pss4_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "hrv_binary":
            df1 = hrv_binarize(df1, stress_cutoff, no_stress_cutoff)
        elif stress_window == "any_stress":
            df1 = hrv_binarize(df1, stress_cutoff, no_stress_cutoff)
            df1 = any_stress(df1)
            
        
        df_nostress, df_stress = iid_windows(df1, buffer, p_features, stress_window)

        #GET NUMBER OF WINDOWS TO SAMPLE
        num_stress = len(df_stress)
        num_nostress = len(df_nostress)
        num_sample = min(num_stress, num_nostress)


        dict_reference[p] = []
        dict_models[p] =[]

        for i in range(bootstraps):
            
            #REFERENCE GRAPHS - RANDOMPLY SAMPLED w/ SAME PROPORTIONS
            #COPY DATA AS WE WILL DROP ROWS AFTER SAMPLING
            df_copy = df1.copy()

            #SAMPLE A AND REMOVE
            df_a = df_copy.sample(num_sample)
            df_copy = df_copy.drop(df_a.index)

            #SAMPLE B FROM REMAINING ROWS
            df_b = df_copy.sample(num_sample)
            
            if len(df_a)>=cutoff and len(df_b)>=cutoff:
                data_a = df_a[p_features]
                data_a = data_a.interpolate().dropna()
                c = PC(data_a)
                if method == "CCIT":
                    model = c.estimate(ci_test = CCIT_test, max_cond_vars=max_cond_vars, significance_level=significance_level)
                else:
                    model = c.estimate(ci_test = "pearsonr", variant = "parallel", max_cond_vars=max_cond_vars, significance_level=significance_level)

                weighted_model = add_weights(model,df_a)
                dict_reference[p].append(("a",weighted_model))
                #print("No Stress")
                #print(model.edges())

                data_b = df_b[p_features]
                data_b = data_b.interpolate().dropna()
                c = PC(data_b)
                if method == "CCIT":
                    model = c.estimate(ci_test = CCIT_test, max_cond_vars=max_cond_vars, significance_level=significance_level)
                else:
                    model = c.estimate(ci_test = "pearsonr", variant = "parallel", max_cond_vars=max_cond_vars, significance_level=significance_level)
                weighted_model = add_weights(model,df_b)
                dict_reference[p].append (("b",weighted_model))
                #print("Stress")
                #print(model.edges())

            #GET STRESS / NON STRESS WINDOWS

            #SAMPLE STRESS 
            df_stress_sample = df_stress.sample(num_sample)

            #SAMPLE NO STRESS
            df_nostress_sample = df_nostress.sample(num_sample)


            if len(df_nostress_sample)>=cutoff and len(df_stress_sample)>=cutoff:
                data_nostress = df_nostress_sample[p_features]
                data_nostress = data_nostress.interpolate().dropna()
                c = PC(data_nostress)
                if method == "CCIT":
                    model = c.estimate(ci_test = CCIT_test, max_cond_vars=max_cond_vars, significance_level=significance_level)
                else:
                    model = c.estimate(ci_test = "pearsonr", variant = "parallel", max_cond_vars=max_cond_vars, significance_level=significance_level)

                weighted_model = add_weights(model,df_nostress_sample)
                dict_models[p].append(("no_stress",weighted_model))
                #print("No Stress")
                #print(model.edges())

                data_stress = df_stress_sample[p_features]
                data_stress = data_stress.interpolate().dropna()
                c = PC(data_stress)
                if method == "CCIT":
                    model = c.estimate(ci_test = CCIT_test, max_cond_vars=max_cond_vars, significance_level=significance_level)
                else:
                    model = c.estimate(ci_test = "pearsonr", variant = "parallel", max_cond_vars=max_cond_vars, significance_level=significance_level)
                weighted_model = add_weights(model,df_stress_sample)
                dict_models[p].append (("stress",weighted_model))
                #print("Stress")
                #print(model.edges())
                dict_features[p] = p_features
                    
    return dict_reference, dict_models, dict_features