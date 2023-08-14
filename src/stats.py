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
import os

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

from time import sleep
from random import random
from multiprocessing import Pool, Process, Manager, set_start_method

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




def worker_func(p,dict_models,dict_reference, dict_features, df,features, method, stress_window, stress_cutoff, 
                no_stress_cutoff, bootstraps, buffer, missing_node, max_cond_vars, significance_level):
        ##Grab an individual
    p_features = features.copy()
    df1 = fetch_participant_df(df, p)

    print("Started: ", p)

    for feature in features:
        if df1[feature].nunique() < 5:
            p_features.remove(feature)

    #Add missing rows
    df1 = missing_rows(df1)

    #Z Score Normalize
    df1 = z_score(df1, p_features)

    df1_mean = df1[p_features].mean()

    #interpolate missing values
    df1 = lin_interp_multiple(df1, p_features)

    #if anything still missing, fill with column mean
    df1[p_features] = df1[p_features].fillna(df1_mean)

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


    #df_nostress, df_stress = iid_windows(df1, buffer, p_features, stress_window)

    #GET NUMBER OF WINDOWS TO SAMPLE
    #num_stress = len(df_stress)
    #num_nostress = len(df_nostress)
    #num_sample = min(num_stress, num_nostress)


    #num_sample = 10
    cutoff = 10
    mu, sigma = 0, 0.001 
    
    for i in range(bootstraps):
        #print("Working on:", p)
        df_nostress, df_stress = iid_windows(df1, buffer, p_features, stress_window)
        num_stress = len(df_stress)
        num_nostress = len(df_nostress)
        num_sample = min(num_stress, num_nostress)
        #print(num_sample)
        if len(df1)>=cutoff and len(df_nostress)>=cutoff and len(df_stress)>=cutoff:
            #REFERENCE GRAPHS - RANDOMPLY SAMPLED w/ SAME PROPORTIONS
            #COPY DATA AS WE WILL DROP ROWS AFTER SAMPLING
            #print("HERE")
            df_copy = df1.copy()

            #SAMPLE A AND REMOVE
            df_a = df_copy.sample(num_sample)
            df_copy = df_copy.drop(df_a.index)

            #SAMPLE B FROM REMAINING ROWS
            df_b = df_copy.sample(num_sample)

            #p_features = features.copy()
            #for feature in features:
            #    if df_a[feature].nunique() <2:
            #        p_features.remove(feature)

            data_a = df_a[p_features]
            data_a = data_a.interpolate().dropna()
            
            # creating a noise with the same dimension as the dataset
            noise = np.random.normal(mu, sigma, data_a.values.shape) 
            added = data_a + noise
            c = PC(added)
            if method == "CCIT":
                model = c.estimate(ci_test = CCIT_test, max_cond_vars=max_cond_vars, significance_level=significance_level)
            else:
                model = c.estimate(ci_test = "pearsonr", variant = "stable", max_cond_vars=max_cond_vars, significance_level=significance_level, show_progress=False)

            weighted_model = add_weights(model,df_a)
            dict_reference[p] +=[("a",weighted_model)]
            #print("A")
            #print(model.edges())

            #p_features = features.copy()
            #for feature in features:
            #    if df_b[feature].nunique() <2:
            #        p_features.remove(feature)

            data_b = df_b[p_features]
            data_b = data_b.interpolate().dropna()
            
            # creating a noise with the same dimension as the dataset
            noise = np.random.normal(mu, sigma, data_b.values.shape) 
            added = data_b + noise
            c = PC(added)
            if method == "CCIT":
                model = c.estimate(ci_test = CCIT_test, max_cond_vars=max_cond_vars, significance_level=significance_level, n_jobs=5)
            else:
                model = c.estimate(ci_test = "pearsonr", variant = "stable", max_cond_vars=max_cond_vars, significance_level=significance_level, n_jobs=5, show_progress=False)
            weighted_model = add_weights(model,df_b)
            dict_reference[p] += [("b",weighted_model)]
            #print("B")
            #print(model.edges())

            #GET STRESS / NON STRESS WINDOWS

            #SAMPLE STRESS 
            df_stress_sample = df_stress.sample(num_sample)

            #SAMPLE NO STRESS
            df_nostress_sample = df_nostress.sample(num_sample)

            #p_features = features.copy()

            #for feature in features:
            #    if df_nostress[feature].nunique() <2:
            #        p_features.remove(feature)

            data_nostress = df_nostress_sample[p_features]
            data_nostress = data_nostress.interpolate().dropna()
            
            # creating a noise with the same dimension as the dataset
            noise = np.random.normal(mu, sigma, data_nostress.values.shape) 
            added = data_nostress + noise
            c = PC(added)
            if method == "CCIT":
                model = c.estimate(ci_test = CCIT_test, max_cond_vars=max_cond_vars, significance_level=significance_level)
            else:
                model = c.estimate(ci_test = "pearsonr", variant = "stable", max_cond_vars=max_cond_vars, significance_level=significance_level, n_jobs=5, show_progress=False)

            weighted_model = add_weights(model,df_nostress_sample)
            dict_models[p] += [("no_stress",weighted_model)]
            #print("No Stress")
            #print(model.edges())

            #p_features = features.copy()

            #for feature in features:
            #    if df_stress[feature].nunique() <2:
            #        p_features.remove(feature)

            data_stress = df_stress_sample[p_features]
            data_stress = data_stress.interpolate().dropna()

            # creating a noise with the same dimension as the dataset
            noise = np.random.normal(mu, sigma, data_stress.values.shape) 
            added = data_stress + noise
            c = PC(added)
            if method == "CCIT":
                model = c.estimate(ci_test = CCIT_test, max_cond_vars=max_cond_vars, significance_level=significance_level)
            else:
                model = c.estimate(ci_test = "pearsonr", variant = "stable", max_cond_vars=max_cond_vars, significance_level=significance_level, n_jobs=5, show_progress=False)
            weighted_model = add_weights(model,df_stress_sample)
            dict_models[p]+=[("stress",weighted_model)]
            #print("Stress")
            #print(model.edges())
    dict_features[p] = p_features

    print("Finished: ", p)

def individual_graph_reference3(df, features, method, stress_window, stress_cutoff, no_stress_cutoff, bootstraps,buffer, missing_node=0,max_cond_vars=2, significance_level = 0.05):

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

    if stress_window == "shift_stress":
        if "daily_shifts" in features:
            features.remove("daily_shifts")

    if stress_window == "daily_control" or stress_window == "control_cat":
        if "daily_stressed" in features:
            features.remove("daily_stressed")

    if 'steps' in features:
        df = load_garmin_subset()

    participant_list = df.participant_id.unique().tolist()
    
    manager = Manager()

    dict_models = manager.dict()
    dict_reference = manager.dict()
    dict_features = manager.dict()
    
    for p in participant_list:
        dict_reference[p] = []
        dict_models[p] =[]
    
    # get number of cpus available to job
    n_cores = multiprocessing.cpu_count()

    print(n_cores)

    # created pool running maximum n_cores
    #set_start_method('spawn', force=True)
    pool = Pool(n_cores)
    # Execute the folding task in parallel
    for p in (participant_list):
        pool.apply_async(worker_func, args=(p,dict_models,
                    dict_reference,
                    dict_features,
                    df, 
                    features, 
                    method, 
                    stress_window,
                    stress_cutoff,
                    no_stress_cutoff,
                    bootstraps, 
                    buffer,
                    missing_node,
                    max_cond_vars, 
                    significance_level))

    # Tell the pool that there are no more tasks to come and join
    pool.close()
    pool.join()
            
                    
    return dict_reference, dict_models, dict_features