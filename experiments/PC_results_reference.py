import os
import time
import sys
import pickle

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0,'..')

import argparse
from random import SystemRandom

from src.definitions import *
from src.plotting import *
from src.processing import *
from src.stats import *

if __name__ == '__main__':


    stress_definitions = ["daily_stressed", "daily_shifts", "shift_stress", "hrv_binary"]

    oura = ['breath_average','efficiency','hr_average','hr_lowest','onset_latency','score','temperature_delta','temperature_trend_deviation','total','log_hrv']
    surveys = ['daily_stressed','daily_control','daily_shifts','sam8','shift_stress','promis_sd5','EBTBP','phq9_score','gad7_score','promis_sri_score','pss4_score','promis_es_score',"fss_score"]

    for stress_definition in stress_definitions:
        print(stress_definition)
        oura_copy = oura.copy()
        surveys_copy = surveys.copy()

        if stress_definition == "daily_stressed":
            surveys_copy.remove("daily_stressed")
        elif stress_definition == "daily_shifts":
            surveys_copy.remove("daily_shifts")
        elif stress_definition == "shift_stress":
            surveys_copy.remove("shift_stress")

        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        filename_a =  os.path.join(parent_dir, "results", "PC",'reference',stress_definition)
        filename_b = "_window_"+stress_definition+"_0.75_0.25_bootstraps_100_buffer_3_miss_0_mcv_2_sig_0.05_pearsonr.pkl"

        oura_surveys_reference = filename_a+"oura+surveys"+filename_b   

        with open(oura_surveys_reference, 'rb') as f:
            dict_reference,dict_models, dict_features  = pickle.load(f)

        df = compare_reference(dict_reference, dict_models, oura_copy+surveys_copy, stress_definition)

        # Saving Val and Test set results
        method_folder = "PC"
        fancy_string = f"results_df"

        #results path
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        if not os.path.exists(os.path.join(parent_dir, "results", "PC","reference", stress_definition)):
            os.makedirs(os.path.join(parent_dir, "results", "PC",'reference', stress_definition))
    
        #Saving Model
        result_path = os.path.join(parent_dir, "results", "PC","reference", stress_definition, fancy_string+".pkl")

        #get a list of tuples from the model and save
        with open(result_path, 'wb') as fp:
            pickle.dump(df, fp)

    