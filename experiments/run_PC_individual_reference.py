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

parser = argparse.ArgumentParser('hypotheses')

parser.add_argument('--features', type=str, default='all', help="Feature sets to include:")
parser.add_argument('--method', type=str, default='pearsonr', help="use PearsonR or CCIT")
parser.add_argument('--stress_window', type =str, default="daily_stressed", help="what stress definition to add")
parser.add_argument('--stress_cutoff', type =float, default=0.75, help="Upper cutoff for ON stress, percentile")
parser.add_argument('--no_stress_cutoff', type =float, default=0.25, help="Lower cutoff for OFF stress, percentile")
parser.add_argument('--bootstraps', type=int, default=100, help="number of random bootstraps to try")
parser.add_argument('--buffer', type=int, default=1, help="number of days to buffer around a stress event")
parser.add_argument('--missing_node', type=int, default=0, help="flag to include missingness node or not")
parser.add_argument('--max_cond_vars', type=int, default=2, help="Max conditional variables to test")
parser.add_argument('--significance_level', type=float, default=0.05, help="significance threshold")

args = parser.parse_args()


#####################################################################################################


if __name__ == '__main__':


    start = time.time()

    print("Running PC at Individual-level...")
    print("Features: ", args.features)
    print("Method: ", args.method)
    print("Max Conditional Variables: ", args.max_cond_vars)
    print("Significance Level: ", args.significance_level)
    print("Stress window: ", args.stress_window)
    print("Stress Cutoff: ", args.stress_cutoff)
    print("No Stress Cutoff: ", args.no_stress_cutoff)
    print("Bootstraps", args.bootstraps)
    print("Buffer: ", args.buffer)
    print("Missingness Node: ", args.missing_node)

    ##################################################################
    # Load and process the data

    oura = ['breath_average','efficiency','hr_average','hr_lowest','onset_latency','score','temperature_delta','temperature_trend_deviation','total','log_hrv']
    garmin = ['steps','minHeartRateInBeatsPerMinute','maxHeartRateInBeatsPerMinute','averageHeartRateInBeatsPerMinute','restingHeartRateInBeatsPerMinute','averageStressLevel','maxStressLevel','stressDurationInSeconds','restStressDurationInSeconds','activityStressDurationInSeconds','lowStressDurationInSeconds','mediumStressDurationInSeconds','highStressDurationInSeconds']
    surveys_truncated = ['daily_stressed','daily_control','daily_shifts','sam8','shift_stress','promis_sd5','EBTBP','phq9_score','gad7_score','promis_sri_score','pss4_score','promis_es_score',"fss_score"]
    
    if args.features == "all":
        features = oura + garmin + surveys_truncated

    elif args.features == "oura":
        features = oura
    elif args.features == "garmin":
        features = garmin
    elif args.features == "surveys":
        features = surveys_truncated
    elif args.features == "oura+surveys":
        features = oura + surveys_truncated
    elif args.features == "garmin+surveys":
        features = garmin + surveys_truncated
    elif args.features == "oura+garmin":
        features = oura + garmin


    ##################################################################

    #load Data
    df = load_merge_all()
    #fit model
    dict_reference, dict_stress, dict_features = individual_graph_reference2(df, 
                                features, 
                                args.method, 
                                args.stress_window,
                                args.stress_cutoff,
                                args.no_stress_cutoff,
                                args.bootstraps, 
                                buffer = args.buffer,
                                missing_node = args.missing_node,
                                max_cond_vars=args.max_cond_vars, 
                                significance_level = args.significance_level)

    ##################################################################
    # Saving Val and Test set results
    method_folder = args.method
    fancy_string = f"{args.features}_window_{args.stress_window}_{args.stress_cutoff}_{args.no_stress_cutoff}_bootstraps_{args.bootstraps}_buffer_{args.buffer}_miss_{args.missing_node}_mcv_{args.max_cond_vars}_sig_{args.significance_level}_{args.method}"

    #results path
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    if not os.path.exists(os.path.join(parent_dir, "results", "PC","reference", args.stress_window)):
        os.makedirs(os.path.join(parent_dir, "results", "PC",'reference', args.stress_window))
   
   #Saving Model
    result_path = os.path.join(parent_dir, "results", "PC","reference", args.stress_window, fancy_string+".pkl")

    #get a list of tuples from the model and save
    with open(result_path, 'wb') as fp:
        pickle.dump([dict_reference, dict_stress, dict_features], fp)