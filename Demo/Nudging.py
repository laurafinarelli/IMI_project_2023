import pandas as pd
import igraph as ig
from igraph import plot
from sklearn.preprocessing import LabelEncoder
import json
import multiprocessing as mp
import scipy.sparse as sp
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import community as community_louvain
import leidenalg
from scipy.sparse import triu
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
np.random.seed(None)
import sys
sys.path.append('Community_detection/')
from Main_community_detection import read_csv_and_return_variables, create_sparse_matrix
from New_user_assignment import create_new_user_df, assign_user_to_community

def find_target_community(nudging, favouriteTopic):
    if nudging == 0:  # no nudging
        community_label = np.random.randint(0, 15)  # Random community between 0 and 15
    elif nudging == 1 and favouriteTopic is not None:  # soft nudging
        if favouriteTopic in ['suisse', 'economie']:
            community_label = 15
        elif favouriteTopic in ['monde', 'culture']:
            community_label = 0
        elif favouriteTopic == 'sciences-tech':
            community_label = 4
        elif favouriteTopic == 'sport':
            community_label = 10
    elif nudging == 1 and favouriteTopic is None:  # hard nudging
        community_label = 15  # Always send to community 15 for hard nudging
    else:
        community_label = None  # Handle the case where neither condition is met
    
    return community_label

combinations = [
    "monde UPDATE", "suisse UPDATE", "monde ANALYSIS", "suisse ANALYSIS",
    "economie UPDATE", "sciences-tech ANALYSIS", "sciences-tech UPDATE",
    "economie ANALYSIS", "culture UPDATE", "sciences-tech EDUCATE",
    "culture DIVERT", "monde EDUCATE", "suisse EDUCATE", "economie EDUCATE",
    "suisse TREND", "culture ANALYSIS", "monde DIVERT", "monde TREND",
    "culture INSPIRE", "culture EDUCATE", "sciences-tech INSPIRE",
    "sciences-tech TREND", "suisse DIVERT", "suisse INSPIRE",
    "monde INSPIRE", "economie TREND", "sciences-tech DIVERT",
    "culture TREND", "sport UPDATE", "sport INSPIRE", "economie INSPIRE",
    "sport ANALYSIS", "sport DIVERT", "sport EDUCATE", "environnement EDUCATE",
    "sport TREND"
]

exclusion_set = [
    "monde UPDATE", "suisse UPDATE", "monde ANALYSIS", "suisse ANALYSIS", 
    "economie UPDATE", "sciences-tech ANALYSIS", "sciences-tech UPDATE", 
    "economie ANALYSIS", "culture UPDATE", "sciences-tech EDUCATE", 
    "culture DIVERT", "monde EDUCATE", "suisse EDUCATE", "economie EDUCATE", 
    "suisse TREND", "culture ANALYSIS"
]
def filter_combinations(combinations):
    """Filters out combinations that are in the exclusion set."""
    return [comb for comb in combinations if comb not in exclusion_set]

def extract_features_from_json(ordered_communities):
    """Extracts features for the communities in order from a JSON file."""
    with open('Community_detection/Community_top4_features.json', 'r') as f:
        community_features = json.load(f)
    
    for community in ordered_communities:
        if str(community) in community_features:
            feature = community_features[str(community)][0]  # Extract the first feature
            print(f"First feature of community {community}: {feature}")
        else:
            print(f"No features found for community {community}")

def main():
     #print(df_rts)
    #df_rts_filtered,_,_,_,_,_ = read_csv_and_return_variables(8000) #to assign the user correctly we need to consider the filtered df
    df_rts, ips, combos, user_ip_encoder, combination_encoder,common_combinations = read_csv_and_return_variables(8000)
    user_combo_scaled_sparse = create_sparse_matrix(df_rts)

    #existing user
    #new_user_combinations = df_rts.loc[df_rts['User_IP_Code'] == 0, 'Combination']
    #new_user_combinations = new_user_combinations.tolist()
    #new_user_connections = df_rts.loc[df_rts['User_IP_Code'] == 0, 'Connections']
    #new_user_connections = new_user_connections.tolist()

    #random user
    
    #filtered_combinations = filter_combinations(new_user_combinations)
    filtered_combinations = [combo for combo in combinations if combo not in exclusion_set]
    random_connections = np.random.randint(0, 31, size=len(filtered_combinations))
    new_user = create_new_user_df(1,filtered_combinations,random_connections)
    print(new_user)
    leiden_partition = pd.read_csv('Community_detection/leiden_communities.csv')
    leiden_communities = leiden_partition['community_id'].values   
    best_community, best_community_combinations,similarity_scores  = assign_user_to_community(new_user, df_rts, user_ip_encoder, combination_encoder, 
    user_combo_scaled_sparse, leiden_communities)

    print(f"User assigned to community {best_community} with combinations: {best_community_combinations}")
    target_community = find_target_community(1, 'culture')
    print(target_community)
    sorted_indices = sorted(range(len(similarity_scores)), key=lambda x: similarity_scores[x], reverse=True)
    ordered_communities = []
    for index in sorted_indices:
        ordered_communities.append(index)
        if index == target_community:
            break
    print(f"Ordered communities by similarity: {ordered_communities}")
  
    extract_features_from_json(ordered_communities)


if __name__ == "__main__":
    main()