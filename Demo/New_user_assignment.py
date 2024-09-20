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
import sys
sys.path.append('Community_detection/')
from Main_community_detection import read_csv_and_return_variables, create_sparse_matrix

''' example of new user
         User_IP                    Combination         Connections  
1      ++mHhyYDJ91Xrz+5XEDJ1w==        culture INSPIRE            10             
2      ++mHhyYDJ91Xrz+5XEDJ1w==           monde DIVERT            13             
3      ++mHhyYDJ91Xrz+5XEDJ1w==            monde TREND            8
'''

def create_new_user_df(user_ip, combinations, connections):
        return pd.DataFrame({
        'User_IP': [user_ip] * len(combinations),
        'Combination': combinations,
        'Connections': connections
    })

def assign_user_to_community(new_user, df_rts_temp, user_ip_encoder, combination_encoder, 
                             user_combo_matrix, leiden_communities):
    
    new_user['encoded_combos'] = combination_encoder.transform(new_user['Combination']) #[0:35] the last values are nudging related
    #print("Encoded combinations:", new_user['encoded_combos'])
    #print("Connections:", new_user['Connections'])    
    ips_temp = user_ip_encoder.classes_
    new_user_vector = sp.coo_matrix((new_user['Connections'], ([0] * len(new_user['encoded_combos']), new_user['encoded_combos'])),
        shape=(1, user_combo_matrix.shape[1])
    )
    
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(user_combo_matrix) 
    existing_users_tfidf = tfidf_transformer.transform(new_user_vector)
    
    community_profiles = []
    for community_label in set(leiden_communities):
        # Get users in the current community
        community_user_indices = [i for i, label in enumerate(leiden_communities) if label == community_label]
        # Get their combinations
        community_data = df_rts_temp[df_rts_temp['User_IP_Code'].isin(community_user_indices)][['Combination', 'Connections']]
    
    # Aggregate Connections by Combination, using the average (or sum)
        aggregated_data = community_data.groupby('Combination')['Connections'].sum().reset_index()  # You can use .sum() instead of .mean() if needed
    
        community_combinations = aggregated_data['Combination'].values
        connections_values = aggregated_data['Connections'].values
    
    # Encode combinations
        encoded_extended_combos = combination_encoder.transform(community_combinations)
    
    # Create sparse vector for the community profile with the actual 'Connections' values
        community_profile_vector = sp.coo_matrix(
        (connections_values, ([0] * len(encoded_extended_combos), encoded_extended_combos)),
        shape=(1, user_combo_matrix.shape[1])
        )
  
        community_profiles.append(tfidf_transformer.transform(community_profile_vector))
        #community_profiles.append((community_profile_vector))
    # Step 4: Compute similarity between the new user and each community profile
    new_user_tfidf = tfidf_transformer.transform(new_user_vector)
    similarity_scores = [cosine_similarity(new_user_tfidf, profile)[0, 0] for profile in community_profiles]
    #similarity_scores = [cosine_similarity(new_user_vector, profile)[0, 0] for profile in community_profiles]
    
    print("Similarity scores for each community:", similarity_scores)
    # Step 5: Assign to the best community (highest similarity score)
    best_community = np.argmax(similarity_scores)
    community_user_indices = [i for i, label in enumerate(leiden_communities) if label == best_community]
    best_community_combinations = df_rts_temp[df_rts_temp['User_IP_Code'].isin(community_user_indices)]['Combination'].unique()
    #print(f"Combinations in the best community {best_community} : {best_community_combinations}")
        
    return best_community, best_community_combinations,similarity_scores

def main():
     #print(df_rts)
    #df_rts_filtered,_,_,_,_,_ = read_csv_and_return_variables(8000) #to assign the user correctly we need to consider the filtered df
    df_rts, ips, combos, user_ip_encoder, combination_encoder,common_combinations = read_csv_and_return_variables(8000)
    print(df_rts)
    user_combo_scaled_sparse = create_sparse_matrix(df_rts)

    #new_user_combinations = ['culture INSPIRE' , 'suisse INSPIRE' , 'monde DIVERT' , 'culture EDUCATE']
    #new_user_connections = [20, 18, 10, 5]
    new_user_combinations = df_rts.loc[df_rts['User_IP_Code'] == 0, 'Combination']
    new_user_combinations = new_user_combinations.tolist()
   
    new_user_connections = df_rts.loc[df_rts['User_IP_Code'] == 0, 'Connections']
    new_user_connections = new_user_connections.tolist()
    print(new_user_combinations)
    print(new_user_connections)
    new_user = create_new_user_df(1,new_user_combinations,new_user_connections)
    print("Loading Leiden communities from file...")
    leiden_partition = pd.read_csv('Community_detection/leiden_communities.csv')
    leiden_communities = leiden_partition['community_id'].values
    print(f"Leiden communities: {len(set(leiden_communities))}")    
    best_community, best_community_combinations,_  = assign_user_to_community(new_user, df_rts, user_ip_encoder, combination_encoder, 
    user_combo_scaled_sparse, leiden_communities)

    print(f"User assigned to community {best_community} with combinations: {best_community_combinations}")

if __name__ == "__main__":
    main()
