import igraph as ig
from igraph import plot
import leidenalg
import pandas as pd
import json
import scipy.sparse as sp
import networkx as nx
import mpld3
import matplotlib.pyplot as plt
import numpy as np
from Main_community_detection import read_csv_and_return_variables,create_sparse_matrix,tfidf_weighted_cosine_similarity,create_graph_from_user_user_matrix 

def reverse_label_encoding(encoded_values, encoder):
    """Helper function to reverse label encoding."""
    return encoder.inverse_transform(encoded_values)

def extract_combinations_for_communities(leiden_membership, df_rts_temp, user_ip_encoder, combination_encoder):
    """Extract the 4 most frequent combinations associated with each community."""
    communities_combinations = {}
    leiden_membership = pd.Series(leiden_membership)
    
    for community_id in set(leiden_membership):
        # Find the users in this community
        community_nodes = df_rts_temp[df_rts_temp['User_IP_Code'].isin(leiden_membership[leiden_membership == community_id].index)]['User_IP_Code']
        
        # Get combinations for these users
        combinations_in_community = df_rts_temp[df_rts_temp['User_IP_Code'].isin(community_nodes)]['Combination_Code']
        
        # Count the occurrences of each combination and get the top 4
        top_combinations = combinations_in_community.value_counts().head(4).index
        
        # Decode combinations
        original_combinations_in_community = reverse_label_encoding(top_combinations, combination_encoder)
        
        # Save the top 4 combinations for this community
        communities_combinations[community_id] = original_combinations_in_community.tolist()  # Convert numpy arrays to lists

    output_file = 'Community_top4_features.json'
    with open(output_file, 'w') as f:
        json.dump(communities_combinations, f, indent=4)
    print(f"Top 4 combinations for each community saved to {output_file}")
    
    return communities_combinations

def visualize_community_connections(partition_list, user_user_matrix, threshold=0.8):
    # Create a new graph where each node represents a community
    communitiesNetwork = nx.Graph()
    
    # Add a node for each community
    for i, cluster in enumerate(partition_list):
        communitiesNetwork.add_node(i, size=len(cluster))
    
    # Loop over pairs of communities
    for i, cluster_a in enumerate(partition_list):
        for j, cluster_b in enumerate(partition_list):
            if i < j:
                connection_count = 0
                
                # Loop over users in each community
                for user_a in cluster_a:
                    for user_b in cluster_b:
                        # Check if the connection between users meets the threshold
                        if user_user_matrix[user_a, user_b] >= threshold:
                            connection_count += 1
                
                # If there are connections, create an edge between communities
                if connection_count > 0:
                    normalized_weight = connection_count / min(len(cluster_a), len(cluster_b))
                    communitiesNetwork.add_edge(i, j, weight=normalized_weight)
    
    # Optionally save or return the graph
    nx.write_gexf(communitiesNetwork, 'community_graph.gexf')
    return communitiesNetwork

def main():
    df_rts, ips, combos, user_ip_encoder, combination_encoder = read_csv_and_return_variables()
    user_combo_scaled_sparse = create_sparse_matrix(df_rts)
    user_user_matrix = tfidf_weighted_cosine_similarity(user_combo_scaled_sparse)
    print(f"User user matrix created")
    threshold = 0.8
    g = create_graph_from_user_user_matrix(sp.csr_matrix(user_user_matrix), threshold)
    
    print("Loading Leiden communities from file...")
    leiden_partition = pd.read_csv('leiden_communities.csv')
    leiden_membership = leiden_partition['community_id'].values
    print(f"Leiden communities: {len(set(leiden_membership))}")
    
    # Extract combinations for each community
    #communities_combinations = extract_combinations_for_communities(leiden_membership, df_rts, user_ip_encoder, combination_encoder)
    #print(communities_combinations)
    partition_list = leiden_partition.groupby('community_id')['user_id'].apply(list).tolist()
    visualize_community_connections(partition_list, user_user_matrix,threshold=0.8)

if __name__ == "__main__":
    main()