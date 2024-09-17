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
#from sklearn.cluster import KMeans
#from sklearn.metrics.pairwise import euclidean_distances
#import networkx as nx 
from scipy.sparse import triu
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

def reverse_label_encoding(encoded_values, encoder):
    return encoder.inverse_transform(encoded_values)

#original_ips = reverse_label_encoding(df_rts_temp['User_IP_Code'], user_ip_encoder)
#original_combinations = reverse_label_encoding(df_rts_temp['Combination_Code'], combination_encoder)

def read_csv_and_return_variables(common_threshold):
    print("Lecture du CSV...")
    start_time = time.time()
    
    # Optimize memory usage
    df_rts_temp_intermediate = pd.read_csv('Cleaned_data_x_clustering_NEW.csv')#, dtype=dtypes)#,nrows=1000 FOR DEBUG
    df_rts_temp_intermediate = df_rts_temp_intermediate.drop_duplicates()
    
        # Group by User_IP and Combination
    df_rts_temp = df_rts_temp_intermediate.groupby(['User_IP', 'Combination']).size().reset_index()
    df_rts_temp = df_rts_temp.rename(columns={0: 'Connections'})
    df_rts_temp['Connections'] = df_rts_temp['Connections'].astype(np.int32)
    #df_rts_temp = df_rts_temp[df_rts_temp['Connections'] > 1]
    #df_rts_temp = df_rts_temp.reset_index(drop=True)
    combination_counts = df_rts_temp['Combination'].value_counts()
    print(combination_counts)

# Set a threshold to filter out very common combinations
    common_threshold = 8000  # Example threshold, adjust based on your data
    common_combinations = combination_counts[combination_counts > common_threshold].index
    #print(len(common_combinations))
    #df_rts_temp['Connections'] = df_rts_temp['Connections'].apply(lambda x: 1 if x > 0 else 0).astype(np.int32)
    #df_rts_temp = df_rts_temp[df_rts_temp['Connections'] > 1]
    #df_rts_temp = df_rts_temp.reset_index(drop=True)
    df_rts_temp = df_rts_temp[~df_rts_temp['Combination'].isin(common_combinations)]

    #df_rts_temp = df_rts_temp[df_rts_temp['Connections'] > 1]
    #df_rts_temp = df_rts_temp.reset_index(drop=True)
    user_ip_encoder = LabelEncoder()
    combination_encoder = LabelEncoder()

    # Fit and transform the data
    df_rts_temp['User_IP_Code'] = user_ip_encoder.fit_transform(df_rts_temp['User_IP']).astype(np.int32)
    df_rts_temp['Combination_Code'] = combination_encoder.fit_transform(df_rts_temp['Combination']).astype(np.int32)

    ips_temp = user_ip_encoder.classes_
    comb_temp = combination_encoder.classes_


    print(f"Nombre de User_IPs uniques: {len(ips_temp)}")
    print(f"Nombre de Combination uniques: {len(comb_temp)}")
    print(f"Lecture du CSV terminée en {time.time() - start_time:.2f} secondes.")
    
    return df_rts_temp, ips_temp, comb_temp,user_ip_encoder, combination_encoder

def create_sparse_matrix(df_rts_temp):
    # Create user-combination matrix
    num_users = len(df_rts_temp['User_IP_Code'].unique())
    num_combinations = len(df_rts_temp['Combination_Code'].unique())
    
    user_combo_matrix = sp.coo_matrix(
        (df_rts_temp['Connections'], (df_rts_temp['User_IP_Code'], df_rts_temp['Combination_Code'])),
        shape=(num_users, num_combinations)
    )
    #df_nonzero = pd.DataFrame({
    #    'User': user_combo_matrix.row,
    #    'Combo': user_combo_matrix.col,
     #   'Connection': user_combo_matrix.data
    #})
    
    # Save the DataFrame to an Excel file
    #df_nonzero.to_excel('user_combo_matrix.xlsx', index=False)
    
    # Convert the sparse matrix to a dense format
    #user_combo_dense = user_combo_matrix.toarray()
    
    # Initialize the Min-Max scaler
    #scaler = MinMaxScaler()
    
    # Apply Min-Max scaling to each row
    #user_combo_scaled_dense = scaler.fit_transform(user_combo_dense)
    
    # Convert back to sparse matrix format
    #user_combo_scaled_sparse = sp.csr_matrix(user_combo_scaled_dense)
    #print(user_combo_matrix)
    return user_combo_matrix

def tfidf_weighted_cosine_similarity(user_combo_matrix):
    # Assuming 'user_combo_matrix' is a sparse matrix (users x combinations)
    
    # Step 1: Apply TF-IDF weighting
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(user_combo_matrix)
    
    # Step 2: Compute cosine similarity between users
    user_user_similarity = cosine_similarity(tfidf_matrix, dense_output=False)
    user_user_similarity.data = user_user_similarity.data.astype(np.float32)
    return user_user_similarity

def compute_user_user_matrix(user_combo_matrix,df):
    # Convert sparse matrix to CSR format (row-wise efficient)
    normalized_matrix = normalize(user_combo_matrix, axis=1, norm='l2')

    user_user_matrix = normalized_matrix.dot(normalized_matrix.T)   
    return user_user_matrix




def create_graph_from_user_user_matrix(user_user_matrix, threshold):
    # Convert to dense array and extract lower triangle without the diagonal (to avoid self-loops)
    lower_triangle = np.tril(user_user_matrix.toarray(), k=-1)
    
    # Get the non-zero values (i.e., existing edges)
    rows, cols = np.nonzero(lower_triangle)
    weights = lower_triangle[rows, cols]
    
    # Plot the weight distribution before applying the threshold
    #plt.figure(figsize=(10, 6))
    #plt.hist(weights, bins=50, edgecolor='black', alpha=0.75)
    #plt.title('Edge Weight Distribution Before Threshold')
    #plt.xlabel('Weight')
    #plt.ylabel('Frequency')
    #plt.grid(True)
    #plt.show()

    # Apply the threshold to filter out low-weight edges
    mask = weights >= threshold
    rows, cols, weights = rows[mask], cols[mask], weights[mask]

    # Create the graph with filtered edges
    g = ig.Graph()
    g.add_vertices(user_user_matrix.shape[0])  # Number of users (nodes)
    
    # Add edges and their weights
    edge_list = [(rows[i], cols[i]) for i in range(len(rows))]
    g.add_edges(edge_list)
    g.es['weight'] = weights
    #g.write_graphml("user_user_graph.graphml")
    print(f"Number of edges after applying threshold of {threshold}: {len(g.es)}")
    
    return g


def save_graph_plot(g, filename='graph_plot.png'):
    ig.plot(g, bbox=(800, 800), vertex_label=g.vs['name'], edge_width=g.es['weight'], target=filename)

def evaluate_modularity(graph, communities):
    return graph.modularity(communities)

def parallel_community_detection(graph):
    methods = ['louvain', 'label_prop', 'fast_greedy']
    
    manager = mp.Manager()
    shared_graph = manager.Namespace()
    shared_graph.graph = graph

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(run_community_detection, [(shared_graph.graph, method) for method in methods])
    
    results_dict = dict(zip(methods, results))
    
    return results_dict

def run_community_detection(graph, method, **kwargs):
    if method == 'louvain':
        return graph.community_multilevel(weights=graph.es['weight'])
    elif method == 'label_prop':
        return graph.community_label_propagation()
    elif method == 'fast_greedy':
        return graph.community_fastgreedy(weights='weight').as_clustering()
    else:
        raise ValueError(f"Unknown method: {method}")

def conductance(graph, communities):
    total_cut_edges = 0
    total_internal_edges = 0

    adjacency_matrix = np.array(graph.get_adjacency(attribute="weight").data)

    for community in communities:
        community_nodes = list(community)

        # Submatrix for the community (internal edges)
        community_submatrix = adjacency_matrix[np.ix_(community_nodes, community_nodes)]
        internal_edges = np.sum(community_submatrix) / 2  # Dividing by 2 to avoid double-counting

        # Edges going outside the community (cut edges)
        community_cut_edges = np.sum(adjacency_matrix[community_nodes, :]) - np.sum(community_submatrix)

        total_internal_edges += internal_edges
        total_cut_edges += community_cut_edges

    if total_internal_edges + total_cut_edges == 0:
        return 0.0

    return total_cut_edges / (total_internal_edges + total_cut_edges)

def save_leiden_communities(leiden_communities, filename='leiden_communities.json'):
    # Save the community membership to a JSON file
    communities_dict = {"user_id": list(range(len(leiden_communities.membership))), 
                        "community_id": leiden_communities.membership}
    
    with open(filename, 'w') as f:
        json.dump(communities_dict, f)
    print(f"Leiden communities saved to {filename}")

def main():
    df_rts, ips, combos, user_ip_encoder, combination_encoder = read_csv_and_return_variables(8000)

    user_combo_scaled_sparse = create_sparse_matrix(df_rts)
    user_user_matrix = tfidf_weighted_cosine_similarity(user_combo_scaled_sparse)
    #user_user_matrix = compute_user_user_matrix(user_combo_scaled_sparse,df_rts)
    #user_user_matrix = sp.csr_matrix(user_user_matrix)
    #user_user_matrix.data = user_user_matrix.data.astype(np.float32)
    print(f"User user matrix created")
    threshold = 0.8# Adjust this based on the plot

    # Create the graph with a weight threshold
    g = create_graph_from_user_user_matrix(sp.csr_matrix(user_user_matrix), threshold)

    print(f"Graph created")

    print("Running Leiden algorithm...")
    leiden_communities = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition,weights=g.es["weight"])
    print(f"Leiden communities: {len(leiden_communities)}")
       # Run the rest of the community detection algorithms in parallel
    #community_results = parallel_community_detection(g)
    louvain_communities = run_community_detection(g, 'louvain',weights=g.es["weight"])
    print(f"Louvain communities: {len(louvain_communities)}")

    #label_prop_communities = run_community_detection(g, 'label_prop')
    #print(f"Label Propagation communities: {len(label_prop_communities)}")

   # fast_greedy_communities = run_community_detection(g, 'fast_greedy')
    #print(f"Fast Greedy communities: {len(fast_greedy_communities)}")

    conductance_louvain = conductance(g, louvain_communities)
    #conductance_label_prop = conductance(g, label_prop_communities)
    #conductance_fast_greedy = conductance(g, fast_greedy_communities)

    print(f"Louvain Conductance: {conductance_louvain}")
    #print(f"Label Propagation Conductance: {conductance_label_prop}")
    #print(f"Fast Greedy Conductance: {conductance_fast_greedy}")

    modularity_louvain = evaluate_modularity(g, louvain_communities)
   # modularity_label_prop = evaluate_modularity(g, label_prop_communities)
    #modularity_fast_greedy = evaluate_modularity(g, fast_greedy_communities)
    modularity_leidenalg = evaluate_modularity(g, leiden_communities)

    print(f"Louvain Modularity: {modularity_louvain}")
   # print(f"Label Propagation Modularity: {modularity_label_prop}")
    #print(f"Fast Greedy Modularity: {modularity_fast_greedy}")
    print(f"Leiden Modularity: {modularity_leidenalg}")
    save_leiden_communities(leiden_communities, filename="leiden_communities.json")
    
    # Optionally, save as CSV
    df_leiden = pd.DataFrame({"user_id": list(range(len(leiden_communities.membership))),
                              "community_id": leiden_communities.membership})
    df_leiden.to_csv("leiden_communities.csv", index=False)
    print("Leiden communities also saved as leiden_communities.csv")

if __name__ == "__main__":
    main()
