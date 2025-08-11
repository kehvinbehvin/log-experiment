#### Train
=== HDBSCAN Log Training with Distance-Based Stable IDs ===

Using Apple Silicon MPS
Loading training data from: data/train/data.log
Loaded 1700 raw logs
Sample raw log: - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNE...
Preprocessing 1700 logs...
Saved preprocessed logs to: data/preprocessed/preprocessed_logs.log
Sample preprocessed log: - . . - - - : - - - - . . . - - - : -
Generating embeddings using device: mps
Data uniqueness: 272 unique patterns out of 1700 logs (16.0%)
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 27/27 [00:05<00:00,  5.07it/s]
Generated embeddings shape: (1700, 384)
Using 'moderate' preset: Balanced clustering for raw logs
Training HDBSCAN with parameters:
  min_cluster_size=15
  min_samples=10
  cluster_selection_epsilon=0.2
/Users/kevin/Documents/Kevin/log-experiment-v5/venv/lib/python3.13/site-packages/sklearn/utils/deprecation.py:132: FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.
  warnings.warn(
/Users/kevin/Documents/Kevin/log-experiment-v5/venv/lib/python3.13/site-packages/sklearn/utils/deprecation.py:132: FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.
  warnings.warn(
Clustering complete. Labels shape: (1700,)
Analyzing cluster distribution...
Total logs: 1700
Number of clusters: 37
Number of outliers: 226 (13.3%)

Cluster size distribution:
  Outliers: 226 logs
  Cluster 0: 88 logs
  Cluster 1: 75 logs
  Cluster 2: 16 logs
  Cluster 3: 24 logs
  Cluster 4: 24 logs
  Cluster 5: 18 logs
  Cluster 6: 32 logs
  Cluster 7: 144 logs
  Cluster 8: 35 logs
  Cluster 9: 26 logs
  Cluster 10: 20 logs
  Cluster 11: 40 logs
  Cluster 12: 35 logs
  Cluster 13: 45 logs
  Cluster 14: 30 logs
  Cluster 15: 30 logs
  Cluster 16: 41 logs
  Cluster 17: 29 logs
  Cluster 18: 26 logs
  Cluster 19: 25 logs
  Cluster 20: 15 logs
  Cluster 21: 35 logs
  Cluster 22: 51 logs
  Cluster 23: 51 logs
  Cluster 24: 39 logs
  Cluster 25: 33 logs
  Cluster 26: 67 logs
  Cluster 27: 31 logs
  Cluster 28: 33 logs
  Cluster 29: 25 logs
  Cluster 30: 16 logs
  Cluster 31: 47 logs
  Cluster 32: 18 logs
  Cluster 33: 84 logs
  Cluster 34: 38 logs
  Cluster 35: 70 logs
  Cluster 36: 18 logs

Sample logs from each cluster:

=== OUTLIERS (226 logs) ===
  . . - - - : - - - - . . . - - - : - : ( . . . :
  . . - - - : - - - - . . . - - - : - : ( . . . :

=== CLUSTER 0 (88 logs) ===
  - ( )
  - ( )

=== CLUSTER 1 (75 logs) ===
  - : : . : . . : . . . . . . $
  . $ :

=== CLUSTER 2 (16 logs) ===
  - . . - - - : - - - - . . . - - - : - , ,
  - . . - - - : - - - - . . . - - - : - , ,

=== CLUSTER 3 (24 logs) ===
  - : : : | | | = # # # # # # # # # #
  - : : : | | | = # # # # # # # # # #

=== CLUSTER 4 (24 logs) ===
  - : : : | | | =
  - : : : | | | =

=== CLUSTER 5 (18 logs) ===
  . : * . : : . . . :
  . : * . : : . . . :

=== CLUSTER 6 (32 logs) ===
  - . . - - - : - - - - . . . - - - : -
  - . . - - - : - - - - . . . - - - : -

=== CLUSTER 7 (144 logs) ===
  - . . - - - : - - - - . . . - - - : - .
  - . . - - - : - - - - . . . - - - : - .

=== CLUSTER 8 (35 logs) ===
  / / : : . : . . ( )
  / / : : . : . . ( )

=== CLUSTER 9 (26 logs) ===
  - : : : | | | :
  - : : : | | | :

=== CLUSTER 10 (20 logs) ===
  - : : : | | |
  - : : : | | |

=== CLUSTER 11 (40 logs) ===
  / / : : . : : ,
  / / : : . : : ,

=== CLUSTER 12 (35 logs) ===
  - : : . : = , = , = , = , = , = , = , = , = , = , = - , = . =
  - : : . : = , = , = , = , = , = , = , = , = , = , = - , = . =

=== CLUSTER 13 (45 logs) ===
  - : : . : ,
  - : : . : ,

=== CLUSTER 14 (30 logs) ===
  - : : . : . .
  - : : . : .

=== CLUSTER 15 (30 logs) ===
  [X] [X] . ( ) / / / / .
  [X] [X] . ( ) / / / / .

=== CLUSTER 16 (41 logs) ===
  [X] [X] ( )
  [X] [X] ( )

=== CLUSTER 17 (29 logs) ===
  [X] [X]
  [X] [X]

=== CLUSTER 18 (26 logs) ===
  : : [X] : ( : ) : ; = = = = = = . . .
  : : [X] : ( : ) : ; = = = = = = - - - - . - - . . . .

=== CLUSTER 19 (25 logs) ===
  : : ( ) [X] : ; = = = = = = . . .
  : : ( ) [X] : ; = = = = = = . . .

=== CLUSTER 20 (15 logs) ===
  : : ( ) [X] : ; = = = = = = - - - . - . . =
  : : ( ) [X] : ; = = = = = = - - - . - . . =

=== CLUSTER 21 (35 logs) ===
  : : ( ) [X] : ;
  : : ( ) [X] : ;

=== CLUSTER 22 (51 logs) ===
  : : ( ) [X] : ( = )
  : : ( ) [X] :

=== CLUSTER 23 (51 logs) ===
  - . . . - - : : - - : : . . . . [X] ... "X" : : : .
  - . . . - - : : - - : : . . . . [X] ... "X" : : : .

=== CLUSTER 24 (39 logs) ===
  - . . . - - : : - - : : . . . . [X] - - - - ( / / / / / / ) :
  - . . . - - : : - - : : . . . . [X] : / / / / / /

=== CLUSTER 25 (33 logs) ===
  [X] . - . . . . : , , ( . ) ,
  [X] . - . . . . : , , ( . ) ,

=== CLUSTER 26 (67 logs) ===
  [X] . - . . . . : . . . . :
  [X] . - . . . . : . . . . :

=== CLUSTER 27 (31 logs) ===
  : : [X] : . . .
  : : [X] : . . .

=== CLUSTER 28 (33 logs) ===
  : : [X] : : [X]
  : : [X] : . . . [X]

=== CLUSTER 29 (25 logs) ===
  - . . . - - : : - - : : . . . [X] [X] ( )
  - . . . - - : : - - : : . . . [X] [X] ( )

=== CLUSTER 30 (16 logs) ===
  : : [X] : . . . ( - - - . . . ) : :
  : : [X] : . . . ( - - - . . . ) : :

=== CLUSTER 31 (47 logs) ===
  - - : : , : . [X]
  - - : : , [X]

=== CLUSTER 32 (18 logs) ===
  - . . : : / [X] : . . . ,
  - . . : : / [X] : . . . ,

=== CLUSTER 33 (84 logs) ===
  - - : : , [X] . . . . . . :
  - - : : , [X] . . . . . . : :

=== CLUSTER 34 (38 logs) ===
  : : - - - - . . [X] : : :
  : : - - - - [X] : : ( : : : : : : : )

=== CLUSTER 35 (70 logs) ===
  - - : : , - [X] - :
  - - : : , - [X] -

=== CLUSTER 36 (18 logs) ===
  - - : : , - [X] - / . . . :
  - - : : , - [X] - / . . . :

=== Building Distance-Based Stable Cluster Registry ===
Building stable registry from 37 clusters...
Registry built with 37 stable IDs
Stored centroids and sample embeddings for distance-based matching
Registry Summary: {'total_stable_ids': 37, 'training_clusters': 37, 'new_patterns_discovered': 0, 'next_available_id': 37, 'has_centroids': True, 'distance_based_enabled': True}
Distance-based matching enabled: 37 centroids stored
KNN sample embeddings stored for: 37 clusters
Saving model artifacts to model/
Saved clusterer: model/hdbscan_clusterer.pkl
Saved embeddings: model/training_embeddings.npy
Saved statistics: model/cluster_stats.json
Stable registry saved to: model/stable_cluster_registry.json
Centroids saved to: model/centroids.pkl
Sample embeddings saved to: model/sample_embeddings.pkl

=== Training Complete ===
Model saved to: model/
Distance-based registry created with 37 base clusters
Centroids saved for inference clustering via cosine distance + KNN verification
Ready for inference on new logs using centroid/KNN matching

#### Inference

=== Distance-Based Log Clustering Pipeline ===

Using Apple Silicon MPS
Loading model artifacts from: model
Loaded HDBSCAN clusterer: model/hdbscan_clusterer.pkl
Loaded centroids for 37 clusters
Loaded sample embeddings for 37 clusters
Stable registry loaded from: model/stable_cluster_registry.json
Registry contains 37 stable cluster definitions
Distance-based matching: Enabled
Loaded stable registry with 37 centroids for distance-based matching
Loading inference logs from: data/inference
Reading: data_1.log
  Loaded 1600 logs from data_1.log
Reading: data_2.log
  Loaded 1600 logs from data_2.log
Total loaded: 3200 logs from 2 files
Sample log: 03-17 16:14:03.281  2227  2227 I PhoneStatusBar: setSystemUiVisibility vis=0 mask=1 oldVal=40000500 ...
Preprocessing 3200 log lines...
Sample preprocessed log: - : : . : = = = = = = = , = ( , - , ) , = ( , - , )
Generating embeddings using device: mps
Batches: 100%|████████████████████████████████████████| 50/50 [00:02<00:00, 18.87it/s]
Generated embeddings shape: (3200, 384)

=== Performing Distance-Based Clustering ===
Using centroids + KNN approach (no HDBSCAN clustering needed)

=== Distance-Based Stable Cluster Assignment ===
Assigning 3200 embeddings to stable clusters...
Assignment Results:
  Direct assignments: 3036
  Verified assignments: 0
  Outliers: 164

Distance-Based Assignment Results:
  Direct assignments: 3036
  KNN verified assignments: 0
  Outliers: 164

Cluster Distribution:
  Outliers: 164 logs
  Stable Cluster 0: 83 logs
  Stable Cluster 1: 8 logs
  Stable Cluster 2: 118 logs
  Stable Cluster 3: 179 logs
  Stable Cluster 4: 188 logs
  Stable Cluster 5: 128 logs
  Stable Cluster 6: 117 logs
  Stable Cluster 7: 72 logs
  Stable Cluster 8: 47 logs
  Stable Cluster 9: 78 logs
  Stable Cluster 10: 63 logs
  Stable Cluster 12: 49 logs
  Stable Cluster 13: 61 logs
  Stable Cluster 14: 50 logs
  Stable Cluster 15: 49 logs
  Stable Cluster 16: 48 logs
  Stable Cluster 17: 33 logs
  Stable Cluster 18: 27 logs
  Stable Cluster 19: 12 logs
  Stable Cluster 20: 86 logs
  Stable Cluster 21: 271 logs
  Stable Cluster 22: 40 logs
  Stable Cluster 23: 65 logs
  Stable Cluster 24: 61 logs
  Stable Cluster 25: 53 logs
  Stable Cluster 27: 28 logs
  Stable Cluster 28: 163 logs
  Stable Cluster 29: 92 logs
  Stable Cluster 30: 102 logs
  Stable Cluster 31: 121 logs
  Stable Cluster 32: 68 logs
  Stable Cluster 33: 80 logs
  Stable Cluster 34: 70 logs
  Stable Cluster 35: 178 logs
  Stable Cluster 36: 148 logs

Saving results to MCP output: data/mcp/data.log
Appended 3200 clustered log entries to data/mcp/data.log
Cluster summary: {-1: 164, 0: 83, 1: 8, 2: 118, 3: 179, 4: 188, 5: 128, 6: 117, 7: 72, 8: 47, 9: 78, 10: 63, 12: 49, 13: 61, 14: 50, 15: 49, 16: 48, 17: 33, 18: 27, 19: 12, 20: 86, 21: 271, 22: 40, 23: 65, 24: 61, 25: 53, 27: 28, 28: 163, 29: 92, 30: 102, 31: 121, 32: 68, 33: 80, 34: 70, 35: 178, 36: 148}

=== Distance-Based Clustering Complete ===
Processed 3200 logs using centroids + KNN approach
Results appended to: data/mcp/data.log
Ready for MCP tools consumption