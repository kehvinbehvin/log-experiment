#### Train
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
Batches: 100%|████████████████████████████████████████| 50/50 [00:03<00:00, 16.21it/s]
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