#!/usr/bin/env python3
"""
Distance-Based Log Clustering Script for log-experiment-v5

Uses trained HDBSCAN model with distance-based stable cluster registry (centroids + KNN)
to assign consistent cluster IDs to new log data. Processes logs from data/inference/
and outputs results to data/mcp/data.log for MCP consumption.

Usage:
    python cluster.py
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime

from stable_cluster_registry import StableClusterRegistry
from log_preprocessing import preprocess_log


def preprocess_logs_batch(raw_logs):
    """Preprocess a batch of raw log lines"""
    print(f"Preprocessing {len(raw_logs)} log lines...")
    preprocessed = [preprocess_log(log.strip()) for log in raw_logs if log.strip()]
    if preprocessed:
        print(f"Sample preprocessed log: {preprocessed[0]}")
    return preprocessed

def setup_device():
    """Setup optimal device for sentence transformers"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Using Apple Silicon MPS")
    else:
        device = 'cpu'
        print("Using CPU")
    return device

def load_model_artifacts(model_dir='model'):
    """Load trained HDBSCAN model and stable cluster registry"""
    model_dir = Path(model_dir)
    
    print(f"Loading model artifacts from: {model_dir}")
    
    # Load HDBSCAN clusterer (for compatibility, though we mainly use centroids)
    clusterer_path = model_dir / 'hdbscan_clusterer.pkl'
    if not clusterer_path.exists():
        raise FileNotFoundError(f"HDBSCAN model not found: {clusterer_path}")
    
    with open(clusterer_path, 'rb') as f:
        clusterer = pickle.load(f)
    print(f"Loaded HDBSCAN clusterer: {clusterer_path}")
    
    # Load stable cluster registry with centroids
    stable_registry = StableClusterRegistry()
    stable_registry.load_registry(model_dir)
    
    # Verify distance-based matching is available
    if not stable_registry.centroids:
        raise ValueError("No centroids found in stable registry - distance-based clustering not available")
    
    print(f"Loaded stable registry with {len(stable_registry.centroids)} centroids for distance-based matching")
    
    # Load training statistics for reference
    stats_path = model_dir / 'cluster_stats.json'
    training_stats = None
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            training_stats = json.load(f)
    
    return clusterer, stable_registry, training_stats

def load_inference_logs_from_directory(inference_dir='data/inference'):
    """Load all log files from data/inference/ directory"""
    inference_dir = Path(inference_dir)
    
    if not inference_dir.exists():
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")
    
    print(f"Loading inference logs from: {inference_dir}")
    
    # Find all .log files in the directory
    log_files = list(inference_dir.glob('*.log'))
    
    if not log_files:
        raise FileNotFoundError(f"No .log files found in: {inference_dir}")
    
    all_logs = []
    file_sources = []  # Track which file each log came from
    
    for log_file in sorted(log_files):
        print(f"Reading: {log_file.name}")
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            file_logs = [line.strip() for line in f if line.strip()]
        
        all_logs.extend(file_logs)
        file_sources.extend([log_file.name] * len(file_logs))
        
        print(f"  Loaded {len(file_logs)} logs from {log_file.name}")
    
    print(f"Total loaded: {len(all_logs)} logs from {len(log_files)} files")
    if all_logs:
        print(f"Sample log: {all_logs[0][:100]}...")
    
    return all_logs, file_sources

def generate_embeddings(logs, device='cuda', batch_size=64):
    """Generate semantic embeddings using SentenceTransformers"""
    print(f"Generating embeddings using device: {device}")
    
    # Load pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Generate embeddings
    embeddings = model.encode(
        logs, 
        batch_size=batch_size, 
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings

def perform_distance_based_clustering(stable_registry, raw_logs, preprocessed_logs, embeddings, file_sources):
    """Perform distance-based stable cluster ID assignment using centroids + KNN"""
    print(f"\n=== Performing Distance-Based Clustering ===")
    print("Using centroids + KNN approach (no HDBSCAN clustering needed)")
    
    # Direct assignment using distance matching
    stable_assignments, assignment_stats = stable_registry.assign_embeddings_to_stable_clusters(
        embeddings, raw_logs, preprocessed_logs
    )
    
    # Create results dataframe with source file information
    results_df = pd.DataFrame({
        'source_file': file_sources,
        'raw_log': raw_logs,
        'preprocessed_log': preprocessed_logs,
        'stable_cluster_id': stable_assignments,
        'timestamp': datetime.now().isoformat()
    })
    
    print(f"\nDistance-Based Assignment Results:")
    print(f"  Direct assignments: {assignment_stats.get('direct_assignments', 0)}")
    print(f"  KNN verified assignments: {assignment_stats.get('verified_assignments', 0)}")  
    print(f"  Outliers: {assignment_stats.get('outliers', 0)}")
    
    # Cluster distribution
    cluster_counts = results_df['stable_cluster_id'].value_counts().sort_index()
    print(f"\nCluster Distribution:")
    for cluster_id, count in cluster_counts.items():
        if cluster_id == -1:
            print(f"  Outliers: {count} logs")
        else:
            print(f"  Stable Cluster {cluster_id}: {count} logs")
    
    return results_df, assignment_stats

def save_results_to_mcp(results_df, output_path='data/mcp/data.log'):
    """Save clustering results to CSV format in MCP output file with patterns"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to MCP output: {output_path}")
    
    # Check if file exists and has content to determine if we need to write headers
    write_headers = not output_path.exists() or output_path.stat().st_size == 0
    
    # Open file in append mode
    with open(output_path, 'a', encoding='utf-8') as f:
        # Write CSV headers if file is empty or doesn't exist
        if write_headers:
            f.write("cluster,raw_log,pattern\n")
        
        # Write CSV rows
        for _, row in results_df.iterrows():
            # Escape quotes in raw_log by doubling them (CSV standard)
            escaped_log = row['raw_log'].replace('"', '""')
            
            # Get pattern by preprocessing the raw log directly
            cluster_id = int(row['stable_cluster_id'])
            
            if cluster_id == -1:
                pattern = "outlier"
            else:
                # Preprocess the raw log to get its pattern
                pattern = preprocess_log(row['raw_log'])
            
            # Escape quotes in pattern by doubling them (CSV standard)
            escaped_pattern = pattern.replace('"', '""')
            
            # Write in format: cluster,"raw_log","pattern"
            f.write(f'{cluster_id},"{escaped_log}","{escaped_pattern}"\n')
    
    print(f"Appended {len(results_df)} clustered log entries to {output_path}")
    
    # Show cluster summary
    cluster_summary = results_df.groupby('stable_cluster_id').size().to_dict()
    print(f"Cluster summary: {cluster_summary}")
    
    return output_path

def main():
    """Main distance-based clustering pipeline"""
    print("=== Distance-Based Log Clustering Pipeline ===\n")
    
    # Setup device
    device = setup_device()
    
    # Load model artifacts and stable registry with centroids
    try:
        clusterer, stable_registry, training_stats = load_model_artifacts()
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        print("Make sure you've run train.py first to create the trained model and centroids.")
        sys.exit(1)
    
    # Load inference logs from data/inference/ directory
    try:
        raw_logs, file_sources = load_inference_logs_from_directory()
    except Exception as e:
        print(f"Error loading inference logs: {e}")
        sys.exit(1)
    
    # Preprocess logs using same logic as training
    preprocessed_logs = preprocess_logs_batch(raw_logs)
    
    # Generate embeddings from preprocessed logs (crucial for consistency)
    embeddings = generate_embeddings(preprocessed_logs, device=device, batch_size=64)
    
    # Perform distance-based clustering using centroids + KNN
    results_df, assignment_stats = perform_distance_based_clustering(
        stable_registry, raw_logs, preprocessed_logs, embeddings, file_sources
    )
    
    # Save results to MCP output file
    output_path = save_results_to_mcp(results_df)
    
    print(f"\n=== Distance-Based Clustering Complete ===")
    print(f"Processed {len(raw_logs)} logs using centroids + KNN approach")
    print(f"Results appended to: {output_path}")
    print(f"Ready for MCP tools consumption")

if __name__ == "__main__":
    main()