#!/usr/bin/env python3
"""
Log Cluster Schema MCP Server

Provides access to log cluster schema information via Model Context Protocol (MCP).
Exposes cluster IDs and their associated patterns from the stable cluster registry.

Usage:
    uv run mcp dev src/log_mcp.py
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("LogClusterServer")

# Global variable to cache registry data
_registry_cache = None

def load_cluster_registry():
    """Load and cache the stable cluster registry data"""
    global _registry_cache
    
    if _registry_cache is not None:
        return _registry_cache
    
    # Look for registry file in model directory (relative to script location)
    script_dir = Path(__file__).parent
    registry_path = script_dir.parent / "model" / "stable_cluster_registry.json"
    
    if not registry_path.exists():
        raise FileNotFoundError(f"Cluster registry not found at: {registry_path.absolute()}")
    
    try:
        with open(registry_path, 'r') as f:
            registry_data = json.load(f)
        
        # Cache the data
        _registry_cache = registry_data
        return registry_data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in cluster registry: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load cluster registry: {e}")

def format_cluster_schema(registry_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format registry data into clean schema format for MCP clients"""
    
    stable_id_to_info = registry_data.get("stable_id_to_info", {})
    
    # Build clean cluster schema
    clusters = {}
    
    for cluster_id_str, cluster_info in stable_id_to_info.items():
        cluster_id = int(cluster_id_str)
        
        # Extract key information
        patterns = cluster_info.get("patterns", [])
        training_size = cluster_info.get("training_size", 0)
        sample_logs = cluster_info.get("sample_logs", [])
        
        clusters[str(cluster_id)] = {
            "patterns": patterns,
            "training_size": training_size,
            "sample_preview": sample_logs[0][:150] + "..." if sample_logs and len(sample_logs[0]) > 150 else (sample_logs[0] if sample_logs else "No sample available")
        }
    
    # Get summary statistics
    total_clusters = len(clusters)
    total_logs = sum(cluster["training_size"] for cluster in clusters.values())
    
    return {
        "total_clusters": total_clusters,
        "total_training_logs": total_logs,
        "clusters": clusters,
        "registry_info": {
            "next_available_id": registry_data.get("next_stable_id", 0),
            "schema_version": "1.0"
        }
    }

@mcp.tool()
def log_schema() -> Dict[str, Any]:
    """
    Get all existing cluster IDs and their associated patterns.
    
    Returns comprehensive schema information about all log clusters,
    including pattern signatures, training data sizes.
    No parameters required.
    
    Returns:
        Dictionary containing:
        - total_clusters: Number of available clusters
        - total_training_logs: Total logs used in training
        - clusters: Map of cluster_id -> {patterns, training_size, sample_preview}
        - registry_info: Metadata about the cluster registry
    """
    
    try:
        # Load registry data
        registry_data = load_cluster_registry()
        
        # Format for MCP client consumption
        schema = format_cluster_schema(registry_data)
        
        return {
            "success": True,
            "data": schema
        }
        
    except FileNotFoundError as e:
        return {
            "success": False,
            "error": "Registry not found",
            "message": str(e),
            "suggestion": "Make sure you've run train.py to generate the cluster registry"
        }
        
    except (ValueError, RuntimeError) as e:
        return {
            "success": False,
            "error": "Registry load error",
            "message": str(e)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": "Unexpected error",
            "message": str(e)
        }

# Server entry point
if __name__ == "__main__":
    # Test the server functionality when run directly
    print("=== Log Cluster Schema MCP Server ===")
    print("Testing log_schema tool...")
    
    try:
        result = log_schema()
        if result.get("success"):
            data = result["data"]
            print(f" Successfully loaded {data['total_clusters']} clusters")
            print(f"   Total training logs: {data['total_training_logs']}")
            print(f"   Next available ID: {data['registry_info']['next_available_id']}")
            print("\nFirst 3 clusters:")
            for i, (cluster_id, info) in enumerate(list(data["clusters"].items())[:3]):
                print(f"   Cluster {cluster_id}: {info['description']} ({info['training_size']} logs)")
                print(f"      Patterns: {len(info['patterns'])} pattern(s)")
        else:
            print(f"L Error: {result.get('error', 'Unknown error')}")
            print(f"   Message: {result.get('message', 'No details available')}")
            
    except Exception as e:
        print(f"L Test failed: {e}")
    
    print("\n=� To use as MCP server, run: uv run mcp dev src/log_mcp.py")
    print("   Or integrate with Claude Desktop via MCP configuration")