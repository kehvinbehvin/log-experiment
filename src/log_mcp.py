#!/usr/bin/env python3
"""
Log Cluster Schema MCP Server

Provides access to log cluster schema information via Model Context Protocol (MCP).
Exposes cluster IDs and their associated patterns from the stable cluster registry.

Usage:
    uv run mcp dev src/log_mcp.py
"""

import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List

from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("LogClusterServer")

# Global variable to cache registry data
_registry_cache = None

# === Shallow Log Extraction Functions (from archive/extract_shallow.py) ===

def reconstruct_log(raw_log: str, preprocessed_pattern: str) -> str:
    """
    Reconstruct log by filling whitespace gaps with real content while preserving masked regions.
    
    Args:
        raw_log: Original log line
        preprocessed_pattern: Structural symbols (e.g., '- - [X] "X"')
    
    Returns:
        Reconstructed log string with real content in gaps, masked content in enclosed regions
        
    Behavior:
        - Whitespace between symbols -> fill with actual alphanumeric content
        - Enclosed symbols (like [X], "X", {X}) -> keep as masked form
        - Standalone symbols -> preserve as-is
    """
    if not raw_log or not preprocessed_pattern:
        return ""
    
    # Parse the preprocessed pattern 
    pattern_tokens = preprocessed_pattern.split()
    
    # Build the reconstruction map
    reconstruction_parts = []
    raw_pos = 0
    
    for i, token in enumerate(pattern_tokens):
        if _is_enclosed_symbol(token):
            # Keep masked form for enclosed symbols like [X], "X", {X}
            reconstruction_parts.append(token)
            # Skip over the corresponding content in raw log
            raw_pos = _skip_enclosed_content_in_raw(raw_log, token, raw_pos)
            
        elif _is_standalone_symbol(token):
            # Keep standalone symbols as-is (like -, ., :)
            reconstruction_parts.append(token)
            # Skip over the symbol in raw log
            raw_pos = _skip_symbol_in_raw(raw_log, token, raw_pos)
            
        else:
            # This shouldn't happen in well-formed patterns, but handle gracefully
            reconstruction_parts.append(token)
    
    # Fill gaps between symbols with real content
    filled_reconstruction = _fill_content_gaps(raw_log, reconstruction_parts, preprocessed_pattern)
    
    return filled_reconstruction

def _is_enclosed_symbol(token: str) -> bool:
    """Check if token is an enclosed symbol like [X], "X", {X}."""
    enclosed_patterns = ['"X"', '[X]', '{X}', '<X>']
    return token in enclosed_patterns

def _is_standalone_symbol(token: str) -> bool:
    """Check if token is a standalone symbol like -, ., :."""
    return re.match(r'^[^\w\sX]+$', token) and 'X' not in token

def _skip_enclosed_content_in_raw(raw_log: str, masked_token: str, start_pos: int) -> int:
    """Skip over the content in raw log that corresponds to the masked token."""
    # Extract the symbol pattern (remove X)
    if masked_token == '"X"':
        open_char, close_char = '"', '"'
    elif masked_token == '[X]':
        open_char, close_char = '[', ']'
    elif masked_token == '{X}':
        open_char, close_char = '{', '}'
    elif masked_token == '<X>':
        open_char, close_char = '<', '>'
    else:
        return start_pos  # Unknown pattern, don't skip
    
    # Find the opening symbol
    open_pos = raw_log.find(open_char, start_pos)
    if open_pos < 0:
        return start_pos
    
    # Find the closing symbol
    close_pos = raw_log.find(close_char, open_pos + 1)
    if close_pos < 0:
        return len(raw_log)
    
    return close_pos + 1

def _skip_symbol_in_raw(raw_log: str, symbol_token: str, start_pos: int) -> int:
    """Skip over the symbol in raw log."""
    # Find the first character of the symbol in the raw log
    for char in symbol_token:
        pos = raw_log.find(char, start_pos)
        if pos >= 0:
            return pos + 1
    return start_pos

def _fill_content_gaps(raw_log: str, reconstruction_parts: List[str], pattern: str) -> str:
    """Fill gaps between symbols with actual content from raw log."""
    pattern_tokens = pattern.split()
    result_parts = []
    raw_pos = 0
    
    for i, token in enumerate(pattern_tokens):
        # Add content before this token (if this is not the first token)
        if i == 0:
            # Handle content before first symbol
            first_symbol_pos = _find_first_symbol_position(raw_log, token)
            if first_symbol_pos > 0:
                pre_content = raw_log[:first_symbol_pos].strip()
                if pre_content:
                    result_parts.append(pre_content)
            raw_pos = first_symbol_pos
        
        # Add the token itself (symbol or masked content)
        result_parts.append(reconstruction_parts[i])
        
        # Skip over this token in the raw log
        if _is_enclosed_symbol(token):
            raw_pos = _skip_enclosed_content_in_raw(raw_log, token, raw_pos)
        elif _is_standalone_symbol(token):
            raw_pos = _skip_symbol_in_raw(raw_log, token, raw_pos)
        
        # Add content between this token and the next (if not last token)
        if i < len(pattern_tokens) - 1:
            next_token = pattern_tokens[i + 1]
            next_symbol_pos = _find_next_symbol_position(raw_log, next_token, raw_pos)
            
            if next_symbol_pos > raw_pos:
                between_content = raw_log[raw_pos:next_symbol_pos].strip()
                if between_content:
                    result_parts.append(between_content)
            raw_pos = next_symbol_pos
    
    # Handle content after last symbol
    if raw_pos < len(raw_log):
        post_content = raw_log[raw_pos:].strip()
        if post_content:
            result_parts.append(post_content)
    
    return ' '.join(result_parts)

def _find_first_symbol_position(raw_log: str, token: str) -> int:
    """Find the position of the first symbol in raw log."""
    if _is_enclosed_symbol(token):
        # Find opening bracket/quote
        open_chars = {'"X"': '"', '[X]': '[', '{X}': '{', '<X>': '<'}
        open_char = open_chars.get(token, '')
        if open_char:
            pos = raw_log.find(open_char)
            return pos if pos >= 0 else 0
    elif _is_standalone_symbol(token):
        # Find first character of symbol
        for char in token:
            pos = raw_log.find(char)
            if pos >= 0:
                return pos
    return 0

def _find_next_symbol_position(raw_log: str, next_token: str, start_pos: int) -> int:
    """Find the position of the next symbol in raw log."""
    if _is_enclosed_symbol(next_token):
        # Find opening bracket/quote
        open_chars = {'"X"': '"', '[X]': '[', '{X}': '{', '<X>': '<'}
        open_char = open_chars.get(next_token, '')
        if open_char:
            pos = raw_log.find(open_char, start_pos)
            return pos if pos >= 0 else len(raw_log)
    elif _is_standalone_symbol(next_token):
        # Find first character of symbol
        for char in next_token:
            pos = raw_log.find(char, start_pos)
            if pos >= 0:
                return pos
    return len(raw_log)

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

@mcp.tool()
def log_query(
    log_cluster_id: int,
    pattern: str,
    limit: int,
    offset: int = 0,
    shallow: bool = False
) -> Dict[str, Any]:
    """
    Query log entries by cluster ID and pattern with pagination support.
    
    Args:
        log_cluster_id: Cluster ID to filter by (exact match)
        pattern: Pattern substring to filter by (partial match)
        limit: Maximum number of results to return
        offset: Number of results to skip (default: 0)
        shallow: Return shallow-formatted logs instead of raw logs (default: False)
    
    Returns:
        Dictionary containing:
        - success: Boolean indicating operation success
        - data: {
            text: String containing logs separated by newlines
            metadata: {
                current_offset: Current offset position
                returned_count: Number of logs returned
                requested_limit: Requested limit
                isEnd: Boolean indicating if no more results available
                total_matches: Total number of matching logs found
                cluster_id: Queried cluster ID
                pattern_filter: Applied pattern filter
            }
        }
        - error/message: Error information if unsuccessful
    """
    
    try:
        # Parameter validation
        if limit <= 0:
            return {
                "success": False,
                "error": "Invalid limit",
                "message": "Limit must be greater than 0"
            }
        
        if offset < 0:
            return {
                "success": False,
                "error": "Invalid offset",
                "message": "Offset must be non-negative"
            }
        
        if limit > 1000:
            return {
                "success": False,
                "error": "Limit too large",
                "message": "Limit cannot exceed 1000 to prevent performance issues"
            }
        
        # Load CSV data
        script_dir = Path(__file__).parent
        csv_path = script_dir.parent / "data" / "mcp" / "data.log"
        
        if not csv_path.exists():
            return {
                "success": False,
                "error": "Data file not found",
                "message": f"Log data file not found at: {csv_path.absolute()}",
                "suggestion": "Make sure you've run cluster.py to generate log data"
            }
        
        # Read and filter CSV data
        matching_logs = []
        total_processed = 0
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            
            for row in csv_reader:
                try:
                    row_cluster_id = int(row['cluster'])
                    row_pattern = row['pattern']
                    row_raw_log = row['raw_log']
                    
                    # Apply filters
                    if (row_cluster_id == log_cluster_id and 
                        pattern.lower() in row_pattern.lower()):
                        matching_logs.append({
                            'cluster': row_cluster_id,
                            'raw_log': row_raw_log,
                            'pattern': row_pattern
                        })
                        
                    total_processed += 1
                    
                except (ValueError, KeyError) as e:
                    # Skip malformed rows
                    continue
        
        # Calculate total matches
        total_matches = len(matching_logs)
        
        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        paginated_logs = matching_logs[start_idx:end_idx]
        
        # Determine if this is the end
        is_end = end_idx >= total_matches
        
        # Format log output
        formatted_logs = []
        for log_entry in paginated_logs:
            if shallow:
                # Apply shallow extraction
                try:
                    shallow_log = reconstruct_log(log_entry['raw_log'], log_entry['pattern'])
                    formatted_logs.append(shallow_log)
                except Exception:
                    # Fallback to raw log if shallow extraction fails
                    formatted_logs.append(log_entry['raw_log'])
            else:
                # Use raw log
                formatted_logs.append(log_entry['raw_log'])
        
        # Join logs with newlines
        log_text = '\n'.join(formatted_logs)
        
        return {
            "success": True,
            "data": {
                "text": log_text,
                "metadata": {
                    "current_offset": offset,
                    "returned_count": len(paginated_logs),
                    "requested_limit": limit,
                    "isEnd": is_end,
                    "total_matches": total_matches,
                    "cluster_id": log_cluster_id,
                    "pattern_filter": pattern,
                    "shallow_mode": shallow
                }
            }
        }
        
    except FileNotFoundError as e:
        return {
            "success": False,
            "error": "Data file access error",
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
        print(f"❌ Test failed: {e}")
    
    # Test log_query tool
    print("\nTesting log_query tool...")
    
    try:
        # Test with sample parameters
        query_result = log_query(
            log_cluster_id=3,
            pattern=": :",
            limit=5,
            offset=0,
            shallow=False
        )
        
        if query_result.get("success"):
            data = query_result["data"]
            metadata = data["metadata"]
            print(f"✅ Query successful: found {metadata['total_matches']} total matches")
            print(f"   Returned {metadata['returned_count']} logs from cluster {metadata['cluster_id']}")
            print(f"   Pattern filter: '{metadata['pattern_filter']}'")
            print(f"   Is end: {metadata['isEnd']}")
            
            if data["text"]:
                first_log = data["text"].split('\n')[0]
                print(f"   First log: {first_log[:80]}...")
        else:
            print(f"❌ Query failed: {query_result.get('error', 'Unknown error')}")
            print(f"   Message: {query_result.get('message', 'No details available')}")
            
    except Exception as e:
        print(f"❌ Query test failed: {e}")
    
    print("\n💡 To use as MCP server, run: uv run mcp dev src/log_mcp.py")
    print("   Or integrate with Claude Desktop via MCP configuration")
    print("\n🔧 Available tools:")
    print("   - log_schema(): Get all cluster IDs and patterns")
    print("   - log_query(cluster_id, pattern, limit, offset, shallow): Query logs with filters")