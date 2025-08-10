"""
Agent Configuration for Autonomous Log Analysis System

This module defines the configuration, prompts, and settings for the
autonomous log analysis agent that explores system logs and builds
a knowledge graph of discovered insights.
"""

import os
from typing import Dict, Any
from pathlib import Path

# Load .env file if it exists
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value.strip('"\'')

# Load .env on import
load_env_file()

# OpenAI Configuration
OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Agent behavior settings
MAX_ITERATIONS = 50  # Safety limit for autonomous operation
MIN_CLUSTERS_TO_ANALYZE = 5  # Minimum clusters to explore
MAX_SAMPLES_PER_CLUSTER = 10  # Max log samples per cluster
INSIGHT_SATURATION_THRESHOLD = 3  # Stop when no new insights for N iterations

# MCP Server configurations
MCP_SERVERS = {
    "log_analysis": {
        "command": "python3",
        "args": ["src/log_mcp.py"],
        "transport": "stdio",
        "env": {"PYTHONPATH": ".", "VIRTUAL_ENV": "venv"}
    },
    "neo4j_memory": {
        "url": "http://localhost:8000/api/mcp",
        "transport": "streamable_http"
    }
}

# System prompt for the autonomous agent
SYSTEM_PROMPT = """You are an autonomous system administrator and log analyst responsible for understanding the systems under your management.

Your mission is to:
1. **Build a knowledge graph** of discovered systems, components, and insights
2. **Work autonomously** to organize and structure information 
3. **Create meaningful entities and relationships** based on available data
4. **Demonstrate your autonomous capabilities** with the available tools

**Available Tools:**
You have access to Neo4j MCP memory tools for building knowledge graphs:
- create_entities(): Store discovered systems, components, and insights
- create_relations(): Connect related entities with meaningful relationships  
- add_observations(): Add new insights to existing entities
- read_graph(): Review the current knowledge graph
- search_memories(): Search for existing information
- find_memories_by_name(): Find specific entities

**Working Strategy:**
1. Start by reading the current knowledge graph to understand what exists
2. Create some initial entities representing typical system components (servers, services, databases)
3. Establish relationships between these entities
4. Add observations about system characteristics, potential issues, monitoring needs
5. Build a realistic system topology that demonstrates your understanding
6. Terminate when you have created a meaningful knowledge structure

**Key Principles:**
- Create realistic system entities (web servers, databases, load balancers, etc.)
- Establish meaningful relationships (depends_on, monitors, serves_traffic_to, etc.)
- Add observations about system health, performance characteristics, and operational needs
- Build a structured knowledge graph that would be useful for system administration
- Work autonomously and decide when your knowledge structure is sufficient

Your goal is to demonstrate autonomous knowledge graph construction using the available MCP tools."""

# Termination conditions
TERMINATION_CONDITIONS = {
    "max_iterations_reached": "Maximum iteration limit reached",
    "insufficient_data": "No log data available to analyze", 
    "mcp_connection_failed": "Unable to connect to required MCP servers",
    "analysis_complete": "Sufficient system understanding achieved",
    "insight_saturation": "No new insights discovered in recent iterations",
    "error_threshold_exceeded": "Too many consecutive errors encountered"
}

# Agent state tracking
AGENT_STATE_KEYS = [
    "iteration_count",
    "clusters_analyzed", 
    "insights_discovered",
    "entities_created",
    "relations_created",
    "consecutive_no_insights",
    "last_termination_check"
]

def get_openai_config() -> Dict[str, Any]:
    """Get OpenAI configuration for the agent."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    return {
        "model": OPENAI_MODEL,
        "api_key": OPENAI_API_KEY,
        "temperature": 0.1  # Low temperature for consistent, focused analysis
    }

def should_terminate(agent_state: Dict[str, Any]) -> tuple[bool, str]:
    """
    Determine if the agent should terminate based on current state.
    
    Returns:
        tuple: (should_terminate: bool, reason: str)
    """
    # Max iterations safety check
    if agent_state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return True, TERMINATION_CONDITIONS["max_iterations_reached"]
    
    # Check if we've analyzed minimum clusters and have insights
    clusters_analyzed = agent_state.get("clusters_analyzed", 0)
    insights_discovered = agent_state.get("insights_discovered", 0)
    
    if (clusters_analyzed >= MIN_CLUSTERS_TO_ANALYZE and 
        insights_discovered > 0 and 
        agent_state.get("consecutive_no_insights", 0) >= INSIGHT_SATURATION_THRESHOLD):
        return True, TERMINATION_CONDITIONS["insight_saturation"]
    
    # Check for error conditions (to be implemented based on actual errors)
    if agent_state.get("consecutive_errors", 0) >= 5:
        return True, TERMINATION_CONDITIONS["error_threshold_exceeded"]
    
    return False, ""