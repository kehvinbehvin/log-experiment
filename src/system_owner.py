#!/usr/bin/env python3
"""
Autonomous Log Analysis Agent

This module implements an autonomous agent using LangGraph that explores
log data via MCP servers and builds a knowledge graph of system insights.

The agent operates autonomously, making intelligent decisions about which
logs to analyze and when it has sufficient understanding to terminate.
"""

import asyncio
import json
import logging
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage

from agent_config import (
    MCP_SERVERS, 
    SYSTEM_PROMPT, 
    get_openai_config,
    should_terminate,
    AGENT_STATE_KEYS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutonomousLogAnalyst:
    """
    Autonomous log analysis agent that explores system logs and builds
    a knowledge graph of insights using LangGraph and MCP servers.
    """
    
    def __init__(self):
        self.client: Optional[MultiServerMCPClient] = None
        self.agent = None
        self.agent_state: Dict[str, Any] = {key: 0 for key in AGENT_STATE_KEYS}
        self.agent_state["start_time"] = datetime.now()
        self.tools = []
        
    async def initialize(self) -> bool:
        """
        Initialize the agent with MCP servers and tools.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing Autonomous Log Analysis Agent...")
            
            # Try to initialize MCP servers one by one for better error handling
            working_servers = {}
            
            # Test Neo4j memory server first (HTTP)
            try:
                neo4j_config = MCP_SERVERS["neo4j_memory"]
                test_client = MultiServerMCPClient({"neo4j_memory": neo4j_config})
                test_tools = await test_client.get_tools()
                working_servers["neo4j_memory"] = neo4j_config
                logger.info(f"Neo4j MCP server connected successfully with {len(test_tools)} tools")
                # Clean up test client
                try:
                    await test_client.cleanup()
                except:
                    pass  # Ignore cleanup errors for test client
            except Exception as e:
                logger.warning(f"Neo4j MCP server failed to connect: {e}")
            
            # Test log analysis server (stdio)
            try:
                log_config = MCP_SERVERS["log_analysis"] 
                logger.info("Attempting to connect to log analysis MCP server...")
                
                # Create a separate client for log analysis
                log_test_client = MultiServerMCPClient({"log_analysis": log_config})
                log_tools = await log_test_client.get_tools()
                working_servers["log_analysis"] = log_config
                logger.info(f"Log analysis MCP server connected successfully with {len(log_tools)} tools")
                
                # Clean up test client
                try:
                    await log_test_client.cleanup()
                except:
                    pass  # Ignore cleanup errors for test client
                    
            except Exception as e:
                logger.warning(f"Log analysis MCP server failed to connect: {e}")
                logger.info("Continuing with Neo4j memory server only")
            
            if not working_servers:
                logger.error("No MCP servers available")
                return False
            
            # Initialize MCP client with working servers
            self.client = MultiServerMCPClient(working_servers)
            logger.info(f"Configured MCP servers: {list(working_servers.keys())}")
            
            # Get available tools from MCP servers
            self.tools = await self.client.get_tools()
            logger.info(f"Available tools: {[tool.name for tool in self.tools]}")
            
            if not self.tools:
                logger.error("No tools available from MCP servers")
                return False
            
            # Initialize OpenAI model
            openai_config = get_openai_config()
            model = ChatOpenAI(
                model=openai_config["model"],
                api_key=openai_config["api_key"],
                temperature=openai_config["temperature"]
            )
            
            # Create the ReAct agent
            self.agent = create_react_agent(
                model=model,
                tools=self.tools,
                prompt=SYSTEM_PROMPT
            )
            
            logger.info("Agent initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def run_autonomous_analysis(self) -> Dict[str, Any]:
        """
        Run the autonomous log analysis process.
        
        Returns:
            Dict containing analysis results and final state
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        logger.info("Starting autonomous log analysis...")
        
        # Initial message to kick off the analysis
        initial_message = {
            "messages": [{
                "role": "user", 
                "content": """Begin your autonomous log analysis mission. 

Start by getting an overview of the available log clusters, then systematically explore the system logs to understand:
- What systems are running
- Common patterns and behaviors  
- Error conditions and issues
- Performance characteristics
- Security-related events

Build a knowledge graph of your findings and terminate when you have sufficient understanding of the systems under management.

Remember to work efficiently - focus on insights, not exhaustive log reading."""
            }]
        }
        
        analysis_results = {
            "start_time": self.agent_state["start_time"],
            "iterations": [],
            "final_state": {},
            "termination_reason": "",
            "success": False
        }
        
        try:
            while True:
                self.agent_state["iteration_count"] += 1
                iteration_start = datetime.now()
                
                logger.info(f"Starting iteration {self.agent_state['iteration_count']}")
                
                # Check termination conditions
                should_stop, reason = should_terminate(self.agent_state)
                if should_stop:
                    logger.info(f"Termination condition met: {reason}")
                    analysis_results["termination_reason"] = reason
                    break
                
                try:
                    # Run agent iteration
                    response = await self.agent.ainvoke(initial_message)
                    
                    # Log tool usage from this iteration
                    tool_usage_records = self._log_tool_usage(response)
                    
                    # Extract insights from the response
                    iteration_result = {
                        "iteration": self.agent_state["iteration_count"],
                        "timestamp": iteration_start,
                        "duration": (datetime.now() - iteration_start).total_seconds(),
                        "response_summary": self._extract_response_summary(response),
                        "tool_usage": tool_usage_records,
                        "agent_state": self.agent_state.copy()
                    }
                    
                    analysis_results["iterations"].append(iteration_result)
                    
                    # Update agent state based on response
                    self._update_agent_state(response)
                    
                    # Use the response to continue the conversation
                    if response.get("messages"):
                        # Get the latest assistant message for the next iteration
                        latest_message = response["messages"][-1]
                        # Handle both dict and AIMessage objects safely
                        try:
                            if hasattr(latest_message, 'role'):
                                message_role = latest_message.role
                            elif isinstance(latest_message, dict):
                                message_role = latest_message.get("role", "")
                            else:
                                message_role = ""
                                
                            if message_role == "assistant":
                                initial_message = {
                                    "messages": response["messages"] + [{
                                        "role": "user",
                                        "content": "Continue your analysis. Focus on discovering new insights or determining if you have sufficient understanding to complete your mission."
                                    }]
                                }
                        except AttributeError as attr_error:
                            logger.warning(f"Error accessing message role: {attr_error}")
                            # Continue with current initial_message if we can't determine the role
                    
                except Exception as e:
                    import traceback
                    logger.error(f"Error in iteration {self.agent_state['iteration_count']}: {e}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    self.agent_state["consecutive_errors"] = self.agent_state.get("consecutive_errors", 0) + 1
                    
                    if self.agent_state["consecutive_errors"] >= 3:
                        analysis_results["termination_reason"] = "Too many consecutive errors"
                        break
            
            # Final state
            analysis_results["final_state"] = self.agent_state
            analysis_results["end_time"] = datetime.now()
            analysis_results["total_duration"] = (
                analysis_results["end_time"] - analysis_results["start_time"]
            ).total_seconds()
            analysis_results["success"] = True
            
            logger.info("Autonomous analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Critical error during analysis: {e}")
            analysis_results["error"] = str(e)
            analysis_results["termination_reason"] = f"Critical error: {e}"
            return analysis_results
        
        finally:
            await self.cleanup()
    
    def _extract_response_summary(self, response: Dict[str, Any]) -> str:
        """Extract a summary from the agent response."""
        try:
            if not response.get("messages"):
                return "No response messages"
            
            last_message = response["messages"][-1]
            # Handle both dict and AIMessage objects
            if hasattr(last_message, 'content'):
                content = last_message.content
            else:
                content = last_message.get("content", "")
            
            # Truncate for summary
            if len(content) > 200:
                return content[:200] + "..."
            return content
        except Exception as e:
            return f"Error extracting summary: {e}"
    
    def _update_agent_state(self, response: Dict[str, Any]) -> None:
        """Update agent state based on the response."""
        try:
            # This is a simplified state update - in a real implementation,
            # you'd parse the response more carefully to extract:
            # - Number of clusters analyzed
            # - Insights discovered  
            # - Entities/relations created
            
            # For now, just update basic counters
            content = ""
            if response.get("messages"):
                last_message = response["messages"][-1]
                # Handle both dict and AIMessage objects
                if hasattr(last_message, 'content'):
                    content = last_message.content.lower()
                else:
                    content = last_message.get("content", "").lower()
            
            # Simple heuristics to update state
            if "cluster" in content:
                self.agent_state["clusters_analyzed"] = self.agent_state.get("clusters_analyzed", 0) + 1
            
            if any(word in content for word in ["insight", "pattern", "discovered", "found", "entity", "relation"]):
                self.agent_state["insights_discovered"] = self.agent_state.get("insights_discovered", 0) + 1
                self.agent_state["consecutive_no_insights"] = 0
            else:
                self.agent_state["consecutive_no_insights"] = self.agent_state.get("consecutive_no_insights", 0) + 1
            
            if "create_entities" in content:
                self.agent_state["entities_created"] = self.agent_state.get("entities_created", 0) + 1
                
            if "create_relations" in content:
                self.agent_state["relations_created"] = self.agent_state.get("relations_created", 0) + 1
                
        except Exception as e:
            logger.warning(f"Error updating agent state: {e}")
            # Don't fail the iteration due to state update errors
    
    def _log_tool_usage(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract and log all tool usage from the agent response.
        
        Returns:
            List of tool usage records for analysis
        """
        tool_usage_records = []
        
        try:
            if not response.get("messages"):
                return tool_usage_records
            
            for message in response["messages"]:
                # Check for tool calls in AI messages
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        # Handle tool_call object attributes safely
                        if hasattr(tool_call, 'name'):
                            tool_name = tool_call.name
                        elif isinstance(tool_call, dict):
                            tool_name = tool_call.get("name", "unknown")
                        else:
                            tool_name = "unknown"
                        
                        if hasattr(tool_call, 'id'):
                            tool_id = tool_call.id
                        elif isinstance(tool_call, dict):
                            tool_id = tool_call.get("id", "unknown")
                        else:
                            tool_id = "unknown"
                        
                        if hasattr(tool_call, 'args'):
                            arguments = tool_call.args
                        elif isinstance(tool_call, dict):
                            arguments = tool_call.get("args", {})
                        else:
                            arguments = {}
                        
                        tool_record = {
                            "timestamp": datetime.now().isoformat(),
                            "tool_name": tool_name,
                            "tool_id": tool_id,
                            "arguments": arguments,
                            "type": "tool_call"
                        }
                        tool_usage_records.append(tool_record)
                        
                        # Log the tool usage
                        logger.info(f"🔧 TOOL CALL: {tool_record['tool_name']}")
                        logger.info(f"   Tool ID: {tool_record['tool_id']}")
                        logger.info(f"   Arguments: {json.dumps(tool_record['arguments'], indent=2)}")
                
                # Check for tool results in ToolMessage objects
                elif isinstance(message, ToolMessage):
                    tool_record = {
                        "timestamp": datetime.now().isoformat(),
                        "tool_call_id": getattr(message, 'tool_call_id', 'unknown'),
                        "content": getattr(message, 'content', ''),
                        "type": "tool_result"
                    }
                    tool_usage_records.append(tool_record)
                    
                    # Log the tool result (truncated for readability)
                    content_preview = str(tool_record['content'])[:200]
                    if len(str(tool_record['content'])) > 200:
                        content_preview += "..."
                    
                    logger.info(f"✅ TOOL RESULT for {tool_record['tool_call_id']}")
                    logger.info(f"   Result: {content_preview}")
                
                # Also check if message has tool_calls as dict (alternative format)
                elif hasattr(message, '__dict__') and 'tool_calls' in message.__dict__:
                    tool_calls = message.__dict__.get('tool_calls', [])
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_record = {
                                "timestamp": datetime.now().isoformat(),
                                "tool_name": tool_call.get("name", "unknown"),
                                "tool_id": tool_call.get("id", "unknown"),
                                "arguments": tool_call.get("args", {}),
                                "type": "tool_call"
                            }
                            tool_usage_records.append(tool_record)
                            
                            logger.info(f"🔧 TOOL CALL: {tool_record['tool_name']}")
                            logger.info(f"   Tool ID: {tool_record['tool_id']}")
                            logger.info(f"   Arguments: {json.dumps(tool_record['arguments'], indent=2)}")
                        else:
                            # Handle non-dict tool_call objects
                            try:
                                tool_name = getattr(tool_call, 'name', 'unknown')
                                tool_id = getattr(tool_call, 'id', 'unknown') 
                                arguments = getattr(tool_call, 'args', {})
                                
                                tool_record = {
                                    "timestamp": datetime.now().isoformat(),
                                    "tool_name": tool_name,
                                    "tool_id": tool_id,
                                    "arguments": arguments,
                                    "type": "tool_call"
                                }
                                tool_usage_records.append(tool_record)
                                
                                logger.info(f"🔧 TOOL CALL: {tool_record['tool_name']}")
                                logger.info(f"   Tool ID: {tool_record['tool_id']}")
                                logger.info(f"   Arguments: {json.dumps(tool_record['arguments'], indent=2)}")
                            except Exception as tool_error:
                                logger.warning(f"Error processing tool_call object: {tool_error}")
        
        except Exception as e:
            logger.warning(f"Error extracting tool usage: {e}")
        
        return tool_usage_records
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.client:
                # MCP client cleanup if needed
                pass
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of the analysis results."""
        print("\n" + "="*60)
        print("AUTONOMOUS LOG ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Start Time: {results.get('start_time', 'N/A')}")
        print(f"End Time: {results.get('end_time', 'N/A')}")
        print(f"Total Duration: {results.get('total_duration', 0):.2f} seconds")
        print(f"Total Iterations: {len(results.get('iterations', []))}")
        print(f"Termination Reason: {results.get('termination_reason', 'N/A')}")
        print(f"Success: {results.get('success', False)}")
        
        if results.get('final_state'):
            state = results['final_state']
            print(f"\nFinal State:")
            print(f"  Clusters Analyzed: {state.get('clusters_analyzed', 0)}")
            print(f"  Insights Discovered: {state.get('insights_discovered', 0)}")
            print(f"  Entities Created: {state.get('entities_created', 0)}")
            print(f"  Relations Created: {state.get('relations_created', 0)}")
        
        # Tool usage summary
        all_tool_usage = []
        for iteration in results.get('iterations', []):
            all_tool_usage.extend(iteration.get('tool_usage', []))
        
        if all_tool_usage:
            print(f"\nTool Usage Summary:")
            tool_counts = {}
            for tool_record in all_tool_usage:
                if tool_record['type'] == 'tool_call':
                    tool_name = tool_record['tool_name']
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            
            for tool_name, count in sorted(tool_counts.items()):
                print(f"  {tool_name}: {count} calls")
            
            print(f"  Total Tool Calls: {sum(tool_counts.values())}")
        
        if results.get('error'):
            print(f"\nError: {results['error']}")
        
        print("="*60)

async def main():
    """Main function to run the autonomous log analysis agent."""
    agent = AutonomousLogAnalyst()
    
    try:
        # Initialize the agent
        if not await agent.initialize():
            logger.error("Failed to initialize agent")
            sys.exit(1)
        
        # Run autonomous analysis
        results = await agent.run_autonomous_analysis()
        
        # Print summary
        agent.print_summary(results)
        
        # Save detailed results to file
        results_file = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            json_results = json.loads(json.dumps(results, default=str))
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Return appropriate exit code
        sys.exit(0 if results.get('success', False) else 1)
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())