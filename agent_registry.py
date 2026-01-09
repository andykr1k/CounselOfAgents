"""Agent registry for managing available agents."""

from typing import Dict, List, Optional
from agent import BaseAgent, AgentCapability, Task


class AgentRegistry:
    """Registry for managing and discovering agents."""
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
    
    def register(self, agent: BaseAgent) -> None:
        """Register an agent in the registry."""
        if agent.agent_id in self._agents:
            raise ValueError(f"Agent with ID '{agent.agent_id}' already registered")
        self._agents[agent.agent_id] = agent
    
    def unregister(self, agent_id: str) -> None:
        """Unregister an agent from the registry."""
        if agent_id not in self._agents:
            raise ValueError(f"Agent with ID '{agent_id}' not found")
        del self._agents[agent_id]
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents."""
        return list(self._agents.values())
    
    def find_agents_by_capability(self, capability: AgentCapability) -> List[BaseAgent]:
        """Find all agents that have a specific capability."""
        return [agent for agent in self._agents.values() if capability in agent.capabilities]
    
    def find_agents_for_task(self, task: Task) -> List[BaseAgent]:
        """Find all agents that can handle a given task."""
        return [agent for agent in self._agents.values() if agent.can_handle(task)]
    
    def get_agent_info(self) -> List[Dict]:
        """Get information about all registered agents."""
        return [agent.get_info() for agent in self._agents.values()]
    
    def get_agents_summary(self) -> str:
        """Get a formatted summary of all agents for LLM context."""
        summaries = []
        for agent in self._agents.values():
            info = agent.get_info()
            summaries.append(
                f"{info['name']} ({info['agent_id']}): {info.get('description', 'No description')}"
            )
        return "\n".join(summaries)
