"""LLM-based task classification system that intelligently routes tasks to agents."""

from typing import List, Dict, Any, Optional
from agent import AgentCapability, Task, BaseAgent
from agent_registry import AgentRegistry

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class TaskClassifier:
    """LLM-based classifier that understands all agents and routes tasks intelligently using only reasoning."""
    
    def __init__(self, agent_registry: Optional[AgentRegistry] = None, model_name: str = "gpt2"):
        """
        Initialize the task classifier.
        
        Args:
            agent_registry: Registry containing all available agents
            model_name: Hugging Face model to use for classification
        """
        self.registry = agent_registry
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = "cpu"
        
        if HF_AVAILABLE:
            self._load_model()
        else:
            raise RuntimeError("transformers and torch are required for task classification. Install with: pip install transformers torch")
    
    def _load_model(self):
        """Load the classification model."""
        if not HF_AVAILABLE:
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            if torch.cuda.is_available():
                self.device = "cuda"
                self.model = self.model.to(self.device)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            raise RuntimeError(f"Could not load classification model: {e}")
    
    def _get_agents_info(self) -> str:
        """Get formatted information about all available agents from the registry."""
        if not self.registry:
            return "No agents available"
        
        agents_info = []
        for agent in self.registry.get_all_agents():
            info = agent.get_info()
            capabilities = ', '.join(info["capabilities"])
            description = info.get("description", "No description available")
            
            agents_info.append(
                f"- {info['name']} (ID: {info['agent_id']})\n"
                f"  Capabilities: {capabilities}\n"
                f"  Description: {description}"
            )
        
        return "\n\n".join(agents_info)
    
    def _build_classification_prompt(self, prompt: str) -> str:
        """Build the prompt for the LLM to classify and route the task."""
        agents_info = self._get_agents_info()
        
        system_prompt = f"""You are an intelligent task router for a multi-agent system. Your job is to analyze tasks and assign them to the most appropriate agent(s) based on their descriptions and capabilities.

        Available Agents:
        {agents_info}

        For each task, you must:
        1. Analyze the task requirements
        2. Identify which agent(s) should handle it based on their descriptions and capabilities
        3. Determine the required capabilities
        4. Provide reasoning for your choice

        Respond in this exact format:
        AGENT_ID: <agent_id>
        CAPABILITIES: <comma-separated capabilities>
        REASONING: <brief explanation of why this agent is best suited>

        If multiple agents are needed, provide multiple lines with the same format.

        Task: {prompt}

        Response:"""
        
        return system_prompt
    
    async def _classify_with_model(self, prompt: str) -> Dict[str, Any]:
        """Use the LLM to classify the task using only reasoning."""
        if not self.pipeline:
            raise RuntimeError("Classification model not available")
        
        classification_prompt = self._build_classification_prompt(prompt)
        
        try:
            # Run pipeline in executor to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.pipeline(
                    classification_prompt,
                    max_length=512,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    return_full_text=False,
                    num_return_sequences=1
                )
            )
            
            if isinstance(result, list) and len(result) > 0:
                response = result[0].get("generated_text", str(result[0]))
            else:
                response = str(result)
            
            # Parse the response
            return self._parse_classification_response(response, prompt)
        
        except Exception as e:
            raise RuntimeError(f"Error in model classification: {e}")
    
    def _parse_classification_response(self, response: str, original_prompt: str) -> Dict[str, Any]:
        """Parse the LLM's classification response."""
        import re
        
        # Extract agent IDs
        agent_ids = re.findall(r'AGENT_ID:\s*(\w+)', response, re.IGNORECASE)
        
        # Extract capabilities
        capabilities_match = re.search(r'CAPABILITIES:\s*([^\n]+)', response, re.IGNORECASE)
        capabilities_str = capabilities_match.group(1) if capabilities_match else ""
        capabilities_list = [c.strip() for c in capabilities_str.split(',') if c.strip()]
        
        # Map capability strings to enum values
        capability_map = {cap.value: cap for cap in AgentCapability}
        required_capabilities = []
        for cap_str in capabilities_list:
            cap_str = cap_str.lower().strip()
            for key, value in capability_map.items():
                if cap_str in key.lower() or key.lower() in cap_str:
                    required_capabilities.append(value)
                    break
        
        # If no capabilities found, try to infer from agent IDs
        if not required_capabilities and agent_ids and self.registry:
            for agent_id in agent_ids:
                agent = self.registry.get_agent(agent_id)
                if agent:
                    required_capabilities.extend(agent.capabilities)
        
        # Default to analysis if nothing found
        if not required_capabilities:
            required_capabilities = [AgentCapability.ANALYSIS]
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\nAGENT_ID:|$)', response, re.IGNORECASE | re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response
        
        return {
            "agent_ids": agent_ids,
            "capabilities": list(set(required_capabilities)),
            "reasoning": reasoning
        }
    
    def classify(self, prompt: str) -> List[AgentCapability]:
        """
        Classify a prompt and return required capabilities (synchronous).
        Note: This is a fallback method. Use classify_async for full LLM reasoning.
        """
        # This should not be used in normal flow, but kept for compatibility
        # It will raise an error if model is not available
        raise RuntimeError("Synchronous classification not supported. Use classify_async() for LLM-based reasoning.")
    
    async def classify_async(self, prompt: str) -> Dict[str, Any]:
        """
        Classify a prompt asynchronously using LLM reasoning.
        
        Args:
            prompt: The user's prompt/task description
            
        Returns:
            Dictionary with agent_ids, capabilities, and reasoning
        """
        return await self._classify_with_model(prompt)
    
    def create_tasks_from_prompt(
        self,
        prompt: str,
        task_id_prefix: str = "task"
    ) -> List[Task]:
        """
        Synchronous version - should not be used. Use create_tasks_from_prompt_async instead.
        """
        raise RuntimeError("Synchronous task creation not supported. Use create_tasks_from_prompt_async() for LLM-based reasoning.")
    
    async def create_tasks_from_prompt_async(
        self,
        prompt: str,
        task_id_prefix: str = "task"
    ) -> List[Task]:
        """
        Asynchronously break down a prompt into tasks with intelligent agent assignment using LLM reasoning.
        
        Args:
            prompt: The user's prompt
            task_id_prefix: Prefix for task IDs
            
        Returns:
            List of Task objects
        """
        classification = await self.classify_async(prompt)
        
        # Create task with recommended agent IDs in metadata
        task = Task(
            id=f"{task_id_prefix}_0",
            description=prompt,
            required_capabilities=classification["capabilities"],
            dependencies=[],
            metadata={
                "original_prompt": prompt,
                "recommended_agents": classification["agent_ids"],
                "classification_reasoning": classification["reasoning"]
            }
        )
        
        return [task]
    
    def create_subtasks(
        self,
        main_task: Task,
        subtask_descriptions: List[str],
        dependencies: List[List[str]] = None
    ) -> List[Task]:
        """
        Create subtasks from a main task.
        Note: This uses async classification internally, so it's less efficient.
        For better performance, use create_tasks_from_prompt_async directly.
        """
        import asyncio
        
        subtasks = []
        for i, desc in enumerate(subtask_descriptions):
            # Use async classification
            try:
                loop = asyncio.get_event_loop()
                classification = loop.run_until_complete(self.classify_async(desc))
                capabilities = classification["capabilities"]
            except:
                # Fallback if async not available
                capabilities = [AgentCapability.ANALYSIS]
            
            deps = dependencies[i] if dependencies and i < len(dependencies) else []
            
            subtask = Task(
                id=f"{main_task.id}_sub_{i}",
                description=desc,
                required_capabilities=capabilities,
                dependencies=deps,
                metadata={"parent_task": main_task.id}
            )
            subtasks.append(subtask)
        
        return subtasks
