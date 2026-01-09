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
        
        system_prompt = f"""You are an intelligent task router for a multi-agent system. Your job is to analyze tasks and break them down into subtasks with dependencies, assigning each to the most appropriate agent(s).

        Available Agents:
        {agents_info}

        For each task, you must:
        1. Break down complex tasks into subtasks if needed (e.g., "make a project" → "generate code" then "create file")
        2. Identify which agent(s) should handle each subtask
        3. Determine dependencies between subtasks
        4. Provide reasoning

        Respond in this exact format (one line per subtask):
        SUBTASK: <subtask description>
        AGENT_ID: <agent_id>
        CAPABILITIES: <comma-separated capabilities>
        DEPENDS_ON: <comma-separated subtask numbers or "none">
        REASONING: <brief explanation>

        Example:
        SUBTASK: Generate Python calculator code
        AGENT_ID: coding_agent
        CAPABILITIES: code_generation
        DEPENDS_ON: none
        REASONING: Need to generate code first

        SUBTASK: Create calculator.py file with the code
        AGENT_ID: filesystem_agent
        CAPABILITIES: file_operations
        DEPENDS_ON: 1
        REASONING: Need to create file after code is generated

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
        """Parse the LLM's classification response, handling multiple subtasks."""
        import re
        
        # Split response into subtask blocks
        subtask_blocks = re.split(r'SUBTASK:', response, re.IGNORECASE)[1:]  # Skip first empty split
        
        if not subtask_blocks:
            # Fallback to single task parsing
            return self._parse_single_task_response(response)
        
        subtasks = []
        for i, block in enumerate(subtask_blocks):
            # Extract subtask description
            desc_match = re.search(r'^(.+?)(?=\nAGENT_ID:)', block, re.IGNORECASE | re.DOTALL)
            description = desc_match.group(1).strip() if desc_match else f"Subtask {i+1}"
            
            # Extract agent ID
            agent_match = re.search(r'AGENT_ID:\s*(\w+)', block, re.IGNORECASE)
            agent_id = agent_match.group(1) if agent_match else None
            
            # Extract capabilities
            cap_match = re.search(r'CAPABILITIES:\s*([^\n]+)', block, re.IGNORECASE)
            capabilities_str = cap_match.group(1) if cap_match else ""
            capabilities_list = [c.strip() for c in capabilities_str.split(',') if c.strip()]
            
            # Map to enum
            capability_map = {cap.value: cap for cap in AgentCapability}
            required_capabilities = []
            for cap_str in capabilities_list:
                cap_str = cap_str.lower().strip()
                for key, value in capability_map.items():
                    if cap_str in key.lower() or key.lower() in cap_str:
                        required_capabilities.append(value)
                        break
            
            # Extract dependencies
            dep_match = re.search(r'DEPENDS_ON:\s*([^\n]+)', block, re.IGNORECASE)
            depends_on_str = dep_match.group(1).strip() if dep_match else "none"
            depends_on = [] if depends_on_str.lower() == "none" else depends_on_str.split(',')
            
            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\nSUBTASK:|$)', block, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            
            subtasks.append({
                "description": description,
                "agent_id": agent_id,
                "capabilities": list(set(required_capabilities)) if required_capabilities else [AgentCapability.ANALYSIS],
                "depends_on": [d.strip() for d in depends_on if d.strip()],
                "reasoning": reasoning
            })
        
        # If no subtasks parsed, fallback to single task
        if not subtasks:
            return self._parse_single_task_response(response)
        
        return {
            "subtasks": subtasks,
            "reasoning": "\n".join([f"{i+1}. {s['reasoning']}" for i, s in enumerate(subtasks)])
        }
    
    def _parse_single_task_response(self, response: str) -> Dict[str, Any]:
        """Parse a single task response (fallback)."""
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
        Automatically breaks down complex tasks into subtasks with dependencies.
        
        Args:
            prompt: The user's prompt
            task_id_prefix: Prefix for task IDs
            
        Returns:
            List of Task objects with dependencies
        """
        classification = await self.classify_async(prompt)
        
        # Check if LLM broke down into multiple subtasks
        if "subtasks" in classification:
            tasks = []
            subtasks = classification["subtasks"]
            
            for i, subtask_info in enumerate(subtasks):
                # Resolve dependencies (convert subtask numbers to task IDs)
                dependencies = []
                for dep in subtask_info.get("depends_on", []):
                    try:
                        # If it's a number, reference previous subtask
                        dep_num = int(dep.strip())
                        if 1 <= dep_num <= len(subtasks):
                            dependencies.append(f"{task_id_prefix}_{dep_num - 1}")
                    except ValueError:
                        # If it's already a task ID, use it
                        dependencies.append(dep.strip())
                
                task = Task(
                    id=f"{task_id_prefix}_{i}",
                    description=subtask_info["description"],
                    required_capabilities=subtask_info["capabilities"],
                    dependencies=dependencies,
                    metadata={
                        "original_prompt": prompt,
                        "recommended_agents": [subtask_info["agent_id"]] if subtask_info.get("agent_id") else [],
                        "classification_reasoning": subtask_info.get("reasoning", ""),
                        "subtask_index": i,
                        "total_subtasks": len(subtasks)
                    }
                )
                tasks.append(task)
            
            return tasks
        else:
            # Single task (fallback or simple task)
            task = Task(
                id=f"{task_id_prefix}_0",
                description=prompt,
                required_capabilities=classification["capabilities"],
                dependencies=[],
                metadata={
                    "original_prompt": prompt,
                    "recommended_agents": classification.get("agent_ids", []),
                    "classification_reasoning": classification.get("reasoning", "")
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
