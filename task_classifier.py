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

        IMPORTANT: Always break down complex tasks into multiple subtasks. For example:
        - "Make a python calculator file" should become: (1) Generate code, (2) Navigate to directory, (3) Create file, (4) Write code to file
        - "Create a project" should become: (1) Generate code, (2) Create directory structure, (3) Create files, (4) Write code to files

        For each task, you must:
        1. ALWAYS break down complex tasks into subtasks (even simple tasks should be broken down if they involve multiple steps)
        2. Identify which agent(s) should handle each subtask
        3. Determine dependencies between subtasks (earlier subtasks have lower numbers)
        4. Provide reasoning

        Respond in this exact format (one subtask per block):
        SUBTASK: <subtask description>
        AGENT_ID: <agent_id>
        CAPABILITIES: <comma-separated capabilities>
        DEPENDS_ON: <subtask number or "none">
        REASONING: <brief explanation>

        Example for "Make a python calculator file at root of my system":
        SUBTASK: Generate Python calculator code with basic operations (add, subtract, multiply, divide)
        AGENT_ID: coding_agent
        CAPABILITIES: code_generation
        DEPENDS_ON: none
        REASONING: First step is to generate the calculator code

        SUBTASK: Navigate to root directory of the system
        AGENT_ID: filesystem_agent
        CAPABILITIES: file_operations
        DEPENDS_ON: none
        REASONING: Can navigate to directory independently

        SUBTASK: Create calculator.py file at root directory
        AGENT_ID: filesystem_agent
        CAPABILITIES: file_operations
        DEPENDS_ON: 2
        REASONING: Create file after navigating to root directory

        SUBTASK: Write the generated calculator code to calculator.py file
        AGENT_ID: filesystem_agent
        CAPABILITIES: file_operations
        DEPENDS_ON: 1, 3
        REASONING: Write code to file after both code is generated and file is created

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
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, create new one (shouldn't happen in async context)
                loop = asyncio.new_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.pipeline(
                    classification_prompt,
                    max_length=1024,  # Increased for better multi-subtask responses
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    return_full_text=False,
                    num_return_sequences=1,
                    truncation=True
                )
            )
            
            if isinstance(result, list) and len(result) > 0:
                response = result[0].get("generated_text", str(result[0]))
            else:
                response = str(result)
            
            # Log the raw response for debugging
            import logging
            logging.debug(f"Task classifier raw response: {response[:500]}")
            
            # Parse the response
            parsed = self._parse_classification_response(response, prompt)
            
            # Log parsed result
            if "subtasks" in parsed:
                logging.debug(f"Parsed {len(parsed['subtasks'])} subtasks")
            else:
                logging.debug("Fell back to single task parsing")
            
            return parsed
        
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
            if depends_on_str.lower() == "none" or not depends_on_str:
                depends_on = []
            else:
                # Handle comma-separated or space-separated numbers
                depends_on = [d.strip() for d in re.split(r'[, ]+', depends_on_str) if d.strip()]
            
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
    
    def _programmatic_breakdown(self, prompt: str) -> Dict[str, Any]:
        """
        Programmatic fallback to break down common task patterns.
        Used when LLM fails to break down tasks properly.
        """
        import re
        
        prompt_lower = prompt.lower()
        
        # Pattern: "make/create/write a [language] [type] file [location]"
        file_pattern = re.search(
            r'(?:make|create|write|generate)\s+(?:a\s+)?(?:python|java|javascript|typescript|c\+\+|c|go|rust|html|css|json|yaml|xml|markdown|txt|text)?\s*(\w+)?\s*(?:file|script|program|code|application)(?:\s+(?:at|in|to)\s+(.+))?',
            prompt_lower
        )
        
        if file_pattern:
            file_type = file_pattern.group(1) or "file"
            location = file_pattern.group(2) or "current directory"
            
            # Extract file extension or name from original prompt
            file_name_match = re.search(r'(\w+\.(?:py|js|ts|java|cpp|c|go|rs|html|css|json|yaml|xml|md|txt))', prompt, re.IGNORECASE)
            file_name = file_name_match.group(1) if file_name_match else f"{file_type}.py"
            
            subtasks = [
                {
                    "description": f"Generate {file_name} code based on: {prompt}",
                    "agent_id": "coding_agent",
                    "capabilities": [AgentCapability.CODE_GENERATION],
                    "depends_on": [],
                    "reasoning": "First, generate the code for the file"
                },
                {
                    "description": f"Navigate to {location}",
                    "agent_id": "filesystem_agent",
                    "capabilities": [AgentCapability.FILE_OPERATIONS],
                    "depends_on": [],
                    "reasoning": f"Navigate to the target location: {location}"
                },
                {
                    "description": f"Create {file_name} file at {location}",
                    "agent_id": "filesystem_agent",
                    "capabilities": [AgentCapability.FILE_OPERATIONS],
                    "depends_on": ["2"],  # Depends on subtask 2 (navigate)
                    "reasoning": "Create the file after navigating to the location"
                },
                {
                    "description": f"Write the generated code to {file_name}",
                    "agent_id": "filesystem_agent",
                    "capabilities": [AgentCapability.FILE_OPERATIONS],
                    "depends_on": ["1", "3"],  # Depends on subtask 1 (generate code) and 3 (create file)
                    "reasoning": "Write code to file after both code is generated and file is created"
                }
            ]
            
            return {
                "subtasks": subtasks,
                "reasoning": "Programmatically broken down into: generate code, navigate, create file, write code"
            }
        
        return None
    
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
        
        # If LLM didn't break down into subtasks, try programmatic fallback
        if "subtasks" not in classification:
            programmatic = self._programmatic_breakdown(prompt)
            if programmatic:
                # Log that we're using programmatic breakdown
                import logging
                logging.info("Using programmatic task breakdown (LLM did not generate subtasks)")
                classification = programmatic
        
        # Check if we have subtasks (either from LLM or programmatic)
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
                        if 1 <= dep_num <= len(subtasks) and dep_num != i + 1:  # Can't depend on itself
                            dep_task_id = f"{task_id_prefix}_{dep_num - 1}"
                            # Validate that the dependency task will exist
                            if dep_num - 1 < len(subtasks):
                                dependencies.append(dep_task_id)
                    except ValueError:
                        # If it's already a task ID, use it (but validate it's in our subtask range)
                        dep_id = dep.strip()
                        # Check if it's a valid task ID format
                        if dep_id.startswith(task_id_prefix):
                            dependencies.append(dep_id)
                
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
                # Try to get running loop, or create new one if needed
                try:
                    loop = asyncio.get_running_loop()
                    # If we're in an async context, we can't use run_until_complete
                    # This method should not be called from async context
                    raise RuntimeError("create_subtasks cannot be called from async context. Use create_tasks_from_prompt_async instead.")
                except RuntimeError:
                    # No running loop, create new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        classification = loop.run_until_complete(self.classify_async(desc))
                        capabilities = classification["capabilities"]
                    finally:
                        loop.close()
            except Exception:
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
