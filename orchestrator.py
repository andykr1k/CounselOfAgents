"""Orchestrator agent for coordinating multiple specialized agents."""

import asyncio
from typing import List, Dict, Any, Optional
from agent import BaseAgent, Task, AgentResult, AgentCapability
from agent_registry import AgentRegistry
from task_graph import TaskGraph
from task_classifier import TaskClassifier


class Orchestrator:
    """Main orchestrator that coordinates agent execution."""
    
    def __init__(self, agent_registry: AgentRegistry, progress_callback=None, user_input_callback=None):
        """
        Initialize the orchestrator.
        
        Args:
            agent_registry: Registry containing all available agents
            progress_callback: Optional callback function(status, message) for progress updates
            user_input_callback: Optional callback function(question: str) -> str for getting user input
        """
        self.registry = agent_registry
        self.classifier = TaskClassifier(agent_registry=agent_registry)
        self.task_graph = TaskGraph()
        self.execution_results: Dict[str, AgentResult] = {}
        self.progress_callback = progress_callback
        self.user_input_callback = user_input_callback
    
    def _update_progress(self, status: str, message: str):
        """Update progress if callback is provided."""
        if self.progress_callback:
            self.progress_callback(status, message)
    
    async def process_prompt(
        self,
        prompt: str,
        max_parallel: int = 5
    ) -> Dict[str, Any]:
        """
        Process a user prompt by orchestrating multiple agents.
        
        Args:
            prompt: The user's prompt/task
            max_parallel: Maximum number of tasks to run in parallel
            
        Returns:
            Dictionary containing results and metadata
        """
        # Step 1: Classify the prompt and create tasks
        self._update_progress("classifying", "Analyzing task and selecting appropriate agents...")
        tasks = await self.classifier.create_tasks_from_prompt_async(prompt)
        self._update_progress("classified", f"Identified {len(tasks)} task(s) to execute")
        
        # Show classification reasoning
        if tasks and tasks[0].metadata.get("classification_reasoning"):
            self._update_progress("reasoning", tasks[0].metadata.get("classification_reasoning", ""))
        
        # Step 2: Build task graph
        self._update_progress("building", "Building task execution plan...")
        self.task_graph = TaskGraph()
        for task in tasks:
            self.task_graph.add_task(task)
        
        # Step 3: Validate graph
        if self.task_graph.has_cycles():
            self._update_progress("error", "Task graph contains cycles")
            return {
                "success": False,
                "error": "Task graph contains cycles",
                "results": {}
            }
        
        # Notify that graph is ready for visualization
        self._update_progress("graph_ready", "Task graph created")
        
        # Step 4: Execute tasks
        try:
            execution_order = self.task_graph.get_execution_order()
        except ValueError as e:
            self._update_progress("error", f"Task graph error: {str(e)}")
            return {
                "success": False,
                "error": f"Task graph error: {str(e)}",
                "results": {}
            }
        
        total_levels = len(execution_order)
        self._update_progress("executing", f"Executing {total_levels} level(s) of tasks...")
        await self._execute_tasks(max_parallel)
        
        # Step 5: Aggregate results
        self._update_progress("aggregating", "Aggregating results...")
        results = self._aggregate_results()
        self._update_progress("complete", "Task completed")
        
        return {
            "success": self.task_graph.is_complete(),
            "prompt": prompt,
            "tasks_created": len(tasks),
            "results": results,
            "task_status": {
                task.id: task.status
                for task in self.task_graph.get_all_tasks()
            }
        }
    
    async def _execute_tasks(self, max_parallel: int) -> None:
        """Execute tasks respecting dependencies and parallelism."""
        execution_order = self.task_graph.get_execution_order()
        
        for level_idx, level in enumerate(execution_order, 1):
            # Execute tasks in this level in parallel (up to max_parallel)
            tasks_to_execute = [
                self.task_graph.get_task(task_id)
                for task_id in level
            ]
            
            self._update_progress("executing", f"Level {level_idx}/{len(execution_order)}: Executing {len(tasks_to_execute)} task(s)...")
            
            # Process in batches to respect max_parallel
            for i in range(0, len(tasks_to_execute), max_parallel):
                batch = tasks_to_execute[i:i + max_parallel]
                await asyncio.gather(*[
                    self._execute_single_task(task)
                    for task in batch
                ])
    
    async def _execute_single_task(self, task: Task) -> None:
        """Execute a single task by finding and using an appropriate agent."""
        # Check if task metadata has recommended agents from the classifier
        recommended_agents = task.metadata.get("recommended_agents", []) if task.metadata else []
        
        # Find agents that can handle this task
        available_agents = self.registry.find_agents_for_task(task)
        
        if not available_agents:
            # No agent available, mark as failed
            self.task_graph.mark_failed(task.id)
            self.execution_results[task.id] = AgentResult(
                task_id=task.id,
                agent_id="none",
                result=None,
                success=False,
                error=f"No agent available for task: {task.description}"
            )
            return
        
        # Prefer recommended agents if available (from LLM classification)
        selected_agent = None
        if recommended_agents:
            for agent_id in recommended_agents:
                agent = self.registry.get_agent(agent_id)
                if agent and agent in available_agents:
                    selected_agent = agent
                    break
        
        # Fallback to first available agent if no recommendation or recommended not available
        if not selected_agent:
            selected_agent = available_agents[0]
        
        # Mark task as in progress
        self.task_graph.mark_in_progress(task.id)
        self._update_progress("executing", f"Using {selected_agent.name} for: {task.description[:50]}...")
        
        try:
            # Execute the task
            result = await selected_agent.execute(task)
            
            # Check if agent needs user input
            if result.needs_user_input and result.question:
                if not self.user_input_callback:
                    # No callback provided, mark as failed
                    self.task_graph.mark_failed(task.id)
                    self.execution_results[task.id] = AgentResult(
                        task_id=task.id,
                        agent_id=selected_agent.agent_id,
                        result=None,
                        success=False,
                        error="Agent requested user input but no user_input_callback provided"
                    )
                    self._update_progress("error", f"✗ {selected_agent.name} needs user input but callback not available")
                    return
                
                # Get user input
                self._update_progress("user_input", f"{selected_agent.name} is asking a question...")
                user_response = self.user_input_callback(result.question)
                
                if user_response is None:
                    # User cancelled or no response
                    self.task_graph.mark_failed(task.id)
                    self.execution_results[task.id] = AgentResult(
                        task_id=task.id,
                        agent_id=selected_agent.agent_id,
                        result=None,
                        success=False,
                        error="User input was cancelled or not provided"
                    )
                    self._update_progress("error", f"✗ User input cancelled for {selected_agent.name}")
                    return
                
                # Update task with user's response and re-execute
                # Add user response to task metadata and update description
                if task.metadata is None:
                    task.metadata = {}
                
                # Store original description if not already stored
                if "original_description" not in task.metadata:
                    task.metadata["original_description"] = task.description
                
                # Append user response to metadata (support multiple questions)
                if "user_responses" not in task.metadata:
                    task.metadata["user_responses"] = []
                task.metadata["user_responses"].append(user_response)
                task.metadata["user_response"] = user_response  # Keep latest for backward compatibility
                
                # Update task description to include user's response
                # Use original description to avoid appending multiple times
                original_desc = task.metadata.get("original_description", task.description)
                all_responses = "\n\n".join([f"User Response: {r}" for r in task.metadata["user_responses"]])
                task.description = f"{original_desc}\n\n{all_responses}"
                
                # Re-execute the agent with user's input
                self._update_progress("executing", f"{selected_agent.name} continuing with your response...")
                result = await selected_agent.execute(task)
            
            self.execution_results[task.id] = result
            
            if result.success:
                task.result = result.result
                self.task_graph.mark_completed(task.id)
                self._update_progress("success", f"✓ {selected_agent.name} completed task")
                # Notify graph update
                self._update_progress("graph_update", f"Task {task.id} completed")
            else:
                self.task_graph.mark_failed(task.id)
                self._update_progress("error", f"✗ {selected_agent.name} failed: {result.error}")
                # Notify graph update
                self._update_progress("graph_update", f"Task {task.id} failed")
        except Exception as e:
            # Handle execution errors
            self.task_graph.mark_failed(task.id)
            self.execution_results[task.id] = AgentResult(
                task_id=task.id,
                agent_id=selected_agent.agent_id,
                result=None,
                success=False,
                error=str(e)
            )
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results from all completed tasks."""
        aggregated = {
            "task_results": {},
            "final_output": None,
            "summary": {
                "total_tasks": len(self.task_graph.get_all_tasks()),
                "completed": sum(1 for t in self.task_graph.get_all_tasks() if t.status == "completed"),
                "failed": sum(1 for t in self.task_graph.get_all_tasks() if t.status == "failed")
            }
        }
        
        # Collect individual task results
        for task_id, result in self.execution_results.items():
            aggregated["task_results"][task_id] = {
                "agent_id": result.agent_id,
                "result": result.result,
                "success": result.success,
                "error": result.error,
                "metadata": result.metadata
            }
        
        # Create final output from completed tasks
        completed_results = [
            result.result
            for result in self.execution_results.values()
            if result.success
        ]
        
        if completed_results:
            # Simple aggregation: combine all results
            if len(completed_results) == 1:
                aggregated["final_output"] = completed_results[0]
            else:
                aggregated["final_output"] = "\n\n".join(str(r) for r in completed_results)
        
        return aggregated
    
    def add_task(self, task: Task) -> None:
        """Manually add a task to the graph."""
        self.task_graph.add_task(task)
    
    def get_task_graph_info(self) -> Dict[str, Any]:
        """Get information about the current task graph."""
        return {
            "total_tasks": len(self.task_graph.get_all_tasks()),
            "execution_order": self.task_graph.get_execution_order(),
            "tasks": [
                {
                    "id": task.id,
                    "description": task.description,
                    "status": task.status,
                    "capabilities": [cap.value for cap in task.required_capabilities],
                    "dependencies": task.dependencies
                }
                for task in self.task_graph.get_all_tasks()
            ]
        }
    
    def get_graph_visualization(self):
        """Get the visual representation of the task graph."""
        return self.task_graph.get_visualization()