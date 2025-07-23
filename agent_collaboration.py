from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from models import run_model_with_fallbacks
from classifier.intent_classifier import classify_task
import uuid
import logging
import time
from datetime import datetime, timedelta

@dataclass
class CollaborationTask:
    task_id: str
    description: str
    assigned_agent: str
    status: str
    result: Optional[Any] = None
    dependencies: Optional[List[str]] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()

class CrossAgentCollaborator:
    def __init__(self):
        self.active_tasks = {}
        self.completed_tasks = {}
        self.agent_capabilities = {
            'text_specialist': {
                'tasks': ['text', 'summarize', 'translate'],
                'strengths': ['natural_language', 'content_creation', 'communication'],
                'model_preference': 'gpt-4o'
            },
            'code_specialist': {
                'tasks': ['code', 'debug', 'optimize'],
                'strengths': ['programming', 'technical_analysis', 'problem_solving'],
                'model_preference': 'claude-sonnet-4'
            },
            'creative_specialist': {
                'tasks': ['image', 'creative_writing', 'design'],
                'strengths': ['creativity', 'visual_design', 'artistic_expression'],
                'model_preference': 'dall-e-3'
            },
            'analysis_specialist': {
                'tasks': ['data_analysis', 'research', 'evaluation'],
                'strengths': ['analytical_thinking', 'research', 'pattern_recognition'],
                'model_preference': 'gpt-4o'
            },
            'integration_specialist': {
                'tasks': ['coordination', 'synthesis', 'quality_assurance'],
                'strengths': ['project_management', 'integration', 'quality_control'],
                'model_preference': 'claude-sonnet-4'
            }
        }
        
    def coordinate_multi_agent_task(self, complex_prompt: str, user_id: str) -> Dict[str, Any]:
        """Coordinate a complex task across multiple specialized agents."""
        try:
            collaboration_analysis = self._analyze_collaboration_need(complex_prompt)
            
            if not collaboration_analysis['needs_collaboration']:
                return {
                    'collaboration_type': 'single_agent',
                    'reason': collaboration_analysis['reason'],
                    'result': None
                }
            
            logging.info(f"Initiating multi-agent collaboration for: {complex_prompt}")
            
            agent_tasks = self._decompose_for_agents(complex_prompt, user_id)
            
            if not agent_tasks:
                return {
                    'collaboration_type': 'failed_decomposition',
                    'reason': 'Could not decompose task for agent collaboration',
                    'result': None
                }
            
            execution_results = self._execute_coordinated_tasks(agent_tasks, user_id)
            
            final_result = self._synthesize_results(execution_results, complex_prompt, user_id)
            
            return {
                'collaboration_type': 'multi_agent',
                'agent_tasks': len(agent_tasks),
                'execution_results': execution_results,
                'final_result': final_result,
                'collaboration_analysis': collaboration_analysis
            }
            
        except Exception as e:
            logging.error(f"Error in multi-agent coordination: {e}")
            return {
                'collaboration_type': 'error',
                'reason': str(e),
                'result': None
            }
    
    def _analyze_collaboration_need(self, prompt: str) -> Dict[str, Any]:
        """Analyze if a prompt would benefit from multi-agent collaboration."""
        try:
            prompt_lower = prompt.lower()
            
            collaboration_indicators = {
                'multiple_domains': [
                    'code and design', 'analysis and visualization', 'research and writing',
                    'technical and creative', 'data and presentation', 'backend and frontend'
                ],
                'complex_workflows': [
                    'end-to-end', 'comprehensive solution', 'full implementation',
                    'complete system', 'integrated approach', 'holistic solution'
                ],
                'quality_requirements': [
                    'high quality', 'production ready', 'professional grade',
                    'enterprise level', 'best practices', 'industry standard'
                ],
                'multiple_outputs': [
                    'documentation and code', 'analysis and recommendations',
                    'design and implementation', 'research and summary'
                ]
            }
            
            collaboration_score = 0
            detected_indicators = []
            
            for category, indicators in collaboration_indicators.items():
                for indicator in indicators:
                    if indicator in prompt_lower:
                        collaboration_score += 1
                        detected_indicators.append(f"{category}: {indicator}")
            
            task_keywords = {
                'code': ['code', 'program', 'script', 'function', 'algorithm'],
                'analysis': ['analyze', 'research', 'study', 'investigate', 'examine'],
                'creative': ['design', 'create', 'artistic', 'visual', 'creative'],
                'text': ['write', 'document', 'explain', 'describe', 'summarize']
            }
            
            detected_task_types = []
            for task_type, keywords in task_keywords.items():
                if any(keyword in prompt_lower for keyword in keywords):
                    detected_task_types.append(task_type)
            
            if len(detected_task_types) >= 2:
                collaboration_score += len(detected_task_types) - 1
                detected_indicators.append(f"multiple_task_types: {detected_task_types}")
            
            needs_collaboration = collaboration_score >= 2
            
            return {
                'needs_collaboration': needs_collaboration,
                'collaboration_score': collaboration_score,
                'detected_indicators': detected_indicators,
                'detected_task_types': detected_task_types,
                'reason': 'Multiple domains/workflows detected' if needs_collaboration else 'Single domain task'
            }
            
        except Exception as e:
            logging.error(f"Error analyzing collaboration need: {e}")
            return {
                'needs_collaboration': False,
                'collaboration_score': 0,
                'detected_indicators': [],
                'detected_task_types': [],
                'reason': f'Analysis error: {str(e)}'
            }
    
    def _decompose_for_agents(self, prompt: str, user_id: str) -> List[CollaborationTask]:
        """Decompose a complex prompt into agent-specific tasks."""
        try:
            decomposition_prompt = f"""
Decompose this complex task into 3-5 specialized subtasks that can be handled by different AI agents:

Original Task: {prompt}

Available Agent Types:
- text_specialist: Natural language, content creation, communication
- code_specialist: Programming, technical analysis, problem solving  
- creative_specialist: Visual design, artistic expression, creativity
- analysis_specialist: Data analysis, research, pattern recognition
- integration_specialist: Coordination, synthesis, quality assurance

For each subtask, specify:
1. Task description
2. Best agent type
3. Dependencies (if any)

Format as JSON array:
[
  {{
    "description": "Task description",
    "agent": "agent_type",
    "dependencies": ["task_id_if_any"]
  }}
]
"""
            
            decomposition_result = run_model_with_fallbacks('gpt-4o', 'text', decomposition_prompt, user_id)
            
            if not decomposition_result or decomposition_result.startswith("❌"):
                return []
            
            import json
            try:
                task_definitions = json.loads(decomposition_result)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\[.*\]', decomposition_result, re.DOTALL)
                if json_match:
                    task_definitions = json.loads(json_match.group())
                else:
                    return []
            
            tasks = []
            for i, task_def in enumerate(task_definitions):
                task_id = f"task_{i+1}_{uuid.uuid4().hex[:8]}"
                
                task = CollaborationTask(
                    task_id=task_id,
                    description=task_def.get('description', ''),
                    assigned_agent=task_def.get('agent', 'text_specialist'),
                    status='pending',
                    dependencies=task_def.get('dependencies', [])
                )
                
                tasks.append(task)
                self.active_tasks[task_id] = task
            
            return tasks
            
        except Exception as e:
            logging.error(f"Error decomposing for agents: {e}")
            return []
    
    def _execute_coordinated_tasks(self, tasks: List[CollaborationTask], user_id: str) -> Dict[str, Any]:
        """Execute tasks in coordination, respecting dependencies."""
        try:
            execution_results = {
                'completed_tasks': [],
                'failed_tasks': [],
                'execution_order': [],
                'total_time': 0
            }
            
            start_time = time.time()
            completed_task_ids = set()
            
            max_iterations = len(tasks) * 2  # Prevent infinite loops
            iteration = 0
            
            while len(completed_task_ids) < len(tasks) and iteration < max_iterations:
                iteration += 1
                progress_made = False
                
                for task in tasks:
                    if task.task_id in completed_task_ids or task.status == 'failed':
                        continue
                    
                    if self._dependencies_satisfied(task, completed_task_ids):
                        task_result = self._execute_single_agent_task(task, user_id, completed_task_ids)
                        
                        if task_result['success']:
                            task.status = 'completed'
                            task.result = task_result['result']
                            task.completed_at = datetime.utcnow().isoformat()
                            completed_task_ids.add(task.task_id)
                            execution_results['completed_tasks'].append({
                                'task_id': task.task_id,
                                'agent': task.assigned_agent,
                                'result': task.result
                            })
                            progress_made = True
                        else:
                            task.status = 'failed'
                            execution_results['failed_tasks'].append({
                                'task_id': task.task_id,
                                'agent': task.assigned_agent,
                                'error': task_result['error']
                            })
                        
                        execution_results['execution_order'].append(task.task_id)
                
                if not progress_made:
                    break
            
            execution_results['total_time'] = time.time() - start_time
            
            return execution_results
            
        except Exception as e:
            logging.error(f"Error executing coordinated tasks: {e}")
            return {
                'completed_tasks': [],
                'failed_tasks': [],
                'execution_order': [],
                'total_time': 0,
                'error': str(e)
            }
    
    def _dependencies_satisfied(self, task: CollaborationTask, completed_task_ids: set) -> bool:
        """Check if all dependencies for a task are satisfied."""
        if not task.dependencies:
            return True
        
        return all(dep_id in completed_task_ids for dep_id in task.dependencies)
    
    def _execute_single_agent_task(self, task: CollaborationTask, user_id: str, completed_task_ids: set) -> Dict[str, Any]:
        """Execute a single agent task."""
        try:
            agent_info = self.agent_capabilities.get(task.assigned_agent, {})
            preferred_model = agent_info.get('model_preference', 'gpt-4o')
            
            task_type, _ = classify_task(task.description)
            if not task_type:
                task_type = 'text'
            
            context = self._build_dependency_context(task, completed_task_ids)
            
            enhanced_prompt = self._enhance_prompt_for_agent(task, agent_info, context)
            
            logging.info(f"Executing task {task.task_id} with {task.assigned_agent}")
            
            result = run_model_with_fallbacks(preferred_model, task_type, enhanced_prompt, user_id)
            
            success = not str(result).startswith("❌")
            
            return {
                'success': success,
                'result': result,
                'model_used': preferred_model,
                'task_type': task_type
            }
            
        except Exception as e:
            logging.error(f"Error executing single agent task: {e}")
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }
    
    def _build_dependency_context(self, task: CollaborationTask, completed_task_ids: set) -> str:
        """Build context from completed dependency tasks."""
        if not task.dependencies:
            return ""
        
        context_parts = []
        for dep_id in task.dependencies:
            if dep_id in completed_task_ids and dep_id in self.active_tasks:
                dep_task = self.active_tasks[dep_id]
                if dep_task.result:
                    context_parts.append(f"Previous task ({dep_task.assigned_agent}): {dep_task.result}")
        
        if context_parts:
            return "Context from previous tasks:\n" + "\n".join(context_parts) + "\n\n"
        
        return ""
    
    def _enhance_prompt_for_agent(self, task: CollaborationTask, agent_info: Dict, context: str) -> str:
        """Enhance prompt with agent-specific instructions."""
        strengths = agent_info.get('strengths', [])
        
        if strengths:
            specialization = f"As a specialist in {', '.join(strengths)}, "
        else:
            specialization = ""
        
        enhanced_prompt = f"{specialization}please complete this task:\n\n{context}{task.description}"
        
        return enhanced_prompt
    
    def _synthesize_results(self, execution_results: Dict[str, Any], original_prompt: str, user_id: str) -> str:
        """Synthesize results from multiple agents into a cohesive final result."""
        try:
            if not execution_results['completed_tasks']:
                return "❌ No tasks were completed successfully."
            
            all_results = []
            for completed_task in execution_results['completed_tasks']:
                result = completed_task['result']
                agent = completed_task['agent']
                all_results.append(f"From {agent}: {result}")
            
            synthesis_prompt = f"""
Synthesize these results from multiple AI specialists into a cohesive, comprehensive response to the original request:

Original Request: {original_prompt}

Results from specialists:
{chr(10).join(all_results)}

Please create a unified, well-structured response that integrates all the specialist contributions into a coherent final answer.
"""
            
            final_result = run_model_with_fallbacks('claude-sonnet-4', 'text', synthesis_prompt, user_id)
            
            return final_result
            
        except Exception as e:
            logging.error(f"Error synthesizing results: {e}")
            return f"❌ Error synthesizing results: {str(e)}"
    
    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get statistics about agent collaboration."""
        return {
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'available_agents': list(self.agent_capabilities.keys()),
            'agent_capabilities': self.agent_capabilities
        }
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Clean up old completed tasks."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            tasks_to_remove = []
            for task_id, task in self.completed_tasks.items():
                if task.completed_at:
                    completed_time = datetime.fromisoformat(task.completed_at)
                    if completed_time < cutoff_time:
                        tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.completed_tasks[task_id]
            
            logging.info(f"Cleaned up {len(tasks_to_remove)} old completed tasks")
            
        except Exception as e:
            logging.error(f"Error cleaning up completed tasks: {e}")
