from typing import List, Dict, Tuple, Optional
from models import run_model_with_fallbacks
from classifier.intent_classifier import classify_task
import re
import logging

class TaskDecomposer:
    def __init__(self):
        self.complex_patterns = [
            r'build a complete', r'create a full', r'develop an entire',
            r'comprehensive analysis', r'end-to-end', r'step by step',
            r'from start to finish', r'complete solution', r'full implementation',
            r'detailed plan', r'thorough investigation', r'in-depth study'
        ]
        
        self.multi_action_indicators = [
            ' and then ', ' followed by ', ' after that ', ' next ',
            ' subsequently ', ' in addition ', ' also ', ' furthermore ',
            ' moreover ', ' plus ', ' as well as '
        ]
        
        self.complexity_keywords = [
            'analyze', 'create', 'build', 'develop', 'design', 'implement',
            'research', 'investigate', 'study', 'examine', 'evaluate',
            'optimize', 'improve', 'enhance', 'refactor', 'test', 'deploy'
        ]
        
    def decompose_if_needed(self, prompt: str, user_id: str) -> Tuple[List[Dict], bool]:
        """Decompose complex tasks into manageable subtasks."""
        try:
            if not self._is_complex_task(prompt):
                task_type, confidence = classify_task(prompt)
                return [{
                    'prompt': prompt, 
                    'task_type': task_type or 'text',
                    'confidence': confidence,
                    'subtask_index': 1,
                    'total_subtasks': 1
                }], False
            
            logging.info(f"Detected complex task for decomposition: '{prompt}'")
            
            decomposition = self._generate_decomposition(prompt, user_id)
            
            if not decomposition:
                task_type, confidence = classify_task(prompt)
                return [{
                    'prompt': prompt, 
                    'task_type': task_type or 'text',
                    'confidence': confidence,
                    'subtask_index': 1,
                    'total_subtasks': 1
                }], False
            
            subtasks = self._parse_subtasks(decomposition, prompt)
            
            if len(subtasks) <= 1:
                task_type, confidence = classify_task(prompt)
                return [{
                    'prompt': prompt, 
                    'task_type': task_type or 'text',
                    'confidence': confidence,
                    'subtask_index': 1,
                    'total_subtasks': 1
                }], False
            
            logging.info(f"Successfully decomposed into {len(subtasks)} subtasks")
            return subtasks, True
            
        except Exception as e:
            logging.error(f"Error in task decomposition: {e}")
            task_type, confidence = classify_task(prompt)
            return [{
                'prompt': prompt, 
                'task_type': task_type or 'text',
                'confidence': confidence,
                'subtask_index': 1,
                'total_subtasks': 1
            }], False
    
    def _is_complex_task(self, prompt: str) -> bool:
        """Determine if a task is complex enough to warrant decomposition."""
        prompt_lower = prompt.lower()
        
        for pattern in self.complex_patterns:
            if re.search(pattern, prompt_lower):
                return True
        
        for indicator in self.multi_action_indicators:
            if indicator in prompt_lower:
                return True
        
        action_count = sum(1 for keyword in self.complexity_keywords if keyword in prompt_lower)
        if action_count >= 3:
            return True
        
        word_count = len(prompt.split())
        if word_count > 25:  # Long prompts are often complex
            return True
        
        if re.search(r'\b\d+\.\s|\b[a-z]\.\s|\b[A-Z]\.\s', prompt):
            return True
        
        return False
    
    def _generate_decomposition(self, prompt: str, user_id: str) -> Optional[str]:
        """Use LLM to generate task decomposition."""
        try:
            decomposition_prompt = f"""
Break down this complex task into 3-6 smaller, manageable subtasks that can be executed sequentially.

Original Task: {prompt}

Requirements:
- Each subtask should be specific and actionable
- Subtasks should build upon each other logically
- Each subtask should be completable independently
- Return as a numbered list (1., 2., 3., etc.)
- Keep each subtask description concise but clear

Subtasks:
"""
            
            decomposition = run_model_with_fallbacks('gpt-4o', 'text', decomposition_prompt, user_id)
            
            if decomposition and not decomposition.startswith("❌"):
                return decomposition
            
            return None
            
        except Exception as e:
            logging.error(f"Error generating decomposition: {e}")
            return None
    
    def _parse_subtasks(self, decomposition: str, original_prompt: str) -> List[Dict]:
        """Parse LLM decomposition into structured subtasks."""
        try:
            subtasks = []
            lines = decomposition.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                match = re.match(r'^(\d+)\.\s*(.+)', line)
                if match:
                    subtask_number = int(match.group(1))
                    subtask_text = match.group(2).strip()
                    
                    if subtask_text:
                        task_type, confidence = classify_task(subtask_text)
                        
                        subtasks.append({
                            'prompt': subtask_text,
                            'task_type': task_type or 'text',
                            'confidence': confidence,
                            'subtask_index': subtask_number,
                            'total_subtasks': 0,  # Will be updated after parsing
                            'original_prompt': original_prompt
                        })
            
            total = len(subtasks)
            for subtask in subtasks:
                subtask['total_subtasks'] = total
            
            return subtasks
            
        except Exception as e:
            logging.error(f"Error parsing subtasks: {e}")
            return []
    
    def execute_decomposed_task(self, subtasks: List[Dict], user_id: str) -> Dict[str, any]:
        """Execute a decomposed task step by step."""
        try:
            results = {
                'original_prompt': subtasks[0].get('original_prompt', '') if subtasks else '',
                'total_subtasks': len(subtasks),
                'subtask_results': [],
                'overall_success': True,
                'execution_summary': ''
            }
            
            context = ""  # Accumulate context from previous subtasks
            
            for i, subtask in enumerate(subtasks):
                try:
                    logging.info(f"Executing subtask {i+1}/{len(subtasks)}: {subtask['prompt']}")
                    
                    enhanced_prompt = subtask['prompt']
                    if context and i > 0:
                        enhanced_prompt = f"Context from previous steps:\n{context}\n\nCurrent task: {subtask['prompt']}"
                    
                    result = run_model_with_fallbacks(
                        'gpt-4o',  # Use consistent model for decomposed tasks
                        subtask['task_type'],
                        enhanced_prompt,
                        user_id
                    )
                    
                    success = not str(result).startswith("❌")
                    
                    subtask_result = {
                        'subtask_index': subtask['subtask_index'],
                        'prompt': subtask['prompt'],
                        'task_type': subtask['task_type'],
                        'result': result,
                        'success': success,
                        'context_used': bool(context)
                    }
                    
                    results['subtask_results'].append(subtask_result)
                    
                    if not success:
                        results['overall_success'] = False
                        logging.warning(f"Subtask {i+1} failed: {result}")
                    else:
                        if len(str(result)) < 500:  # Avoid too much context
                            context += f"\nStep {i+1} result: {result}"
                        else:
                            context += f"\nStep {i+1}: Completed successfully"
                    
                except Exception as e:
                    logging.error(f"Error executing subtask {i+1}: {e}")
                    results['subtask_results'].append({
                        'subtask_index': subtask.get('subtask_index', i+1),
                        'prompt': subtask.get('prompt', ''),
                        'task_type': subtask.get('task_type', 'text'),
                        'result': f"Error: {str(e)}",
                        'success': False,
                        'context_used': False
                    })
                    results['overall_success'] = False
            
            successful_tasks = sum(1 for r in results['subtask_results'] if r['success'])
            results['execution_summary'] = f"Completed {successful_tasks}/{len(subtasks)} subtasks successfully"
            
            return results
            
        except Exception as e:
            logging.error(f"Error executing decomposed task: {e}")
            return {
                'original_prompt': '',
                'total_subtasks': 0,
                'subtask_results': [],
                'overall_success': False,
                'execution_summary': f"Execution failed: {str(e)}"
            }
    
    def analyze_task_complexity(self, prompt: str) -> Dict[str, any]:
        """Analyze the complexity level of a task."""
        try:
            prompt_lower = prompt.lower()
            complexity_score = 0
            complexity_indicators = []
            
            for pattern in self.complex_patterns:
                if re.search(pattern, prompt_lower):
                    complexity_score += 2
                    complexity_indicators.append("comprehensive_scope")
            
            for indicator in self.multi_action_indicators:
                if indicator in prompt_lower:
                    complexity_score += 1
                    complexity_indicators.append("multiple_actions")
            
            action_count = sum(1 for keyword in self.complexity_keywords if keyword in prompt_lower)
            if action_count >= 3:
                complexity_score += action_count - 2
                complexity_indicators.append("multiple_verbs")
            
            word_count = len(prompt.split())
            if word_count > 25:
                complexity_score += 1
                complexity_indicators.append("long_description")
            
            if re.search(r'\b\d+\.\s|\b[a-z]\.\s|\b[A-Z]\.\s', prompt):
                complexity_score += 1
                complexity_indicators.append("enumerated_items")
            
            if complexity_score >= 4:
                complexity_level = "high"
            elif complexity_score >= 2:
                complexity_level = "medium"
            else:
                complexity_level = "low"
            
            return {
                'complexity_score': complexity_score,
                'complexity_level': complexity_level,
                'indicators': complexity_indicators,
                'word_count': word_count,
                'action_verb_count': action_count,
                'should_decompose': complexity_score >= 2
            }
            
        except Exception as e:
            logging.error(f"Error analyzing task complexity: {e}")
            return {
                'complexity_score': 0,
                'complexity_level': 'unknown',
                'indicators': [],
                'word_count': 0,
                'action_verb_count': 0,
                'should_decompose': False
            }
