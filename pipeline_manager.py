from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from models import run_model_with_fallbacks
from classifier.intent_classifier import classify_task
import logging
import json
import re

@dataclass
class PipelineStep:
    model: str
    task_type: str
    prompt_template: str
    output_key: str
    condition: Optional[str] = None  # Optional condition for conditional execution

class ModelPipelineManager:
    def __init__(self):
        self.predefined_pipelines = {
            'image_analysis_summary': [
                PipelineStep('gpt-4o', 'text', 'Analyze this image in detail: {input}', 'analysis'),
                PipelineStep('claude-sonnet-4', 'summarize', 'Create a concise summary of this analysis: {analysis}', 'summary')
            ],
            'code_review_improve': [
                PipelineStep('gpt-4o', 'code', 'Review this code for issues, bugs, and improvements: {input}', 'review'),
                PipelineStep('claude-sonnet-4', 'code', 'Based on this review, provide improved code:\n\nOriginal Code:\n{input}\n\nReview:\n{review}', 'improved_code')
            ],
            'research_and_summarize': [
                PipelineStep('gpt-4o', 'text', 'Research and gather comprehensive information about: {input}', 'research'),
                PipelineStep('claude-sonnet-4', 'summarize', 'Summarize this research into key points: {research}', 'summary'),
                PipelineStep('gpt-4o', 'text', 'Based on this summary, provide actionable recommendations: {summary}', 'recommendations')
            ],
            'creative_writing_enhance': [
                PipelineStep('gpt-4o', 'text', 'Create a creative story or content based on: {input}', 'draft'),
                PipelineStep('claude-sonnet-4', 'text', 'Enhance and polish this creative content: {draft}', 'enhanced'),
                PipelineStep('gpt-4o', 'text', 'Add final touches and ensure consistency: {enhanced}', 'final')
            ],
            'data_analysis_insights': [
                PipelineStep('gpt-4o', 'text', 'Analyze this data and identify patterns: {input}', 'analysis'),
                PipelineStep('claude-sonnet-4', 'text', 'Extract key insights from this analysis: {analysis}', 'insights'),
                PipelineStep('gpt-4o', 'text', 'Provide business recommendations based on these insights: {insights}', 'recommendations')
            ],
            'translate_and_localize': [
                PipelineStep('gpt-4o', 'translate', 'Translate this content: {input}', 'translation'),
                PipelineStep('claude-sonnet-4', 'text', 'Localize this translation for cultural appropriateness: {translation}', 'localized')
            ]
        }
        
        self.pipeline_detection_patterns = {
            'image_analysis_summary': [
                r'analyze.*image.*summarize',
                r'describe.*image.*summary',
                r'image.*analysis.*brief'
            ],
            'code_review_improve': [
                r'review.*code.*improve',
                r'check.*code.*fix',
                r'analyze.*code.*better'
            ],
            'research_and_summarize': [
                r'research.*summarize',
                r'investigate.*summary',
                r'study.*key points'
            ],
            'creative_writing_enhance': [
                r'write.*story.*improve',
                r'create.*content.*enhance',
                r'creative.*writing.*polish'
            ],
            'data_analysis_insights': [
                r'analyze.*data.*insights',
                r'examine.*data.*recommendations',
                r'data.*analysis.*business'
            ],
            'translate_and_localize': [
                r'translate.*localize',
                r'translate.*cultural',
                r'localize.*translation'
            ]
        }
        
    def execute_pipeline(self, pipeline_name: str, initial_input: str, user_id: str) -> Dict[str, Any]:
        """Execute a predefined pipeline with the given input."""
        try:
            if pipeline_name not in self.predefined_pipelines:
                raise ValueError(f"Pipeline {pipeline_name} not found")
            
            pipeline = self.predefined_pipelines[pipeline_name]
            context = {'input': initial_input}
            results = {'pipeline_name': pipeline_name, 'initial_input': initial_input}
            
            logging.info(f"Executing pipeline: {pipeline_name} for user: {user_id}")
            
            for i, step in enumerate(pipeline):
                try:
                    if step.condition and not self._evaluate_condition(step.condition, context):
                        logging.info(f"Skipping step {i+1} due to condition: {step.condition}")
                        continue
                    
                    formatted_prompt = step.prompt_template.format(**context)
                    
                    logging.info(f"Executing pipeline step {i+1}/{len(pipeline)}: {step.output_key}")
                    
                    result = run_model_with_fallbacks(step.model, step.task_type, formatted_prompt, user_id)
                    
                    results[step.output_key] = result
                    context[step.output_key] = result
                    
                    logging.info(f"Pipeline step {step.output_key} completed successfully")
                    
                except Exception as e:
                    logging.error(f"Error in pipeline step {step.output_key}: {e}")
                    results[f"{step.output_key}_error"] = str(e)
                    context[step.output_key] = f"[Error in {step.output_key}: {str(e)}]"
            
            results['execution_metadata'] = {
                'steps_completed': len([k for k in results.keys() if not k.endswith('_error') and k not in ['pipeline_name', 'initial_input', 'execution_metadata']]),
                'total_steps': len(pipeline),
                'errors': [k for k in results.keys() if k.endswith('_error')]
            }
            
            logging.info(f"Pipeline {pipeline_name} execution completed")
            return results
            
        except Exception as e:
            logging.error(f"Error executing pipeline {pipeline_name}: {e}")
            return {
                'pipeline_name': pipeline_name,
                'initial_input': initial_input,
                'error': str(e),
                'execution_metadata': {'steps_completed': 0, 'total_steps': 0, 'errors': ['pipeline_error']}
            }
    
    def detect_pipeline_need(self, prompt: str, task_type: str) -> Optional[str]:
        """Detect if prompt needs pipeline processing."""
        try:
            prompt_lower = prompt.lower()
            
            for pipeline_name, patterns in self.pipeline_detection_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, prompt_lower):
                        logging.info(f"Detected pipeline need: {pipeline_name}")
                        return pipeline_name
            
            if self._is_complex_multi_step_task(prompt):
                if task_type == 'code' and any(word in prompt_lower for word in ['review', 'improve', 'fix']):
                    return 'code_review_improve'
                elif task_type == 'image' and any(word in prompt_lower for word in ['analyze', 'summarize']):
                    return 'image_analysis_summary'
                elif any(word in prompt_lower for word in ['research', 'investigate', 'study']) and any(word in prompt_lower for word in ['summary', 'summarize']):
                    return 'research_and_summarize'
                elif task_type == 'text' and any(word in prompt_lower for word in ['creative', 'story', 'writing']) and any(word in prompt_lower for word in ['improve', 'enhance']):
                    return 'creative_writing_enhance'
                elif any(word in prompt_lower for word in ['data', 'analyze']) and any(word in prompt_lower for word in ['insights', 'recommendations']):
                    return 'data_analysis_insights'
                elif task_type == 'translate' and any(word in prompt_lower for word in ['localize', 'cultural']):
                    return 'translate_and_localize'
            
            return None
            
        except Exception as e:
            logging.error(f"Error detecting pipeline need: {e}")
            return None
    
    def _is_complex_multi_step_task(self, prompt: str) -> bool:
        """Determine if a prompt represents a complex multi-step task."""
        prompt_lower = prompt.lower()
        
        action_verbs = ['analyze', 'create', 'build', 'review', 'improve', 'research', 'summarize', 'translate', 'enhance', 'optimize']
        verb_count = sum(1 for verb in action_verbs if verb in prompt_lower)
        
        conjunctions = [' and ', ' then ', ' after ', ' followed by ', ' next ']
        has_conjunctions = any(conj in prompt_lower for conj in conjunctions)
        
        multi_step_indicators = ['step by step', 'comprehensive', 'detailed analysis', 'end-to-end', 'complete solution']
        has_multi_step = any(indicator in prompt_lower for indicator in multi_step_indicators)
        
        return verb_count >= 2 or has_conjunctions or has_multi_step
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition for conditional pipeline execution."""
        try:
            if ':' in condition:
                parts = condition.split(':')
                if len(parts) == 2:
                    key, expected_value = parts
                    return context.get(key, '').lower() == expected_value.lower()
                elif len(parts) == 3:
                    key, operator, value = parts
                    context_value = context.get(key, '').lower()
                    if operator == 'contains':
                        return value.lower() in context_value
                    elif operator == 'not_contains':
                        return value.lower() not in context_value
            
            return True  # Default to true if condition can't be evaluated
            
        except Exception as e:
            logging.error(f"Error evaluating condition {condition}: {e}")
            return True
    
    def create_custom_pipeline(self, name: str, steps: List[Dict[str, str]]) -> bool:
        """Create a custom pipeline from step definitions."""
        try:
            pipeline_steps = []
            for step_def in steps:
                step = PipelineStep(
                    model=step_def.get('model', 'gpt-4o'),
                    task_type=step_def.get('task_type', 'text'),
                    prompt_template=step_def.get('prompt_template', ''),
                    output_key=step_def.get('output_key', ''),
                    condition=step_def.get('condition')
                )
                pipeline_steps.append(step)
            
            self.predefined_pipelines[name] = pipeline_steps
            logging.info(f"Created custom pipeline: {name}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating custom pipeline {name}: {e}")
            return False
    
    def get_available_pipelines(self) -> List[str]:
        """Get list of available pipeline names."""
        return list(self.predefined_pipelines.keys())
    
    def get_pipeline_info(self, pipeline_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific pipeline."""
        if pipeline_name not in self.predefined_pipelines:
            return None
        
        pipeline = self.predefined_pipelines[pipeline_name]
        return {
            'name': pipeline_name,
            'steps': len(pipeline),
            'step_details': [
                {
                    'model': step.model,
                    'task_type': step.task_type,
                    'output_key': step.output_key,
                    'has_condition': step.condition is not None
                }
                for step in pipeline
            ]
        }
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about pipeline usage and availability."""
        return {
            'total_pipelines': len(self.predefined_pipelines),
            'pipeline_names': list(self.predefined_pipelines.keys()),
            'detection_patterns': len(self.pipeline_detection_patterns)
        }
