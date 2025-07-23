from models import ComprehensiveMetrics
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from datetime import datetime, timedelta
import statistics

class OutcomeBasedScorer:
    def __init__(self):
        self.metrics = ComprehensiveMetrics()
        self.outcome_weights = {
            'user_feedback': 0.4,
            'task_completion': 0.3,
            'response_quality': 0.2,
            'efficiency': 0.1
        }
        self.outcome_file = "outcome_scores.jsonl"
        self.model_scores = {}
        self.task_type_scores = {}
        
        self._load_outcome_data()
        
    def score_model_outcome(self, model: str, task_type: str, user_feedback: Optional[int], 
                          completion_success: bool, response_time: float, 
                          prompt_length: int, response_length: int, 
                          user_id: str = None) -> float:
        """Score model outcome based on multiple factors."""
        try:
            if user_feedback is not None:
                feedback_score = max(0, min(1, (user_feedback - 1) / 4))
            else:
                feedback_score = 0.5  # Neutral if no feedback
            
            completion_score = 1.0 if completion_success else 0.0
            
            quality_score = self._calculate_quality_score(prompt_length, response_length, completion_success)
            
            efficiency_score = self._calculate_efficiency_score(response_time, task_type)
            
            final_score = (
                feedback_score * self.outcome_weights['user_feedback'] +
                completion_score * self.outcome_weights['task_completion'] +
                quality_score * self.outcome_weights['response_quality'] +
                efficiency_score * self.outcome_weights['efficiency']
            )
            
            outcome_data = {
                'model': model,
                'task_type': task_type,
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat(),
                'final_score': final_score,
                'component_scores': {
                    'feedback': feedback_score,
                    'completion': completion_score,
                    'quality': quality_score,
                    'efficiency': efficiency_score
                },
                'raw_metrics': {
                    'user_feedback': user_feedback,
                    'completion_success': completion_success,
                    'response_time': response_time,
                    'prompt_length': prompt_length,
                    'response_length': response_length
                }
            }
            
            self._store_outcome(outcome_data)
            self._update_model_scores(model, task_type, final_score)
            
            logging.info(f"Scored outcome for {model} on {task_type}: {final_score:.3f}")
            return final_score
            
        except Exception as e:
            logging.error(f"Error scoring model outcome: {e}")
            return 0.5  # Return neutral score on error
    
    def _calculate_quality_score(self, prompt_length: int, response_length: int, completion_success: bool) -> float:
        """Calculate response quality score based on length ratio and success."""
        try:
            if not completion_success:
                return 0.0
            
            if prompt_length == 0 or response_length == 0:
                return 0.3  # Low but not zero for edge cases
            
            length_ratio = response_length / prompt_length
            
            if 2 <= length_ratio <= 10:
                ratio_score = 1.0
            elif 1 <= length_ratio < 2:
                ratio_score = 0.7
            elif 10 < length_ratio <= 20:
                ratio_score = 0.8
            elif 0.5 <= length_ratio < 1:
                ratio_score = 0.5
            else:
                ratio_score = 0.3
            
            if response_length < 10:  # Very short response
                ratio_score *= 0.5
            elif response_length > 5000:  # Very long response
                ratio_score *= 0.8
            
            return min(1.0, ratio_score)
            
        except Exception as e:
            logging.error(f"Error calculating quality score: {e}")
            return 0.5
    
    def _calculate_efficiency_score(self, response_time: float, task_type: str) -> float:
        """Calculate efficiency score based on response time and task type."""
        try:
            baseline_times = {
                'text': 5.0,
                'code': 8.0,
                'image': 15.0,
                'translate': 3.0,
                'summarize': 4.0,
                'multimodal': 12.0
            }
            
            baseline = baseline_times.get(task_type, 6.0)
            
            if response_time <= baseline:
                return 1.0
            else:
                import math
                decay_factor = (response_time - baseline) / baseline
                return max(0.1, math.exp(-decay_factor))
            
        except Exception as e:
            logging.error(f"Error calculating efficiency score: {e}")
            return 0.5
    
    def _store_outcome(self, outcome_data: Dict[str, Any]):
        """Store outcome data to persistent storage."""
        try:
            with open(self.outcome_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(outcome_data, default=str) + '\n')
        except Exception as e:
            logging.error(f"Error storing outcome data: {e}")
    
    def _update_model_scores(self, model: str, task_type: str, score: float):
        """Update running averages for model scores."""
        try:
            if model not in self.model_scores:
                self.model_scores[model] = {
                    'total_score': 0.0,
                    'count': 0,
                    'average': 0.0,
                    'by_task_type': {}
                }
            
            model_data = self.model_scores[model]
            model_data['total_score'] += score
            model_data['count'] += 1
            model_data['average'] = model_data['total_score'] / model_data['count']
            
            if task_type not in model_data['by_task_type']:
                model_data['by_task_type'][task_type] = {
                    'total_score': 0.0,
                    'count': 0,
                    'average': 0.0
                }
            
            task_data = model_data['by_task_type'][task_type]
            task_data['total_score'] += score
            task_data['count'] += 1
            task_data['average'] = task_data['total_score'] / task_data['count']
            
            if task_type not in self.task_type_scores:
                self.task_type_scores[task_type] = {
                    'total_score': 0.0,
                    'count': 0,
                    'average': 0.0,
                    'by_model': {}
                }
            
            task_type_data = self.task_type_scores[task_type]
            task_type_data['total_score'] += score
            task_type_data['count'] += 1
            task_type_data['average'] = task_type_data['total_score'] / task_type_data['count']
            
            if model not in task_type_data['by_model']:
                task_type_data['by_model'][model] = {
                    'total_score': 0.0,
                    'count': 0,
                    'average': 0.0
                }
            
            model_in_task = task_type_data['by_model'][model]
            model_in_task['total_score'] += score
            model_in_task['count'] += 1
            model_in_task['average'] = model_in_task['total_score'] / model_in_task['count']
            
        except Exception as e:
            logging.error(f"Error updating model scores: {e}")
    
    def _load_outcome_data(self):
        """Load existing outcome data from file."""
        try:
            import os
            if os.path.exists(self.outcome_file):
                with open(self.outcome_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                outcome_data = json.loads(line)
                                model = outcome_data.get('model')
                                task_type = outcome_data.get('task_type')
                                score = outcome_data.get('final_score')
                                
                                if model and task_type and score is not None:
                                    self._update_model_scores(model, task_type, score)
                                    
                            except json.JSONDecodeError:
                                continue
                                
                logging.info(f"Loaded outcome data for {len(self.model_scores)} models")
        except Exception as e:
            logging.error(f"Error loading outcome data: {e}")
    
    def get_best_model_for_task(self, task_type: str, min_samples: int = 5) -> Optional[str]:
        """Get the best performing model for a specific task type."""
        try:
            if task_type not in self.task_type_scores:
                return None
            
            task_data = self.task_type_scores[task_type]
            best_model = None
            best_score = 0.0
            
            for model, model_data in task_data['by_model'].items():
                if model_data['count'] >= min_samples and model_data['average'] > best_score:
                    best_score = model_data['average']
                    best_model = model
            
            return best_model
            
        except Exception as e:
            logging.error(f"Error getting best model for task: {e}")
            return None
    
    def get_model_performance(self, model: str, task_type: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for a model."""
        try:
            if model not in self.model_scores:
                return {'error': 'Model not found in performance data'}
            
            model_data = self.model_scores[model]
            
            if task_type:
                if task_type in model_data['by_task_type']:
                    task_data = model_data['by_task_type'][task_type]
                    return {
                        'model': model,
                        'task_type': task_type,
                        'average_score': task_data['average'],
                        'total_requests': task_data['count'],
                        'performance_level': self._get_performance_level(task_data['average'])
                    }
                else:
                    return {'error': f'No data for {model} on {task_type}'}
            else:
                return {
                    'model': model,
                    'overall_average': model_data['average'],
                    'total_requests': model_data['count'],
                    'task_breakdown': {
                        task: data['average'] for task, data in model_data['by_task_type'].items()
                    },
                    'performance_level': self._get_performance_level(model_data['average'])
                }
                
        except Exception as e:
            logging.error(f"Error getting model performance: {e}")
            return {'error': str(e)}
    
    def get_task_performance_ranking(self, task_type: str) -> List[Dict[str, Any]]:
        """Get performance ranking for all models on a specific task."""
        try:
            if task_type not in self.task_type_scores:
                return []
            
            task_data = self.task_type_scores[task_type]
            rankings = []
            
            for model, model_data in task_data['by_model'].items():
                rankings.append({
                    'model': model,
                    'average_score': model_data['average'],
                    'total_requests': model_data['count'],
                    'performance_level': self._get_performance_level(model_data['average'])
                })
            
            rankings.sort(key=lambda x: x['average_score'], reverse=True)
            
            for i, ranking in enumerate(rankings):
                ranking['rank'] = i + 1
            
            return rankings
            
        except Exception as e:
            logging.error(f"Error getting task performance ranking: {e}")
            return []
    
    def _get_performance_level(self, score: float) -> str:
        """Convert numeric score to performance level."""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.6:
            return "fair"
        elif score >= 0.5:
            return "poor"
        else:
            return "very_poor"
    
    def get_outcome_stats(self) -> Dict[str, Any]:
        """Get comprehensive outcome statistics."""
        try:
            total_models = len(self.model_scores)
            total_task_types = len(self.task_type_scores)
            
            all_scores = []
            total_requests = 0
            
            for model_data in self.model_scores.values():
                all_scores.extend([model_data['average']] * model_data['count'])
                total_requests += model_data['count']
            
            if all_scores:
                overall_avg = statistics.mean(all_scores)
                overall_median = statistics.median(all_scores)
                overall_std = statistics.stdev(all_scores) if len(all_scores) > 1 else 0
            else:
                overall_avg = overall_median = overall_std = 0
            
            return {
                'total_models_tracked': total_models,
                'total_task_types': total_task_types,
                'total_requests': total_requests,
                'overall_statistics': {
                    'average_score': overall_avg,
                    'median_score': overall_median,
                    'std_deviation': overall_std
                },
                'model_count_by_performance': {
                    level: sum(1 for model_data in self.model_scores.values() 
                              if self._get_performance_level(model_data['average']) == level)
                    for level in ['excellent', 'good', 'fair', 'poor', 'very_poor']
                }
            }
            
        except Exception as e:
            logging.error(f"Error getting outcome stats: {e}")
            return {'error': str(e)}
    
    def record_user_feedback(self, model: str, task_type: str, user_feedback: int, 
                           response_time: float, prompt_length: int, response_length: int,
                           user_id: str = None) -> float:
        """Record user feedback and calculate outcome score."""
        return self.score_model_outcome(
            model=model,
            task_type=task_type,
            user_feedback=user_feedback,
            completion_success=True,  # Assume success if user is providing feedback
            response_time=response_time,
            prompt_length=prompt_length,
            response_length=response_length,
            user_id=user_id
        )
