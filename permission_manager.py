from typing import Dict, List, Set, Tuple, Optional, Any
from enum import Enum
from datetime import datetime, timedelta
import json
import os
import logging

class PermissionTier(Enum):
    FREE = "free"
    PRO = "pro" 
    ENTERPRISE = "enterprise"
    ADMIN = "admin"

class PermissionManager:
    def __init__(self):
        self.tier_permissions = {
            PermissionTier.FREE: {
                'models': ['gpt-4o-mini', 'claude-haiku'],
                'tasks': ['text', 'summarize'],
                'daily_requests': 100,
                'sensitive_tasks': False,
                'pipeline_access': False,
                'multimodal_access': False,
                'advanced_features': []
            },
            PermissionTier.PRO: {
                'models': ['gpt-4o', 'claude-sonnet-4', 'dall-e-3', 'gpt-4o-mini', 'claude-haiku'],
                'tasks': ['text', 'image', 'summarize', 'translate', 'code'],
                'daily_requests': 1000,
                'sensitive_tasks': True,
                'pipeline_access': True,
                'multimodal_access': True,
                'advanced_features': ['semantic_rewriting', 'culture_aware', 'emotion_processing']
            },
            PermissionTier.ENTERPRISE: {
                'models': ['*'],  # All models
                'tasks': ['*'],   # All tasks
                'daily_requests': 10000,
                'sensitive_tasks': True,
                'pipeline_access': True,
                'multimodal_access': True,
                'advanced_features': ['*']  # All features
            },
            PermissionTier.ADMIN: {
                'models': ['*'],
                'tasks': ['*'],
                'daily_requests': -1,  # Unlimited
                'sensitive_tasks': True,
                'pipeline_access': True,
                'multimodal_access': True,
                'advanced_features': ['*']
            }
        }
        
        self.user_tiers = {}
        self.usage_tracking = {}
        self.permissions_file = "user_permissions.json"
        self.usage_file = "usage_tracking.json"
        
        self._load_permissions()
        self._load_usage_tracking()
        
    def check_permission(self, user_id: str, model: str, task_type: str) -> Tuple[bool, str]:
        """Check if user has permission to use specific model and task type."""
        try:
            user_tier = self.user_tiers.get(user_id, PermissionTier.FREE)
            permissions = self.tier_permissions[user_tier]
            
            if not self._check_usage_limit(user_id, permissions['daily_requests']):
                return False, f"Daily request limit exceeded for {user_tier.value} tier"
            
            if '*' not in permissions['models'] and model not in permissions['models']:
                return False, f"Model {model} not available in {user_tier.value} tier"
            
            if '*' not in permissions['tasks'] and task_type not in permissions['tasks']:
                return False, f"Task {task_type} not available in {user_tier.value} tier"
            
            return True, "Permission granted"
            
        except Exception as e:
            logging.error(f"Error checking permissions: {e}")
            return False, f"Permission check failed: {str(e)}"
    
    def check_feature_permission(self, user_id: str, feature: str) -> bool:
        """Check if user has permission to use advanced feature."""
        try:
            user_tier = self.user_tiers.get(user_id, PermissionTier.FREE)
            permissions = self.tier_permissions[user_tier]
            
            advanced_features = permissions.get('advanced_features', [])
            return '*' in advanced_features or feature in advanced_features
            
        except Exception as e:
            logging.error(f"Error checking feature permission: {e}")
            return False
    
    def track_usage(self, user_id: str, model: str, task_type: str, success: bool = True):
        """Track user usage for rate limiting."""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            if user_id not in self.usage_tracking:
                self.usage_tracking[user_id] = {}
            
            if today not in self.usage_tracking[user_id]:
                self.usage_tracking[user_id][today] = {
                    'requests': 0,
                    'successful_requests': 0,
                    'models_used': set(),
                    'tasks_used': set()
                }
            
            daily_usage = self.usage_tracking[user_id][today]
            daily_usage['requests'] += 1
            
            if success:
                daily_usage['successful_requests'] += 1
            
            daily_usage['models_used'].add(model)
            daily_usage['tasks_used'].add(task_type)
            
            daily_usage['models_used'] = list(daily_usage['models_used'])
            daily_usage['tasks_used'] = list(daily_usage['tasks_used'])
            
            self._save_usage_tracking()
            
        except Exception as e:
            logging.error(f"Error tracking usage: {e}")
    
    def set_user_tier(self, user_id: str, tier: PermissionTier):
        """Set user permission tier."""
        try:
            self.user_tiers[user_id] = tier
            self._save_permissions()
            logging.info(f"Set user {user_id} to tier {tier.value}")
            
        except Exception as e:
            logging.error(f"Error setting user tier: {e}")
    
    def get_user_tier(self, user_id: str) -> PermissionTier:
        """Get user permission tier."""
        return self.user_tiers.get(user_id, PermissionTier.FREE)
    
    def get_user_usage(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Get user usage statistics for specified number of days."""
        try:
            if user_id not in self.usage_tracking:
                return {'total_requests': 0, 'daily_breakdown': {}}
            
            user_usage = self.usage_tracking[user_id]
            total_requests = 0
            daily_breakdown = {}
            
            for i in range(days):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                if date in user_usage:
                    daily_data = user_usage[date]
                    daily_breakdown[date] = daily_data
                    total_requests += daily_data.get('requests', 0)
            
            return {
                'total_requests': total_requests,
                'daily_breakdown': daily_breakdown,
                'user_tier': self.get_user_tier(user_id).value
            }
            
        except Exception as e:
            logging.error(f"Error getting user usage: {e}")
            return {'total_requests': 0, 'daily_breakdown': {}}
    
    def _check_usage_limit(self, user_id: str, daily_limit: int) -> bool:
        """Check if user is within daily usage limit."""
        if daily_limit == -1:  # Unlimited
            return True
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        if user_id not in self.usage_tracking:
            return True
        
        if today not in self.usage_tracking[user_id]:
            return True
        
        today_requests = self.usage_tracking[user_id][today].get('requests', 0)
        return today_requests < daily_limit
    
    def _load_permissions(self):
        """Load user permissions from file."""
        try:
            if os.path.exists(self.permissions_file):
                with open(self.permissions_file, 'r') as f:
                    data = json.load(f)
                    for user_id, tier_name in data.items():
                        try:
                            self.user_tiers[user_id] = PermissionTier(tier_name)
                        except ValueError:
                            logging.warning(f"Invalid tier {tier_name} for user {user_id}")
                            self.user_tiers[user_id] = PermissionTier.FREE
                            
                logging.info(f"Loaded permissions for {len(self.user_tiers)} users")
        except Exception as e:
            logging.error(f"Error loading permissions: {e}")
    
    def _save_permissions(self):
        """Save user permissions to file."""
        try:
            data = {user_id: tier.value for user_id, tier in self.user_tiers.items()}
            
            with open(self.permissions_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving permissions: {e}")
    
    def _load_usage_tracking(self):
        """Load usage tracking data from file."""
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r') as f:
                    self.usage_tracking = json.load(f)
                    
                for user_id, user_data in self.usage_tracking.items():
                    for date, daily_data in user_data.items():
                        if 'models_used' in daily_data:
                            daily_data['models_used'] = set(daily_data['models_used'])
                        if 'tasks_used' in daily_data:
                            daily_data['tasks_used'] = set(daily_data['tasks_used'])
                            
                logging.info(f"Loaded usage tracking for {len(self.usage_tracking)} users")
        except Exception as e:
            logging.error(f"Error loading usage tracking: {e}")
    
    def _save_usage_tracking(self):
        """Save usage tracking data to file."""
        try:
            data = {}
            for user_id, user_data in self.usage_tracking.items():
                data[user_id] = {}
                for date, daily_data in user_data.items():
                    data[user_id][date] = daily_data.copy()
                    if 'models_used' in daily_data:
                        data[user_id][date]['models_used'] = list(daily_data['models_used'])
                    if 'tasks_used' in daily_data:
                        data[user_id][date]['tasks_used'] = list(daily_data['tasks_used'])
            
            with open(self.usage_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving usage tracking: {e}")
    
    def get_tier_info(self, tier: PermissionTier) -> Dict[str, Any]:
        """Get information about a permission tier."""
        return self.tier_permissions.get(tier, {})
    
    def get_all_tiers_info(self) -> Dict[str, Any]:
        """Get information about all permission tiers."""
        return {tier.value: permissions for tier, permissions in self.tier_permissions.items()}
    
    def cleanup_old_usage_data(self, days_to_keep: int = 30):
        """Clean up old usage tracking data."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.strftime('%Y-%m-%d')
            
            for user_id in self.usage_tracking:
                dates_to_remove = []
                for date in self.usage_tracking[user_id]:
                    if date < cutoff_str:
                        dates_to_remove.append(date)
                
                for date in dates_to_remove:
                    del self.usage_tracking[user_id][date]
            
            self._save_usage_tracking()
            logging.info(f"Cleaned up usage data older than {days_to_keep} days")
            
        except Exception as e:
            logging.error(f"Error cleaning up usage data: {e}")
    
    def get_permission_stats(self) -> Dict[str, Any]:
        """Get statistics about permission usage."""
        tier_counts = {}
        for tier in PermissionTier:
            tier_counts[tier.value] = 0
        
        for user_tier in self.user_tiers.values():
            tier_counts[user_tier.value] += 1
        
        return {
            'total_users': len(self.user_tiers),
            'tier_distribution': tier_counts,
            'total_tracked_users': len(self.usage_tracking)
        }
