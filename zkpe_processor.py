import hashlib
import re
import base64
import logging
from typing import Tuple, Dict, Any, List, Optional

try:
    from cryptography.fernet import Fernet
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logging.warning("cryptography not available, using fallback encryption")

class ZKPEProcessor:
    def __init__(self):
        if CRYPTOGRAPHY_AVAILABLE:
            self.encryption_key = Fernet.generate_key()
            self.cipher = Fernet(self.encryption_key)
        else:
            self.encryption_key = None
            self.cipher = None
            
        self.sensitive_patterns = {
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'api_key': r'\b[A-Za-z0-9]{32,}\b',
            'password': r'(?i)(?:password|pwd|pass)[\s:=]+[^\s]+',
            'token': r'(?i)(?:token|auth|bearer)[\s:=]+[A-Za-z0-9._-]+',
            'personal_id': r'\b[A-Z]{2}\d{6,}\b'
        }
        
        self.context_keywords = {
            'financial': ['bank', 'account', 'credit', 'payment', 'transaction'],
            'personal': ['name', 'address', 'birthday', 'personal', 'private'],
            'medical': ['medical', 'health', 'diagnosis', 'prescription', 'patient'],
            'legal': ['legal', 'court', 'lawsuit', 'contract', 'confidential'],
            'business': ['proprietary', 'confidential', 'internal', 'classified']
        }
        
    def process_with_privacy(self, prompt: str, user_id: str) -> Tuple[str, Dict[str, Any]]:
        """Process prompt with zero-knowledge privacy protection."""
        try:
            sensitivity_analysis = self._analyze_sensitivity(prompt)
            
            if not sensitivity_analysis['is_sensitive']:
                return prompt, {
                    'privacy_applied': False,
                    'sensitivity_level': 'low',
                    'analysis': sensitivity_analysis
                }
            
            anonymized_prompt, token_map = self._anonymize_sensitive_data(prompt)
            
            privacy_metadata = {
                'privacy_applied': True,
                'sensitivity_level': sensitivity_analysis['level'],
                'token_count': len(token_map),
                'sensitive_types': list(sensitivity_analysis['detected_types']),
                'analysis': sensitivity_analysis,
                'user_id_hash': self._hash_user_id(user_id),
                'token_map': token_map
            }
            
            logging.info(f"Applied ZKPE to prompt with {len(token_map)} sensitive tokens")
            return anonymized_prompt, privacy_metadata
            
        except Exception as e:
            logging.error(f"Error in ZKPE processing: {e}")
            return prompt, {
                'privacy_applied': False,
                'error': str(e),
                'sensitivity_level': 'unknown'
            }
    
    def _analyze_sensitivity(self, text: str) -> Dict[str, Any]:
        """Analyze text for sensitive content."""
        try:
            detected_patterns = {}
            detected_types = set()
            sensitivity_score = 0
            
            for pattern_name, pattern in self.sensitive_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    detected_patterns[pattern_name] = len(matches)
                    detected_types.add(pattern_name)
                    sensitivity_score += len(matches)
            
            text_lower = text.lower()
            detected_contexts = []
            for context, keywords in self.context_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    detected_contexts.append(context)
                    sensitivity_score += 0.5
            
            if sensitivity_score >= 3:
                level = 'high'
            elif sensitivity_score >= 1:
                level = 'medium'
            elif detected_contexts:
                level = 'low'
            else:
                level = 'none'
            
            is_sensitive = level != 'none'
            
            return {
                'is_sensitive': is_sensitive,
                'level': level,
                'score': sensitivity_score,
                'detected_patterns': detected_patterns,
                'detected_types': detected_types,
                'detected_contexts': detected_contexts
            }
            
        except Exception as e:
            logging.error(f"Error analyzing sensitivity: {e}")
            return {
                'is_sensitive': False,
                'level': 'unknown',
                'score': 0,
                'detected_patterns': {},
                'detected_types': set(),
                'detected_contexts': []
            }
    
    def _anonymize_sensitive_data(self, text: str) -> Tuple[str, Dict[str, Dict]]:
        """Anonymize sensitive data in text."""
        try:
            anonymized_text = text
            token_map = {}
            token_counter = 0
            
            for pattern_name, pattern in self.sensitive_patterns.items():
                matches = list(re.finditer(pattern, anonymized_text, re.IGNORECASE))
                
                for match in reversed(matches):
                    sensitive_data = match.group()
                    token_counter += 1
                    token = f"[ZKPE_TOKEN_{token_counter}]"
                    
                    if self.cipher:
                        encrypted_data = self.cipher.encrypt(sensitive_data.encode())
                        encrypted_b64 = base64.b64encode(encrypted_data).decode()
                    else:
                        encrypted_b64 = self._fallback_encrypt(sensitive_data)
                    
                    token_map[token] = {
                        'type': pattern_name,
                        'encrypted': encrypted_b64,
                        'hash': hashlib.sha256(sensitive_data.encode()).hexdigest()[:8],
                        'length': len(sensitive_data),
                        'position': match.start()
                    }
                    
                    start, end = match.span()
                    anonymized_text = anonymized_text[:start] + token + anonymized_text[end:]
            
            return anonymized_text, token_map
            
        except Exception as e:
            logging.error(f"Error anonymizing sensitive data: {e}")
            return text, {}
    
    def _fallback_encrypt(self, data: str) -> str:
        """Fallback encryption when cryptography is not available."""
        salt = "zkpe_salt_2024"
        combined = f"{salt}:{data}:{salt}"
        encoded = base64.b64encode(combined.encode()).decode()
        return encoded
    
    def restore_sensitive_data(self, response: str, token_map: Dict[str, Dict], 
                             privacy_level: str = 'redacted') -> str:
        """Restore sensitive data in response based on privacy level."""
        try:
            if not token_map:
                return response
            
            restored_response = response
            
            for token, data_info in token_map.items():
                if token in response:
                    if privacy_level == 'full_restore' and self.cipher:
                        try:
                            encrypted_data = base64.b64decode(data_info['encrypted'])
                            decrypted_data = self.cipher.decrypt(encrypted_data).decode()
                            restored_response = restored_response.replace(token, decrypted_data)
                        except Exception as e:
                            logging.error(f"Error restoring token {token}: {e}")
                            restored_response = restored_response.replace(
                                token, f"[RESTORE_ERROR_{data_info['hash']}]")
                    elif privacy_level == 'partial_restore':
                        data_type = data_info['type']
                        hash_short = data_info['hash']
                        restored_response = restored_response.replace(
                            token, f"[{data_type.upper()}_{hash_short}]")
                    else:
                        data_type = data_info['type']
                        hash_short = data_info['hash']
                        restored_response = restored_response.replace(
                            token, f"[REDACTED_{data_type.upper()}_{hash_short}]")
            
            return restored_response
            
        except Exception as e:
            logging.error(f"Error restoring sensitive data: {e}")
            return response
    
    def _hash_user_id(self, user_id: str) -> str:
        """Create a hash of user ID for privacy tracking."""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    def validate_privacy_compliance(self, text: str) -> Dict[str, Any]:
        """Validate that text complies with privacy requirements."""
        try:
            sensitivity_analysis = self._analyze_sensitivity(text)
            
            remaining_sensitive = []
            for pattern_name, pattern in self.sensitive_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    remaining_sensitive.extend([(pattern_name, match) for match in matches])
            
            is_compliant = len(remaining_sensitive) == 0
            
            return {
                'is_compliant': is_compliant,
                'remaining_sensitive_data': remaining_sensitive,
                'sensitivity_analysis': sensitivity_analysis,
                'compliance_score': 1.0 if is_compliant else max(0.0, 1.0 - len(remaining_sensitive) * 0.2)
            }
            
        except Exception as e:
            logging.error(f"Error validating privacy compliance: {e}")
            return {
                'is_compliant': False,
                'error': str(e),
                'compliance_score': 0.0
            }
    
    def get_privacy_stats(self) -> Dict[str, Any]:
        """Get statistics about privacy processing capabilities."""
        return {
            'encryption_available': CRYPTOGRAPHY_AVAILABLE,
            'supported_patterns': list(self.sensitive_patterns.keys()),
            'context_categories': list(self.context_keywords.keys()),
            'privacy_levels': ['full_restore', 'partial_restore', 'redacted']
        }
    
    def create_privacy_report(self, original_text: str, processed_text: str, 
                            token_map: Dict[str, Dict]) -> Dict[str, Any]:
        """Create a privacy processing report."""
        try:
            original_analysis = self._analyze_sensitivity(original_text)
            processed_analysis = self._analyze_sensitivity(processed_text)
            
            return {
                'original_sensitivity': original_analysis,
                'processed_sensitivity': processed_analysis,
                'tokens_created': len(token_map),
                'data_types_processed': list(set(info['type'] for info in token_map.values())),
                'privacy_improvement': {
                    'before_score': original_analysis['score'],
                    'after_score': processed_analysis['score'],
                    'improvement_ratio': max(0, 1 - (processed_analysis['score'] / max(original_analysis['score'], 1)))
                },
                'compliance_status': processed_analysis['score'] == 0
            }
            
        except Exception as e:
            logging.error(f"Error creating privacy report: {e}")
            return {'error': str(e)}
