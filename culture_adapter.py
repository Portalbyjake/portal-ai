from typing import Dict, Optional, Tuple
from utils import detect_language
import logging

class CultureAwareAdapter:
    def __init__(self):
        self.cultural_contexts = {
            'en': {
                'formality': 'casual', 
                'directness': 'direct', 
                'examples': 'western',
                'greeting_style': 'informal',
                'politeness_level': 'moderate'
            },
            'ja': {
                'formality': 'formal', 
                'directness': 'indirect', 
                'examples': 'japanese',
                'greeting_style': 'respectful',
                'politeness_level': 'high'
            },
            'de': {
                'formality': 'formal', 
                'directness': 'direct', 
                'examples': 'german',
                'greeting_style': 'professional',
                'politeness_level': 'moderate'
            },
            'es': {
                'formality': 'warm', 
                'directness': 'expressive', 
                'examples': 'hispanic',
                'greeting_style': 'friendly',
                'politeness_level': 'high'
            },
            'zh': {
                'formality': 'respectful', 
                'directness': 'indirect', 
                'examples': 'chinese',
                'greeting_style': 'formal',
                'politeness_level': 'high'
            },
            'fr': {
                'formality': 'elegant', 
                'directness': 'diplomatic', 
                'examples': 'french',
                'greeting_style': 'polite',
                'politeness_level': 'high'
            },
            'it': {
                'formality': 'expressive', 
                'directness': 'passionate', 
                'examples': 'italian',
                'greeting_style': 'warm',
                'politeness_level': 'moderate'
            },
            'pt': {
                'formality': 'friendly', 
                'directness': 'warm', 
                'examples': 'portuguese',
                'greeting_style': 'cordial',
                'politeness_level': 'moderate'
            },
            'ru': {
                'formality': 'formal', 
                'directness': 'straightforward', 
                'examples': 'russian',
                'greeting_style': 'reserved',
                'politeness_level': 'moderate'
            },
            'ar': {
                'formality': 'respectful', 
                'directness': 'courteous', 
                'examples': 'arabic',
                'greeting_style': 'honorific',
                'politeness_level': 'high'
            }
        }
        
        self.cultural_adaptations = {
            'business_context': {
                'en': 'professional and efficient',
                'ja': 'respectful and consensus-building',
                'de': 'precise and structured',
                'es': 'relationship-focused and collaborative',
                'zh': 'hierarchical and face-saving'
            },
            'creative_context': {
                'en': 'innovative and bold',
                'ja': 'harmonious and refined',
                'de': 'systematic and thorough',
                'es': 'passionate and colorful',
                'zh': 'balanced and meaningful'
            },
            'educational_context': {
                'en': 'interactive and questioning',
                'ja': 'respectful and attentive',
                'de': 'methodical and comprehensive',
                'es': 'engaging and supportive',
                'zh': 'disciplined and reverent'
            }
        }
        
    def adapt_prompt(self, prompt: str, user_id: str, detected_language: Optional[str] = None, context_type: str = 'general') -> Tuple[str, Dict[str, str]]:
        """Adapt prompt based on cultural context and language."""
        try:
            if not detected_language:
                detected_language = detect_language(prompt)
            
            if detected_language not in self.cultural_contexts:
                logging.info(f"No cultural adaptation available for language: {detected_language}")
                return prompt, {'language': detected_language, 'adaptation': 'none'}
            
            context = self.cultural_contexts[detected_language]
            
            cultural_instruction = self._generate_cultural_instruction(context, detected_language, context_type)
            
            if cultural_instruction:
                adapted_prompt = f"{cultural_instruction}\n\nUser request: {prompt}"
            else:
                adapted_prompt = prompt
            
            adaptation_info = {
                'language': detected_language,
                'formality': context['formality'],
                'directness': context['directness'],
                'context_type': context_type,
                'adaptation': 'applied' if cultural_instruction else 'minimal'
            }
            
            logging.info(f"Applied cultural adaptation for {detected_language}: {context['formality']}, {context['directness']}")
            return adapted_prompt, adaptation_info
            
        except Exception as e:
            logging.error(f"Error in cultural adaptation: {e}")
            return prompt, {'language': detected_language or 'unknown', 'adaptation': 'error'}
    
    def _generate_cultural_instruction(self, context: Dict[str, str], language: str, context_type: str) -> str:
        """Generate cultural instruction based on context."""
        try:
            formality = context['formality']
            directness = context['directness']
            politeness = context['politeness_level']
            
            if formality == 'formal':
                tone_instruction = "Please respond in a formal, respectful tone"
            elif formality == 'casual':
                tone_instruction = "Please respond in a friendly, casual tone"
            elif formality == 'warm':
                tone_instruction = "Please respond in a warm, personable tone"
            elif formality == 'elegant':
                tone_instruction = "Please respond in an elegant, refined tone"
            elif formality == 'expressive':
                tone_instruction = "Please respond in an expressive, enthusiastic tone"
            elif formality == 'respectful':
                tone_instruction = "Please respond in a respectful, considerate tone"
            else:
                tone_instruction = f"Please respond in a {formality} tone"
            
            if directness == 'indirect':
                style_instruction = "Be diplomatic and tactful in your communication"
            elif directness == 'direct':
                style_instruction = "Be clear and straightforward in your communication"
            elif directness == 'expressive':
                style_instruction = "Be expressive and emotionally engaging"
            elif directness == 'diplomatic':
                style_instruction = "Be diplomatic and nuanced in your approach"
            elif directness == 'passionate':
                style_instruction = "Be passionate and engaging in your delivery"
            elif directness == 'courteous':
                style_instruction = "Be courteous and considerate in your manner"
            else:
                style_instruction = f"Be {directness} in your communication style"
            
            context_instruction = ""
            if context_type in self.cultural_adaptations and language in self.cultural_adaptations[context_type]:
                context_adaptation = self.cultural_adaptations[context_type][language]
                context_instruction = f" Approach this in a {context_adaptation} manner."
            
            politeness_instruction = ""
            if politeness == 'high':
                politeness_instruction = " Use appropriate honorifics and show high respect."
            elif politeness == 'moderate':
                politeness_instruction = " Maintain appropriate politeness."
            
            return f"{tone_instruction}. {style_instruction}.{context_instruction}{politeness_instruction}"
            
        except Exception as e:
            logging.error(f"Error generating cultural instruction: {e}")
            return ""
    
    def detect_context_type(self, prompt: str) -> str:
        """Detect the context type of the prompt for cultural adaptation."""
        prompt_lower = prompt.lower()
        
        business_keywords = [
            'business', 'company', 'corporate', 'professional', 'meeting', 'proposal',
            'strategy', 'marketing', 'sales', 'revenue', 'profit', 'investment',
            'contract', 'negotiation', 'partnership', 'client', 'customer'
        ]
        
        creative_keywords = [
            'creative', 'art', 'design', 'story', 'poem', 'music', 'painting',
            'drawing', 'writing', 'novel', 'screenplay', 'artistic', 'imagination',
            'inspiration', 'aesthetic', 'beautiful', 'elegant'
        ]
        
        educational_keywords = [
            'learn', 'teach', 'education', 'study', 'research', 'academic',
            'university', 'school', 'course', 'lesson', 'tutorial', 'explain',
            'understand', 'knowledge', 'science', 'mathematics', 'history'
        ]
        
        business_score = sum(1 for keyword in business_keywords if keyword in prompt_lower)
        creative_score = sum(1 for keyword in creative_keywords if keyword in prompt_lower)
        educational_score = sum(1 for keyword in educational_keywords if keyword in prompt_lower)
        
        if business_score > creative_score and business_score > educational_score:
            return 'business_context'
        elif creative_score > educational_score:
            return 'creative_context'
        elif educational_score > 0:
            return 'educational_context'
        else:
            return 'general'
    
    def get_cultural_preferences(self, language: str) -> Optional[Dict[str, str]]:
        """Get cultural preferences for a specific language."""
        return self.cultural_contexts.get(language)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages for cultural adaptation."""
        return list(self.cultural_contexts.keys())
    
    def analyze_cultural_needs(self, prompt: str, user_id: str) -> Dict[str, any]:
        """Analyze if prompt would benefit from cultural adaptation."""
        try:
            detected_language = detect_language(prompt)
            context_type = self.detect_context_type(prompt)
            
            needs_adaptation = (
                detected_language in self.cultural_contexts and
                detected_language != 'en'  # English is default
            )
            
            return {
                'detected_language': detected_language,
                'context_type': context_type,
                'needs_adaptation': needs_adaptation,
                'available_adaptations': detected_language in self.cultural_contexts,
                'cultural_context': self.cultural_contexts.get(detected_language, {})
            }
            
        except Exception as e:
            logging.error(f"Error analyzing cultural needs: {e}")
            return {
                'detected_language': 'unknown',
                'context_type': 'general',
                'needs_adaptation': False,
                'available_adaptations': False,
                'cultural_context': {}
            }
