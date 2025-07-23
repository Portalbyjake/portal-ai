from prompt_optimizer import IntelligentPromptOptimizer, OptimizationContext
from memory import memory_manager
from models import run_model_with_fallbacks
import re
from typing import Tuple, List, Dict, Optional
import logging

class SemanticTaskRewriter:
    def __init__(self):
        self.optimizer = IntelligentPromptOptimizer()
        self.ambiguity_patterns = [
            r'\b(write|create|make|build|do)\s+(something|anything|it|this|that)\b',
            r'\b(help|assist)\s+(me|with)\s*$',
            r'\b(do|fix|solve|handle)\s+(this|that|it)\s*$',
            r'^\s*(write|create|make|build|do)\s*$',
            r'\b(analyze|research|study)\s+(data|trends|patterns)\s*$',
            r'\b(investigate|examine)\s+(market|user behavior)\s*$'
        ]
        self.context_indicators = [
            'based on', 'considering', 'given that', 'taking into account',
            'following up on', 'continuing from', 'building on'
        ]
        
    def rewrite_if_needed(self, prompt: str, user_id: str, task_type: str) -> Tuple[str, bool]:
        """Rewrite prompt if it's ambiguous or vague, using conversation context."""
        try:
            if not self._is_ambiguous(prompt):
                return prompt, False
            
            logging.info(f"Detected ambiguous prompt for user {user_id}: '{prompt}'")
            
            conversation_history = memory_manager.get_recent_memory(user_id, 10)
            
            if not conversation_history:
                clarification = self._generate_clarification_request(prompt, task_type)
                return clarification, True
            
            opt_context = OptimizationContext(
                task_type=task_type,
                model="semantic-rewriter",
                original_prompt=prompt,
                conversation_history=conversation_history
            )
            
            rewritten = self._generate_clarified_prompt(prompt, conversation_history, opt_context)
            
            if rewritten != prompt:
                logging.info(f"Rewritten prompt: '{rewritten}'")
                return rewritten, True
            
            return prompt, False
            
        except Exception as e:
            logging.error(f"Error in semantic rewriting: {e}")
            return prompt, False
    
    def _is_ambiguous(self, prompt: str) -> bool:
        """Check if prompt is ambiguous or vague."""
        prompt_lower = prompt.lower().strip()
        
        for pattern in self.ambiguity_patterns:
            if re.search(pattern, prompt_lower):
                return True
        
        if len(prompt.split()) <= 2:
            vague_words = ['help', 'do', 'make', 'create', 'write', 'build', 'fix']
            if any(word in prompt_lower for word in vague_words):
                return True
        
        pronouns = ['it', 'this', 'that', 'them', 'they']
        if any(pronoun in prompt_lower for pronoun in pronouns):
            if not any(indicator in prompt_lower for indicator in self.context_indicators):
                return True
        
        return False
    
    def _generate_clarification_request(self, prompt: str, task_type: str) -> str:
        """Generate a clarification request when no context is available."""
        base_clarification = "I'd like to help you with that! To provide the best assistance, could you clarify:"
        
        if 'something' in prompt.lower() or 'anything' in prompt.lower():
            return f"{base_clarification}\n\n• What specific type of content would you like me to create?\n• What's the purpose or goal of this request?\n• Who is the intended audience?"
        
        if any(word in prompt.lower() for word in ['write', 'create', 'make', 'build']):
            return f"{base_clarification}\n\n• What specific item or content should I {prompt.split()[0].lower()}?\n• What are the requirements or specifications?\n• What's the intended use or context?"
        
        if any(word in prompt.lower() for word in ['analyze', 'research', 'study']):
            return f"{base_clarification}\n\n• What specific data or topic should I analyze?\n• What type of analysis are you looking for?\n• What questions are you trying to answer?"
        
        # Generic clarification
        return f"{base_clarification}\n\n• What specific task would you like me to help with?\n• What's the context or background?\n• What outcome are you looking for?"
    
    def _generate_clarified_prompt(self, prompt: str, conversation_history: List[Dict], opt_context: OptimizationContext) -> str:
        """Generate a clarified prompt using conversation context."""
        try:
            context_summary = self._extract_context_summary(conversation_history)
            
            if not context_summary:
                return prompt
            
            rewriting_prompt = f"""
Based on the conversation context below, rewrite this ambiguous request to be more specific and actionable:

Conversation Context:
{context_summary}

Ambiguous Request: "{prompt}"

Rewrite the request to be clear, specific, and actionable based on the conversation context. Keep it concise but complete. Only return the rewritten request, nothing else.
"""
            
            rewritten = run_model_with_fallbacks('gpt-4o', 'text', rewriting_prompt, 'system')
            
            rewritten = rewritten.strip()
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            
            if self._is_valid_rewrite(prompt, rewritten):
                return rewritten
            
            return prompt
            
        except Exception as e:
            logging.error(f"Error generating clarified prompt: {e}")
            return prompt
    
    def _extract_context_summary(self, conversation_history: List[Dict]) -> str:
        """Extract relevant context from conversation history."""
        if not conversation_history:
            return ""
        
        recent_history = conversation_history[-10:]
        
        context_parts = []
        for entry in recent_history:
            role = entry.get('role', 'unknown')
            content = entry.get('content', '')
            if content and len(content) < 200:  # Avoid very long entries
                context_parts.append(f"{role.title()}: {content}")
        
        return "\n".join(context_parts[-6:])  # Last 6 entries
    
    def _is_valid_rewrite(self, original: str, rewritten: str) -> bool:
        """Validate that the rewritten prompt is better than the original."""
        if rewritten.lower() == original.lower():
            return False
        
        if len(rewritten.split()) <= len(original.split()):
            return False
        
        if self._is_ambiguous(rewritten):
            return False
        
        if len(rewritten) > 500:  # Too long
            return False
        
        return True
    
    def analyze_ambiguity_level(self, prompt: str) -> Dict[str, any]:
        """Analyze the level and type of ambiguity in a prompt."""
        ambiguity_score = 0
        ambiguity_types = []
        
        prompt_lower = prompt.lower()
        
        for pattern in self.ambiguity_patterns:
            if re.search(pattern, prompt_lower):
                ambiguity_score += 1
                ambiguity_types.append("vague_action")
        
        pronouns = ['it', 'this', 'that', 'them', 'they']
        if any(pronoun in prompt_lower for pronoun in pronouns):
            ambiguity_score += 1
            ambiguity_types.append("pronoun_reference")
        
        if len(prompt.split()) <= 3:
            ambiguity_score += 1
            ambiguity_types.append("too_short")
        
        return {
            'is_ambiguous': ambiguity_score > 0,
            'ambiguity_score': ambiguity_score,
            'ambiguity_types': ambiguity_types,
            'needs_rewriting': ambiguity_score >= 2
        }
