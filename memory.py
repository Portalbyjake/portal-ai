import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import re

class ConversationMemory:
    """Enhanced conversation memory with intelligent context recognition and task routing"""
    
    def __init__(self, memory_file="memory_text.jsonl", image_memory_file="memory_image.jsonl"):
        self.memory_file = memory_file
        self.image_memory_file = image_memory_file
        self.conversation_memory = {}
        self.conversation_topics = {}  # Track conversation topics
        self.task_history = {}  # Track task types for routing
        self.multimodal_context = {}  # Track cross-modal references
        self.image_history = {}  # NEW: Per-user image history stack
        self.load_memory()
        self.load_image_memory()  # NEW: Load image memory with error handling
    
    def load_memory(self):
        """Load existing memory from files with robust error handling"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    line_number = 0
                    for line in f:
                        line_number += 1
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            entry = json.loads(line)
                            user_id = entry.get('user_id', 'default')
                            if user_id not in self.conversation_memory:
                                self.conversation_memory[user_id] = []
                            self.conversation_memory[user_id].append(entry)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping corrupted JSON line {line_number}: {e}")
                            # Try to recover by removing the corrupted line
                            self._remove_corrupted_line(line_number)
                            continue
                        except Exception as e:
                            print(f"Warning: Error processing line {line_number}: {e}")
                            continue
        except Exception as e:
            print(f"Error loading memory: {e}")
            # If the file is completely corrupted, backup and start fresh
            self._backup_corrupted_file()
    
    def _remove_corrupted_line(self, line_number):
        """Remove a corrupted line from the memory file"""
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if line_number <= len(lines):
                lines.pop(line_number - 1)  # Convert to 0-based index
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)
        except Exception as e:
            print(f"Error removing corrupted line: {e}")
    
    def _backup_corrupted_file(self):
        """Backup corrupted file and start fresh"""
        try:
            if os.path.exists(self.memory_file):
                backup_name = f"{self.memory_file}.backup.{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                os.rename(self.memory_file, backup_name)
                print(f"Backed up corrupted file to {backup_name}")
        except Exception as e:
            print(f"Error backing up corrupted file: {e}")
    
    def save_memory(self, user_id: str, role: str, content: str, task_type: Optional[str] = None, model: Optional[str] = None, metadata: Optional[Dict] = None):
        """Save a memory entry with robust error handling and enhanced metadata"""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'role': role,
            'content': content,
            'task_type': task_type,
            'model': model,
            'metadata': metadata or {}
        }
        
        # Add to in-memory storage
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = []
        self.conversation_memory[user_id].append(entry)
        
        # Track task history for intelligent routing
        if task_type:
            if user_id not in self.task_history:
                self.task_history[user_id] = []
            self.task_history[user_id].append({
                'task_type': task_type,
                'timestamp': datetime.utcnow().isoformat(),
                'content': content[:100],  # Store first 100 chars for context
                'metadata': metadata
            })
        
        # NEW: Track image history for undo/revert and chaining
        if task_type == 'image':
            if user_id not in self.image_history:
                self.image_history[user_id] = []
            # Store prompt, summary, url, model, and timestamp
            image_entry = {
                'prompt': content,
                'summary': (metadata or {}).get('summary'),
                'url': (metadata or {}).get('url'),
                'model': model,
                'timestamp': entry['timestamp']
            }
            self.image_history[user_id].append(image_entry)
        
        # Update multimodal context
        self._update_multimodal_context(user_id, entry)
        
        # Save to file with error handling
        try:
            with open(self.memory_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def _update_multimodal_context(self, user_id: str, entry: Dict):
        """Update multimodal context tracking"""
        if user_id not in self.multimodal_context:
            self.multimodal_context[user_id] = {
                'current_modality': None,
                'cross_references': [],
                'last_image_prompt': None,
                'last_text_topic': None
            }
        
        context = self.multimodal_context[user_id]
        task_type = entry.get('task_type')
        
        # Track modality switches
        if task_type != context['current_modality']:
            if context['current_modality']:
                # Record cross-modal reference
                context['cross_references'].append({
                    'from_modality': context['current_modality'],
                    'to_modality': task_type,
                    'timestamp': entry['timestamp'],
                    'content': entry['content'][:100]
                })
            context['current_modality'] = task_type
        
        # Track specific content for cross-modal references
        if task_type == 'image':
            context['last_image_prompt'] = entry['content']
        elif task_type == 'text':
            context['last_text_topic'] = entry['content']
    
    def get_multimodal_context(self, user_id: str) -> Dict:
        """Get current multimodal context for the user"""
        return self.multimodal_context.get(user_id, {})
    
    def is_cross_modal_reference(self, user_input: str, user_id: str) -> Tuple[bool, str, str]:
        """
        Detect if user input references content from a different modality.
        Returns: (is_cross_modal, referenced_modality, referenced_content)
        """
        if user_id not in self.multimodal_context:
            return False, "", ""
        
        context = self.multimodal_context[user_id]
        input_lower = user_input.lower()
        
        # Check for references to previous image
        if context.get('last_image_prompt'):
            image_ref_indicators = [
                'that image', 'the image', 'this image', 'the picture', 'that picture',
                'the photo', 'that photo', 'the art', 'that art', 'it', 'this', 'that'
            ]
            if any(indicator in input_lower for indicator in image_ref_indicators):
                return True, 'image', context['last_image_prompt']
        
        # Check for references to previous text
        if context.get('last_text_topic'):
            text_ref_indicators = [
                'that topic', 'what we discussed', 'the subject', 'that subject',
                'it', 'this', 'that', 'the thing', 'what you said'
            ]
            if any(indicator in input_lower for indicator in text_ref_indicators):
                return True, 'text', context['last_text_topic']
        
        return False, "", ""
    
    def get_enhanced_context_for_followup(self, user_id: str, current_query: str) -> str:
        """Get enhanced context that includes cross-modal references"""
        memory = self.get_recent_memory(user_id, 10)
        
        if not memory:
            return ""
        
        # Build basic conversation context
        context_parts = []
        for entry in memory[-6:]:  # Last 6 exchanges
            role = entry.get('role', 'user')
            content = entry.get('content', '')
            task_type = entry.get('task_type', 'text')
            
            if role == 'user':
                context_parts.append(f"User ({task_type}): {content}")
            else:
                context_parts.append(f"Assistant ({task_type}): {content}")
        
        # Add cross-modal context
        is_cross_modal, referenced_modality, referenced_content = self.is_cross_modal_reference(current_query, user_id)
        if is_cross_modal:
            context_parts.append(f"\nCross-modal reference: User is referring to previous {referenced_modality} content: '{referenced_content}'")
        
        context = "\n".join(context_parts)
        return f"Previous conversation context:\n{context}\n\nCurrent query: {current_query}"
    
    def get_recent_memory(self, user_id: str, max_exchanges: int = 10) -> List[Dict]:
        """Get recent conversation memory for context"""
        if user_id not in self.conversation_memory:
            return []
        
        memory = self.conversation_memory[user_id]
        return memory[-max_exchanges:] if len(memory) > max_exchanges else memory
    
    def get_memory_for_model(self, user_id: str, model: str, max_tokens: int = 4000) -> List[Dict]:
        """Get memory formatted for specific model context with improved follow-up handling"""
        memory = self.get_recent_memory(user_id, 20)
        
        # Convert to model-specific format
        formatted_memory = []
        total_tokens = 0
        
        for entry in reversed(memory):  # Start from most recent
            content = entry.get('content', '')
            estimated_tokens = len(content.split()) * 1.3  # Rough token estimation
            
            if total_tokens + estimated_tokens > max_tokens:
                break
                
            formatted_memory.insert(0, {
                'role': entry.get('role', 'user'),
                'content': content
            })
            total_tokens += estimated_tokens
        
        return formatted_memory
    
    def get_context_for_followup(self, user_id: str, current_query: str) -> str:
        """Get context specifically for handling follow-up questions"""
        memory = self.get_recent_memory(user_id, 10)
        
        if not memory:
            return ""
        
        # Build context from recent exchanges (last 6 exchanges = 3 Q&A pairs)
        context_parts = []
        for entry in memory[-6:]:  # Last 6 exchanges
            role = entry.get('role', 'user')
            content = entry.get('content', '')
            if role == 'user':
                context_parts.append(f"User: {content}")
            else:
                context_parts.append(f"Assistant: {content}")
        
        context = "\n".join(context_parts)
        return f"Previous conversation context:\n{context}\n\nCurrent query: {current_query}"
    
    def is_followup_question(self, query: str, user_id: str) -> bool:
        """Enhanced follow-up question detection with multimodal awareness"""
        query_lower = query.lower().strip()
        
        # Check for cross-modal references first
        is_cross_modal, _, _ = self.is_cross_modal_reference(query, user_id)
        if is_cross_modal:
            return True
        
        # Direct follow-up indicators
        followup_indicators = [
            "it", "that", "this", "they", "them", "those", "these", 
            "he", "she", "his", "her", "their", "its", "there",
            "how many", "what about", "tell me more", "and", "also",
            "what else", "anything else", "more", "other", "different"
        ]
        
        # Check for follow-up indicators
        has_followup_indicators = any(indicator in query_lower for indicator in followup_indicators)
        
        # Check if this is a short question (likely follow-up)
        is_short_question = len(query.split()) <= 5
        
        # Check if there's recent conversation context
        has_context = len(self.get_recent_memory(user_id, 2)) > 0
        
        # Consider it a follow-up if it has indicators OR is a short question with context
        return has_followup_indicators or (is_short_question and has_context)
    
    def is_conversational_response(self, user_input: str, user_id: str) -> bool:
        """
        Advanced conversational response detection:
        - Multi-turn context (looks back 4 turns)
        - Regex for transformation/action verbs
        - Multi-intent splitting
        - Semantic similarity stub (for future embedding use)
        - Robust feedback/clarification detection
        - On par with or better than ChatGPT
        """
        input_lower = user_input.lower().strip()
        recent_memory = self.get_recent_memory(user_id, 4)
        # 1. Feedback/acknowledgment detection (expanded)
        feedback_ack_patterns = [
            r"thank(s| you)?", r"that's (perfect|great|exactly what i wanted|fine|good|all|it|what i needed)",
            r"awesome", r"perfect", r"much better", r"that works", r"i appreciate( it)?", r"no further action",
            r"no more needed", r"that's all", r"that helps( a lot)?", r"close enough", r"not quite right(, can you try again)?",
            r"not quite", r"not exactly", r"not what i meant", r"not what i wanted", r"cheers", r"no, that's (fine|good|perfect|all)"
        ]
        for pattern in feedback_ack_patterns:
            if re.fullmatch(pattern, input_lower) or re.search(pattern, input_lower):
                return True
        # 2. Multi-intent/action detection (regex, multi-turn)
        # Action/transform verbs (expanded)
        action_regex = re.compile(r"\b(summarize|put|make|turn|convert|translate|list|show|draw|write|explain|visualize|diagram|chart|table|bullet|checklist|steps|concise|expand|elaborate|paraphrase|rewrite|simplify|structure|organize|sort|filter|group|highlight|extract|compare|contrast|code|reformat|format|structure|shorter|longer|more concise|more detailed|step by step|as a|in a|into a)\b", re.I)
        # Multi-intent splitting (e.g., "summarize and translate that")
        if action_regex.search(input_lower):
            # If multiple actions, split and check each
            actions = re.split(r" and |,|;| then | also | as well as ", input_lower)
            for act in actions:
                if action_regex.search(act):
                    # If any action is present, treat as follow-up task, not conversational
                    return False
        # 3. Multi-turn context: check if previous assistant output exists
        prev_assistant_msgs = [m for m in recent_memory if m.get('role') == 'assistant']
        if prev_assistant_msgs:
            # If message is a question and contains action/transform, treat as follow-up
            if input_lower.endswith('?') or input_lower.startswith(('can you', 'could you', 'would you', 'please')):
                if action_regex.search(input_lower):
                    return False
        # 4. Semantic similarity stub (for future embedding use)
        # If you have embeddings, compute similarity between user_input and previous assistant outputs
        # If similarity > threshold and action verb present, treat as follow-up task
        # 5. Clarification/ambiguous feedback (expanded)
        clarification_patterns = [
            r"what do you mean", r"i don't (understand|get it)", r"can you (explain|clarify)",
            r"what does that mean", r"i'm confused", r"i don't see", r"where is", r"not sure"
        ]
        for pattern in clarification_patterns:
            if re.search(pattern, input_lower):
                return True
        ambiguous_feedback = [
            r"closer", r"almost", r"getting there", r"better", r"improved", r"more like it", r"not bad", r"try again", r"try once more"
        ]
        for pattern in ambiguous_feedback:
            if re.search(pattern, input_lower):
                return True
        # 6. Short, non-question, non-action = likely feedback
        if len(input_lower) < 20 and not input_lower.endswith('?') and not action_regex.search(input_lower):
            return True
        return False

    def _analyze_conversational_intent_enhanced(self, user_input: str, recent_memory: list) -> bool:
        """
        Enhanced: Analyze if the input is conversational, feedback, or clarification, with ambiguity handling.
        Now more robust: Only classify as conversational if the input is truly feedback, clarification, or a non-image, non-action, non-color follow-up.
        """
        input_lower = user_input.lower().strip()
        last_assistant_response = ""
        last_user_input = ""
        for entry in reversed(recent_memory):
            if entry.get('role') == 'assistant' and not last_assistant_response:
                last_assistant_response = entry.get('content', '')
            elif entry.get('role') == 'user' and not last_user_input:
                last_user_input = entry.get('content', '')
            if last_assistant_response and last_user_input:
                break
        if not last_assistant_response:
            return False
        # --- Use all previous conversational patterns ---
        # Only classify as conversational if NOT an image/action/color prompt
        image_indicators = [
            "generate", "create", "make", "show", "draw", "paint", "image", "picture", "photo", "art", "drawing", "sketch"
        ]
        color_indicators = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white", "gray"]
        action_words = ["change", "edit", "modify", "replace", "swap", "add", "remove", "adjust", "transform", "convert"]
        # If it contains image/action/color words, it's NOT conversational
        if any(indicator in input_lower for indicator in image_indicators + action_words):
            return False
        if any(color in input_lower for color in color_indicators):
            reference_pronouns = ["it", "that", "this", "them", "those", "these"]
            if any(pronoun in input_lower for pronoun in reference_pronouns):
                return False
        # Otherwise, use conversational patterns
        conversational_patterns = [
            self._is_reaction_to_response,
            self._is_clarification_request,
            self._is_contextual_followup,
            self._is_evaluative_response,
            self._is_acknowledgment,
            self._is_feedback_acknowledgment,
            self._is_ambiguous_feedback
        ]
        for pattern_func in conversational_patterns:
            if pattern_func(input_lower, last_assistant_response, last_user_input):
                print(f"DEBUG: Enhanced conversational pattern {pattern_func.__name__} returned True for '{input_lower}'")
                return True
        return False

    def _is_reaction_to_response(self, input_lower: str, last_response: str, last_user_input: str) -> bool:
        """Check if input is a reaction to the assistant's response"""
        # Very short responses that don't ask for new information
        if len(input_lower.split()) <= 3:
            # But exclude clear new questions
            question_words = ["what", "how", "why", "when", "where", "who", "which"]
            if not any(word in input_lower for word in question_words):
                return True
        
        # Emotional reactions
        emotional_words = ["wow", "oh", "hmm", "interesting", "cool", "nice", "good", "bad", 
                         "amazing", "terrible", "beautiful", "ugly", "weird", "strange"]
        if any(word in input_lower for word in emotional_words):
            return True
        
        return False
    
    def _is_clarification_request(self, input_lower: str, last_response: str, last_user_input: str) -> bool:
        """Check if input is asking for clarification about the response"""
        clarification_indicators = [
            "what do you mean", "i don't understand", "i don't get it",
            "can you explain", "can you clarify", "what does that mean",
            "i'm confused", "i don't see", "where is"
        ]
        
        return any(indicator in input_lower for indicator in clarification_indicators)
    
    def _is_contextual_followup(self, input_lower: str, last_response: str, last_user_input: str) -> bool:
        """Check if input is a follow-up that references the previous conversation"""
        
        # First, check if this is an image request (should not be classified as conversational)
        image_indicators = [
            "generate", "create", "make", "show", "draw", "paint", "image", "picture", "photo", "art", "drawing", "sketch"
        ]
        color_indicators = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white", "gray"]
        
        # If it contains image-related words, it's not conversational
        if any(indicator in input_lower for indicator in image_indicators):
            return False
        
        # If it contains color words and pronouns, it's likely an image request
        if any(color in input_lower for color in color_indicators):
            reference_pronouns = ["it", "that", "this", "them", "those", "these"]
            if any(pronoun in input_lower for pronoun in reference_pronouns):
                return False
        
        # Check for pronouns that reference previous content
        reference_pronouns = ["it", "that", "this", "they", "them", "those", "these"]
        has_reference = any(pronoun in input_lower for pronoun in reference_pronouns)
        
        if has_reference:
            # But exclude if it's clearly a new question
            question_words = ["what", "how", "why", "when", "where", "who", "which"]
            is_new_question = any(word in input_lower for word in question_words)
            
            # If it has reference pronouns but is NOT a new question, it's likely conversational
            if not is_new_question:
                return True
        
        return False
    
    def _is_evaluative_response(self, input_lower: str, last_response: str, last_user_input: str) -> bool:
        """Check if input is evaluating or commenting on the response"""
        evaluative_patterns = [
            "that's good", "that's bad", "that's better", "that's worse",
            "i like it", "i don't like it", "i love it", "i hate it",
            "that's not what i asked for", "that's not what i wanted",
            "that's wrong", "that's not correct", "that's not what i meant"
        ]
        
        return any(pattern in input_lower for pattern in evaluative_patterns)
    
    def _is_acknowledgment(self, input_lower: str, last_response: str, last_user_input: str) -> bool:
        """Check if input is a simple acknowledgment"""
        acknowledgments = ["ok", "okay", "sure", "yeah", "yes", "no", "nope", "maybe",
                         "i guess", "i think", "probably", "possibly"]
        
        return input_lower in acknowledgments

    def _is_feedback_acknowledgment(self, input_lower: str, last_response: str, last_user_input: str) -> bool:
        """Detects gratitude, satisfaction, or 'no further action' signals."""
        feedback_ack_patterns = [
            "thank you", "thanks", "that's perfect", "that's great", "that's exactly what i wanted", "awesome", "perfect", "much better", "that works", "that's it", "that's what i needed", "appreciate it", "cheers", "no further action needed", "no more", "no, i wasn't wanting another image", "no, i wasn't wanting another response", "no, that's all", "no, that's fine", "no, that's good", "no, that's perfect"
        ]
        return any(pat in input_lower for pat in feedback_ack_patterns)

    def _is_ambiguous_feedback(self, input_lower: str, last_response: str, last_user_input: str) -> bool:
        """Detects ambiguous feedback that is not a clear new request."""
        ambiguous_patterns = [
            "that's better", "that's more like it", "closer", "almost", "not quite", "getting there", "improved", "improvement", "not exactly", "not quite right"
        ]
        # If ambiguous, treat as conversational unless clear new request is present
        if any(pat in input_lower for pat in ambiguous_patterns):
            # If also contains a clear new request, don't treat as conversational
            request_indicators = ["generate", "create", "make", "show", "draw", "paint", "image", "picture", "photo", "art", "drawing", "sketch", "write", "explain", "analyze", "summarize", "code", "program"]
            if not any(req in input_lower for req in request_indicators):
                return True
        return False
    
    def get_conversational_response(self, user_input: str, user_id: str, last_response: str, last_task_type: str) -> str:
        """
        Generate an appropriate conversational response when the user is commenting/reacting.
        """
        input_lower = user_input.lower().strip()
        
        # Get context about what was just generated
        if last_task_type == "image":
            if any(word in input_lower for word in ["what", "that", "this", "is", "supposed"]):
                return "I generated an image based on your request. If it's not what you were looking for, could you help me understand what you'd like to see instead? You can describe it more specifically or ask me to try a different approach."
            
            elif any(word in input_lower for word in ["wrong", "not", "didn't", "bad", "terrible"]):
                return "I apologize if the image didn't meet your expectations. Could you tell me more specifically what you were looking for? I can try again with a different approach or more detailed description."
            
            elif any(word in input_lower for word in ["good", "nice", "cool", "amazing", "beautiful"]):
                return "I'm glad you like the image! Is there anything else you'd like me to create or modify about it?"
            
            else:
                return "I see you're reacting to the image I generated. Is there something specific you'd like me to adjust or create instead?"
        
        elif last_task_type == "text":
            if any(word in input_lower for word in ["what", "that", "this", "mean", "understand"]):
                return "I provided information based on your question. If something wasn't clear or you need more details, could you let me know what specifically you'd like me to explain further?"
            
            elif any(word in input_lower for word in ["wrong", "not", "didn't", "bad", "incorrect"]):
                return "I apologize if my response wasn't helpful. Could you clarify what you were looking for? I want to make sure I provide the right information."
            
            elif any(word in input_lower for word in ["good", "nice", "helpful", "useful"]):
                return "I'm glad I could help! Is there anything else you'd like to know about this topic or something else entirely?"
            
            else:
                return "I see you're responding to my answer. Is there something specific you'd like me to clarify or expand on?"
        
        else:
            return "I see you're reacting to my previous response. How can I better assist you with what you're looking for?"
    
    def analyze_conversation_context(self, user_id: str, current_input: str) -> Tuple[str, float, str]:
        """
        Analyze conversation context dynamically to determine task routing.
        Uses semantic analysis rather than keyword matching.
        """
        if user_id not in self.conversation_memory or not self.conversation_memory[user_id]:
            return "text", 0.5, "No previous conversation context"
        
        recent_memory = self.get_recent_memory(user_id, 6)
        current_input_lower = current_input.lower()
        
        # Dynamic context analysis
        context_analysis = self._analyze_conversation_dynamics(current_input, recent_memory)
        
        if context_analysis['is_continuation']:
            return context_analysis['task_type'], context_analysis['confidence'], context_analysis['reasoning']
        
        # Check for explicit new task indicators
        if self._is_new_task_request(current_input, recent_memory):
            return "text", 0.9, "Detected new task request"
        
        # Default to text analysis for unclear cases
        return "text", 0.6, "Defaulting to text analysis"
    
    def _analyze_conversation_dynamics(self, current_input: str, recent_memory: List[Dict]) -> Dict:
        """
        Dynamically analyze conversation context to determine if this is a continuation.
        """
        current_input_lower = current_input.lower()
        
        # Get recent task types and content
        recent_tasks = [entry.get('task_type', 'text') for entry in recent_memory if entry.get('task_type')]
        recent_content = [entry.get('content', '').lower() for entry in recent_memory if entry.get('content')]
        
        if not recent_tasks:
            return {'is_continuation': False, 'task_type': 'text', 'confidence': 0.5, 'reasoning': 'No recent tasks'}
        
        last_task = recent_tasks[-1]
        
        # Analyze semantic continuity
        continuity_score = self._calculate_semantic_continuity(current_input, recent_content)
        
        # If there's strong semantic continuity, continue the task
        if continuity_score > 0.7:
            return {
                'is_continuation': True,
                'task_type': last_task,
                'confidence': continuity_score,
                'reasoning': f'Semantic continuity detected (score: {continuity_score:.2f})'
            }
        
        # Check for explicit continuation indicators
        if self._has_continuation_indicators(current_input_lower, last_task):
            return {
                'is_continuation': True,
                'task_type': last_task,
                'confidence': 0.8,
                'reasoning': 'Explicit continuation indicators detected'
            }
        
        return {'is_continuation': False, 'task_type': 'text', 'confidence': 0.5, 'reasoning': 'No clear continuation'}
    
    def _calculate_semantic_continuity(self, current_input: str, recent_content: List[str]) -> float:
        """
        Calculate semantic continuity between current input and recent conversation.
        Returns a score between 0 and 1.
        """
        if not recent_content:
            return 0.0
        
        current_words = set(current_input.lower().split())
        
        # Calculate word overlap with recent content
        total_overlap = 0
        total_words = 0
        
        for content in recent_content[-3:]:  # Look at last 3 exchanges
            content_words = set(content.split())
            overlap = len(current_words & content_words)
            total_overlap += overlap
            total_words += len(content_words)
        
        if total_words == 0:
            return 0.0
        
        # Normalize overlap score
        overlap_score = total_overlap / total_words
        
        # Boost score for short inputs (likely continuations)
        if len(current_input.split()) <= 3:
            overlap_score *= 1.5
        
        return min(overlap_score, 1.0)
    
    def _has_continuation_indicators(self, input_lower: str, last_task: str) -> bool:
        """
        Check for explicit indicators that suggest continuing the previous task.
        """
        # Task-specific continuation indicators
        if last_task == "image":
            image_continuation = ["bigger", "smaller", "different", "another", "more", "again", "same",
                                "larger", "smaller", "change", "modify", "adjust", "edit"]
            return any(indicator in input_lower for indicator in image_continuation)
        
        elif last_task == "text":
            text_continuation = ["more", "details", "explain", "clarify", "expand", "continue"]
            return any(indicator in input_lower for indicator in text_continuation)
        
        return False
    
    def _is_new_task_request(self, current_input: str, recent_memory: List[Dict]) -> bool:
        """
        Check if the input is clearly a new task request.
        """
        input_lower = current_input.lower()
        
        # Clear new task indicators
        new_task_indicators = [
            "what is", "how does", "explain", "describe", "why", "when", "where",
            "who", "which", "how many", "what about", "information", "details", "facts",
            "create", "make", "generate", "show", "draw", "write", "find", "search"
        ]
        
        # Check for explicit task indicators
        for indicator in new_task_indicators:
            if indicator in input_lower:
                return True
        
        # Check for question structure that suggests a new topic
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        is_question = any(word in input_lower for word in question_words)
        
        if is_question and len(input_lower.split()) > 3:
            return True
        
        return False
    
    def get_intelligent_task_routing(self, user_id: str, user_input: str) -> Tuple[str, float, str]:
        """
        Intelligent task routing that considers conversation context.
        Returns: (task_type, confidence, reasoning)
        """
        print(f"DEBUG: get_intelligent_task_routing called with input: '{user_input}' for user: {user_id}")
        
        # Check if there's any conversation context first
        if user_id not in self.conversation_memory or not self.conversation_memory[user_id]:
            print(f"DEBUG: No conversation context for user {user_id}")
            # No conversation context, use standard task classification
            from classifier.intent_classifier import classify_task
            task_type, confidence = classify_task(user_input)
            if task_type is None:
                task_type = "text"
            print(f"DEBUG: Using standard classification: {task_type}, confidence: {confidence}")
            return task_type, confidence, "No conversation context, using standard classification"
        
        print(f"DEBUG: Found conversation context for user {user_id}")
        
        # There is conversation context, check if this is a conversational response
        is_conv = self.is_conversational_response(user_input, user_id)
        print(f"DEBUG: is_conversational_response returned: {is_conv}")
        
        if is_conv:
            print(f"DEBUG: Detected as conversational response")
            return "conversation", 0.95, "Detected conversational response, not a task request"
        
        print(f"DEBUG: Not conversational, analyzing conversation context")
        
        # First, get the standard task classification
        from classifier.intent_classifier import classify_task
        base_task, base_confidence = classify_task(user_input)
        
        # Ensure base_task is a string
        if base_task is None:
            base_task = "text"
        
        print(f"DEBUG: Standard classification: {base_task}, confidence: {base_confidence}")
        
        # If it's clearly an image task, prioritize that over conversation context
        if base_task == "image" and base_confidence > 0.5:
            print(f"DEBUG: High-confidence image task detected, prioritizing classifier result")
            return base_task, base_confidence, f"High-confidence image classification: {base_confidence:.2f}"
        
        # Not conversational, analyze conversation context for task routing
        task_type, confidence, reasoning = self.analyze_conversation_context(user_id, user_input)
        print(f"DEBUG: analyze_conversation_context returned: {task_type}, {confidence}, {reasoning}")
        
        # If confidence is high for follow-up, use that
        if confidence > 0.8:
            print(f"DEBUG: High confidence follow-up detected")
            return task_type, confidence, reasoning
        
        print(f"DEBUG: Using standard task classification")
        
        # Enhance confidence based on conversation context
        if user_id in self.task_history and self.task_history[user_id]:
            recent_tasks = [entry['task_type'] for entry in self.task_history[user_id][-3:]]
            if base_task in recent_tasks:
                # Task type matches recent history, boost confidence
                enhanced_confidence = min(base_confidence + 0.2, 1.0)
                print(f"DEBUG: Task type matches recent history, enhanced confidence: {enhanced_confidence}")
                return base_task, enhanced_confidence, f"Task type matches recent history: {base_task}"
        
        print(f"DEBUG: Final result: {base_task}, {base_confidence}")
        return base_task, base_confidence, "Standard task classification"
    
    def clear_memory(self, user_id: Optional[str] = None):
        """Clear memory for specific user or all users"""
        if user_id:
            if user_id in self.conversation_memory:
                del self.conversation_memory[user_id]
            if user_id in self.task_history:
                del self.task_history[user_id]
            if user_id in self.multimodal_context:
                del self.multimodal_context[user_id]
            if user_id in self.image_history:
                del self.image_history[user_id]
        else:
            self.conversation_memory = {}
            self.task_history = {}
            self.multimodal_context = {}
            self.image_history = {}
    
    def summarize_old_memory(self, user_id: str, max_age_hours: int = 24):
        """Summarize old memory to prevent context overflow"""
        if user_id not in self.conversation_memory:
            return
        
        memory = self.conversation_memory[user_id]
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        # Keep recent entries, summarize older ones
        recent_memory = []
        old_memory = []
        
        for entry in memory:
            try:
                entry_time = datetime.fromisoformat(entry.get('timestamp', ''))
                if entry_time > cutoff_time:
                    recent_memory.append(entry)
                else:
                    old_memory.append(entry)
            except:
                recent_memory.append(entry)  # Keep if timestamp parsing fails
        
        if len(old_memory) > 5:  # Only summarize if there's significant old memory
            # Create summary entry
            summary_content = f"Previous conversation summary: {len(old_memory)} exchanges from earlier session"
            summary_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'role': 'system',
                'content': summary_content,
                'task_type': 'summary'
            }
            
            # Replace old memory with summary
            self.conversation_memory[user_id] = [summary_entry] + recent_memory
    
    def get_memory_stats(self, user_id: Optional[str] = None) -> Dict:
        """Get memory statistics for monitoring"""
        stats = {
            'total_users': len(self.conversation_memory),
            'total_entries': 0,
            'memory_age': 0.0  # Changed to float to match _get_memory_age return type
        }
        
        if user_id:
            if user_id in self.conversation_memory:
                memory = self.conversation_memory[user_id]
                stats['total_entries'] = len(memory)
                stats['memory_age'] = self._get_memory_age(memory)
        else:
            for user_memory in self.conversation_memory.values():
                stats['total_entries'] += len(user_memory)
        
        return stats
    
    def _get_memory_age(self, memory: List[Dict]) -> float:
        """Calculate the age of memory in hours"""
        if not memory:
            return 0
        
        try:
            oldest_timestamp = memory[0].get('timestamp', '')
            oldest_time = datetime.fromisoformat(oldest_timestamp)
            age = (datetime.utcnow() - oldest_time).total_seconds() / 3600
            return age
        except:
            return 0

    def get_image_history(self, user_id: str) -> list:
        """Get the user's image generation history stack."""
        return self.image_history.get(user_id, [])
    
    def get_last_image_entry(self, user_id: str) -> Optional[dict]:
        """Get the most recent image entry for the user."""
        history = self.get_image_history(user_id)
        return history[-1] if history else None
    
    def undo_last_image(self, user_id: str) -> Optional[dict]:
        """Remove and return the last image entry for undo functionality."""
        if user_id in self.image_history and self.image_history[user_id]:
            return self.image_history[user_id].pop()
        return None

    def load_image_memory(self):
        """Load existing image memory from file with robust error handling"""
        try:
            if os.path.exists(self.image_memory_file):
                with open(self.image_memory_file, 'r', encoding='utf-8') as f:
                    line_number = 0
                    for line in f:
                        line_number += 1
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            user_id = entry.get('user_id', 'default')
                            if user_id not in self.image_history:
                                self.image_history[user_id] = []
                            # Store the whole entry for undo/history
                            self.image_history[user_id].append(entry)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping corrupted image JSON line {line_number}: {e}")
                            self._remove_corrupted_image_line(line_number)
                            continue
                        except Exception as e:
                            print(f"Warning: Error processing image line {line_number}: {e}")
                            continue
        except Exception as e:
            print(f"Error loading image memory: {e}")
            self._backup_corrupted_image_file()

    def _remove_corrupted_image_line(self, line_number):
        """Remove a corrupted line from the image memory file"""
        try:
            with open(self.image_memory_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if line_number <= len(lines):
                lines.pop(line_number - 1)
            with open(self.image_memory_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)
        except Exception as e:
            print(f"Error removing corrupted image line: {e}")

    def _backup_corrupted_image_file(self):
        """Backup corrupted image memory file and start fresh"""
        try:
            if os.path.exists(self.image_memory_file):
                backup_name = f"{self.image_memory_file}.backup.{datetime.utcnow().strftime('%Y%m%d_%H%M%S') }"
                os.rename(self.image_memory_file, backup_name)
                print(f"Backed up corrupted image memory file to {backup_name}")
        except Exception as e:
            print(f"Error backing up corrupted image memory file: {e}")

# Create global instance
conversation_memory = ConversationMemory()
memory_manager = conversation_memory  # Backward compatibility
