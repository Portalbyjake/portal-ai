#!/usr/bin/env python3
"""
Enhanced Intelligent Prompt Optimizer with conversation context awareness.
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationContext:
    """Context for prompt optimization including conversation history."""
    task_type: str
    model: str
    original_prompt: str
    conversation_history: Optional[List[Dict[str, Any]]] = None
    user_preferences: Optional[Dict[str, Any]] = None
    conversation_tone: Optional[str] = None  # 'formal', 'casual', 'technical', 'creative'
    context_hints: Optional[List[str]] = None

class IntelligentPromptOptimizer:
    """Enhanced prompt optimizer with conversation awareness and clarification handling."""
    
    def __init__(self):
        # Enhanced role detection patterns - now more dynamic, broad, and intuitive
        self.role_patterns = {
            # Marketing & Business
            'CMO': [
                'marketing strategy', 'sales strategy', 'marketing campaign', 'brand strategy',
                'market strategy', 'promotional strategy', 'advertising strategy', 'go-to-market',
                'growth strategy', 'customer acquisition', 'digital marketing', 'content marketing',
                'social media strategy', 'influencer campaign', 'product launch', 'brand awareness',
                'market positioning', 'competitive positioning', 'marketing plan', 'marketing roadmap',
                'marketing professional', 'marketing expert', 'marketing specialist', 'marketing manager',
            ],
            'Copywriter': [
                'marketing copy', 'advertising content', 'promotional materials', 'ad copy',
                'marketing content', 'sales copy', 'advertising copy', 'website copy', 'landing page copy',
                'email campaign', 'newsletter', 'slogan', 'tagline', 'product description', 'press release',
                'copywriting', 'copywriter', 'write ad', 'write copy', 'write content',
            ],
            'Business Analyst': [
                'business plan', 'business strategy', 'business proposal', 'strategic plan',
                'business model', 'business analysis', 'business case', 'business document',
                'swot analysis', 'market analysis', 'financial analysis', 'business requirements',
                'business analyst', 'business report', 'business review',
            ],
            'Entrepreneur': [
                'startup idea', 'startup plan', 'founder', 'entrepreneur', 'pitch deck', 'startup strategy',
                'business pitch', 'startup roadmap', 'startup growth',
            ],
            'Product Manager': [
                'product roadmap', 'product requirements', 'product strategy', 'product launch',
                'product manager', 'product vision', 'product features', 'product backlog',
            ],
            'Sales Director': [
                'sales pitch', 'sales email', 'sales strategy', 'sales plan', 'sales manager',
                'sales enablement', 'sales funnel', 'sales process', 'sales deck', 'sales call',
            ],
            # Development & Technical
            'Senior Developer': [
                'build web app', 'create api', 'develop mobile app', 'build software',
                'create application', 'develop application', 'build with python', 'build with javascript',
                'write code', 'program', 'develop software', 'create software', 'software engineer',
                'software developer', 'full stack', 'backend', 'frontend', 'write script', 'write function',
                'write algorithm', 'codebase', 'refactor', 'debug', 'unit test', 'integration test',
                'fix bug', 'implement feature', 'add feature', 'software project',
            ],
            'System Architect': [
                'design system architecture', 'system architecture', 'architectural design',
                'system design', 'architecture design', 'infrastructure design', 'scalability',
                'system integration', 'microservices', 'cloud architecture', 'solution architect',
            ],
            'Code Reviewer': [
                'debug code', 'review code', 'optimize code', 'refactor code',
                'code review', 'debugging', 'code optimization', 'pull request review',
                'static analysis', 'linting', 'code quality', 'test coverage',
            ],
            'DevOps Engineer': [
                'deploy with docker', 'deploy application', 'ci/cd', 'infrastructure',
                'deployment', 'devops', 'containerization', 'kubernetes', 'cloud deployment',
                'infrastructure as code', 'monitoring', 'logging', 'automation', 'devops pipeline',
            ],
            'Data Engineer': [
                'data pipeline', 'etl', 'data warehouse', 'data engineering', 'big data',
                'data ingestion', 'data transformation', 'data integration', 'data lake',
            ],
            # Creative Writing
            'Creative Writer': [
                'write story', 'write content', 'write blog post', 'write something',
                'create something amazing', 'write something about', 'creative writing',
                'short story', 'novel', 'fiction', 'nonfiction', 'write narrative', 'write scene',
                'write dialogue', 'write character', 'write plot', 'creative writer',
            ],
            'Poet': [
                'write poem', 'write poetry', 'compose poem', 'poetry', 'haiku', 'sonnet',
                'limerick', 'free verse', 'acrostic', 'poet',
            ],
            'Songwriter': [
                'write lyrics', 'compose song', 'write song', 'lyrics', 'songwriting', 'songwriter',
                'chorus', 'verse', 'bridge', 'music lyrics',
            ],
            'Screenwriter': [
                'write script', 'write screenplay', 'screenplay', 'script writing', 'screenwriter',
                'film script', 'tv script', 'dialogue', 'scene', 'screenplay outline',
            ],
            'Journalist': [
                'write article', 'write report', 'write news', 'journalism', 'news story',
                'press release', 'news report', 'investigative report', 'journalist',
            ],
            'Technical Writer': [
                'write documentation', 'technical documentation', 'write technical',
                'documentation writing', 'api docs', 'user manual', 'technical writer',
                'write guide', 'write instructions', 'write manual',
            ],
            # Analysis & Research
            'Data Scientist': [
                'analyze data', 'data analysis', 'statistical analysis', 'machine learning',
                'predictive modeling', 'data mining', 'big data analysis', 'data science',
                'data visualization', 'data scientist', 'data model', 'data prediction',
            ],
            'Market Research Analyst': [
                'investigate market', 'market research', 'market analysis', 'competitive analysis',
                'market trends', 'market investigation', 'market research analyst',
                'industry analysis', 'customer research', 'market segmentation',
            ],
            'UX Researcher': [
                'analyze user behavior', 'user research', 'usability study', 'user experience research',
                'user behavior analysis', 'user testing', 'ux researcher', 'user persona',
                'user journey', 'user feedback',
            ],
            'Financial Analyst': [
                'financial analysis', 'financial model', 'investment analysis', 'budget analysis',
                'forecasting', 'valuation', 'finance report', 'financial statement', 'financial analyst',
            ],
            # Design & UX
            'UX/UI Designer': [
                'design user interface', 'design logo', 'create brand identity',
                'ui design', 'ux design', 'user interface design', 'wireframe', 'mockup',
                'prototype', 'user flow', 'user experience', 'ui designer', 'ux designer',
            ],
            'Graphic Designer': [
                'graphic design', 'design poster', 'design brochure', 'design flyer',
                'design graphics', 'visual design', 'illustration', 'infographic', 'graphic designer',
            ],
            'Product Designer': [
                'product design', 'industrial design', 'design product', 'product designer',
                'design prototype', 'product concept',
            ],
            # Education & Support
            'Teacher': [
                'lesson plan', 'teach', 'explain', 'educate', 'curriculum', 'syllabus',
                'classroom', 'homework', 'quiz', 'test', 'exam', 'teacher', 'instructor',
            ],
            'Tutor': [
                'tutor', 'tutoring', 'help me learn', 'explain to me', 'study guide',
                'practice problems', 'test prep', 'homework help',
            ],
            'Coach': [
                'coach', 'coaching', 'life coach', 'career coach', 'mentor', 'mentoring',
                'guidance', 'advice', 'personal development',
            ],
            'Customer Support': [
                'customer support', 'customer service', 'support ticket', 'help desk',
                'technical support', 'support agent', 'customer care', 'troubleshoot',
                'resolve issue', 'support request',
            ],
            # Healthcare & Wellness
            'Doctor': [
                'diagnose', 'medical advice', 'doctor', 'physician', 'treatment plan',
                'symptoms', 'healthcare', 'medical report', 'prescription',
            ],
            'Therapist': [
                'therapy', 'therapist', 'mental health', 'counseling', 'psychologist',
                'emotional support', 'wellness', 'self-care', 'stress management',
            ],
            'Nutritionist': [
                'nutrition', 'diet plan', 'meal plan', 'nutritionist', 'healthy eating',
                'diet advice', 'weight loss', 'meal prep',
            ],
            # Science & Engineering
            'Scientist': [
                'scientific research', 'experiment', 'hypothesis', 'lab report', 'scientist',
                'research paper', 'science project', 'scientific method',
            ],
            'Engineer': [
                'engineering', 'engineer', 'design circuit', 'mechanical design',
                'electrical engineering', 'civil engineering', 'structural analysis',
                'engineering report', 'engineering drawing',
            ],
            # Law & Policy
            'Lawyer': [
                'legal advice', 'lawyer', 'attorney', 'contract review', 'legal document',
                'policy analysis', 'compliance', 'regulation', 'legal opinion',
            ],
            'Policy Analyst': [
                'policy analysis', 'public policy', 'policy recommendation', 'policy brief',
                'regulatory analysis', 'government policy', 'policy analyst',
            ],
            # General Assistant (fallback for generic help/support)
            'General Assistant': [
                'help', 'assist', 'support', 'guide', 'general question', 'general inquiry',
                'how do i', 'what is', 'can you', 'could you', 'would you', 'please help',
            ],
        }
        
        # Patterns that indicate need for clarification - EXPANDED
        self.clarification_patterns = [
            r'\b(write|create|build|make)\s+(something|anything)\b',
            r'\b(write|create|build|make)\s+(it|this|that)\b',
            r'\b(write|create|build|make)\s+(a|an)\s*\w{1,3}\b',  # Very short objects
            r'\b(write|create|build|make)\s*$',  # Just the verb
            r'\b(write|create|build|make)\s+\w+\s+and\s+\w+',  # Compound requests
            # NEW: Ambiguous patterns that need clarification - MORE SPECIFIC
            r'\b(create|write)\s+a\s+(plan|report)\b',  # Generic plans/reports (but not specific ones)
            r'\b(build|design)\s+a\s+solution\b',  # Ambiguous solutions
            r'\b(analyze|research|study)\s+(data|trends|patterns)\s*$',  # Generic analysis without context
            r'\b(investigate|analyze)\s+(market|user behavior)\s*$',  # Could be multiple roles
        ]
        
        # Tone indicators
        self.tone_indicators = {
            'formal': ['please', 'kindly', 'would you', 'could you', 'business', 'professional'],
            'casual': ['hey', 'hi', 'thanks', 'cool', 'awesome', 'great'],
            'technical': ['api', 'code', 'debug', 'optimize', 'architecture', 'system'],
            'creative': ['story', 'poem', 'creative', 'artistic', 'imaginative']
        }

    def _analyze_conversation_context(self, context: OptimizationContext) -> Dict[str, Any]:
        """Analyze conversation history and context for better role assignment."""
        analysis = {
            'tone': 'neutral',
            'complexity_level': 'medium',
            'user_expertise': 'general',
            'conversation_style': 'mixed',
            'ongoing_context': None,
            'primary_intent': None,
            'domain_focus': None
        }
        
        if not context.conversation_history:
            return analysis
            
        # Analyze recent messages for tone and context
        recent_messages = context.conversation_history[-5:]  # Last 5 messages
        
        # Detect tone from conversation
        formal_count = 0
        casual_count = 0
        technical_count = 0
        creative_count = 0
        
        for msg in recent_messages:
            content = msg.get('content', '').lower()
            
            # Count tone indicators
            for tone, indicators in self.tone_indicators.items():
                if any(indicator in content for indicator in indicators):
                    if tone == 'formal':
                        formal_count += 1
                    elif tone == 'casual':
                        casual_count += 1
                    elif tone == 'technical':
                        technical_count += 1
                    elif tone == 'creative':
                        creative_count += 1
        
        # Determine dominant tone
        tone_counts = {
            'formal': formal_count,
            'casual': casual_count,
            'technical': technical_count,
            'creative': creative_count
        }
        
        if max(tone_counts.values()) > 0:
            analysis['tone'] = max(tone_counts, key=lambda k: tone_counts[k])
        
        # DYNAMIC CONTEXT ANALYSIS - Analyze full conversation intent
        all_content = ' '.join([msg.get('content', '') for msg in recent_messages]).lower()
        current_prompt = context.original_prompt.lower()
        
        # Analyze the PRIMARY INTENT and DOMAIN FOCUS from the full conversation
        intent_analysis = self._analyze_conversation_intent(all_content, current_prompt)
        
        analysis['primary_intent'] = intent_analysis['primary_intent']
        analysis['domain_focus'] = intent_analysis['domain_focus']
        analysis['ongoing_context'] = intent_analysis['context']
        analysis['domain_scores'] = intent_analysis['domain_scores']
        
        return analysis
    
    def _analyze_conversation_intent(self, conversation_content: str, current_prompt: str) -> Dict[str, Any]:
        """Analyze the full conversation to determine primary intent and domain focus."""
        
        # Combine conversation content with current prompt for full context
        full_context = f"{conversation_content} {current_prompt}"
        
        # Define domain patterns with context awareness
        domain_patterns = {
            'marketing': {
                'strong_indicators': ['marketing strategy', 'brand strategy', 'advertising campaign', 'digital marketing', 'social media marketing', 'brand awareness', 'marketing plan'],
                'weak_indicators': ['marketing', 'brand', 'advertising', 'campaign'],
                'exclude_contexts': ['sales process', 'lead generation', 'revenue pipeline', 'technical development']
            },
            'sales': {
                'strong_indicators': ['sales strategy', 'lead generation', 'sales pipeline', 'revenue growth', 'sales process', 'lead qualification', 'sales team'],
                'weak_indicators': ['sales', 'leads', 'pipeline', 'revenue'],
                'exclude_contexts': ['marketing campaign', 'brand strategy', 'technical development']
            },
            'technical': {
                'strong_indicators': ['software development', 'web application', 'mobile app', 'api development', 'debug code', 'programming', 'coding', 'system architecture'],
                'weak_indicators': ['code', 'programming', 'development', 'software', 'app', 'api', 'debug'],
                'exclude_contexts': ['marketing strategy', 'brand awareness', 'sales process', 'business strategy']
            },
            'business': {
                'strong_indicators': ['business strategy', 'strategic planning', 'business plan', 'business development', 'market expansion', 'business operations'],
                'weak_indicators': ['business', 'strategy', 'plan', 'operations', 'growth'],
                'exclude_contexts': ['marketing campaign', 'sales process', 'technical development']
            },
            'creative': {
                'strong_indicators': ['write story', 'creative writing', 'poetry', 'song lyrics', 'screenplay', 'artistic content'],
                'weak_indicators': ['creative', 'writing', 'story', 'poetry', 'music'],
                'exclude_contexts': ['business strategy', 'technical development', 'sales process']
            }
        }
        
        # Analyze each domain with context awareness
        domain_scores = {}
        
        for domain, patterns in domain_patterns.items():
            score = 0
            
            # Check for strong indicators (higher weight)
            for indicator in patterns['strong_indicators']:
                if indicator in full_context:
                    score += 3
            
            # Check for weak indicators (lower weight)
            for indicator in patterns['weak_indicators']:
                if indicator in full_context:
                    score += 1
            
            # Check for exclusion contexts (negative weight)
            for exclude in patterns['exclude_contexts']:
                if exclude in full_context:
                    score -= 2
            
            domain_scores[domain] = max(0, score)  # Don't go below 0
        
        # Determine primary intent based on highest score
        if max(domain_scores.values()) > 0:
            primary_intent = max(domain_scores, key=lambda k: domain_scores[k])
        else:
            primary_intent = None
        
        # Determine domain focus (more specific than context)
        domain_focus = None
        if primary_intent:
            if primary_intent == 'marketing':
                domain_focus = 'marketing'
            elif primary_intent == 'sales':
                domain_focus = 'sales'
            elif primary_intent == 'technical':
                domain_focus = 'technical'
            elif primary_intent == 'business':
                domain_focus = 'business'
            elif primary_intent == 'creative':
                domain_focus = 'creative'
        
        return {
            'primary_intent': primary_intent,
            'domain_focus': domain_focus,
            'context': primary_intent,  # Use primary intent as context
            'domain_scores': domain_scores
        }

    def _needs_clarification(self, prompt: str, context_analysis: Optional[Dict[str, Any]] = None) -> bool:
        """Check if the prompt needs clarification before optimization. Now only triggers for truly ambiguous prompts."""
        prompt_lower = prompt.lower().strip()
        # If we have clear context, don't ask for clarification
        if context_analysis and context_analysis.get('ongoing_context'):
            return False
        # Only ask for clarification on truly ambiguous cases (not just short or slightly vague)
        clarification_patterns = [
            r'\b(write|create|build|make)\s+(something|anything)\b',
            r'\b(write|create|build|make)\s+(it|this|that)\b',
            r'\b(write|create|build|make)\s*$',  # Just the verb
            r'\b(analyze|research|study)\s+(data|trends|patterns)\s*$',  # Just "analyze data"
            r'\b(investigate|analyze)\s+(market|user behavior)\s*$',  # Just "investigate market"
        ]
        for pattern in clarification_patterns:
            if re.search(pattern, prompt_lower):
                return True
        return False

    def _get_clarification_prompt(self, original_prompt: str, context_analysis: Dict[str, Any]) -> str:
        """Generate a clarification prompt based on the context."""
        base_clarification = "I'd like to help you with that! To provide the best assistance, could you clarify:"
        
        if 'something' in original_prompt.lower():
            return f"{base_clarification}\n\n• What specific type of content would you like me to create?\n• What's the purpose or goal of this request?\n• Who is the intended audience?"
        
        if ' and ' in original_prompt.lower():
            return f"{base_clarification}\n\n• Which aspect would you like me to focus on primarily?\n• Are these related tasks or separate requests?\n• What's the most important outcome you're looking for?"
        
        # Generic clarification
        return f"{base_clarification}\n\n• What specific type of content or solution are you looking for?\n• What's the context or purpose of this request?\n• Any specific requirements or preferences?"

    def _get_appropriate_role(self, prompt: str, context_analysis: Dict[str, Any]) -> Optional[str]:
        """Get the most appropriate role based on prompt and context analysis. Only assign if strong match or user context."""
        primary_intent = context_analysis.get('primary_intent')
        domain_focus = context_analysis.get('domain_focus')
        prompt_lower = prompt.lower()
        # Only assign role if strong match (intent + domain score or direct pattern match)
        role_mapping = {
            'marketing': 'CMO',
            'sales': 'Sales Director', 
            'technical': 'Senior Developer',
            'business': 'Business Strategist',
            'creative': 'Creative Writer',
            'journalism': 'Journalist',
            'poetry': 'Poet',
            'songwriting': 'Songwriter',
            'screenwriting': 'Screenwriter',
            'documentation': 'Technical Writer',
            'data': 'Data Scientist',
            'ux': 'UX/UI Designer',
            'review': 'Code Reviewer',
            'devops': 'DevOps Engineer',
            'architecture': 'System Architect',
        }
        if primary_intent and domain_focus and domain_focus == primary_intent:
            if primary_intent in role_mapping:
                return role_mapping[primary_intent]
        # Only assign if direct, strong pattern match
        for role, patterns in self.role_patterns.items():
            for pattern in patterns:
                if pattern in prompt_lower:
                    return role
        return None

    def _should_add_role_context(self, prompt: str, context_analysis: Dict[str, Any]) -> bool:
        """Determine if role context should be added based on prompt or context analysis. Only if strong match."""
        primary_intent = context_analysis.get('primary_intent')
        domain_scores = context_analysis.get('domain_scores', {})
        # Add role if context is strong
        if primary_intent and domain_scores.get(primary_intent, 0) >= 2:
            return True
        # Add role if prompt matches a role pattern directly
        prompt_lower = prompt.lower()
        for role, patterns in self.role_patterns.items():
            for pattern in patterns:
                if pattern in prompt_lower:
                    return True
        return False

    def _would_benefit_from_structure(self, context: OptimizationContext) -> bool:
        """Check if the prompt would benefit from structured optimization."""
        prompt_lower = context.original_prompt.lower()
        
        # Keywords that indicate need for structure
        structure_keywords = [
            'write', 'create', 'build', 'develop', 'design', 'analyze', 'research',
            'study', 'investigate', 'debug', 'review', 'optimize', 'refactor',
            'compose', 'craft', 'generate', 'produce', 'construct', 'implement',
            'make', 'do', 'help', 'assist', 'support', 'guide'
        ]
        
        return any(keyword in prompt_lower for keyword in structure_keywords)

    def _is_very_vague(self, prompt: str) -> bool:
        """Check if the prompt is too vague for meaningful optimization."""
        prompt_lower = prompt.lower()
        
        # Very vague patterns
        vague_patterns = [
            r'^\s*(write|create|build|make|do)\s*$',
            r'^\s*(write|create|build|make|do)\s+(something|anything)\s*$',
            r'^\s*(help|assist|support)\s*$'
        ]
        
        return any(re.match(pattern, prompt_lower) for pattern in vague_patterns)

    def _reduce_redundancy(self, prompt: str, model: str) -> str:
        """Reduce redundancy and verbosity in the prompt if it would benefit the model."""
        # Only apply for text models (not image/audio)
        if not model or not any(m in model for m in ["gpt", "claude", "gemini", "llama", "phind", "wizardcoder"]):
            return prompt
        
        # Remove repeated words/phrases (simple version)
        import re
        # Remove repeated polite phrases
        polite_patterns = [
            r"can you please ", r"could you please ", r"would you please ", r"please kindly ", r"if you don't mind", r"i was wondering if you could", r"i think i need", r"i'm just wondering if", r"maybe you could", r"would you be so kind as to", r"if it's not too much trouble"
        ]
        for pat in polite_patterns:
            prompt = re.sub(pat, "please ", prompt, flags=re.IGNORECASE)
        # Remove duplicate whitespace
        prompt = re.sub(r"\s+", " ", prompt)
        # Remove repeated phrases (e.g., "explain, explain this")
        prompt = re.sub(r"(\b\w+\b)(,? \1\b)+", r"\1", prompt, flags=re.IGNORECASE)
        # Remove trailing polite/verbose endings
        prompt = re.sub(r"(if you don't mind|if possible|if that's okay|if that's alright|if you could|if you would|if you can|if you may|if you will|if you please)[.!?]*$", "", prompt, flags=re.IGNORECASE)
        # Remove excessive length if not needed
        if len(prompt) > 200:
            # Try to keep the main request, cut rambling
            sentences = re.split(r'(?<=[.!?]) +', prompt)
            if len(sentences) > 1:
                prompt = sentences[0] + (" " + sentences[1] if len(sentences[1]) < 100 else "")
        return prompt.strip()

    def _enhance_image_prompt(self, prompt: str, model: str) -> str:
        """Dynamically enhance image prompts for models like DALL-E, Stable Diffusion, etc., based on user intent, subject, and style."""
        prompt = prompt.strip()
        if not model or not any(m in model.lower() for m in ["dall-e", "stablediffusion", "midjourney", "sdxl", "imagegen", "firefly"]):
            return prompt
        import re
        # Extract subject
        subject_match = re.search(r'(?:of|a|an|the)\s+([\w\s]+)', prompt, re.IGNORECASE)
        subject = subject_match.group(1).strip() if subject_match else prompt
        # Detect style/intent
        style = ''
        if any(kw in prompt.lower() for kw in ['photo', 'photograph', 'photorealistic', 'realistic']):
            style = 'photorealistic, high-resolution photograph'
        elif any(kw in prompt.lower() for kw in ['painting', 'oil painting', 'watercolor', 'art', 'artwork']):
            style = 'detailed painting, vibrant colors'
        elif any(kw in prompt.lower() for kw in ['sketch', 'drawing', 'line art']):
            style = 'clean line art, high contrast'
        elif any(kw in prompt.lower() for kw in ['anime', 'manga', 'cartoon']):
            style = 'anime style, colorful, dynamic composition'
        elif any(kw in prompt.lower() for kw in ['logo', 'icon', 'symbol']):
            style = 'minimalist, vector, flat design'
        elif any(kw in prompt.lower() for kw in ['fantasy', 'castle', 'dragon', 'magic']):
            style = 'fantasy art, epic, dramatic lighting'
        else:
            style = 'highly detailed, professional, visually stunning'
        # Add lighting/composition if not present
        if not any(kw in prompt.lower() for kw in ['lighting', 'composition', 'angle', 'perspective']):
            style += ', soft lighting, balanced composition'
        # Compose enhanced prompt
        enhanced = f"{subject}, {style}"
        # Add 'create an image of' if prompt is very short
        if len(prompt.split()) <= 3:
            enhanced = f"Create an image of {enhanced}"
        return enhanced

    def _is_factual_question(self, prompt: str) -> bool:
        """Detect if the prompt is a simple factual question (who, what, when, where, why, how, define, calculate, etc.)."""
        prompt_lower = prompt.strip().lower()
        factual_starts = [
            'what is', 'who is', 'where is', 'when is', 'why is', 'how is',
            'what are', 'who are', 'where are', 'when are', 'why are', 'how are',
            'define', 'calculate', 'explain', 'tell me about', 'give me the', 'list', 'show me', 'find', 'how many', 'how much', 'what was', 'who was', 'where was', 'when was', 'why was', 'how was',
        ]
        return any(prompt_lower.startswith(start) for start in factual_starts)

    def _should_optimize_text(self, context: OptimizationContext) -> str:
        """Determine how to optimize text prompts with context awareness."""
        prompt = context.original_prompt
        context_analysis = self._analyze_conversation_context(context)
        
        # Check if clarification is needed
        if self._needs_clarification(prompt, context_analysis):
            return self._get_clarification_prompt(prompt, context_analysis)
        
        # Check if too vague
        if self._is_very_vague(prompt):
            return prompt  # Return unchanged for very vague prompts
        
        # Get appropriate role
        role = self._get_appropriate_role(prompt, context_analysis)
        
        # Determine if role context should be added (lower threshold)
        should_add_role = self._should_add_role_context(prompt, context_analysis)
        if not should_add_role and role:
            # If a clear role is detected from the prompt itself, allow role assignment
            should_add_role = True
        
        # Only assign roles when there's a clear benefit
        if role and should_add_role:
            if context_analysis['tone'] == 'formal':
                role_preface = f"You are a {role}. "
            elif context_analysis['tone'] == 'casual':
                role_preface = f"You're a {role}. "
            else:
                role_preface = f"You are a {role}. "
            optimized_prompt = role_preface + prompt
        else:
            optimized_prompt = prompt
        
        # Redundancy reduction: only apply if it would improve prompt quality
        reduced = self._reduce_redundancy(optimized_prompt, context.model)
        if reduced != optimized_prompt:
            return reduced
        return optimized_prompt

    def _needs_optimization(self, context: OptimizationContext) -> bool:
        """Check if the prompt needs optimization."""
        return (
            self._would_benefit_from_structure(context) and
            not self._is_very_vague(context.original_prompt)
        )

    def optimize_prompt(self, context: OptimizationContext) -> str:
        """Optimize the prompt for the given context, including text and image prompts. Always enhance image prompts for image tasks."""
        prompt = context.original_prompt.strip()
        model = context.model
        task_type = context.task_type
        # Analyze context
        context_analysis = self._analyze_conversation_context(context)
        # Always enhance image prompt for image tasks
        if task_type == "image" or (model and any(m in model.lower() for m in ["dall-e", "stablediffusion", "midjourney", "sdxl", "imagegen", "firefly"])):
            prompt = self._enhance_image_prompt(prompt, model)
            return prompt
        # Factual question filter: never assign a role
        if self._is_factual_question(prompt):
            optimized = self._reduce_redundancy(prompt, model)
            return optimized
        # Collaborative logic for text prompts
        role = self._get_appropriate_role(prompt, context_analysis)
        needs_clarification = self._needs_clarification(prompt, context_analysis)
        optimized = prompt
        if role:
            optimized = f"You are a {role}. {optimized}"
        if needs_clarification:
            clarification = self._get_clarification_prompt(prompt, context_analysis)
            # If role was added, append clarification; else, just clarification
            if role:
                optimized = f"{optimized}\n{clarification}"
            else:
                optimized = clarification
        # Redundancy reduction and other optimizations
        optimized = self._reduce_redundancy(optimized, model)
        return optimized
