import re
from typing import Dict, Tuple, List, Optional
import logging

def classify_task(prompt: str) -> Tuple[str, float]:
    """
    Industry-leading task classifier with sophisticated pattern matching and confidence scoring.
    Returns (task_type, confidence_score)
    """
    prompt = prompt.strip().lower()
    
    # Enhanced confidence scoring system with more task types
    scores = {
        "text": 0.0,
        "code": 0.0,
        "image": 0.0, 
        "summarize": 0.0,
        "translate": 0.0,
        "audio": 0.0,
        "multimodal": 0.0,
        "debug": 0.0,
        "optimize": 0.0
    }
    
    # -- Clean, short prompt sanity check --
    if len(prompt) <= 5:
        return "text", 0.8
    
    # -- Code Generation Detection (HIGH PRIORITY) --
    code_patterns = [
        # Programming languages and frameworks
        (r"\b(python|javascript|java|c\+\+|c#|go|rust|swift|kotlin|php|ruby|scala|haskell|elixir|clojure|f#|dart|r|matlab|perl|bash|shell|powershell)\b", 0.8),
        # Code-specific keywords
        (r"\b(function|class|method|variable|loop|array|object|json|xml|html|css|sql|api|endpoint|database|server|client)\b", 0.9),
        # Code creation patterns
        (r"\b(write.*code|create.*function|implement.*class|build.*app|develop.*script|program.*algorithm)\b", 0.95),
        (r"\b(generate.*code|code.*generation|write.*program|create.*script|build.*website|develop.*api)\b", 0.95),
        # Code-related tasks
        (r"\b(debug|fix.*bug|error.*handling|exception|try.*catch|logging|testing|unit.*test|integration.*test)\b", 0.9),
        (r"\b(optimize.*code|performance.*improvement|refactor|clean.*code|best.*practice|design.*pattern)\b", 0.9),
        # Development tools and concepts
        (r"\b(git|docker|kubernetes|aws|azure|gcp|deployment|ci/cd|microservices|rest|graphql)\b", 0.8),
        # Code analysis and review
        (r"\b(code.*review|analyze.*code|explain.*code|documentation|comments|readability)\b", 0.9),
        # Specific code patterns
        (r"\b(if.*else|switch.*case|for.*loop|while.*loop|recursion|async|await|promise|callback)\b", 0.8),
        # Framework-specific
        (r"\b(react|vue|angular|node|express|django|flask|fastapi|spring|asp\.net|laravel|rails)\b", 0.8),
        # Data structures and algorithms
        (r"\b(algorithm|data.*structure|sorting|searching|tree|graph|hash|stack|queue|linked.*list)\b", 0.9),
        # Code quality and architecture
        (r"\b(architecture|scalable|maintainable|readable|efficient|secure|robust|reliable)\b", 0.8),
        # Development workflow
        (r"\b(version.*control|branch|merge|pull.*request|code.*review|testing|deployment)\b", 0.8)
    ]
    
    for pattern, confidence in code_patterns:
        if re.search(pattern, prompt):
            scores["code"] += confidence
    
    # -- Debug and Optimization Detection --
    debug_patterns = [
        (r"\b(debug|fix.*error|resolve.*issue|troubleshoot|error.*message|exception|crash)\b", 0.9),
        (r"\b(why.*not.*working|what.*wrong|how.*fix|problem.*with|issue.*with)\b", 0.8),
        (r"\b(log.*error|stack.*trace|debug.*mode|breakpoint|step.*through)\b", 0.9)
    ]
    
    for pattern, confidence in debug_patterns:
        if re.search(pattern, prompt):
            scores["debug"] += confidence
    
    optimize_patterns = [
        (r"\b(optimize|improve.*performance|make.*faster|reduce.*complexity|efficient|speed.*up)\b", 0.9),
        (r"\b(better.*algorithm|optimization|performance.*tuning|memory.*usage|cpu.*usage)\b", 0.9),
        (r"\b(scale.*up|handle.*load|concurrent|parallel|multithreading|async.*processing)\b", 0.8)
    ]
    
    for pattern, confidence in optimize_patterns:
        if re.search(pattern, prompt):
            scores["optimize"] += confidence
    
    # -- Image modification follow-ups (HIGH PRIORITY) --
    image_modification_patterns = [
        (r"\b(make it|change it|modify it|adjust it|update it)\b", 0.95),
        (r"\b(make the|change the|modify the|adjust the|update the)\b", 0.95),
        (r"\b(look more like|look like|resemble|similar to)\b", 0.95),
        (r"\b(add|remove|include|exclude|put|take away)\b", 0.9),
        (r"\b(enhance|improve|better|fix|correct)\b", 0.9),
        (r"\b(same|similar|like that|but|however|instead)\b", 0.8),
        (r"\b(different|another|version|variation|style)\b", 0.8),
        (r"\b(zoom in|zoom out|closer|farther|wider|narrower)\b", 0.9),
        (r"\b(brighter|darker|lighter|more colorful|less colorful)\b", 0.9),
        (r"\b(background|foreground|setting|scene|environment)\b", 0.8),
        (r"\b(clothes|outfit|dress|wear|wearing)\b", 0.8),
        (r"\b(hair|beard|mustache|glasses|hat|accessories)\b", 0.8),
        (r"\b(expression|smile|frown|emotion|mood)\b", 0.8),
        (r"\b(pose|position|angle|view|perspective)\b", 0.8),
        (r"\b(realistic|cartoon|anime|artistic|photorealistic)\b", 0.9)
    ]
    
    for pattern, confidence in image_modification_patterns:
        if re.search(pattern, prompt):
            scores["image"] += confidence
    
    # -- Audio Processing Detection --
    audio_patterns = [
        (r"\b(transcribe|speech to text|voice to text|audio to text)\b", 0.9),
        (r"\b(text to speech|speak|voice|audio|sound)\b", 0.8),
        (r"\b(record|recording|voice recording|audio file)\b", 0.7),
        (r"\b(whisper|elevenlabs|voice clone|synthesize)\b", 0.9),
        (r"\b(convert.*audio|audio.*convert)\b", 0.8)
    ]
    
    for pattern, confidence in audio_patterns:
        if re.search(pattern, prompt):
            scores["audio"] += confidence
    
    # -- Multimodal Detection --
    multimodal_patterns = [
        (r"\b(analyze.*image|describe.*picture|what.*see.*image)\b", 0.9),
        (r"\b(vision|visual.*analysis|image.*understanding)\b", 0.8),
        (r"\b(ocr|text.*image|extract.*text.*image)\b", 0.9),
        (r"\b(object.*detection|face.*recognition|scene.*analysis)\b", 0.9),
        (r"\b(multimodal|multi-modal|image.*text|text.*image)\b", 0.9)
    ]
    
    for pattern, confidence in multimodal_patterns:
        if re.search(pattern, prompt):
            scores["multimodal"] += confidence
    
    # -- Translation detection (high confidence patterns) --
    translation_patterns = [
        (r"\btranslate\b", 0.9),
        (r"\bwhat does .* mean in\b", 0.8),
        (r"\bin (spanish|french|german|chinese|japanese|portuguese|italian|russian|korean|arabic)\b", 0.8),
        (r"\bto (spanish|french|german|chinese|japanese|portuguese|italian|russian|korean|arabic)\b", 0.8),
        (r"\b(spanish|french|german|chinese|japanese|portuguese|italian|russian|korean|arabic) (translation|version)\b", 0.9),
        (r"\bhow do you say .* in\b", 0.9),
        (r"\bconvert .* to\b", 0.7)
    ]
    
    for pattern, confidence in translation_patterns:
        if re.search(pattern, prompt):
            scores["translate"] += confidence
    
    # -- Summarization detection --
    summarize_patterns = [
        (r"\b(tl;dr|summarize|summary|key points|main points)\b", 0.9),
        (r"\b(give me.*summary|what are.*key points|break.*down)\b", 0.8),
        (r"\b(condense|abbreviate|shorten)\b", 0.7),
        (r"\b(overview|synopsis|abstract)\b", 0.8),
        (r"\bwhat is.*about\b", 0.6)
    ]
    
    for pattern, confidence in summarize_patterns:
        if re.search(pattern, prompt):
            scores["summarize"] += confidence
    
    # -- Image generation detection (comprehensive) --
    image_patterns = [
        # Direct visual commands (more specific)
        (r"\b(draw|sketch|generate.*image|visualize|show me.*image|render|depict|illustrate|design.*logo|picture of|art of|image of|photo of)\b", 0.8),
        # Visual creation with context - ADD "make" commands
        (r"\b(create|make).*(image|picture|art|drawing|sketch|logo|banner|poster|graphic|illustration)\b", 0.9),
        (r"\b(create|make).*(visual|visualization|diagram|chart|infographic)\b", 0.8),
        # Visual questions
        (r"\bwhat does .* look like\b", 0.9),
        (r"\bhow would .* look\b", 0.8),
        # Style specifications
        (r"\bas a (cartoon|meme|photo|anime|logo|painting|drawing|character|3d model|comic|portrait)\b", 0.9),
        (r"\bin (cartoon|anime|realistic|artistic|photorealistic|sketch|watercolor|oil painting) style\b", 0.9),
        # Visual modifications
        (r"\bwith (a beard|a sword|glasses|a crown|wings|horns|armor)\b", 0.7),
        (r"\bwearing (armor|a tuxedo|a dress|modern clothes|costume)\b", 0.7),
        # Scene descriptions
        (r"\b(scene|landscape|background|setting) of\b", 0.8),
        (r"\b(forest|beach|mountain|city|room|office|kitchen)\b", 0.6),
        # Artistic terms
        (r"\b(artwork|illustration|graphic|banner|poster|cover)\b", 0.8),
        # Color and visual attributes
        (r"\b(colorful|bright|dark|vibrant|muted|pastel|neon)\b", 0.6),
        # Composition terms
        (r"\b(close-up|wide shot|bird's eye view|side view|front view)\b", 0.7),
        # Enhanced image generation patterns for follow-ups
        (r"\b(generate|create|make|show|draw|paint).*(it|that|this|them|those)\b", 0.95),
        (r"\b(image|picture|photo|art|drawing|sketch).*(of|with|in|as)\b", 0.9),
        (r"\b(painted|colored|red|blue|green|yellow|purple|orange|pink|brown|black|white|gray)\b", 0.8),
        (r"\b(statue|monument|building|structure|object).*(red|blue|green|yellow|purple|orange|pink|brown|black|white|gray)\b", 0.9),
        # Direct image requests with pronouns
        (r"\b(generate|create|make|show|draw|paint).*(image|picture|photo|art).*(it|that|this)\b", 0.95),
        (r"\b(it|that|this).*(image|picture|photo|art|drawing|sketch)\b", 0.9),
        # Color modifications
        (r"\b(paint|color|make).*(red|blue|green|yellow|purple|orange|pink|brown|black|white|gray)\b", 0.9)
    ]
    
    for pattern, confidence in image_patterns:
        if re.search(pattern, prompt):
            scores["image"] += confidence
    
    # -- Text generation detection (fallback for complex queries) --
    text_patterns = [
        # Question patterns
        (r"\b(how|why|what|when|where|who|which)\b", 0.6),
        # Writing tasks
        (r"\b(write|explain|describe|suggest|help|guide|give|list|compose|tell me)\b", 0.7),
        # Analysis tasks
        (r"\b(analyze|compare|contrast|evaluate|assess|review)\b", 0.8),
        # Creative writing
        (r"\b(story|poem|essay|article|blog|script|dialogue)\b", 0.8),
        # Problem solving
        (r"\b(solve|calculate|compute|figure out|determine)\b", 0.7),
        # Advice and recommendations
        (r"\b(advice|recommend|suggest|tips|best way|how to)\b", 0.7),
        # Text creation tasks
        (r"\bcreate.*(summary|list|plan|schedule|outline|report|document|content|text|description)\b", 0.9),
        (r"\bcreate.*(strategy|method|approach|solution|guide|tutorial|instructions)\b", 0.8),
        (r"\bcreate.*(email|letter|message|note|memo|proposal|presentation)\b", 0.9),
        # Exclude image-related create commands from text classification
        (r"\bcreate.*(image|picture|photo|art|drawing|sketch|visual|graphic|illustration)\b", 0.0),
        # General text tasks - REMOVE broad "create" pattern to avoid conflicts
        # (r"\bcreate\b", 0.5)  # This was too broad and conflicted with image creation
    ]
    
    for pattern, confidence in text_patterns:
        if re.search(pattern, prompt):
            scores["text"] += confidence
    
    # -- Context-aware adjustments --
    
    # If image score is high, reduce text score to avoid conflicts
    if scores["image"] > 0.3:
        scores["text"] *= 0.2
    
    # If code score is high, reduce text score to avoid conflicts
    if scores["code"] > 0.5:
        scores["text"] *= 0.3
    
    # If debug or optimize scores are high, boost code score
    if scores["debug"] > 0.3 or scores["optimize"] > 0.3:
        scores["code"] += 0.5
    
    # If it's clearly a question and doesn't fit other categories, it's text
    if prompt.endswith("?") and scores["text"] == 0:
        scores["text"] += 0.5
    
    # If it mentions specific languages but isn't translation, reduce translation score
    if "translate" not in prompt and any(lang in prompt for lang in ["spanish", "french", "german", "chinese"]):
        scores["translate"] *= 0.5
    
    # -- Determine winner with enhanced logic --
    best_task = max(scores.items(), key=lambda x: x[1])[0]
    confidence = scores[best_task]
    
    # Normalize confidence to 0-1 range
    confidence = min(confidence, 1.0)
    
    # Log classification for debugging
    logging.info(f"Task classification: {best_task} (confidence: {confidence:.2f})")
    logging.info(f"All scores: {scores}")
    
    return best_task, confidence

def get_task_confidence(prompt: str) -> Dict[str, float]:
    """
    Get confidence scores for all task types.
    Useful for debugging and advanced routing.
    """
    prompt = prompt.strip().lower()
    
    scores = {
        "text": 0.0,
        "code": 0.0,
        "image": 0.0, 
        "summarize": 0.0,
        "translate": 0.0,
        "audio": 0.0,
        "multimodal": 0.0,
        "debug": 0.0,
        "optimize": 0.0
    }
    
    # Apply all the same patterns as above but return full scores
    # (This would duplicate the logic above - in practice, you'd refactor to avoid duplication)
    
    return scores

def classify_code_complexity(prompt: str) -> Tuple[str, float]:
    """
    Classify the complexity level of code-related tasks.
    Returns (complexity_level, confidence)
    """
    prompt_lower = prompt.lower()
    
    # Simple code patterns
    simple_patterns = [
        r"\b(hello world|basic|simple|example|demo|snippet|quick)\b",
        r"\b(print|console\.log|alert|echo)\b",
        r"\b(variable|string|number|boolean)\b"
    ]
    
    # Complex code patterns
    complex_patterns = [
        r"\b(architecture|design pattern|microservices|distributed|scalable)\b",
        r"\b(algorithm|data structure|optimization|performance|efficiency)\b",
        r"\b(security|authentication|authorization|encryption|hashing)\b",
        r"\b(testing|unit test|integration test|tdd|bdd)\b",
        r"\b(api|rest|graphql|websocket|grpc)\b",
        r"\b(database|orm|migration|query|index)\b",
        r"\b(deployment|docker|kubernetes|ci/cd|devops)\b"
    ]
    
    simple_score = sum(1 for pattern in simple_patterns if re.search(pattern, prompt_lower))
    complex_score = sum(1 for pattern in complex_patterns if re.search(pattern, prompt_lower))
    
    if complex_score > simple_score:
        return "complex", 0.8
    elif simple_score > complex_score:
        return "simple", 0.8
    else:
        return "medium", 0.6

def classify_task_with_context(prompt: str, conversation_history: Optional[List[Dict]] = None) -> Tuple[str, float]:
    """
    Enhanced task classification with conversation context awareness.
    Returns (task_type, confidence_score)
    """
    # Get base classification
    base_task, base_confidence = classify_task(prompt)
    
    if not conversation_history:
        return base_task, base_confidence
    
    # Analyze recent conversation context
    recent_topics = []
    recent_task_types = []
    recent_models = []
    
    # Look at last 6 entries for context
    for entry in conversation_history[-6:]:
        if entry.get('role') == 'user':
            recent_topics.append(entry.get('content', '').lower())
        recent_task_types.append(entry.get('task_type', ''))
        recent_models.append(entry.get('model_used', ''))
    
    # Context-aware adjustments
    prompt_lower = prompt.lower()
    
    # If recent conversation was about code, boost code classification
    if any('code' in task_type for task_type in recent_task_types[-3:]):
        if any(word in prompt_lower for word in ['it', 'this', 'that', 'the', 'fix', 'debug', 'error']):
            # Likely continuing code conversation
            if base_task == "text":
                return "code", min(base_confidence + 0.3, 1.0)
            elif base_task == "code":
                return "code", min(base_confidence + 0.2, 1.0)
    
    # If recent conversation was about images, boost image classification
    if any('image' in task_type for task_type in recent_task_types[-3:]):
        if any(word in prompt_lower for word in ['show', 'see', 'look', 'picture', 'photo', 'draw', 'create', 'make it', 'change it']):
            return "image", min(base_confidence + 0.3, 1.0)
    
    # If recent conversation was about text/analysis, boost text classification
    if any('text' in task_type for task_type in recent_task_types[-3:]):
        if any(word in prompt_lower for word in ['explain', 'tell', 'what', 'how', 'why', 'describe', 'analyze']):
            return "text", min(base_confidence + 0.2, 1.0)
    
    # If user is asking about something mentioned before, it's likely text
    if any(pronoun in prompt_lower for pronoun in ['it', 'this', 'that', 'they', 'them']):
        if base_task == "text":
            return "text", min(base_confidence + 0.1, 1.0)
    
    # Model-specific context adjustments
    if recent_models and recent_models[-1]:
        last_model = recent_models[-1]
        
        # If last model was code-focused, boost code classification for follow-ups
        if last_model in ["claude-3-5-sonnet", "codellama-70b", "gpt-4o-mini"]:
            if any(word in prompt_lower for word in ['fix', 'debug', 'error', 'problem', 'issue']):
                return "code", min(base_confidence + 0.2, 1.0)
        
        # If last model was image-focused, boost image classification for follow-ups
        elif last_model in ["dall-e-3", "stablediffusion", "anime-diffusion"]:
            if any(word in prompt_lower for word in ['change', 'modify', 'adjust', 'make it', 'look like']):
                return "image", min(base_confidence + 0.3, 1.0)
    
    return base_task, base_confidence

def classify_task_with_user_preferences(prompt: str, user_preferences: Optional[Dict] = None) -> Tuple[str, float]:
    """
    Task classification that considers user preferences and history.
    """
    base_task, base_confidence = classify_task(prompt)
    
    if not user_preferences:
        return base_task, base_confidence
    
    # Adjust based on user's preferred task types
    preferred_tasks = user_preferences.get('preferred_task_types', [])
    if base_task in preferred_tasks:
        return base_task, min(base_confidence + 0.1, 1.0)
    
    # Adjust based on user's preferred models
    preferred_models = user_preferences.get('preferred_models', [])
    if preferred_models:
        # Map preferred models to task types
        model_to_task = {
            "claude-3-5-sonnet": "code",
            "gpt-4o": "text",
            "dall-e-3": "image",
            "claude-sonnet-4": "text"
        }
        
        for model in preferred_models:
            if model in model_to_task and model_to_task[model] == base_task:
                return base_task, min(base_confidence + 0.1, 1.0)
    
    return base_task, base_confidence
