from openai import OpenAI
import requests
import anthropic
from datetime import datetime
from memory import conversation_memory
import google.generativeai as genai  # type: ignore
import os
import logging
from typing import cast
from openai.types.chat import ChatCompletionUserMessageParam

NUCLEAR_DEBUG = True

# Add at the top, after imports
ALWAYS_USE_CLAUDE = True  # Set to False to allow fallback to GPT-4

# Model name mapping for Anthropic/Claude
MODEL_NAME_MAP = {
    'claude-sonnet-4': 'claude-3-haiku-20240307',
    'claude-haiku': 'claude-3-haiku-20240307',
    # Add more mappings as needed
}

client = OpenAI()
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Initialize Google Gemini
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # type: ignore
    gemini_model = genai.GenerativeModel('gemini-pro')  # type: ignore
except Exception as e:
    print(f"Warning: Google GenerativeAI not properly configured: {e}")
    gemini_model = None

def log_interaction(entry):
    entry["timestamp"] = datetime.utcnow().isoformat()

    # Universal log (all tasks)
    with open("usage_logs.jsonl", "a") as f:
        f.write(f"{entry}\n")

    # Task-specific memory
    task = entry.get("task_type")
    if task:
        task_log_file = f"memory_{task}.jsonl"
        with open(task_log_file, "a") as f:
            f.write(f"{entry}\n")

def run_model(task_type, model, prompt, user_id="default", user_lang="en"):
    print(f"🔍 Optimized prompt for {model}: {prompt}")

    original_prompt = prompt
    image_intent = None

    try:
        if task_type == "image":
            from memory import memory_manager
            conversation_context = memory_manager.get_recent_memory(user_id, 6)
            print(f"DEBUG: [run_model] Conversation context for image: {conversation_context}")
            if model == "dall-e-3":
                output_url = run_dalle(prompt, user_id, conversation_context)
            elif model == "stablediffusion":
                output_url = run_stable_diffusion(prompt)
            else:
                output_url = run_dalle(prompt, user_id, conversation_context)
            from models import llm_based_coreference_resolution
            summary_prompt = f"Summarize this image prompt in a short, clear phrase: '{prompt}'"
            try:
                summary = llm_based_coreference_resolution(summary_prompt, prompt)
            except Exception:
                summary = prompt[:80]
            memory_manager.save_memory(
                user_id,
                role="user",
                content=prompt,
                task_type="image",
                model=model,
                metadata={"summary": summary, "url": output_url}
            )
            # UNIVERSAL POST-PROCESSING: Strip data URL prefix if present
            if isinstance(output_url, str) and output_url.startswith("data:image/png;base64,"):
                print("DEBUG: Stripping data URL prefix from image output before returning to frontend.")
                output_url = output_url[len("data:image/png;base64,"):]
            print(f"DEBUG: FINAL image output returned to frontend: {str(output_url)[:80]}...")
            return output_url, None
        elif model == "gpt-4o":
            output = run_gpt(prompt, user_id)
        elif model == "gpt-4-turbo":
            output = run_gpt_turbo(prompt, user_id)
        elif model == "claude-sonnet-4":
            output = run_claude(prompt, user_id)
        elif model == "claude-haiku":
            output = run_claude_haiku(prompt, user_id)
        elif model == "gemini-pro":
            output = run_gemini(prompt)
        elif model == "deepl":
            output = run_deepl(prompt, user_lang)
        elif model == "dall-e-3":
            from memory import memory_manager
            conversation_context = memory_manager.get_recent_memory(user_id, 6)
            print(f"DEBUG: [run_model] Conversation context for image: {conversation_context}")
            output = run_dalle(prompt, user_id, conversation_context)
        elif model == "stablediffusion":
            output = run_stable_diffusion(prompt)
        elif model == "midjourney":
            output = run_midjourney(prompt)
        elif model == "anime-diffusion":
            output = run_anime_diffusion(prompt)
        elif model == "whisper":
            output = run_whisper(prompt)
        elif model == "elevenlabs":
            output = run_elevenlabs(prompt)
        elif model == "claude-3-5-sonnet":
            output = run_claude_3_5_sonnet(prompt, user_id)
        elif model == "gpt-4o-mini":
            output = run_gpt_4o_mini(prompt, user_id)
        elif model == "codellama-70b":
            output = run_codellama_70b(prompt)
        elif model == "wizardcoder":
            output = run_wizardcoder(prompt)
        elif model == "phind-codellama":
            output = run_phind_codellama(prompt)
        else:
            raise Exception("Model not implemented")

        log_interaction({
            "task_type": task_type,
            "model_used": model,
            "user_input": original_prompt,
            "optimized_prompt": prompt,
            "success": True,
            "fallback_used": None,
            "image_intent": image_intent
        })

        return output, None

    except Exception as e:
        # For image tasks, don't use fallback - just return the error
        if task_type == "image":
            log_interaction({
                "task_type": task_type,
                "model_used": model,
                "user_input": original_prompt,
                "optimized_prompt": prompt,
                "success": False,
                "fallback_used": None,
                "image_intent": image_intent
            })
            return f"❌ Image generation failed: {str(e)}", None
        
        # For non-image tasks, use fallback logic
        from models import MODEL_REGISTRY
        # Get fallback model from registry
        fallback = None
        if ALWAYS_USE_CLAUDE and model.startswith("claude"):
            print(f"DEBUG: ALWAYS_USE_CLAUDE is True. Not falling back from {model}.")
            log_interaction({
                "task_type": task_type,
                "model_used": model,
                "user_input": original_prompt,
                "optimized_prompt": prompt,
                "success": False,
                "fallback_used": None,
                "image_intent": image_intent
            })
            return f"❌ Model {model} failed. Error: {str(e)}", None
        for model_name, model_info in MODEL_REGISTRY.items():
            if model_info.get("provider") != "openai":  # Use non-OpenAI as fallback
                fallback = model_name
                break

        if fallback and fallback != model:
            try:
                fallback_prompt = original_prompt  # Use original prompt as fallback
                output = run_model(task_type, fallback, fallback_prompt, user_id)[0]

                log_interaction({
                    "task_type": task_type,
                    "model_used": fallback,
                    "user_input": original_prompt,
                    "optimized_prompt": fallback_prompt,
                    "success": True,
                    "fallback_used": model,
                    "image_intent": image_intent
                })

                return output, f"⚠️ Primary model ({model}) failed. Used fallback: {fallback}"

            except Exception as e2:
                log_interaction({
                    "task_type": task_type,
                    "model_used": model,
                    "user_input": original_prompt,
                    "optimized_prompt": prompt,
                    "success": False,
                    "fallback_used": fallback,
                    "image_intent": image_intent
                })
                return f"❌ All models failed. Error: {str(e2)}", None

        log_interaction({
            "task_type": task_type,
            "model_used": model,
            "user_input": original_prompt,
            "optimized_prompt": prompt,
            "success": False,
            "fallback_used": None,
            "image_intent": image_intent
        })

        return f"❌ Model {model} failed. Error: {str(e)}", None

def run_gpt(prompt, user_id="default"):
    print("=== RUN_GPT FUNCTION CALLED ===")
    logging.info("=== RUN_GPT FUNCTION CALLED ===")
    
    # Write to a separate debug file
    with open("debug.log", "a") as f:
        f.write(f"[{datetime.now()}] RUN_GPT CALLED - prompt: {prompt}\n")
    
    from memory import memory_manager
    
    # Get the last user question and assistant response
    recent_memory = memory_manager.get_recent_memory(user_id, 4)
    last_user_question = None
    last_assistant_response = None
    for entry in reversed(recent_memory):
        if entry.get('role') == 'assistant' and not last_assistant_response:
            last_assistant_response = entry.get('content', '')
        if entry.get('role') == 'user' and not last_user_question:
            last_user_question = entry.get('content', '')
        if last_user_question and last_assistant_response:
            break
    
    followup_indicators = ["it", "that", "this", "they", "them", "those", "these", "he", "she", "his", "her", "their", "its", "there"]
    is_followup = any(indicator in prompt.lower() for indicator in followup_indicators)
    
    logging.info("\n=== DEBUG INFO ===")
    logging.info(f"Original prompt: {prompt}")
    logging.info(f"Is follow-up: {is_followup}")
    logging.info(f"Last user question: {last_user_question}")
    logging.info(f"Last assistant response: {last_assistant_response}")
    
    if is_followup and last_user_question and last_assistant_response:
        enhanced_prompt = f"""Previous conversation:\nUser: {last_user_question}\nAssistant: {last_assistant_response}\n\nCurrent question: {prompt}\n\nPlease answer the current question using the context from the previous conversation. When the user says \"there\", \"it\", \"that\", etc., refer to the subject from the previous conversation."""
        logging.info(f"Enhanced prompt: {enhanced_prompt}")
    else:
        enhanced_prompt = prompt
        logging.info(f"Using original prompt (no enhancement)")
    
    logging.info(f"=== END DEBUG ===\n")
    
    messages = memory_manager.get_memory_for_model(user_id, "gpt-4o", max_tokens=4000)
    # Set ChatGPT-style system prompt
    system_prompt = {"role": "system", "content": "You are ChatGPT, a helpful assistant. If the user asks for a prediction or speculative answer, you may provide a plausible, hypothetical range or scenario, but always include a disclaimer that this is not financial advice."}
    messages = [system_prompt] + messages
    messages.append({"role": "user", "content": enhanced_prompt})
    
    logging.info("==== SENDING TO MODEL ====")
    for m in messages:
        logging.info(m)
    logging.info("========================\n")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages  # type: ignore
    )
    reply = getattr(response.choices[0].message, "content", "")
    if reply is None:
        reply = ""
    elif not isinstance(reply, str):
        reply = str(reply)
    memory_manager.save_memory(user_id, "assistant", cast(str, reply), "text", "gpt-4o")  # type: ignore
    return reply

def run_gpt_turbo(prompt, user_id="default"):
    from memory import memory_manager
    
    # Enhanced prompt engineering for text generation
    enhanced_prompt = apply_text_prompt_engineering(prompt, user_id)
    
    # Get context for follow-up questions
    context = memory_manager.get_context_for_followup(user_id, prompt)
    
    # Check if this looks like a follow-up question using improved detection
    is_followup = memory_manager.is_followup_question(prompt, user_id)
    
    if is_followup and context:
        # For follow-up questions, provide explicit context with enhanced prompting
        enhanced_context = f"{context}\n\nCurrent question: {enhanced_prompt}\n\nPlease provide a comprehensive answer using the context provided. Be specific and detailed in your response."
        messages = [
            {"role": "system", "content": get_enhanced_system_prompt()},
            {"role": "user", "content": enhanced_context}
        ]
    else:
        # For new questions, use standard memory approach with enhanced prompting
        messages = memory_manager.get_memory_for_model(user_id, "gpt-4-turbo", max_tokens=4000)
        system_prompt = {"role": "system", "content": get_enhanced_system_prompt()}
        messages = [system_prompt] + messages
        messages.append({"role": "user", "content": enhanced_prompt})
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,  # type: ignore
    )
    reply = getattr(response.choices[0].message, "content", "")
    if reply is None:
        reply = ""
    elif not isinstance(reply, str):
        reply = str(reply)
    memory_manager.save_memory(user_id, "assistant", cast(str, reply), "text", "gpt-4-turbo")  # type: ignore
    return reply

def apply_text_prompt_engineering(prompt, user_id):
    """
    Intelligently integrated prompt engineering that can combine role assignment 
    and enhancements when both would benefit the output, avoiding redundancy.
    """
    from prompt_optimizer import IntelligentPromptOptimizer, OptimizationContext
    from memory import memory_manager
    
    prompt_lower = prompt.lower().strip()
    
    # Smart Enhancement Detection - Simple interactions that need no enhancement
    simple_patterns = [
        'hello', 'hi', 'hey', 'how are you', 'good morning', 'good afternoon', 'good evening',
        'thanks', 'thank you', 'bye', 'goodbye', 'see you', 'ok', 'okay', 'yes', 'no',
        'test', 'testing', 'test message'
    ]
    
    # Check if this is a simple interaction that doesn't need any enhancement
    if any(pattern in prompt_lower for pattern in simple_patterns):
        return prompt  # Return original prompt unchanged
    
    # Get conversation context for role assignment
    conversation_history = memory_manager.get_recent_memory(user_id)
    
    # Create optimization context for the existing role assignment logic
    context = OptimizationContext(
        task_type="text",
        model="gpt-4-turbo",
        original_prompt=prompt,
        conversation_history=conversation_history
    )
    
    # Use the existing intelligent prompt optimizer
    optimizer = IntelligentPromptOptimizer()
    
    # Apply the existing role assignment and optimization logic
    optimized_prompt = optimizer.optimize_prompt(context)
    
    # Analyze what the optimizer did
    has_role_assignment = optimized_prompt.startswith("You are a ")
    is_enhanced = optimized_prompt != prompt
    
    # Intent-Based Enhancement Detection
    technical_indicators = [
        'explain', 'analyze', 'compare', 'discuss', 'evaluate', 'assess', 'examine',
        'research', 'investigate', 'explore', 'delve into', 'break down', 'outline',
        'comprehensive', 'detailed', 'thorough', 'in-depth', 'extensive',
        'code', 'programming', 'algorithm', 'function', 'class', 'method',
        'database', 'api', 'framework', 'library', 'syntax', 'debug',
        'architecture', 'design pattern', 'optimization', 'performance'
    ]
    
    creative_indicators = [
        'write', 'create', 'design', 'compose', 'generate', 'develop',
        'imagine', 'brainstorm', 'conceptualize', 'craft', 'build',
        'story', 'poem', 'lyrics', 'fiction', 'creative', 'artistic'
    ]
    
    # Detect intent for smart enhancement
    is_technical = any(indicator in prompt_lower for indicator in technical_indicators)
    is_creative = any(indicator in prompt_lower for indicator in creative_indicators)
    is_complex_analysis = any(word in prompt_lower for word in ['analyze', 'comprehensive', 'detailed', 'thorough'])
    
    # Intelligent Combination Strategy:
    # 1. Check if role assignment is appropriate and beneficial
    # 2. Check if enhancement would be beneficial
    # 3. Combine intelligently, avoiding redundancy
    
    # Determine if role assignment is appropriate
    role_appropriate_patterns = [
        'write a poem', 'write lyrics', 'write a story', 'write fiction',
        'design', 'create', 'compose', 'generate',
        'debug', 'review code', 'optimize code',
        'analyze', 'research', 'investigate'
    ]
    
    role_is_appropriate = any(pattern in prompt_lower for pattern in role_appropriate_patterns)
    
    # Determine if enhancement would be beneficial
    enhancement_needed = is_technical or is_creative or is_complex_analysis
    
    # Check for redundancy - if role already provides the enhancement we want
    role_provides_enhancement = False
    if has_role_assignment:
        # More sophisticated redundancy detection
        role_lower = optimized_prompt.lower()
        
        # Check for creative redundancy
        if is_creative:
            creative_role_indicators = [
                'creative writer', 'author', 'poet', 'novelist', 'storyteller', 'creative',
                'artistic', 'imaginative', 'creative writing', 'fiction writer', 'lyricist',
                'playwright', 'screenwriter', 'content creator', 'creative professional'
            ]
            if any(indicator in role_lower for indicator in creative_role_indicators):
                role_provides_enhancement = True
        
        # Check for technical redundancy
        elif is_technical:
            technical_role_indicators = [
                'technical expert', 'engineer', 'developer', 'programmer', 'scientist',
                'analyst', 'consultant', 'specialist', 'expert', 'professional',
                'technical writer', 'instructor', 'teacher', 'educator', 'researcher'
            ]
            if any(indicator in role_lower for indicator in technical_role_indicators):
                role_provides_enhancement = True
        
        # Check for comprehensive analysis redundancy
        elif is_complex_analysis:
            analysis_role_indicators = [
                'analyst', 'researcher', 'consultant', 'expert', 'specialist',
                'comprehensive', 'detailed', 'thorough', 'in-depth', 'extensive',
                'professional', 'senior', 'lead', 'principal', 'chief'
            ]
            if any(indicator in role_lower for indicator in analysis_role_indicators):
                role_provides_enhancement = True
        
        # Check for specific instruction overlap
        if not role_provides_enhancement:
            # Check if role already contains similar instructions
            if is_creative and any(phrase in role_lower for phrase in ['creative', 'original', 'engaging', 'imaginative']):
                role_provides_enhancement = True
            elif is_technical and any(phrase in role_lower for phrase in ['clear', 'practical', 'explanation', 'examples']):
                role_provides_enhancement = True
            elif is_complex_analysis and any(phrase in role_lower for phrase in ['comprehensive', 'detailed', 'structured', 'thorough']):
                role_provides_enhancement = True
    
    # Intelligent Decision Making
    if has_role_assignment and role_is_appropriate:
        if enhancement_needed and not role_provides_enhancement:
            # Role is good, but enhancement would add value without redundancy
            if is_technical:
                return f"{optimized_prompt}\n\nPlease provide a clear, practical explanation with examples if helpful."
            elif is_creative:
                return f"{optimized_prompt}\n\nPlease be creative and provide original, engaging content."
            elif is_complex_analysis:
                return f"{optimized_prompt}\n\nPlease provide a comprehensive and well-structured response."
            else:
                return optimized_prompt  # Role is sufficient
        else:
            # Role assignment is sufficient (either no enhancement needed or role already provides it)
            return optimized_prompt
    
    elif has_role_assignment and not role_is_appropriate:
        # Role was assigned but inappropriate - remove it and apply enhancement if needed
        if enhancement_needed:
            if is_technical:
                return f"{prompt}\n\nPlease provide a clear, practical explanation with examples if helpful."
            elif is_creative:
                return f"{prompt}\n\nPlease be creative and provide original, engaging content."
            elif is_complex_analysis:
                return f"{prompt}\n\nPlease provide a comprehensive and well-structured response."
            else:
                return prompt  # Keep simple questions simple
        else:
            return prompt  # No enhancement needed
    
    else:
        # No role assignment - apply enhancement if beneficial
        if enhancement_needed:
            if is_technical:
                return f"{prompt}\n\nPlease provide a clear, practical explanation with examples if helpful."
            elif is_creative:
                return f"{prompt}\n\nPlease be creative and provide original, engaging content."
            elif is_complex_analysis:
                return f"{prompt}\n\nPlease provide a comprehensive and well-structured response."
            else:
                return prompt  # Keep simple questions simple
        else:
            return prompt  # No enhancement needed

def get_enhanced_system_prompt():
    """
    Get a balanced system prompt that provides guidance without being overly prescriptive.
    """
    return """You are a helpful AI assistant. Be accurate, clear, and genuinely helpful. 

- Provide accurate, up-to-date information
- Use clear, accessible language
- Be conversational but professional
- Consider context from the conversation
- If unsure about something, acknowledge it
- For technical topics, provide practical explanations
- For creative requests, be imaginative and engaging

Your goal is to be genuinely helpful while maintaining accuracy and safety."""

def apply_image_prompt_engineering(prompt, conversation_context=None):
    """
    Apply comprehensive prompt engineering best practices for image generation.
    Combines context enhancement with industry best practices.
    """
    # Step 1: Context enhancement
    enhanced_prompt = prompt
    if conversation_context:
        context_info = extract_context_for_image_generation(conversation_context, prompt)
        if context_info:
            enhanced_prompt = context_info
            print(f"DEBUG: Context-enhanced prompt: {enhanced_prompt}")
    
    # Step 2: Apply DALL-E specific best practices
    enhanced_prompt = apply_dalle_best_practices(enhanced_prompt)
    
    # Step 3: Final optimization
    enhanced_prompt = optimize_for_dalle(enhanced_prompt)
    
    return enhanced_prompt

def apply_dalle_best_practices(prompt):
    """
    Apply DALL-E specific best practices for optimal image generation.
    Based on OpenAI's DALL-E documentation and community best practices.
    """
    prompt_lower = prompt.lower()
    
    # DALL-E best practices
    best_practices = {
        'be_specific': 'Use specific, descriptive language',
        'avoid_ambiguity': 'Be clear about what you want',
        'include_style': 'Specify artistic style when relevant',
        'quality_indicators': 'Add quality and detail specifications',
        'composition_hints': 'Include composition and framing hints'
    }
    
    # Add quality indicators if not present
    quality_indicators = ['high quality', 'detailed', 'sharp', 'professional']
    if not any(indicator in prompt_lower for indicator in quality_indicators):
        prompt += ", high quality, detailed"
    
    # Add style guidance for better results
    style_enhancements = []
    
    # Determine if it's a realistic or artistic request
    if any(word in prompt_lower for word in ['realistic', 'photograph', 'photo', 'real']):
        style_enhancements.extend(['professional photography', 'sharp focus'])
    elif any(word in prompt_lower for word in ['artistic', 'painting', 'drawing', 'illustration']):
        style_enhancements.extend(['artistic style', 'creative composition'])
    else:
        # Default to high-quality realistic style
        style_enhancements.extend(['professional photography', 'sharp focus'])
    
    # Add lighting hints for better composition
    if 'portrait' in prompt_lower or 'person' in prompt_lower:
        style_enhancements.append('professional lighting')
    elif 'landscape' in prompt_lower or 'nature' in prompt_lower:
        style_enhancements.append('natural lighting')
    
    # Combine enhancements
    if style_enhancements:
        prompt += f", {', '.join(style_enhancements)}"
    
    return prompt

def optimize_for_dalle(prompt):
    """
    Final optimization for DALL-E, removing problematic elements and improving clarity.
    """
    # Remove problematic phrases that can confuse DALL-E
    problematic_phrases = [
        'generate an image of',
        'create an image of',
        'draw a picture of',
        'make an image of',
        'show me',
        'please create',
        'please generate'
    ]
    
    optimized = prompt
    for phrase in problematic_phrases:
        optimized = optimized.replace(phrase, '').strip()
    
    # Clean up extra spaces and punctuation
    optimized = ' '.join(optimized.split())
    
    # Ensure proper capitalization for better AI interpretation
    if optimized:
        optimized = optimized[0].upper() + optimized[1:]
    
    # Add final quality assurance
    if not optimized.endswith(('high quality', 'detailed', 'sharp')):
        optimized += ", high quality"
    
    return optimized

def run_claude(prompt, user_id="default", model_name="claude-sonnet-4"):
    from memory import memory_manager
    import anthropic
    # Get context for follow-up questions
    context = memory_manager.get_context_for_followup(user_id, prompt)
    is_followup = memory_manager.is_followup_question(prompt, user_id)
    if is_followup and context:
        enhanced_prompt = f"{context}\n\nPlease answer the current query using the context provided."
        messages = [{"role": "user", "content": enhanced_prompt}]
    else:
        messages = memory_manager.get_memory_for_model(user_id, model_name, max_tokens=4000)
        messages.append({"role": "user", "content": prompt})
    api_model = MODEL_NAME_MAP.get(model_name, model_name)
    try:
        response = claude_client.messages.create(
            model=api_model,
            max_tokens=1000,
            temperature=0.7,
            messages=messages  # type: ignore
        )  # type: ignore
        block = response.content[0]
        reply = getattr(block, "text", "")
        if not isinstance(reply, str):
            reply = str(reply)
        memory_manager.save_memory(user_id, "assistant", reply, "text", model_name)
        return reply
    except anthropic.APIStatusError as e:
        print(f"ERROR: Anthropic API error for model {api_model}: {e}")
        return f"❌ Claude model failed: {e}", None
    except Exception as e:
        print(f"ERROR: Claude model {api_model} failed: {e}")
        return f"❌ Claude model failed: {e}", None

def run_claude_haiku(prompt, user_id="default", model_name="claude-haiku"):
    from memory import memory_manager
    import anthropic
    context = memory_manager.get_context_for_followup(user_id, prompt)
    is_followup = memory_manager.is_followup_question(prompt, user_id)
    if is_followup and context:
        enhanced_prompt = f"{context}\n\nPlease answer the current query using the context provided."
        messages = [{"role": "user", "content": enhanced_prompt}]
    else:
        messages = memory_manager.get_memory_for_model(user_id, model_name, max_tokens=4000)
        messages.append({"role": "user", "content": prompt})
    api_model = MODEL_NAME_MAP.get(model_name, model_name)
    try:
        response = claude_client.messages.create(
            model=api_model,
            max_tokens=1000,
            temperature=0.7,
            messages=messages  # type: ignore
        )  # type: ignore
        block = response.content[0]
        reply = getattr(block, "text", "")
        if not isinstance(reply, str):
            reply = str(reply)
        memory_manager.save_memory(user_id, "assistant", reply, "text", model_name)
        return reply
    except anthropic.APIStatusError as e:
        print(f"ERROR: Anthropic API error for model {api_model}: {e}")
        return f"❌ Claude model failed: {e}", None
    except Exception as e:
        print(f"ERROR: Claude model {api_model} failed: {e}")
        return f"❌ Claude model failed: {e}", None

def run_gemini(prompt):
    if gemini_model:
        response = gemini_model.generate_content(prompt)
        return response.text
    else:
        return "Gemini model not available."

def extract_context_for_image_generation(conversation_context, current_prompt):
    print(f"DEBUG: extract_context_for_image_generation called with prompt: {current_prompt}")
    print(f"DEBUG: conversation_context length: {len(conversation_context) if conversation_context else 0}")
    if NUCLEAR_DEBUG:
        print(f"NUCLEAR_DEBUG: [extract_context_for_image_generation] Full context:")
        for i, entry in enumerate(conversation_context):
            print(f"  [{i}] role: {entry.get('role')}, content: {entry.get('content')}")
    if not conversation_context:
        print("DEBUG: No conversation context available")
        if NUCLEAR_DEBUG:
            print("NUCLEAR_DEBUG: [extract_context_for_image_generation] No context! Cannot merge.")
        return None
    context_info = extract_entities_from_context(conversation_context)
    modification_request = extract_modification_request(current_prompt)
    enhanced_prompt = apply_prompt_engineering_best_practices(
        context_info, 
        modification_request, 
        current_prompt
    )
    if enhanced_prompt:
        print(f"DEBUG: Enhanced prompt with best practices: {enhanced_prompt}")
        return enhanced_prompt
    vague_followup_indicators = [
        'it', 'what it would look like', 'show me', 'make it', 'paint it', 'what it looks like', 'what would it look like', 'what would this look like', 'what would that look like', 'what would they look like', 'what would these look like', 'what would those look like'
    ]
    prompt_lower = current_prompt.lower()
    if any(ind in prompt_lower for ind in vague_followup_indicators):
        last_assistant = None
        for entry in reversed(conversation_context):
            if entry.get('role') == 'assistant' and entry.get('content'):
                last_assistant = entry['content']
                break
        print(f"DEBUG: Last assistant answer found: {last_assistant}")
        if NUCLEAR_DEBUG and not last_assistant:
            print("NUCLEAR_DEBUG: [extract_context_for_image_generation] No last assistant answer found!")
        if last_assistant:
            mod = current_prompt.lower()
            for indicator in vague_followup_indicators:
                mod = mod.replace(indicator, '').strip()
            if not mod:
                mod = current_prompt.strip()
            merged_prompt = f"Show me what the {last_assistant} would look like {mod}."
            print(f"DEBUG: STRONG FALLBACK merged prompt: {merged_prompt}")
            return merged_prompt
    if NUCLEAR_DEBUG:
        print("NUCLEAR_DEBUG: [extract_context_for_image_generation] Merging failed! Returning None.")
    print("DEBUG: No context enhancement applied")
    return None

def extract_entities_from_context(conversation_context):
    """
    Extract specific entities and their attributes from conversation context.
    Uses advanced pattern matching and entity recognition.
    """
    entities = {
        'objects': [],
        'locations': [],
        'people': [],
        'attributes': [],
        'actions': []
    }
    
    # Enhanced entity detection patterns
    entity_patterns = {
        'statue': {
            'patterns': ['statue', 'monument', 'sculpture'],
            'specific': {
                'unity': 'Statue of Unity',
                'liberty': 'Statue of Liberty',
                'christ': 'Christ the Redeemer'
            }
        },
        'vehicle': {
            'patterns': ['car', 'vehicle', 'automobile', 'sports car', 'supercar'],
            'attributes': ['fastest', 'luxury', 'sports', 'electric', 'hybrid']
        },
        'building': {
            'patterns': ['building', 'tower', 'skyscraper', 'monument', 'landmark'],
            'specific': {
                'burj': 'Burj Khalifa',
                'eiffel': 'Eiffel Tower',
                'empire': 'Empire State Building'
            }
        },
        'person': {
            'patterns': ['person', 'man', 'woman', 'figure', 'portrait'],
            'attributes': ['famous', 'celebrity', 'president', 'leader']
        }
    }
    
    for entry in conversation_context[-8:]:  # Increased context window
        content = entry.get('content', '').lower()
        role = entry.get('role', '')
        
        # Extract entities using pattern matching
        for category, patterns in entity_patterns.items():
            for pattern in patterns['patterns']:
                if pattern in content:
                    # Check for specific named entities
                    if 'specific' in patterns:
                        for key, value in patterns['specific'].items():
                            if key in content:
                                entities['objects'].append(value)
                                break
                        else:
                            entities['objects'].append(pattern)
                    else:
                        entities['objects'].append(pattern)
                    
                    # Extract attributes
                    if 'attributes' in patterns:
                        for attr in patterns['attributes']:
                            if attr in content:
                                entities['attributes'].append(attr)
    
    return entities

def apply_prompt_engineering_best_practices(context_info, modification_request, original_prompt):
    """
    Apply prompt engineering best practices for image generation.
    Based on research and industry standards for effective prompting.
    """
    if not context_info['objects']:
        return None
    
    # Best practices for image generation prompts
    best_practices = {
        'structure': 'subject + modifier + style + quality',
        'specificity': 'be specific about visual elements',
        'style_guidance': 'include artistic style direction',
        'quality_indicators': 'add quality and detail specifications',
        'context_preservation': 'maintain original intent while enhancing clarity'
    }
    
    # Extract primary subject
    primary_subject = context_info['objects'][0] if context_info['objects'] else None
    
    if not primary_subject or not modification_request:
        return None
    
    # Build enhanced prompt following best practices
    enhanced_parts = []
    
    # 1. Subject specification (be specific)
    enhanced_parts.append(f"realistic image of {primary_subject}")
    
    # 2. Modification request (cleaned and enhanced)
    enhanced_parts.append(modification_request)
    
    # 3. Style and quality specifications
    style_enhancements = []
    
    # Add photographic quality indicators
    style_enhancements.append("high quality")
    style_enhancements.append("detailed")
    
    # Add lighting and composition hints
    style_enhancements.append("professional photography")
    
    # Add context-appropriate style
    if any(attr in context_info['attributes'] for attr in ['fastest', 'sports', 'luxury']):
        style_enhancements.append("dynamic composition")
    elif 'statue' in primary_subject.lower():
        style_enhancements.append("monumental scale")
    
    # Combine all parts
    enhanced_prompt = f"{' '.join(enhanced_parts)}, {', '.join(style_enhancements)}"
    
    # Clean up the prompt
    enhanced_prompt = clean_prompt_for_ai(enhanced_prompt)
    
    return enhanced_prompt

def clean_prompt_for_ai(prompt):
    """
    Clean and optimize prompt for AI image generation.
    Removes redundant words and improves clarity.
    """
    # Remove common redundant phrases
    redundant_phrases = [
        'show me what it looks like',
        'show me what',
        'what it looks like',
        'what it would look like',
        'generate an image of',
        'create an image of',
        'draw',
        'make',
        'picture of',
        'image of'
    ]
    
    cleaned = prompt.lower()
    for phrase in redundant_phrases:
        cleaned = cleaned.replace(phrase, '').strip()
    
    # Remove extra spaces and common filler words
    cleaned = ' '.join(cleaned.split())  # Normalize whitespace
    
    # Remove leading filler words
    filler_words = ['it', 'this', 'that', 'the', 'a', 'an', 'and']
    words = cleaned.split()
    while words and words[0] in filler_words:
        words = words[1:]
    
    result = ' '.join(words)
    
    # Capitalize first letter for better AI interpretation
    if result:
        result = result[0].upper() + result[1:]
    
    return result

def extract_modification_request(prompt):
    """
    Extract the actual modification request from a prompt, removing redundant phrases.
    """
    prompt_lower = prompt.lower()
    
    # Remove common redundant phrases
    redundant_phrases = [
        'show me what it looks like',
        'show me what',
        'what it looks like',
        'what it would look like',
        'generate an image of',
        'create an image of',
        'draw',
        'make'
    ]
    
    cleaned_prompt = prompt_lower
    for phrase in redundant_phrases:
        cleaned_prompt = cleaned_prompt.replace(phrase, '').strip()
    
    # Clean up extra spaces and common words
    cleaned_prompt = cleaned_prompt.replace('  ', ' ').strip()
    
    # Remove common filler words at the beginning
    filler_words = ['it', 'this', 'that', 'the', 'a', 'an']
    words = cleaned_prompt.split()
    if words and words[0] in filler_words:
        words = words[1:]
    
    result = ' '.join(words)
    print(f"DEBUG: Extracted modification request: '{result}' from '{prompt}'")
    return result

def run_dalle(prompt, user_id="default", conversation_context=None):
    print(f"DEBUG: run_dalle called with prompt: {prompt}")
    if NUCLEAR_DEBUG:
        print(f"NUCLEAR_DEBUG: [run_dalle] Full conversation context:")
        if conversation_context:
            for i, entry in enumerate(conversation_context):
                print(f"  [{i}] role: {entry.get('role')}, content: {entry.get('content')}")
        else:
            print("  [WARNING] No conversation context available!")
    merged_prompt = None
    if conversation_context:
        merged_prompt = extract_context_for_image_generation(conversation_context, prompt)
        if merged_prompt:
            print(f"DEBUG: Using merged prompt for DALL-E: {merged_prompt}")
    else:
        if NUCLEAR_DEBUG:
            print("NUCLEAR_DEBUG: [run_dalle] No conversation context, cannot merge prompt!")
    final_prompt = merged_prompt if merged_prompt else prompt
    print(f"DEBUG: FINAL prompt sent to DALL-E: {final_prompt}")
    if not conversation_context or not merged_prompt:
        if NUCLEAR_DEBUG:
            print("NUCLEAR_DEBUG: [run_dalle] Merging failed or context missing! Returning error to UI.")
        return "❌ Unable to merge previous answer for follow-up image prompt. Please try again after a factual question.", None
    try:
        print(f"DEBUG: Making DALL-E API call with enhanced prompt: {final_prompt}")
        response = client.images.generate(
            model="dall-e-3",
            prompt=final_prompt,
            size="1024x1024",
            n=1
        )
        print(f"DEBUG: DALL-E response received: {response}")
        print(f"DEBUG: Response type: {type(response)}")
        print(f"DEBUG: Response has data: {hasattr(response, 'data')}")
        if response and hasattr(response, "data") and response.data:
            print(f"DEBUG: Data length: {len(response.data)}")
            url = getattr(response.data[0], "url", None)
            print(f"DEBUG: DALL-E URL: {url}")
            if url:
                print(f"DEBUG: Returning URL: {url}")
                return url
            else:
                print("DEBUG: No URL found in response.data[0]")
                print(f"DEBUG: response.data[0] attributes: {dir(response.data[0])}")
        else:
            print("DEBUG: No data in response")
            print(f"DEBUG: Response attributes: {dir(response)}")
        print("DEBUG: DALL-E failed - no URL found")
        return "Image generation failed - no URL returned from DALL-E API."
    except Exception as e:
        print(f"❌ DALL-E error: {e}")
        print(f"DEBUG: Exception type: {type(e)}")
        print(f"DEBUG: Exception details: {str(e)}")
        # Provide user-friendly error messages
        error_str = str(e)
        if "content_policy_violation" in error_str:
            return "🛡️ **Content Policy Violation**\n\nYour image request was rejected by OpenAI's safety system. This usually happens when the prompt contains:\n\n• **Violent content** (words like 'attack', 'destroy', 'kill')\n• **Inappropriate content** (explicit or adult material)\n• **Harmful content** (hate speech, discrimination)\n\n**Try rephrasing your request** to avoid these terms. For example:\n• Instead of 'attacking', use 'approaching' or 'near'\n• Instead of 'destroying', use 'transforming' or 'changing'\n• Focus on the visual elements rather than violent actions"
        elif "billing" in error_str.lower() or "quota" in error_str.lower():
            return "💳 **Billing/Quota Error**\n\nYour OpenAI account has reached its usage limit or billing quota. Please check your OpenAI account settings."
        elif "invalid" in error_str.lower() and "prompt" in error_str.lower():
            return "📝 **Invalid Prompt**\n\nYour image request couldn't be processed. Try:\n\n• Making your description more specific\n• Using simpler language\n• Avoiding very long descriptions\n• Focusing on visual elements"
        elif "rate" in error_str.lower() and "limit" in error_str.lower():
            return "⏱️ **Rate Limit Exceeded**\n\nYou're making requests too quickly. Please wait a moment and try again."
        else:
            return f"❌ **Image Generation Failed**\n\nError: {error_str}\n\nPlease try again with a different prompt."

def run_stable_diffusion(prompt, base_image_url=None):
    try:
        if base_image_url:
            # Download the base image
            base_image_response = requests.get(base_image_url)
            if base_image_response.status_code != 200:
                return f"❌ Failed to download base image for img2img: {base_image_url}"
            base_image_bytes = base_image_response.content
            # Call Stability img2img endpoint
            response = requests.post(
                "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image",
                headers={
                    "Authorization": f"Bearer {os.getenv('STABLE_DIFFUSION_API_KEY')}",
                },
                files={"init_image": ("init_image.png", base_image_bytes, "image/png")},
                data={
                    "text_prompts": '[{"text": "%s"}]' % prompt,
                    "cfg_scale": "7",
                    "height": "1024",
                    "width": "1024",
                    "samples": "1",
                    "steps": "30"
                }
            )
        else:
            response = requests.post(
                "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                headers={
                    "Authorization": f"Bearer {os.getenv('STABLE_DIFFUSION_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "text_prompts": [{"text": prompt}],
                    "cfg_scale": 7,
                    "height": 1024,
                    "width": 1024,
                    "samples": 1,
                    "steps": 30
                }
            )
        response.raise_for_status()
        result = response.json()
        # Extract base64 image from all possible fields and strip prefix if present
        b64 = None
        if "artifacts" in result and result["artifacts"]:
            b64 = result["artifacts"][0].get("base64") or result["artifacts"][0].get("b64_json")
        if not b64 and "data" in result and result["data"]:
            b64 = result["data"][0].get("b64_json")
        if b64:
            if b64.startswith("data:image/png;base64,"):
                b64 = b64[len("data:image/png;base64,"):]
            return b64
        return "❌ No image returned from Stable Diffusion API."
    except Exception as e:
        return f"❌ Stable Diffusion error: {e}"

def run_midjourney(prompt):
    # Midjourney API implementation (would need Midjourney API access)
    try:
        # This is a placeholder - actual implementation would use Midjourney's API
        return f"🎨 Midjourney-style image: {prompt} (API access required)"
    except Exception as e:
        return f"Midjourney generation failed: {str(e)}"

def run_anime_diffusion(prompt):
    # Anime Diffusion via Hugging Face
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
            headers={"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"},
            json={"inputs": f"anime style, {prompt}"}
        )
        if response.status_code == 200:
            return response.content  # Returns image bytes
        else:
            return "Anime diffusion generation failed."
    except Exception as e:
        return f"Anime diffusion failed: {str(e)}"

def run_deepl(prompt, target_lang="EN"):
    try:
        response = requests.post(
            "https://api.deepl.com/v2/translate",
            data={
                "auth_key": os.getenv("DEEPL_API_KEY"),
                "text": prompt,
                "target_lang": target_lang
            }
        )
        result = response.json()
        print("🔍 DeepL raw response:", result)

        if "translations" not in result:
            print("❌ DeepL API error:", result)
            return "Translation failed."
        return result["translations"][0]["text"]
    except Exception as e:
        print("❌ Exception during DeepL translation:", str(e))
        return "Translation failed."

def run_whisper(audio_file_path):
    # Speech-to-text using OpenAI Whisper
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return response.text
    except Exception as e:
        return f"Speech-to-text failed: {str(e)}"

def run_elevenlabs(text):
    # Text-to-speech using ElevenLabs
    try:
        response = requests.post(
            "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM",
            headers={
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": os.getenv("ELEVENLABS_API_KEY")
            },
            json={
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
        )
        if response.status_code == 200:
            # Save audio file and return path
            with open("output_audio.mp3", "wb") as f:
                f.write(response.content)
            return "output_audio.mp3"
        else:
            return "Text-to-speech failed."
    except Exception as e:
        return f"ElevenLabs failed: {str(e)}"

def run_claude_3_5_sonnet(prompt, user_id="default"):
    from memory import memory_manager
    context = memory_manager.get_context_for_followup(user_id, prompt)
    is_followup = memory_manager.is_followup_question(prompt, user_id)
    if is_followup and context:
        enhanced_prompt = f"{context}\n\nPlease answer the current query using the context provided."
        messages = [{"role": "user", "content": enhanced_prompt}]
    else:
        messages = memory_manager.get_memory_for_model(user_id, "claude-3-5-sonnet", max_tokens=4000)
        messages.append({"role": "user", "content": prompt})
    # Anthropic expects a list of dicts with 'role' and 'content'
    response = claude_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=0.7,
        messages=messages  # type: ignore
    )  # type: ignore
    block = response.content[0]
    reply = getattr(block, "text", "")
    if not isinstance(reply, str):
        reply = str(reply)
    memory_manager.save_memory(user_id, "assistant", reply, "text", "claude-3-5-sonnet")
    return reply

def run_gpt_4o_mini(prompt, user_id="default"):
    from memory import memory_manager
    messages = memory_manager.get_memory_for_model(user_id, "gpt-4o-mini", max_tokens=4000)
    chat_messages = []
    for m in messages:
        if m.get("role") in ("user", "assistant"):
            chat_messages.append({"role": m.get("role"), "content": m.get("content", "")})
    chat_messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=chat_messages
    )
    reply = getattr(response.choices[0].message, "content", "")
    if reply is None:
        reply = ""
    elif not isinstance(reply, str):
        reply = str(reply)
    memory_manager.save_memory(user_id, "assistant", reply, "text", "gpt-4o-mini")
    return reply

# CodeLlama-70B, WizardCoder, and Phind-CodeLlama handlers (if not already present)
def run_codellama_70b(prompt):
    api_url = "https://api-inference.huggingface.co/models/codellama/CodeLlama-70b-Instruct-hf"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}", "Content-Type": "application/json"}
    data = {"inputs": prompt}
    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
            return result[0]['generated_text']
        elif isinstance(result, dict) and 'generated_text' in result:
            return result['generated_text']
        else:
            return str(result)
    else:
        return f"CodeLlama-70B generation failed: {response.text}"

def run_wizardcoder(prompt):
    return "[WizardCoder integration required: connect to Hugging Face or Replicate API for real code generation.]"

def run_phind_codellama(prompt):
    return "[Phind-CodeLlama integration required: connect to Hugging Face or Replicate API for real code generation.]"
