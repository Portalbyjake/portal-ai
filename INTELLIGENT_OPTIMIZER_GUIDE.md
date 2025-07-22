# Intelligent Prompt Optimizer - Complete Guide

## 🎯 Overview

The **Intelligent Prompt Optimizer** is a task-aware, model-aligned optimization layer that enhances prompts only when it clearly improves clarity, output quality, or model performance. It intelligently decides whether to optimize or pass through prompts unchanged.

## ✅ Core Optimization Conditions

### **DO Optimize When:**
- Prompt is vague, unclear, or overly verbose
- Task involves creative generation, formatting, or role alignment
- Prompt would benefit from structure, tone, or model guidance
- User intent is ambiguous and needs clarification

### **DO NOT Optimize When:**
- Simple factual queries (e.g., "What's the capital of France?")
- Prompts that are already clear, concise, and effective
- User has already specified role/tone explicitly
- Prompt contains clear instructions or context

## 🧠 Optimization Decision Logic

### **Factual Query Detection**
The system identifies simple factual queries that don't need optimization:

```python
factual_patterns = [
    r"what is the capital of",
    r"how many",
    r"when was",
    r"who is",
    r"where is",
    r"define",
    r"meaning of",
    r"temperature",
    r"convert",
    r"calculate"
]
```

### **Vague/Verbose Detection**
Identifies prompts that need optimization:

```python
vague_indicators = [
    "something", "anything", "whatever", "you know", "kind of",
    "sort of", "maybe", "perhaps", "I think", "I guess",
    "please kindly", "if you don't mind", "would you be so kind",
    "I was wondering if", "I would like to know"
]
```

### **Role Context Detection**
Checks if user already specified a role:

```python
role_indicators = [
    "as a", "you are", "act as", "pretend to be", "imagine you are",
    "formal", "casual", "professional", "friendly", "serious",
    "expert", "specialist", "consultant", "advisor"
]
```

## ✂️ Prompt Refinement Examples

### **Verbose → Concise**
```
Raw: "Please kindly give me a helpful and informative list of suggestions"
Optimized: "Give me a helpful list of suggestions"
```

### **Vague → Clear**
```
Raw: "Write something about money"
Optimized: "You are a witty poet. Write a rhyming, humorous poem about the stock market and investing anxiety."
```

### **Unclear Intent → Structured**
```
Raw: "Build a to-do app"
Optimized: "You are a senior developer. Build a to-do app using Python and Flask. Include inline comments."
```

## 🎭 Role-Based Prefacing

### **Dynamic Role Assignment**
The system automatically assigns appropriate expert roles based on task context:

#### **Code Roles**
- **Senior Developer**: build, create, develop, implement, architecture
- **Code Reviewer**: review, debug, optimize, refactor, improve
- **System Architect**: design, architecture, scalable, distributed
- **DevOps Engineer**: deploy, infrastructure, ci/cd, docker, kubernetes

#### **Writing Roles**
- **Creative Writer**: write, create, compose, story, poem, novel
- **Technical Writer**: document, manual, guide, tutorial, explain
- **Copywriter**: marketing, advertisement, persuasive, sales
- **Journalist**: news, report, article, investigation, interview

#### **Analysis Roles**
- **Data Scientist**: analyze, data, statistics, trends, patterns
- **Business Analyst**: business, strategy, market, competition, growth
- **Researcher**: research, study, investigation, academic, thesis

#### **Creative Roles**
- **Motivational Coach**: uplifting, encouraging, inspirational, motivation
- **Comedian**: funny, humorous, joke, comedy, entertaining
- **Poet**: poem, poetry, rhyming, verse, lyrical
- **Artist**: creative, artistic, visual, design, aesthetic

## 🤖 Model-Specific Formatting

### **Text Models**

#### **GPT Models (gpt-4o, gpt-4-turbo, gpt-4o-mini)**
- Add conversational tone
- Format cues and brevity when needed
- Example: "You are a helpful assistant. Provide a clear, engaging response."

#### **Claude Models (claude-sonnet-4, claude-haiku, claude-3-5-sonnet)**
- Add logical, step-by-step structure
- Clear role context
- Example: "You are an expert analyst. Provide a thorough, well-reasoned analysis."

#### **Other Text Models (gemini-pro, llama-3-70b)**
- Emphasize synthesis and clarity
- Open-ended reasoning
- Example: "You are a synthetic thinker. Provide clear, comprehensive analysis."

### **Code Models**

#### **Claude 3.5 Sonnet**
- Add task, language, comment/format expectations
- Example: "Write a minimal Node.js API. Include only essential routes and add inline comments."

#### **CodeLlama, WizardCoder, Phind**
- Focus on practical, working code
- Include language specification when missing

### **Image Models**

#### **DALL-E 3**
- Use full scene descriptions in natural language
- Subject + background + lighting + style
- Example: "Create a photorealistic image: {prompt}"

#### **Stable Diffusion**
- Use stylized, comma-separated keywords
- Art tags and negative prompts
- Example: "ultra-detailed photograph of {prompt}, golden hour lighting, realistic shadows, DSLR quality, 4K resolution"

#### **Anime Diffusion**
- Add anime-specific aesthetic terms
- Example: "Create an anime-style image: {prompt}"

### **Translation Models**

#### **DeepL, Google Translate, Azure Translator**
- Add tone and dialect context
- Example: "Translate to formal Mexican Spanish for a professional email"

### **Audio Models**

#### **Whisper**
- Add context: accent, noise level, speed
- Example: "Transcribe accurately with context: {prompt}"

#### **ElevenLabs**
- Specify voice tone, gender, pace, emotion
- Example: "Convert to natural speech with appropriate tone: {prompt}"

## 🔄 Memory & Chaining Support

### **Cross-Modal References**
- Detects when users reference previous images in text questions
- Identifies when users reference previous text in image requests
- Uses semantic analysis rather than simple keyword matching

### **Follow-up Chaining**
- Supports natural follow-ups like "Make it more visual," "Turn it into a poem," "Add more blue"
- Doesn't duplicate prior context — summarizes what's needed from history

### **User Preference Memory**
- If memory indicates user tone/format preference, includes that automatically
- Maintains conversation flow across modalities

## 💬 Vague Input Handling

When input is too vague, the system can ask clarifying questions:

- **"Would you like a list, paragraph, or table?"**
- **"Should the tone be casual, formal, or professional?"**
- **"What scene or visual style are you imagining for this image?"**
- **"What's the target language, tone, and audience for this translation?"**

## 📊 Usage Examples

### **Text Generation**
```python
# Simple factual query - passes through unchanged
prompt = "What is the capital of France?"
# Result: "What is the capital of France?" (unchanged)

# Vague prompt - gets optimized
prompt = "Can you please kindly give me a helpful and informative list of suggestions if you don't mind?"
# Result: "Give me a helpful list of suggestions" (optimized)

# Creative prompt - gets role context
prompt = "Write a poem about money"
# Result: "You are a witty poet. Write a rhyming, humorous poem about the stock market and investing anxiety."
```

### **Code Generation**
```python
# Code with context - passes through
prompt = "Write a Python function to calculate fibonacci numbers"
# Result: "Write a Python function to calculate fibonacci numbers" (unchanged)

# Code without context - gets enhanced
prompt = "Build a web app"
# Result: "You are a senior developer. Build a web app using Python and Flask. Include inline comments."
```

### **Image Generation**
```python
# Clear image prompt - passes through
prompt = "Create a photorealistic image of a cat in a garden"
# Result: "Create a photorealistic image of a cat in a garden" (unchanged)

# Simple image prompt - gets enhanced
prompt = "make an image of a cat"
# Result: "Create a high-quality image: make an image of a cat"
```

## 🚀 Integration with Portal

The intelligent optimizer integrates seamlessly with your Portal application:

1. **Automatic Detection**: Analyzes prompts in real-time
2. **Model Alignment**: Applies model-specific optimizations
3. **Memory Integration**: Uses conversation history for context
4. **Performance Tracking**: Monitors optimization effectiveness

## 🎯 Key Benefits

1. **Intelligent Decision Making**: Only optimizes when beneficial
2. **Model-Aware**: Tailors optimizations to specific model strengths
3. **Role-Based**: Automatically assigns appropriate expert roles
4. **Memory-Integrated**: Uses conversation context for better optimization
5. **Performance-Focused**: Improves output quality without unnecessary changes

## 🔧 Technical Implementation

### **Core Classes**
- `IntelligentPromptOptimizer`: Main optimization engine
- `OptimizationContext`: Context for optimization decisions

### **Key Methods**
- `optimize_prompt()`: Main optimization function
- `_needs_optimization()`: Decision logic
- `_should_add_role_context()`: Role detection
- `_get_appropriate_role()`: Role assignment

### **Configuration**
- Model-specific guidelines
- Role-based prefacing rules
- Optimization conditions
- Fluff removal patterns

This intelligent optimization system ensures that your Portal application provides the best possible user experience by enhancing prompts only when it clearly improves clarity, output quality, or model performance. 