# Portal AI - Unified AI Gateway

Portal AI is a world-class AI orchestration platform that intelligently interprets user intent, selects the best model across 15+ providers, optimizes the prompt for that model, and returns 
high-quality results — all through a unified interface.

---

## 💡 Vision

Our mission is to make AI radically more accessible, powerful, and personalized. Portal acts as your AI assistant *for using AI assistants* — routing tasks to the best models, optimizing 
inputs, and learning over time to deliver smarter, more intuitive results.

---

## 🚀 Core Features

- **🔍 Intent Detection**  
  Advanced classifier determines task type (text, image, translation, etc.) with confidence scoring.

- **🧠 Prompt Optimization**  
  Rewrites user inputs to match the strengths and formatting expectations of the selected model.

- **🎯 Dynamic Model Selection**  
  Smart routing across GPT-4o, Claude, Gemini, DeepL, DALL·E, Stable Diffusion, and more.

- **🗃️ Memory System**  
  Stores past inputs/outputs for context-aware follow-ups across modalities (text, image, audio).

- **🖼️ Image Generation & Understanding**  
  Supports fantasy, anime, logos, architectural renderings, OCR, and more.

- **🗣️ Audio Capabilities**  
  Speech-to-text and text-to-speech support via Whisper, ElevenLabs, and others.

- **📊 Analytics**  
  Tracks usage, token costs, model performance, and error rates for iterative improvement.

- **💻 Clean Interface**  
  Modern frontend with real-time feedback, transparent model display, and dark mode support.

- **🧠 Self-Improving Logic**  
  System designed to learn from usage patterns and continuously refine performance.

---

## 🔮 Roadmap: World-Class Upgrades

- [x] Contextual memory chaining across sessions  
- [ ] API billing and usage tier support  
- [ ] Developer-facing API gateway for third-party apps  
- [ ] Auto-suggestion of best model based on past performance  
- [ ] Reinforcement learning loop based on user feedback  
- [ ] Real-time data plugin support (search, stock tickers, etc.)  
- [ ] Secure video generation via leading diffusion models  
- [ ] Autonomous agents with long-term memory and planning  
- [ ] Web automation & agentic task execution (via headless browser agents)  
- [ ] Personalized behavior based on individual usage history  

---

## 🧪 Next-Gen Feature Enhancements

- **🔄 Self-Healing Model Routing**  
  Detect and reroute failed/slow calls in real time using fallback logic and cached outputs.

- **🔍 Semantic Task Rewriting Engine**  
  Reformulates vague/ambiguous prompts based on history and intent.

- **🧠 Multimodal Memory Embeddings**  
  Store/retrieve across text, image, voice, and video — with continuity across modalities.

- **🧩 Composable Model Pipeline Support**  
  Chain models dynamically (e.g., caption → image gen → summarization).

- **🛡 Granular Permission Layers for Sensitive Tasks**  
  Restrict model/task/data access by role (great for enterprise/dev tiers).

- **🌍 Language- and Culture-Aware Behavior Switching**  
  Dynamically adjust tone/content based on user’s language and region.

- **💡 Task Planning & Decomposition Engine**  
  Decomposes complex tasks into subtasks and executes them in parallel/sequence.

- **📊 Outcome-Based Model Scoring Dashboard**  
  Internal benchmarks per task/domain/cohort with feedback integration.

- **🤝 Cross-Agent Collaboration Protocols**  
  Autonomous agents can delegate and collaborate in shared spaces.

- **🗣️ Voice & Emotion Interface Layer**  
  Accept/generate output based on tone, urgency, or emotional cues.

- **🔐 Zero-Knowledge Prompt Execution (ZKPE)**  
  Execute on private/encrypted data without exposing it to the models.

- **🧪 Autonomous Prompt A/B Testing**  
  Routinely tests variations to maximize prompt performance over time.

- **📘 Model Behavior Notebooking**  
  Logs nuanced model quirks and adapts future prompt strategies.

- **🧬 Continual Learning Loop**  
  Learns from votes, follow-ups, and user corrections.

- **🧰 Plugin Ecosystem & SDKs**  
  Let devs extend Portal via SDK and pluggable modules.

- **🌐 Open API Hub Integration**  
  Allow use of Perplexity, Mistral, Groq, etc. in real time.

- **🔁 Version Control for Prompt Templates**  
  Track prompt changes per model/task with rollback support.

- **🧯 Model Bias & Safety Filter Layer**  
  Intercept noncompliant or biased model outputs before display.

- **📜 Transparent Prompt + Model Audit Logs**  
  Full traceability of every request (prompt, model, response).

- **🌑 Shadow Execution Mode (Stealth Eval)**  
  Background eval of alternate models to improve robustness.

- **🧭 Conversational Goal-Mapping UI**  
  Break down multi-step goals in a transparent visual format.

- **🎥 Live-Generated Instructional Output**  
  AI can create how-to GIFs/videos using synthesis models.

- **🖇️ Prompt Copilot**  
  A helper that iteratively improves the user’s input before model call.

---

## 🧩 Supported Models

| Category       | Providers |
|----------------|-----------|
| Text           | OpenAI GPT-4o, Claude Sonnet, Gemini 1.5 |
| Image          | DALL·E 3, Stable Diffusion, Anime Diffusion |
| Translation    | DeepL, Google Translate |
| Summarization  | OpenAI, Claude |
| Audio (TTS/STT)| Whisper, ElevenLabs |

---

## 🔑 Required API Keys

### Core (Required)
```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic (Claude)
ANTHROPIC_API_KEY=sk-ant-...

# Stability AI
STABLE_DIFFUSION_API_KEY=sk-...
```

### Enhanced (Optional)
```bash
# Google (Gemini)
GOOGLE_API_KEY=...

# DeepL (Translation)
DEEPL_API_KEY=...

# ElevenLabs (Text-to-Speech)
ELEVENLABS_API_KEY=...

# Hugging Face (Anime Diffusion)
HUGGINGFACE_API_KEY=hf_...
```

---

## 📦 Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
python main.py
```

---

## 🧠 Architecture Overview

- `/main.py`: Entry point with smart routing
- `/models.py`: Handles model selection and API calls
- `/prompt_optimizer.py`: Rewrites prompts per model formatting
- `/classifier/`: Task and intent classifiers
- `/memory.py`: Stores and retrieves past inputs/outputs
- `/templates/`: Frontend HTML templates
- `/routes.py`: Flask endpoints for each task

---

## 🧪 Testing

```bash
# Run all test cases
pytest
```

---

## 🔒 Security Notes

- API keys stored in environment variables (never hardcoded)
- Secure logging without exposing user data
- Rate limiting and failover logic for all model endpoints

---

## 🐛 Troubleshooting

### Missing API Key Error
- Ensure all `.env` keys are properly set
- Check for typos or invalid formats

### Model Timeout or Fallback
- If primary model fails, fallback will be triggered
- Check logs for model failure details

---

Portal is continuously evolving — designed to be *as intuitive, intelligent, and dynamic as the future of AI itself.*


