# Portal AI - Unified AI Gateway

A sophisticated AI orchestration platform that intelligently routes tasks to the best specialized models.

## 🚀 Features

- **Intelligent Task Classification** with confidence scoring
- **Dynamic Model Selection** across 15+ AI providers
- **Smart Memory Management** with token optimization
- **Modern Chat Interface** with dark mode support
- **Comprehensive Analytics** and performance tracking
- **Audio Processing** (speech-to-text, text-to-speech)
- **Multimodal Capabilities** (image analysis, OCR)
- **Specialized Image Generation** (anime, 3D, artistic styles)

## 🔑 Required API Keys

### Core Models (Required)
```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic (Claude)
ANTHROPIC_API_KEY=sk-ant-api03-...

# Stability AI (Stable Diffusion)
STABLE_DIFFUSION_API_KEY=sk-...
```

### Enhanced Models (Optional)
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

### Environment Setup
Create a `.env` file in your project root:

```bash
# Core APIs (Required)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
STABLE_DIFFUSION_API_KEY=your_stability_key_here

# Enhanced APIs (Optional - system will use fallbacks if not provided)
GOOGLE_API_KEY=your_google_key_here
DEEPL_API_KEY=your_deepl_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
HUGGINGFACE_API_KEY=your_huggingface_key_here
```

## 📊 Model Capabilities

| **Task Type** | **Primary Models** | **Fallback** | **API Required** |
|---------------|-------------------|--------------|------------------|
| **Text Generation** | GPT-4o, Claude Sonnet, Gemini Pro | GPT-4 Turbo | OpenAI, Anthropic, Google |
| **Image Generation** | DALL-E 3, Stable Diffusion | - | OpenAI, Stability AI |
| **Translation** | DeepL | Google Translate | DeepL |
| **Audio STT** | Whisper | - | OpenAI |
| **Audio TTS** | ElevenLabs | - | ElevenLabs |
| **Multimodal** | Gemini Pro | GPT-4o | Google, OpenAI |

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Portal
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Run the application**
```bash
python main.py
```

## 🎯 Usage Examples

### Text Generation
- "Write a creative story about space exploration" → GPT-4o
- "Explain quantum physics in simple terms" → Claude Sonnet
- "Quick response to a simple question" → Claude Haiku

### Image Generation
- "Create a logo for my coffee shop" → DALL-E 3
- "Generate an anime character" → Anime Diffusion
- "3D architectural render of a modern house" → Midjourney

### Audio Processing
- "Transcribe this meeting recording" → Whisper
- "Convert this text to speech" → ElevenLabs

### Multimodal
- "What do you see in this image?" → Gemini Pro
- "Extract text from this document" → GPT-4o

## 🔧 Configuration

### Model Selection Logic
The system automatically selects the best model based on:
- **Task type** (text, image, audio, multimodal)
- **Content characteristics** (style, complexity, language)
- **Cost optimization** (uses cheaper models when appropriate)
- **Speed requirements** (fast models for simple tasks)

### Memory Management
- **Smart inclusion**: Only includes conversation history when needed
- **Token optimization**: Limits to last 10 exchanges
- **Auto-summarization**: Summarizes old conversations when memory gets large

## 📈 Analytics

The system tracks comprehensive metrics:
- **Model usage statistics**
- **Success rates** and error analysis
- **Processing times** and performance
- **Task distribution** and user patterns
- **Cost optimization** insights

Access analytics at: `GET /analytics`

## 🚀 Advanced Features

### Custom Model Integration
Add new models to `MODEL_REGISTRY` in `models.py`:

```python
"custom-model": {
    "provider": "custom",
    "strengths": ["specific task"],
    "cost": "low",
    "speed": "fast"
}
```

### Task Type Extension
Add new task types in `classifier/intent_classifier.py`:

```python
scores["new_task"] = 0.0
# Add detection patterns
new_task_patterns = [
    (r"\bpattern\b", 0.9),
]
```

## 🔒 Security Notes

- **API keys** are stored in environment variables
- **No hardcoded credentials** in the codebase
- **Secure logging** without sensitive data exposure
- **Rate limiting** and error handling for all APIs

## 🐛 Troubleshooting

### Common Issues

1. **Missing API Key Error**
   - Ensure all required API keys are set in `.env`
   - Check key format and validity

2. **Model Not Available**
   - System will automatically fallback to available models
   - Check API quotas and limits

3. **High Token Usage**
   - Memory management should prevent this
   - Check conversation length and summarization

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review analytics for system performance
- Monitor API usage and costs

---

**Portal AI** - Your intelligent gateway to the best AI models! 🚀 