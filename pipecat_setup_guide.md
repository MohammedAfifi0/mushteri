# Pipecat + Twilio Voice Agent - Complete Setup Guide

**Production-ready voice AI agent with:**
- 🤖 Groq Llama 3.3 Versatile LLM
- 🎤 Azure Speech-to-Text (Arabic)
- 🔊 Groq TTS (Arabic with multiple voices)
- 👂 Silero VAD (Voice Activity Detection)
- ☎️ Twilio (Phone integration)

## Why Your Previous Setup Failed

Based on the Pipecat documentation and your symptoms (call connects but LLM never responds), here are the likely issues:

### 1. **Incorrect Pipeline Order** ❌
The aggregators MUST be in the correct position:
```python
# WRONG - This won't work
pipeline = Pipeline([
    transport.input(),
    stt,
    llm,  # LLM before aggregators
    user_aggregator,  # Wrong position
    tts,
    assistant_aggregator,  # Wrong position
    transport.output(),
])

# CORRECT - This works
pipeline = Pipeline([
    transport.input(),
    stt,
    user_aggregator,        # AFTER STT, BEFORE LLM
    llm,
    tts,
    transport.output(),
    assistant_aggregator,   # AFTER OUTPUT
])
```

### 2. **Missing Context Setup** ❌
You need to properly create and link the context:
```python
# Create context from LLM
context = llm.create_context(messages)

# Create context aggregator
context_aggregator = llm.create_context_aggregator(context)

# Create user and assistant aggregators with the SAME context
user_aggregator = LLMUserContextAggregator(context)
assistant_aggregator = LLMAssistantContextAggregator(context)
```

### 3. **Wrong Twilio Serializer Usage** ❌
The serializer needs the `stream_sid` from Twilio's start event:
```python
# WRONG - No stream_sid
serializer = TwilioFrameSerializer()

# CORRECT - Wait for Twilio start event
async for message in websocket.iter_text():
    data = json.loads(message)
    if data.get("event") == "start":
        stream_sid = data["start"]["streamSid"]
        serializer = TwilioFrameSerializer(stream_sid=stream_sid)
        break
```

### 4. **VAD Configuration Issues** ❌
VAD parameters are critical for proper turn-taking:
```python
# Use recommended settings from docs
vad_analyzer = SileroVADAnalyzer(
    params=VADParams(
        confidence=0.7,
        start_secs=0.2,  # Quick response to speech
        stop_secs=0.8,   # Natural pauses
        min_volume=0.6,
    )
)
```

## Prerequisites

### 1. API Keys Needed

| Service | Purpose | Get Key |
|---------|---------|---------|
| Groq | LLM + TTS | [console.groq.com](https://console.groq.com/) |
| Azure Speech | STT | [portal.azure.com](https://portal.azure.com/) |
| Twilio | Phone | [console.twilio.com](https://console.twilio.com/) |

### 2. Tools Needed

- **Python 3.10+**
- **ngrok** for local testing: [ngrok.com/download](https://ngrok.com/download)
- **Railway** for deployment: [railway.app](https://railway.app/)

## Installation

### Step 1: Create Project Directory

```bash
mkdir pipecat-voice-agent
cd pipecat-voice-agent
```

### Step 2: Install Dependencies

Create `requirements.txt`:
```
pipecat-ai[azure,groq,silero]
fastapi
uvicorn[standard]
websockets
python-dotenv
loguru
aiohttp
```

Install:
```bash
pip install -r requirements.txt
```

### Step 3: Environment Configuration

Create `.env` file:
```bash
# Groq (for LLM and TTS)
GROQ_API_KEY=gsk_your_key_here

# Azure Speech Services
AZURE_SPEECH_KEY=your_azure_key_here
AZURE_SPEECH_REGION=eastus

# Server (set this when deploying)
SERVER_HOST=your-ngrok-url.ngrok.io
PORT=7860
```

## Getting API Keys

### Groq API Key

1. Go to [console.groq.com](https://console.groq.com/)
2. Sign up with Google/GitHub
3. Navigate to "API Keys" in sidebar
4. Click "Create API Key"
5. Copy the key (starts with `gsk_`)

**Free Tier:**
- Llama 3.3 70B: 30 requests/minute
- TTS: Generous free quota

### Azure Speech Key

1. Go to [Azure Portal](https://portal.azure.com/)
2. Create new resource → Search "Speech"
3. Create "Speech Services" resource
4. Choose region (e.g., East US)
5. Select Free tier (F0) or Standard (S0)
6. After creation, go to "Keys and Endpoint"
7. Copy Key 1 and Region

**Free Tier:**
- STT: 5 hours/month free
- After: $1 per audio hour

### Twilio Setup

1. Sign up at [twilio.com](https://www.twilio.com/)
2. Get $15 trial credit
3. Buy a phone number (Voice enabled)
4. Save Account SID and Auth Token

## Local Testing

### Terminal 1: Start ngrok

```bash
ngrok http 7860
```

You'll see output like:
```
Forwarding https://abc123.ngrok.io -> http://localhost:7860
```

**Copy this URL** - you'll need it for Twilio.

### Terminal 2: Run the Bot

```bash
# Make sure .env is configured
python bot.py
```

You should see:
```
Starting Pipecat Twilio Voice Agent on port 7860
Configuration:
  - LLM: Groq Llama 3.3 Versatile
  - STT: Azure Speech Services (Arabic)
  - TTS: Groq PlayAI Arabic (Ahmad voice)
  - VAD: Silero (local, CPU-based)
```

### Configure Twilio Webhook

1. Go to [Twilio Console](https://console.twilio.com/)
2. Navigate to: **Phone Numbers → Manage → Active Numbers**
3. Click your phone number
4. Scroll to **Voice Configuration**
5. Under "A call comes in":
   - Select **Webhook**
   - Enter: `https://your-ngrok-url.ngrok.io/incoming-call`
   - Method: **HTTP POST**
6. Click **Save**

### Test It!

Call your Twilio number. You should hear:
- Arabic greeting: "مرحبا، أنا مساعدك الذكي"
- Bot listens to your Arabic speech
- Responds in Arabic with Groq TTS

## Troubleshooting Common Issues

### Issue 1: Call Connects But No LLM Response

**Symptoms:**
- Phone rings and connects
- No audio from bot
- Or bot speaks initial greeting but doesn't respond to you

**Solutions:**

1. **Check Pipeline Order:**
```bash
# Look for this in logs when call starts:
# Pipeline: transport.input → stt → user_aggregator → llm → tts → transport.output → assistant_aggregator
```

2. **Verify Context Creation:**
```python
# Make sure you're creating context from the LLM:
context = llm.create_context(messages)  # NOT creating context manually
```

3. **Check API Keys:**
```bash
# Test Groq API
curl https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY"

# Test Azure Speech
# Try a simple STT test first
```

4. **Monitor Logs:**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python bot.py

# Look for:
# - "Media stream started"
# - "User started speaking"
# - "Transcription received"
# - "LLM processing"
```

### Issue 2: Audio is Garbled or Robotic

**Solutions:**

1. **Check Sample Rates:**
```python
# Groq TTS is fixed at 48kHz
tts = GroqTTSService(
    sample_rate=48000,  # Don't change this
)

# Twilio uses 8kHz by default
# The transport should handle resampling
```

2. **VAD Too Aggressive:**
```python
# Increase stop_secs if cutting off speech
vad_params = VADParams(
    stop_secs=1.0,  # Was 0.8, try higher
)
```

### Issue 3: Bot Responds in English Instead of Arabic

**Solutions:**

1. **Check STT Language:**
```python
stt = AzureSTTService(
    language="ar-SA",  # Arabic (Saudi)
    # Also try: "ar-EG" (Egyptian), "ar-AE" (UAE)
)
```

2. **Check TTS Configuration:**
```python
tts = GroqTTSService(
    model_name="playai-tts-arabic",  # Must be Arabic model
    voice_id="Ahmad-PlayAI",  # Arabic voice
    params=GroqTTSService.InputParams(
        language="ar",
    ),
)
```

3. **Check System Prompt:**
```python
messages = [
    {
        "role": "system",
        "content": "أنت مساعد ذكاء اصطناعي. تحدث بالعربية فقط.",
    },
]
```

### Issue 4: Webhook Fails / Can't Connect

**Solutions:**

1. **Verify ngrok is Running:**
```bash
# Check ngrok dashboard
open http://localhost:4040

# Look for incoming POST requests
```

2. **Test Webhook Manually:**
```bash
curl -X POST https://your-ngrok-url.ngrok.io/incoming-call \
  -d "From=+1234567890" \
  -d "To=+1234567890"

# Should return TwiML XML
```

3. **Check Twilio Configuration:**
- URL must include `/incoming-call`
- Must be HTTPS (ngrok provides this)
- Method must be POST

### Issue 5: WebSocket Disconnects Immediately

**Solutions:**

1. **Check Stream SID Parsing:**
```python
# Add logging
logger.info(f"Received message: {message}")

# Verify you're getting the 'start' event
if event == "start":
    logger.info(f"Stream SID: {data['start']['streamSid']}")
```

2. **Handle All Twilio Events:**
```python
# Twilio sends: connected, start, media, stop
# Make sure you handle them all
```

## Deploy to Railway

### Why Railway?

- Automatic HTTPS (required for Twilio)
- Easy deployment from GitHub
- Environment variable management
- Free tier available

### Step 1: Prepare for Deployment

Create `railway.json`:
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python bot.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

Create `Procfile`:
```
web: python bot.py
```

Create `.dockerignore`:
```
__pycache__
*.pyc
.env
.git
```

### Step 2: Deploy to Railway

1. **Initialize Git:**
```bash
git init
git add .
git commit -m "Initial Pipecat voice agent"
```

2. **Push to GitHub:**
```bash
gh repo create pipecat-voice-agent --public --source=. --remote=origin
git push -u origin main
```

3. **Deploy on Railway:**
   - Go to [railway.app](https://railway.app/)
   - Click "New Project"
   - Choose "Deploy from GitHub repo"
   - Select your repository
   - Railway auto-detects and deploys

4. **Add Environment Variables:**
   - Go to project → Variables tab
   - Add all variables from `.env`:
     - `GROQ_API_KEY`
     - `AZURE_SPEECH_KEY`
     - `AZURE_SPEECH_REGION`
     - `PORT` (set to `7860`)
   - Click "Redeploy"

5. **Get Railway URL:**
   - Go to Settings → Generate Domain
   - Copy URL: `https://your-app.up.railway.app`

6. **Update Twilio:**
   - Go to Twilio Console → Your Phone Number
   - Update webhook to: `https://your-app.up.railway.app/incoming-call`
   - Save

### Step 3: Test Production Deployment

Call your Twilio number - it should work exactly like local!

## Advanced Configuration

### Use Different Arabic Voice

Groq provides multiple Arabic voices:
```python
# Male voices
voice_id = "Ahmad-PlayAI"   # Default male
voice_id = "Khalid-PlayAI"  # Alternative male
voice_id = "Nasser-PlayAI"  # Alternative male

# Female voices
voice_id = "Amira-PlayAI"   # Female
```

### Add Function Calling

Enable your bot to call external APIs:

```python
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

# Define a function
async def get_weather(params: FunctionCallParams):
    location = params.arguments.get("location")
    # Call weather API
    weather_data = {"temp": "25°C", "conditions": "sunny"}
    await params.result_callback(weather_data)

# Create function schema
weather_function = FunctionSchema(
    name="get_weather",
    description="Get current weather",
    properties={
        "location": {
            "type": "string",
            "description": "City name",
        }
    },
    required=["location"]
)

# Add to LLM context
tools = ToolsSchema(standard_tools=[weather_function])
context = llm.create_context(messages, tools=tools)

# Register function handler
llm.register_function("get_weather", get_weather)
```

### Switch to English

```python
# STT
stt = AzureSTTService(
    language="en-US",  # English
)

# TTS
tts = GroqTTSService(
    model_name="playai-tts",  # English model
    voice_id="Celeste-PlayAI",  # English voice
    params=GroqTTSService.InputParams(
        language="en-US",
    ),
)

# System prompt
messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant. Speak naturally and concisely.",
    },
]
```

### Enable Turn Detection

For more natural conversations (experimental):
```python
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

# Add turn analyzer
turn_analyzer = LocalSmartTurnAnalyzerV3()

# Update VAD params
vad_params = VADParams(
    stop_secs=0.2,  # Lower with turn detection
)

# Add to transport
transport = DailyTransport(
    params=DailyParams(
        vad_analyzer=SileroVADAnalyzer(params=vad_params),
        turn_analyzer=turn_analyzer,  # Add this
    ),
)
```

## Architecture Deep Dive

### How Audio Flows Through the Pipeline

```
1. User speaks → Twilio captures audio (8kHz)
2. Twilio sends audio via WebSocket
3. Transport receives and buffers audio
4. VAD detects speech start/stop
5. STT converts to text (Azure)
6. User aggregator adds to context
7. LLM processes context (Groq Llama)
8. TTS generates audio (Groq Arabic)
9. Transport sends audio to Twilio
10. Twilio plays to user
11. Assistant aggregator updates context
```

### Critical Components

**VAD (Voice Activity Detection):**
- Runs locally on CPU (Silero)
- Detects speech vs silence
- Controls turn-taking timing
- ~1ms processing time per chunk

**Context Aggregators:**
- `LLMUserContextAggregator`: Collects user transcriptions
- `LLMAssistantContextAggregator`: Collects bot responses
- Both maintain conversation history
- Critical for multi-turn conversations

**Interruption Handling:**
- Enabled by default (`allow_interruptions=True`)
- User can interrupt bot mid-sentence
- Pipeline clears queues and resets
- Natural conversation flow

## Performance Tips

### 1. Optimize for Latency

```python
# Use fast VAD settings
vad_params = VADParams(
    start_secs=0.1,  # Very responsive (but may catch noise)
    stop_secs=0.5,   # Quick turn-taking
)

# Use faster LLM model
llm = GroqLLMService(
    model="llama-3.1-70b-versatile",  # Faster than 3.3
)
```

### 2. Monitor Performance

```python
# Enable metrics
task = PipelineTask(
    pipeline,
    params=PipelineParams(
        enable_metrics=True,
        enable_usage_metrics=True,
    ),
)

# Log metrics
@task.event_handler("on_metrics")
async def on_metrics(metrics):
    logger.info(f"Pipeline metrics: {metrics}")
```

### 3. Handle Errors Gracefully

```python
# Add error handlers
@transport.event_handler("on_error")
async def on_error(error):
    logger.error(f"Transport error: {error}")
    # Notify user, retry, or gracefully end call

@task.event_handler("on_error")
async def on_task_error(error):
    logger.error(f"Pipeline error: {error}")
    # Handle pipeline failures
```

## Debugging Checklist

When your bot isn't working, check these in order:

- [ ] All API keys are correct in `.env`
- [ ] ngrok is running and URL is correct
- [ ] Twilio webhook is configured correctly
- [ ] Bot server is running (`python bot.py`)
- [ ] WebSocket connection established (check logs)
- [ ] Stream SID received from Twilio
- [ ] VAD is detecting speech (check for "User started speaking")
- [ ] STT is producing transcriptions (check logs)
- [ ] LLM is processing (check for "LLM processing" or similar)
- [ ] TTS is generating audio
- [ ] Audio is being sent to Twilio

## Resources

- [Pipecat Documentation](https://docs.pipecat.ai/)
- [Pipecat Examples](https://github.com/pipecat-ai/pipecat-examples)
- [Pipecat Discord](https://discord.gg/pipecat)
- [Twilio Media Streams](https://www.twilio.com/docs/voice/twiml/stream)
- [Groq API Docs](https://console.groq.com/docs)
- [Azure Speech Docs](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/)

## Next Steps

1. **Test locally** with ngrok
2. **Deploy to Railway** for production
3. **Add features** like function calling
4. **Monitor performance** and optimize
5. **Scale** as needed with load balancing

## Common Questions

**Q: Why use Azure STT instead of Groq?**
A: Groq doesn't offer STT yet. Azure has excellent Arabic support and low latency.

**Q: Can I use OpenAI instead of Groq?**
A: Yes! Just replace `GroqLLMService` with `OpenAILLMService`. TTS you can use Groq or OpenAI.

**Q: How much does this cost to run?**
A: With free tiers:
- Groq: Free for reasonable usage
- Azure: 5 hours STT free/month
- Twilio: $1-2 per number/month + per-minute charges
- Railway: Free tier available

**Q: Can I use this for production?**
A: Yes, but:
- Monitor API quotas
- Add error handling
- Implement logging and monitoring
- Consider rate limiting
- Test thoroughly with real users

**Q: How do I add call recording?**
A: Check Twilio's `<Record>` TwiML verb or use Pipecat's recording features.

---

**Need help?** Join the [Pipecat Discord](https://discord.gg/pipecat) community!
