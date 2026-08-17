## Voice Agent (Twilio + Azure + Groq + Pipecat)

Arabic voice bot running over Twilio Media Streams. Answers real inbound phone calls.

- STT: Azure Speech, `ar-KW` by default (override with `AZURE_SPEECH_LANGUAGE`)
- LLM: Groq `openai/gpt-oss-120b`
- TTS: Groq PlayAI Arabic `playai-tts-arabic`
- VAD/endpointing: Silero, with `stop_secs` raised above the default so the agent stops cutting speakers off mid-sentence

Main entrypoint: `pipecat_twilio_bot.py`

### Notes on the audio path

- The opening greeting is synthesised once at process boot, downmixed to mono, resampled to 8kHz and
  pre-chunked into 20ms frames, then pushed the moment the WebSocket opens. It never changes, so it
  does not need a model on the critical path. Without this the caller hears roughly 1.5 seconds of
  silence while STT, the LLM and TTS warm up.
- Services are not blindly pooled between calls. The speech recogniser and the TTS wrapper both hold
  per-session state, so reusing them across calls produces silence on the second call. Only the LLM
  client is safely a singleton.
- Turn-detection defaults are tuned on English speech. The thresholds here are deliberately slower,
  trading a little latency for not interrupting the caller.

### Local run

1. Create `.env` in the project root (see example below).
2. Install deps:

```bash
py -3.12 -m pip install -r requirements.txt
```

3. Start the bot:

```bash
py -3.12 pipecat_twilio_bot.py
```

4. Expose with ngrok (example):

```bash
ngrok http 8765
```

5. In Twilio, set your number's Voice webhook to:

```text
https://YOUR-NGROK-URL/webhook
```

### Deploying on Railway

- Railway will detect a Python app; set the start command to:

```bash
python pipecat_twilio_bot.py
```

- Configure env vars in Railway:
  - `GROQ_API_KEY`
  - `AZURE_SPEECH_KEY` (or `AZURE_SPEECH_API_KEY`)
  - `AZURE_SPEECH_REGION`
  - `TWILIO_ACCOUNT_SID`
  - `TWILIO_AUTH_TOKEN`
  - `PORT` (e.g. `8765`)

- After deploy, point Twilio to:

```text
https://YOUR-RAILWAY-URL/webhook
```

### .env example (do NOT commit this file)

```env
GROQ_API_KEY=your_groq_key
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=your_azure_region
AZURE_SPEECH_LANGUAGE=ar-KW
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth
PORT=8765
```

Optional:

```env
SERVER_HOST=0.0.0.0
LIMIT_CONCURRENCY=10
```
