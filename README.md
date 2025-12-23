## Mushteri Voice Agent (Twilio + Groq + Pipecat)

Python voice bot for Mushteri, running over Twilio Media Streams with:

- STT: Groq Whisper `whisper-large-v3-turbo` (Arabic)
- LLM: Groq `llama-3.3-70b-versatile`
- TTS: Groq PlayAI Arabic `playai-tts-arabic` (Nasser)

Main entrypoint: `pipecat_twilio_bot.py`

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
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth
AZURE_SPEECH_KEY=unused_now
AZURE_SPEECH_REGION=unused_now
AZURE_SPEECH_LANGUAGE=ar-KW
PORT=8765
```


