"""
Mushtari Voice Agent - Twilio Integration
Status: PRODUCTION READY (Updated for Groq 2026 Standards)
"""

from __future__ import annotations

import os
import asyncio
import io
import wave
import numpy as np
from scipy import signal
from datetime import datetime
from typing import Dict, Any

# Third-party imports
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
from loguru import logger
import uvicorn
from groq import Groq

# Pipecat core imports
from pipecat.frames.frames import (
    TTSAudioRawFrame,
    TTSSpeakFrame,
    EndFrame,
    TranscriptionFrame,
    LLMTextFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

# Pipecat services
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.tts import GroqTTSService
from pipecat.services.groq.stt import GroqSTTService

# Pipecat processors
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.processors.frame_processor import FrameProcessor

# Pipecat utilities
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

# Pipecat transport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

# Pipecat audio/VAD
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams

# Pipecat serializers
from pipecat.serializers.twilio import TwilioFrameSerializer

# Pipecat utilities
from pipecat.runner.utils import parse_telephony_websocket

load_dotenv()

# In-memory storage for lead data (will be replaced with DB later)
lead_storage: Dict[str, Dict[str, Any]] = {}

# Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY") or os.getenv("AZURE_SPEECH_API_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
# Force ar-KW for Kuwaiti Arabic (better for Kuwaiti dialect)
# Azure supports ar-KW for Kuwaiti Arabic recognition
# Default to ar-KW, but allow override via env var
env_language = os.getenv("AZURE_SPEECH_LANGUAGE")
if env_language:
    AZURE_SPEECH_LANGUAGE = env_language
    if env_language == "ar-SA":
        logger.warning(
            "AZURE_SPEECH_LANGUAGE is set to ar-SA. "
            "For better Kuwaiti dialect recognition, update your .env file: AZURE_SPEECH_LANGUAGE=ar-KW"
        )
else:
    AZURE_SPEECH_LANGUAGE = "ar-KW"  # Default to Kuwaiti Arabic

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")

# Reserved agent: Pre-generated initial greeting audio for instant pickup
# This eliminates TTS latency on first response by pre-generating audio at startup
INITIAL_GREETING_AUDIO: bytes | None = None
INITIAL_GREETING_SAMPLE_RATE: int = 48000  # Groq PlayTTS Arabic outputs at 48kHz


def pre_generate_greeting_audio() -> tuple[bytes | None, int]:
    """
    Pre-generate initial greeting using Groq PlayTTS Arabic (Nasser voice).
    This is called at startup to eliminate TTS latency on first response.
    
    Returns:
        Tuple of (audio_bytes, sample_rate) or (None, 48000) if generation fails
    """
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not found, cannot pre-generate greeting audio")
        return None, 48000
    
    try:
        logger.info("Pre-generating initial greeting audio with Groq Orpheus Arabic Saudi (sultan voice)...")
        client = Groq(api_key=GROQ_API_KEY)
        
        # Exact greeting text as specified
        greeting_text = "السلام عليكم، معاك سالم من تطبيق مُشْتَري، شلون أقدر أخدمك اليوم؟"
        
        # Generate audio using Groq Orpheus Arabic Saudi model (migrated from deprecated playai-tts-arabic)
        # Available voices: sultan, fahad, lulwa, noura
        response = client.audio.speech.create(
            model="canopylabs/orpheus-arabic-saudi",  # New model replacing deprecated playai-tts-arabic
            voice="sultan",  # Saudi male voice
            response_format="wav",
            input=greeting_text,
        )
        
        # Read the response content properly
        # Try different methods to read the binary response
        try:
            # Method 1: Try content attribute (most common)
            audio_bytes = response.content
        except AttributeError:
            try:
                # Method 2: Try read() if it's a file-like object
                audio_bytes = response.read()
            except AttributeError:
                try:
                    # Method 3: Try iter_bytes() if it's a streaming response
                    audio_bytes = b""
                    for chunk in response.iter_bytes():
                        audio_bytes += chunk
                except AttributeError:
                    logger.error("Could not read audio response - unknown response type")
                    return None, 48000
        
        # Parse WAV header to get sample rate
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
        
        logger.info(f"Reserved agent audio pre-generated successfully ({len(audio_bytes)} bytes, {sample_rate}Hz)")
        return audio_bytes, sample_rate
        
    except Exception as e:
        logger.error(f"Failed to pre-generate greeting audio: {e}")
        return None, 48000


# Initial pre-generated greeting is disabled now to keep behavior simple.
INITIAL_GREETING_AUDIO, INITIAL_GREETING_SAMPLE_RATE = None, 0


async def run_bot(websocket: WebSocket):
    """
    Main bot logic with proper Pipecat pipeline.
    
    Following docs pattern: parse websocket inside run_bot, then set up transport and pipeline.
    
    Args:
        websocket: FastAPI WebSocket connection from Twilio
    """
    # Parse Twilio WebSocket data (following docs pattern)
    transport_type, call_data = await parse_telephony_websocket(websocket)
    
    if transport_type != "twilio":
        logger.error(f"Expected Twilio transport, got {transport_type}")
        return
    
    # Extract Twilio-specific call data
    stream_sid = call_data.get("stream_id")
    call_sid = call_data.get("call_id")
    
    if not stream_sid:
        logger.error("Missing stream_id in call_data")
        return
    
    logger.info(
        f"Media stream started - Stream SID: {stream_sid}, "
        f"Call SID: {call_sid or 'N/A'}"
    )
    
    # Configure VAD - optimized for low-latency turn-taking on PSTN
    # stop_secs controls when we decide "user stopped speaking"
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            confidence=0.5,
            start_secs=0.25,  # quick to start
            stop_secs=0.4,    # faster finalization for lower latency
            min_volume=0.3,
        )
    )
    
    # Create Twilio serializer for proper audio handling and optional auto hang-up
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=TWILIO_ACCOUNT_SID or None,
        auth_token=TWILIO_AUTH_TOKEN or None,
    )
    
    # WebSocket transport for Twilio Media Streams
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=vad_analyzer,
            serializer=serializer,
        ),
    )
    
    # Initialize Groq STT (Whisper) for Arabic – closer to Vapi-style configs
    # This removes one extra network hop (Azure) and usually gives faster, cleaner text.
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is required")

    logger.info("Using Groq STT (whisper-large-v3-turbo, language=ar)")
    stt = GroqSTTService(
        api_key=GROQ_API_KEY,
        model="whisper-large-v3-turbo",
        language="ar",  # focus on Arabic / Gulf
    )
    
    # Initialize Groq LLM (openai/gpt-oss-120b) for high-quality, low-latency responses
    llm = GroqLLMService(
        api_key=GROQ_API_KEY,
        model="openai/gpt-oss-120b",
    )
    
    
    # Initialize Groq TTS for Arabic using PlayAI Arabic / Nasser
    # NOTE: This model is scheduled for deprecation by Groq, but still works as of now.
    # It often sounds more natural for Gulf Arabic than Orpheus for some content.
    tts = GroqTTSService(
        api_key=GROQ_API_KEY,
        model_name="playai-tts-arabic",
        voice_id="Nasser-PlayAI",
        params=GroqTTSService.InputParams(
            language="ar",
            speed=1.15,  # slightly faster than default for snappier feel
        ),
    )
    
    # Super simple Salem behavior for Twilio – low latency, natural flow
    messages = [
        {
            "role": "system",
            "content": (
                "انت سالم، موظف استقبال لتطبيق مشتري في الكويت.\n"
                "- ردودك قصيرة جداً وطبيعية، جملة أو جملتين فقط.\n"
                "- لا تعيد السلام ولا التعريف بنفسك أكثر من مرة.\n"
                "- أول رد فقط: 'معاك سالم من تطبيق مشتري، شلون أقدر أساعدك؟'.\n"
                "- إذا المتصل يبي يبيع مشروع → اسأله بهدوء عن نوع المشروع، بعدين الموقع، بعدين السعر.\n"
                "- إذا المتصل يبي يشتري مشروع → اسأله عن الميزانية، بعدين نوع المشروع اللي يدوره.\n"
                "- إذا كان بس يسأل أو يستفسر عن التطبيق → جاوب باختصار وارجع تسأله إذا يبي يشتري أو يبيع مشروع.\n"
                "- استخدم لهجة كويتية بسيطة بدون كلمات إنجليزية وبدون تشكيل.\n"
                "- لا تستخدم كلمات غريبة أو جمل طويلة، خلّ الأسلوب واضح وسلس زي المكالمة العادية.\n"
            ),
        },
    ]
    
    # Create context and aggregator (universal pattern from Pipecat docs)
    # LLMContextAggregatorPair gives us .user() and .assistant() for the pipeline.
    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)
    
    # Function to save lead data (will be connected to DB later)
    def save_lead_data(call_sid: str, data: Dict[str, Any]):
        """Save lead information to storage (temporary - will connect to DB async later)"""
        if call_sid not in lead_storage:
            lead_storage[call_sid] = {
                "call_sid": call_sid,
                "timestamp": datetime.now().isoformat(),
                "lead_quality": None,  # "Hot" or "Cold"
                "data": {}
            }
        lead_storage[call_sid]["data"].update(data)
        logger.info(f"Lead data saved for {call_sid}: {data}")
    
    # Custom frame logger to debug what's happening in the pipeline
    class FrameLogger(FrameProcessor):
        """Log all frames flowing through pipeline for debugging"""
        def __init__(self, logger_name: str):
            super().__init__()
            self.logger_name = logger_name
            
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            
            # Log important frames for debugging
            if isinstance(frame, TranscriptionFrame):
                logger.info(f"[{self.logger_name}] TranscriptionFrame: {frame.text}")
            elif isinstance(frame, LLMTextFrame):
                logger.info(f"[{self.logger_name}] LLMTextFrame: '{frame.text}'")
            elif isinstance(frame, TTSTextFrame):
                logger.info(f"[{self.logger_name}] TTSTextFrame: '{frame.text}'")
            elif isinstance(frame, TTSAudioRawFrame):
                logger.info(f"[{self.logger_name}] TTSAudioRawFrame: {len(frame.audio)} bytes")
            elif hasattr(frame, '__class__'):
                frame_type = frame.__class__.__name__
                if 'LLM' in frame_type or 'TTS' in frame_type or 'Transcription' in frame_type:
                    logger.debug(f"[{self.logger_name}] {frame_type}")
            
            await self.push_frame(frame, direction)
    
    # Create frame loggers for debugging
    logger_stt = FrameLogger("STT_OUT")
    logger_llm_in = FrameLogger("LLM_IN")
    logger_llm_out = FrameLogger("LLM_OUT")
    logger_tts_in = FrameLogger("TTS_IN")
    logger_tts_out = FrameLogger("TTS_OUT")
    
    # Build the pipeline - STANDARD Pipecat streaming (no bridge needed)
    # Groq GPT-OSS streams LLMTextFrames which Groq TTS can consume directly
    pipeline = Pipeline(
        [
            transport.input(),                 # Audio input from Twilio
            stt,                               # Azure Speech-to-Text (Arabic)
            logger_stt,                        # DEBUG: Log STT output
            context_aggregator.user(),         # User → context
            logger_llm_in,                     # DEBUG: Log frames before LLM
            llm,                               # Groq openai/gpt-oss-120b processes context → streams LLMTextFrames
            logger_llm_out,                    # DEBUG: Log LLM output
            logger_tts_in,                     # DEBUG: Log frames before TTS
            tts,                               # Groq TTS processes LLMTextFrames → creates TTSAudioRawFrames
            logger_tts_out,                    # DEBUG: Log TTS output
            transport.output(),                # Audio output to Twilio
            context_aggregator.assistant(),    # Assistant → context
        ]
    )
    
    # Create task with interruption support and optimized for low latency
    # Following Twilio docs: Set audio_out_sample_rate to 8000 (Twilio requirement)
    # Pipeline will automatically resample Groq TTS 48kHz output to 8kHz for Twilio
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,   # Twilio Media Streams uses 8kHz (docs requirement)
            audio_out_sample_rate=8000,  # Twilio output rate (docs requirement) - pipeline resamples from Groq's 48kHz
            allow_interruptions=True,     # Enable natural interruptions
            enable_metrics=True,          # Track performance metrics
            enable_usage_metrics=True,    # Track token usage
        ),
    )
    
    
    # Run the pipeline
    runner = PipelineRunner()
    
    try:
        # Start pipeline in background
        # FastAPIWebsocketTransport will automatically start reading WebSocket messages
        # in a background task when the pipeline runs
        logger.info("Starting pipeline...")
        pipeline_task = asyncio.create_task(runner.run(task))
        
        # Immediately hand control to the normal pipeline (no pre-recorded greeting).
        # Salem will speak only when the LLM/TTS respond.

        # Wait for pipeline to complete (runs until call ends)
        # The transport will continue reading WebSocket messages in the background
        await pipeline_task
        
    except asyncio.CancelledError:
        logger.info("Pipeline cancelled - call ended")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
    finally:
        # Save conversation summary when call ends
        if call_sid:
            try:
                conversation_summary = {
                    "call_ended": datetime.now().isoformat(),
                    "reason": "call_ended",
                }
                save_lead_data(call_sid, conversation_summary)
                logger.info(f"Conversation summary saved for call {call_sid}")
            except Exception as e:
                logger.error(f"Failed to save conversation summary: {e}")


# FastAPI application
app = FastAPI(title="Mushtari Voice Agent", version="1.0.0")


@app.get("/")
async def root():
    return {"status": "Mushtari Voice Agent Running"}


@app.get("/leads")
async def get_leads():
    """Get all stored lead data (for testing - will be replaced with DB later)"""
    return {"leads": lead_storage, "count": len(lead_storage)}


@app.get("/leads/{call_sid}")
async def get_lead(call_sid: str):
    """Get specific lead data by call SID"""
    if call_sid in lead_storage:
        return lead_storage[call_sid]
    return {"error": "Lead not found"}


@app.post("/incoming-call")
@app.post("/twilio/voice")
@app.post("/webhook")
@app.post("/")
async def handle_incoming_call(request: Request):
    """
    Twilio webhook handler for incoming calls.
    Returns TwiML that connects to WebSocket endpoint.
    """
    try:
        logger.info("Twilio webhook received - processing incoming call")
        
        # Get server host from environment or request headers
        # Priority: Valid SERVER_HOST env var > X-Forwarded-Host header > Host header
        # Always prefer request headers if SERVER_HOST is not a valid domain
        server_host_env = os.getenv("SERVER_HOST", "")
        
        # Check if SERVER_HOST is a valid domain (contains dot and is not just a partial name)
        is_valid_domain = server_host_env and "." in server_host_env and len(server_host_env) > 3
        
        if is_valid_domain:
            host = server_host_env
        else:
            # Use request headers - these are more reliable for ngrok
            host = (
                request.headers.get("x-forwarded-host") 
                or request.headers.get("host")
                or server_host_env  # Fallback to env var even if invalid
            )
        
        # Ensure host doesn't include protocol
        if host:
            if host.startswith("http://"):
                host = host.replace("http://", "")
            if host.startswith("https://"):
                host = host.replace("https://", "")
            # Remove port if present (ngrok URLs don't need ports)
            if ":" in host and not host.startswith("["):  # IPv6 check
                host = host.split(":")[0]
        
        # Determine protocol - ALWAYS use wss for ngrok and production
        # ngrok always uses HTTPS/WSS, so detect ngrok domains
        is_ngrok = host and ("ngrok" in host.lower() or "ngrok-free.dev" in host.lower() or "ngrok.io" in host.lower())
        is_production = host and ("railway.app" in host or "vercel.app" in host or "herokuapp.com" in host)
        
        protocol = "wss" if (is_ngrok or is_production) else "ws"
        
        # Final validation - if host looks invalid, log warning
        if not host or len(host) < 3 or "." not in host:
            logger.warning(f"Invalid host detected: '{host}'. Using request headers instead.")
            host = request.headers.get("host") or request.headers.get("x-forwarded-host") or "localhost"
            protocol = "wss" if "ngrok" in host.lower() else "ws"
        
        logger.info(f"Incoming call detected - Host: {host}, Protocol: {protocol}")
        logger.info(f"Connecting to {protocol}://{host}/ws")
        
        # Return TwiML that starts media stream.
        # We intentionally skip <Say> so that all audio comes from Groq TTS (Orpheus Arabic Saudi - sultan voice),
        # instead of Twilio's built-in TTS voice.
        twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{protocol}://{host}/ws">
            <Parameter name="aCustomParameter" value="customValue"/>
        </Stream>
    </Connect>
</Response>'''
        
        logger.info("Returning TwiML response to Twilio")
        return Response(content=twiml, media_type="application/xml")
        
    except Exception as e:
        logger.error(f"Error in webhook handler: {e}", exc_info=True)
        # Return error TwiML so Twilio doesn't show busy
        error_twiml = '''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">Sorry, there was an error connecting. Please try again later.</Say>
</Response>'''
        return Response(content=error_twiml, media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams.
    
    Following docs pattern: accept websocket and pass to run_bot.
    run_bot will parse the websocket and set up the pipeline.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted from Twilio")
    
    try:
        # Pass websocket to run_bot - it will parse and set up everything
        await run_bot(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        logger.info("Closing WebSocket connection")
        try:
            await websocket.close()
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")


if __name__ == "__main__":
    # Get port from environment or use default
    # Default to 8765 to match ngrok forwarding
    port = int(os.getenv("PORT", 8765))
    
    logger.info(f"Starting Mushtari Voice Agent on port {port}")
    logger.info("Configuration:")
    logger.info(f"  - LLM: Groq openai/gpt-oss-120b")
    logger.info(f"  - STT: Groq Whisper (whisper-large-v3-turbo, ar)")
    logger.info(f"  - TTS: Groq PlayAI Arabic (playai-tts-arabic, Nasser-PlayAI)")
    logger.info(f"  - VAD: Silero (confidence=0.5, start=0.25s, stop=0.4s)")
    
    # Run FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
