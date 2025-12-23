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
    InterimTranscriptionFrame,
    LLMTextFrame,
    AggregatedTextFrame,
    TTSTextFrame,
    InputAudioRawFrame,
    LLMMessagesAppendFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

# Pipecat services
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.tts import GroqTTSService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.azure import AzureSTTService

# Pipecat processors
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

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

# Reserved agent: Pre-initialized services for low-latency startup
# Global service instances that are reused across calls to reduce initialization time
_reserved_services: Dict[str, Any] = {}


def initialize_reserved_services():
    """
    Pre-initialize services at startup to reduce cold-start latency.
    This is a simple reserved agent pattern for self-hosted deployments.
    """
    global _reserved_services
    
    try:
        logger.info("Initializing reserved agent services...")
        
        # Pre-initialize Azure STT (most time-consuming)
        if AZURE_SPEECH_KEY and AZURE_SPEECH_REGION:
            azure_language = os.getenv("AZURE_SPEECH_LANGUAGE", "ar-SA")
            logger.info(f"Pre-initializing Azure STT (language={azure_language})...")
            _reserved_services['stt'] = AzureSTTService(
                api_key=AZURE_SPEECH_KEY,
                region=AZURE_SPEECH_REGION,
                language=azure_language,
                audio_passthrough=True,
            )
            logger.info("✅ Azure STT pre-initialized")
        
        # Pre-initialize Groq LLM
        if GROQ_API_KEY:
            logger.info("Pre-initializing Groq LLM...")
            _reserved_services['llm'] = GroqLLMService(
                api_key=GROQ_API_KEY,
                model="openai/gpt-oss-120b",
            )
            logger.info("✅ Groq LLM pre-initialized")
            
            # Pre-initialize Groq TTS
            logger.info("Pre-initializing Groq TTS...")
            _reserved_services['tts'] = GroqTTSService(
                api_key=GROQ_API_KEY,
                model_name="playai-tts-arabic",
                voice_id="Nasser-PlayAI",
                params=GroqTTSService.InputParams(
                    language="ar",
                    speed=1.15,
                ),
            )
            logger.info("✅ Groq TTS pre-initialized")
        
        logger.info("Reserved agent services initialized successfully")
        
    except Exception as e:
        logger.warning(f"Failed to initialize reserved services (will create per-call): {e}")
        _reserved_services = {}


def get_reserved_service(service_name: str):
    """
    Get a pre-initialized service instance, or return None if not available.
    Note: Services are reused across calls. This works if services are stateless
    or properly handle concurrent usage. If issues occur, we'll create new instances per call.
    """
    service = _reserved_services.get(service_name)
    if service:
        logger.debug(f"Using reserved {service_name} service")
    return service


def pre_generate_greeting_frames() -> list[TTSAudioRawFrame] | None:
    """
    Pre-generate initial greeting audio and convert to TTSAudioRawFrame chunks.
    This is called at startup to eliminate TTS latency on first response.
    Audio is resampled to 8kHz and chunked for immediate queuing.
    
    Returns:
        List of TTSAudioRawFrame chunks ready to queue, or None if generation fails
    """
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not found, cannot pre-generate greeting audio")
        return None
    
    try:
        logger.info("Pre-generating initial greeting audio frames...")
        client = Groq(api_key=GROQ_API_KEY)
        
        # Greeting text matching the system prompt
        greeting_text = INITIAL_GREETING_TEXT
        
        # Use same TTS model/voice as the service (playai-tts-arabic with Nasser-PlayAI)
        # Note: Using Groq audio.speech API directly (same as TTS service uses internally)
        response = client.audio.speech.create(
            model="playai-tts-arabic",
            voice="Nasser-PlayAI",
            response_format="wav",
            input=greeting_text,
        )
        
        # Read the response content
        try:
            audio_bytes = response.content
        except AttributeError:
            try:
                audio_bytes = response.read()
            except AttributeError:
                try:
                    audio_bytes = b""
                    for chunk in response.iter_bytes():
                        audio_bytes += chunk
                except AttributeError:
                    logger.error("Could not read audio response")
                    return None
        
        # Parse WAV and convert to 8kHz PCM for Twilio
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
            source_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            pcm_data = wav_file.readframes(wav_file.getnframes())
        
        # Convert to numpy array
        audio_array = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Stereo to mono if needed
        if n_channels == 2:
            audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
        
        # Resample to 8kHz for Twilio (if not already 8kHz)
        target_rate = 8000
        if source_rate != target_rate:
            num_samples = int(len(audio_array) * target_rate / source_rate)
            audio_array = signal.resample(audio_array, num_samples).astype(np.int16)
        
        # Convert back to bytes
        final_bytes = audio_array.tobytes()
        
        # Chunk into 20ms frames (320 bytes @ 8kHz = 20ms)
        chunk_size = 320
        frames = []
        for i in range(0, len(final_bytes), chunk_size):
            chunk = final_bytes[i:i + chunk_size]
            if len(chunk) > 0:
                # Pad last chunk if needed to maintain consistent chunk size
                if len(chunk) < chunk_size:
                    chunk = chunk + b'\x00' * (chunk_size - len(chunk))
                frames.append(TTSAudioRawFrame(
                    audio=chunk,
                    sample_rate=target_rate,
                    num_channels=1
                ))
        
        logger.info(f"✅ Pre-generated {len(frames)} greeting frames ({len(final_bytes)} bytes, {target_rate}Hz)")
        return frames
        
    except Exception as e:
        logger.error(f"Failed to pre-generate greeting audio: {e}", exc_info=True)
        return None


# Global storage for pre-generated greeting frames and text
INITIAL_GREETING_FRAMES: list[TTSAudioRawFrame] | None = None
INITIAL_GREETING_TEXT = "هلا، معاك سالم من تطبيق مشتري، شلون أقدر أساعدك؟"


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
    
    # Initialize lead data tracking for this call
    if call_sid:
        save_lead_data(call_sid, {"call_started": datetime.now().isoformat()})
    
    # Configure VAD - optimized for low-latency turn-taking on PSTN
    # Balance between quick response and not cutting off speech
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            confidence=0.4,    # Lower threshold for better detection in noisy PSTN
            start_secs=0.2,    # Faster start detection for lower latency
            stop_secs=0.5,     # Balanced pause detection (not too short, not too long)
            min_volume=0.2,    # Lower minimum volume for better sensitivity
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
    
    # Use reserved services if available, otherwise create new instances
    # Reserved services reduce cold-start latency by pre-initializing at startup
    # Each call still gets its own pipeline, so services should handle concurrent usage
    
    # Initialize Azure STT for Arabic (KSA)
    if not AZURE_SPEECH_KEY:
        raise ValueError("AZURE_SPEECH_API_KEY or AZURE_SPEECH_KEY environment variable is required")
    if not AZURE_SPEECH_REGION:
        raise ValueError("AZURE_SPEECH_REGION environment variable is required")

    # Use ar-SA as requested (Saudi Arabic)
    azure_language = os.getenv("AZURE_SPEECH_LANGUAGE", "ar-SA")
    reserved_stt = get_reserved_service('stt')
    if reserved_stt:
        stt = reserved_stt
        logger.info(f"Using reserved Azure STT (language={azure_language})")
    else:
        logger.info(f"Creating new Azure STT (language={azure_language})")
        stt = AzureSTTService(
            api_key=AZURE_SPEECH_KEY,
            region=AZURE_SPEECH_REGION,
            language=azure_language,
            audio_passthrough=True,  # Allow audio to continue downstream for continuous recognition
        )
    
    # Initialize Groq LLM (openai/gpt-oss-120b) for high-quality, low-latency responses
    reserved_llm = get_reserved_service('llm')
    if reserved_llm:
        llm = reserved_llm
        logger.info("Using reserved Groq LLM")
    else:
        logger.info("Creating new Groq LLM")
        llm = GroqLLMService(
            api_key=GROQ_API_KEY,
            model="openai/gpt-oss-120b",
        )
    
    # Initialize Groq TTS for Arabic using PlayAI Arabic / Nasser
    # NOTE: This model is scheduled for deprecation by Groq, but still works as of now.
    # It often sounds more natural for Gulf Arabic than Orpheus for some content.
    # Since we use LLMTextProcessor upstream, TTS will receive AggregatedTextFrames
    # and should NOT use its own text aggregator (to avoid double aggregation issues)
    reserved_tts = get_reserved_service('tts')
    if reserved_tts:
        tts = reserved_tts
        logger.info("Using reserved Groq TTS")
    else:
        logger.info("Creating new Groq TTS")
        tts = GroqTTSService(
            api_key=GROQ_API_KEY,
            model_name="playai-tts-arabic",
            voice_id="Nasser-PlayAI",
            params=GroqTTSService.InputParams(
                language="ar",
                speed=1.2,  # Slightly faster for lower perceived latency
            ),
        )
    
    # نظام محادثة طبيعي ومتوازن – ردود متوسطة الطول، طبيعية، ومريحة
    messages = [
        {
            "role": "system",
            "content": (
                "انت سالم، موظف استقبال ودود ومحترف في تطبيق مُشْتَري في الكويت.\n\n"
                "**مهم جداً:**\n"
                "- لقد قلت الترحيب الأولي بالفعل: 'هلا، معاك سالم من تطبيق مشتري، شلون أقدر أساعدك؟'\n"
                "- لا تكرر الترحيب أو تعيد نفس الجملة. استمع لما يقوله المتصل ورد عليه بشكل طبيعي.\n\n"
                "**الأسلوب واللهجة:**\n"
                "- تكلم بلهجة كويتية طبيعية وبسيطة، بدون كلمات إنجليزية وبدون تشكيل.\n"
                "- ردودك متوسطة الطول: ليست قصيرة جداً (مثل 'نعم' فقط) وليست طويلة جداً (أكثر من 3-4 جمل).\n"
                "- خلك طبيعي ومريح، مثل محادثة عادية مع صديق أو زميل.\n"
                "- استخدم تعابير كويتية طبيعية مثل 'هلا'، 'شلونك'، 'يعطيك العافية'.\n\n"
                "**التدفق الطبيعي للمحادثة:**\n"
                "- استمع جيداً لما يقوله المتصل قبل الرد.\n"
                "- إذا قال إنه يبي يبيع مشروع:\n"
                "  * ابدأ بجملة تأكيد ودودة مثل 'ممتاز' أو 'زين'.\n"
                "  * اسأله عن نوع المشروع (مطعم، كوفي، صالون، إلخ) وسجّل المعلومات.\n"
                "  * بعد ما يجيب، اسأله عن الموقع وسجّله.\n"
                "  * بعدين اسأله عن السعر التقريبي والربحية الشهرية وسجّلها.\n"
                "- إذا قال إنه يبي يشتري مشروع:\n"
                "  * ابدأ بجملة تأكيد ودودة.\n"
                "  * اسأله: 'أي نوع من المشاريع تبي تشتريه؟ مثلاً مطعم، كوفي، صالون، ولا شي ثاني؟' وسجّل نوع المشروع.\n"
                "  * بعد ما يجيب، اسأله عن ميزانيته المتوقعة وسجّلها.\n"
                "- إذا بس يستفسر عن التطبيق:\n"
                "  * جاوبه باختصار ووضوح (2-3 جمل كافية).\n"
                "  * بعدين اسأله: 'حاب تشتري مشروع ولا تبيع مشروع؟'\n\n"
                "**مبادئ مهمة:**\n"
                "- لا تستخدم كلمات غريبة أو غير مألوفة.\n"
                "- استعمل كلمات بسيطة وواضحة: تشتري، تبيع، تدير، تبي تعرض مشروعك.\n"
                "- لا تطوّل في الرد، لكن أيضاً لا تقصر جداً. اهدف لرد طبيعي ومريح.\n"
                "- خلك صبور ومتفهم. إذا المتصل محتاج توضيح، وضّح له بوضوح.\n"
                "- لا تكرر نفس الجملة مرتين متتاليتين (مثل 'يعطيك العافية' أكثر من مرة).\n"
                "- إذا المتصل قال شكراً أو مع السلامة، اختم بلطف: 'يعطيك العافية، تشرفنا فيك'.\n"
            ),
        },
    ]
    
    # Create context and aggregator (universal pattern from Pipecat docs)
    # LLMContextAggregatorPair gives us .user() and .assistant() for the pipeline.
    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    # LLM text processor – يجمع التوكِنات المتقطعة ويطلع جُمَل كاملة للـ TTS
    # SimpleTextAggregator يجمّع النص حتى نهاية الجملة (؟، .، !) قبل إرساله للـ TTS
    # هذا يحسّن النطق ويقلّل التقطيع
    llm_text_processor = LLMTextProcessor(
        text_aggregator=SimpleTextAggregator()
    )
    
    # Custom frame logger to debug what's happening in the pipeline
    class FrameLogger(FrameProcessor):
        """Log all frames flowing through pipeline for debugging"""
        def __init__(self, logger_name: str):
            super().__init__()
            self.logger_name = logger_name
            
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            
            # Log important frames for debugging (errors in logging shouldn't break pipeline)
            try:
                if isinstance(frame, TranscriptionFrame):
                    logger.info(f"[{self.logger_name}] TranscriptionFrame: {frame.text}")
                elif isinstance(frame, InterimTranscriptionFrame):
                    logger.debug(f"[{self.logger_name}] InterimTranscriptionFrame: {frame.text}")
                elif isinstance(frame, InputAudioRawFrame):
                    # Only log occasionally to avoid spam (every 50 frames = ~1 second at 8kHz)
                    if not hasattr(self, '_audio_frame_count'):
                        self._audio_frame_count = 0
                    self._audio_frame_count += 1
                    if self._audio_frame_count % 50 == 0:
                        logger.debug(f"[{self.logger_name}] InputAudioRawFrame: {len(frame.audio)} bytes (frame {self._audio_frame_count})")
                elif isinstance(frame, LLMTextFrame):
                    logger.debug(f"[{self.logger_name}] LLMTextFrame: '{frame.text}'")
                elif isinstance(frame, AggregatedTextFrame):
                    # Safely access text attribute
                    text = getattr(frame, 'text', '')
                    logger.info(f"[{self.logger_name}] AggregatedTextFrame: '{text}'")
                elif isinstance(frame, TTSTextFrame):
                    logger.info(f"[{self.logger_name}] TTSTextFrame: {frame.text}")
                elif isinstance(frame, TTSAudioRawFrame):
                    logger.debug(f"[{self.logger_name}] TTSAudioRawFrame: {len(frame.audio)} bytes")
                elif hasattr(frame, '__class__'):
                    frame_type = frame.__class__.__name__
                    # Log all frame types for STT debugging
                    if 'Transcription' in frame_type or 'Interim' in frame_type or 'User' in frame_type:
                        logger.debug(f"[{self.logger_name}] {frame_type}")
            except Exception as e:
                # Log error but don't break the pipeline
                logger.warning(f"[{self.logger_name}] Error logging frame: {e}")
            
            await self.push_frame(frame, direction)
    
    # Create frame loggers for debugging
    logger_audio_in = FrameLogger("AUDIO_IN")  # Log audio before STT
    logger_stt = FrameLogger("STT_OUT")
    logger_llm_in = FrameLogger("LLM_IN")
    logger_llm_out = FrameLogger("LLM_OUT")
    logger_tts_in = FrameLogger("TTS_IN")
    logger_tts_out = FrameLogger("TTS_OUT")
    
    # Build the pipeline – STT → context → LLM (streaming) → sentence aggregation → TTS
    pipeline = Pipeline(
        [
            transport.input(),                 # Audio input from Twilio
            logger_audio_in,                   # DEBUG: Log audio input
            stt,                               # Azure Speech-to-Text (Arabic)
            logger_stt,                        # DEBUG: Log STT output
            context_aggregator.user(),         # User → context
            logger_llm_in,                     # DEBUG: Log frames before LLM
            llm,                               # Groq openai/gpt-oss-120b (streams LLMTextFrames)
            llm_text_processor,                # Aggregate LLMTextFrames into AggregatedTextFrames (sentences)
            logger_llm_out,                    # DEBUG: Log aggregated text frames
            logger_tts_in,                     # DEBUG: Log frames before TTS
            tts,                               # Groq TTS processes AggregatedTextFrames → creates TTSAudioRawFrames + TTSTextFrames
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
        
        # Queue pre-generated greeting frames immediately to mask cold start latency
        # This plays the greeting while STT/LLM are initializing in the background
        global INITIAL_GREETING_FRAMES, INITIAL_GREETING_TEXT
        if INITIAL_GREETING_FRAMES:
            logger.info(f"Queueing {len(INITIAL_GREETING_FRAMES)} pre-generated greeting frames...")
            # Small delay to ensure pipeline is ready to receive frames
            await asyncio.sleep(0.15)  # Slightly longer delay to ensure pipeline is fully ready
            await task.queue_frames(INITIAL_GREETING_FRAMES)
            
            # IMPORTANT: Add greeting to context so LLM knows it was already spoken
            # This prevents the LLM from repeating the greeting
            greeting_message = {"role": "assistant", "content": INITIAL_GREETING_TEXT}
            await task.queue_frames([
                LLMMessagesAppendFrame([greeting_message], run_llm=False)
            ])
            logger.info("✅ Greeting frames queued and added to context - LLM will not repeat greeting")
        else:
            logger.warning("No pre-generated greeting frames available - using TTSSpeakFrame fallback")
            # Fallback: use TTSSpeakFrame if pre-generation failed
            await asyncio.sleep(0.15)
            await task.queue_frames([TTSSpeakFrame(text=INITIAL_GREETING_TEXT)])
            # Add to context for fallback too
            greeting_message = {"role": "assistant", "content": INITIAL_GREETING_TEXT}
            await task.queue_frames([
                LLMMessagesAppendFrame([greeting_message], run_llm=False)
            ])

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


@app.on_event("startup")
async def startup_event():
    """Initialize reserved agent services and pre-generate greeting on startup for low-latency cold starts"""
    global INITIAL_GREETING_FRAMES
    
    # Initialize services in parallel with greeting generation
    initialize_reserved_services()
    
    # Pre-generate greeting frames in background (non-blocking)
    # This masks cold start latency by playing greeting immediately on call start
    INITIAL_GREETING_FRAMES = pre_generate_greeting_frames()


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
    logger.info(f"  - STT: Azure Speech Services ({os.getenv('AZURE_SPEECH_LANGUAGE', 'ar-SA')})")
    logger.info(f"  - TTS: Groq PlayAI Arabic (playai-tts-arabic, Nasser-PlayAI)")
    logger.info(f"  - VAD: Silero (confidence=0.4, start=0.25s, stop=0.6s)")
    logger.info(f"  - Reserved Agent: Enabled (services pre-initialized for low latency)")
    
    # Run FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
