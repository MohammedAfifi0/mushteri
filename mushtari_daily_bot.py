import asyncio
import os
from urllib.parse import urlparse

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.groq.tts import GroqTTSService

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)

from pipecat.transports.services.daily import DailyParams, DailyTransport


load_dotenv(override=True)


async def _get_daily_token(api_key: str, room_url: str) -> str:
    """Create a meeting token for the given Daily room using the REST API."""
    parsed = urlparse(room_url)
    room_name = parsed.path.strip("/").split("/")[-1]

    if not room_name:
        raise RuntimeError(f"Could not parse room name from DAILY_ROOM_URL={room_url}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "properties": {
            "room_name": room_name,
            "is_owner": False,
        }
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.daily.co/v1/meeting-tokens", json=payload, headers=headers
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(
                    f"Failed to create Daily meeting token ({resp.status}): {text}"
                )
            data = await resp.json()
            token = data.get("token")
            if not token:
                raise RuntimeError("Daily API response missing 'token' field")
            return token


async def main() -> None:
    """High‑quality WebRTC Mushtari bot using Daily + Groq (STT + LLM + Orpheus Arabic)."""

    groq_api_key = os.getenv("GROQ_API_KEY")
    daily_room_url = os.getenv("DAILY_ROOM_URL")
    daily_api_key = os.getenv("DAILY_API_KEY")

    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required in .env")
    if not daily_room_url:
        raise RuntimeError("DAILY_ROOM_URL is required in .env")
    if not daily_api_key:
        raise RuntimeError("DAILY_API_KEY is required in .env (from Daily dashboard)")

    logger.info("Requesting Daily meeting token via REST API...")
    daily_token = await _get_daily_token(daily_api_key, daily_room_url)
    logger.info("Got Daily meeting token.")

    logger.info("Starting Mushtari Daily bot (WebRTC, high‑quality Arabic)...")

    # Daily WebRTC transport (Opus @ 48kHz internally)
    transport = DailyTransport(
        daily_room_url,
        daily_token,
        "Salem Mushtari Bot",
        DailyParams(
            audio_in_enabled=True,
            audio_in_passthrough=True,
            audio_out_enabled=True,
            # Aggressive VAD for snappy turn‑taking
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(confidence=0.5, start_secs=0.2, stop_secs=0.4)
            ),
        ),
    )

    # Groq STT – multilingual Whisper turbo (better than Azure + fewer hops)
    stt = GroqSTTService(
        api_key=groq_api_key,
        model="whisper-large-v3-turbo",
        language="ar",  # focus on Arabic, still understands some mixed speech
    )

    # Groq LLM – same family you use on Twilio, but now in WebRTC path
    llm = GroqLLMService(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile",
    )

    # Groq TTS – Orpheus Arabic Saudi, Sultan voice, high‑quality 48kHz
    tts = GroqTTSService(
        api_key=groq_api_key,
        model_name="canopylabs/orpheus-arabic-saudi",
        voice_id="sultan",
        params=GroqTTSService.InputParams(
            language="ar",
            speed=1.2,  # snappier feel
        ),
    )

    # Salem system prompt – Arabic only, tuned for conversational WebRTC
    messages = [
        {
            "role": "system",
            "content": (
                "### IDENTITY\n"
                "Name: Salem (سالم). Role: Professional receptionist for تطبيق مشتري (Kuwait).\n"
                "Tone: طبيعي، واثق، محترم، لهجة كويتية واضحة.\n"
                "في أول رد فقط، عرّف بنفسك بـ: 'معاك سالم من تطبيق مشتري'.\n\n"
                "### BEHAVIOR\n"
                "- اسأل دائماً: 'تبي تشتري مشروع أو تبيع مشروع؟'.\n"
                "- استخدم كلمات: مشروع، شركة، محل تجاري. لا تستخدم 'شي'.\n"
                "- ردود قصيرة جداً (جملة أو جملتين كحد أقصى) عشان الإحساس يكون سريع.\n"
                "- لا تستخدم تشكيل (اكتب: مشتري، رخصة، مشروع بدون حركات).\n"
                "- لا تستخدم حروف إنجليزية أو كلام ممزوج مثل 'المercial'، استخدم عربي واضح مثل 'تجارية'.\n"
                "- استخدم علامات الترقيم: '،' داخل الجملة، '؟' للأسئلة، '.' للنهاية.\n\n"
                "### GREETING / HOW‑ARE‑YOU\n"
                "- إذا قال لك المستخدم: السلام، هلا، شلونك، كيف حالك، رد بجملة واحدة:\n"
                "  'معاك سالم من تطبيق مشتري، أنا بخير. تبي تشتري مشروع أو تبيع مشروع؟'\n"
                "- لا تضيف 'الحمدلله أنا ممتاز' أو أي سوالف إضافية.\n\n"
                "### FLOW\n"
                "1) رحّب بسرعة وادخل مباشرة في سؤال 'تبي تشتري مشروع أو تبيع مشروع؟'.\n"
                "2) إذا قال يبي يبيع: اسأل عن الرخصة، الموقع، الأرباح، السعر، خطوة خطوة.\n"
                "3) إذا قال يبي يشتري: اسأل عن الميزانية، نوع المشروع، المنطقة.\n"
            ),
        }
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            # Let Daily/Opus run at full quality; we don't force resampling here.
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    runner = PipelineRunner(handle_sigint=False, force_gc=True)

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info("Participant joined Daily room – starting Salem context.")
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"Participant left ({reason}), stopping task.")
        await task.cancel()

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())


