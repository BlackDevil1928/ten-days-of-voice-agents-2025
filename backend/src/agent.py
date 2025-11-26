#!/usr/bin/env python3
# ======================================================
# 💼 DAY 5: AI SALES DEVELOPMENT REP (SDR) - VipuXAi
# ======================================================
import logging
import json
import os
from datetime import datetime
from typing import Annotated, Literal, Optional, List
from dataclasses import dataclass, asdict

print("\n" + "💼" * 40)
print("🚀 VipuXAi SDR AGENT - READY")
print("💼" * 40 + "\n")

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

# Plugins
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# ------------------------
# Basic config
# ------------------------
logger = logging.getLogger("vipu_sdr")
logging.basicConfig(level=logging.INFO)
load_dotenv(".env.local")

# Files (safe paths)
LEADS_FILE = os.path.join(os.getcwd(), "leads_db.json")
FAQ_FILE = os.path.join(os.getcwd(), "store_faq.json")

# ------------------------
# VipuXAi FAQ (company-specific)
# ------------------------
DEFAULT_FAQ = [
    {
        "question": "What is VipuXAi?",
        "answer": "VipuXAi is a SaaS platform offering Voice AI solutions, voice agents, training, and enterprise consulting for automating customer support, bookings, and lead outreach."
    },
    {
        "question": "What products do you offer?",
        "answer": "We offer: (1) VipuVoice Platform — hosted voice-agent platform, (2) Custom Voice Agent development & integration, (3) Training & workshops, and (4) Managed consulting services."
    },
    {
        "question": "What are typical pricing options?",
        "answer": "We have three tiers: Starter (for SMBs), Pro (growing teams), and Enterprise (custom pricing). For exact quotes, we provide tailored proposals based on scope."
    },
    {
        "question": "Do you provide integrations?",
        "answer": "Yes — we integrate with CRMs (HubSpot, Salesforce), calendar systems, and ticketing platforms. We also offer MCP connectors for Notion, Todoist, and Zapier workflows."
    },
    {
        "question": "Can you build a voice agent for my business?",
        "answer": "Absolutely — we offer end-to-end development from design to production and monitoring. We'll typically ask about your use-case, traffic, required channels, and timeline."
    }
]

def ensure_faq_file():
    """Create default FAQ file if missing and return FAQ text for the prompt."""
    try:
        if not os.path.exists(FAQ_FILE):
            with open(FAQ_FILE, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_FAQ, f, indent=4, ensure_ascii=False)
        with open(FAQ_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Return joined FAQ text for prompt embedding
            return "\n".join([f"Q: {i['question']}\nA: {i['answer']}" for i in data])
    except Exception as e:
        logger.exception("Could not load or create FAQ file: %s", e)
        return "\n".join([f"Q: {q['question']}\nA: {q['answer']}" for q in DEFAULT_FAQ])

STORE_FAQ_TEXT = ensure_faq_file()

# ------------------------
# Lead structure
# ------------------------
@dataclass
class LeadProfile:
    name: Optional[str] = None
    company: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    use_case: Optional[str] = None
    team_size: Optional[str] = None
    timeline: Optional[str] = None

    def is_qualified(self) -> bool:
        return bool(self.name and self.email and self.use_case)

@dataclass
class Userdata:
    lead_profile: LeadProfile

# ------------------------
# Tools
# ------------------------
@function_tool
async def update_lead_profile(
    ctx: RunContext[Userdata],
    name: Annotated[Optional[str], Field(description="Customer's name")] = None,
    company: Annotated[Optional[str], Field(description="Customer's company")] = None,
    email: Annotated[Optional[str], Field(description="Customer's email")] = None,
    role: Annotated[Optional[str], Field(description="Job title")] = None,
    use_case: Annotated[Optional[str], Field(description="What they want to build or learn")] = None,
    team_size: Annotated[Optional[str], Field(description="Team size")] = None,
    timeline: Annotated[Optional[str], Field(description="Desired timeline")] = None,
) -> str:
    profile = ctx.userdata.lead_profile
    # Update fields if provided (non-empty)
    if name: profile.name = name.strip()
    if company: profile.company = company.strip()
    if email: profile.email = email.strip()
    if role: profile.role = role.strip()
    if use_case: profile.use_case = use_case.strip()
    if team_size: profile.team_size = team_size.strip()
    if timeline: profile.timeline = timeline.strip()

    logger.info("Lead updated: %s", profile)
    # Friendly confirmation message
    return f"Thanks — I noted that. Current lead summary: {profile.name or '—'} / {profile.company or '—'} / {profile.email or '—'}."

@function_tool
async def submit_lead_and_end(
    ctx: RunContext[Userdata],
) -> str:
    profile = ctx.userdata.lead_profile
    entry = asdict(profile)
    entry["timestamp"] = datetime.now().isoformat()

    data = []
    if os.path.exists(LEADS_FILE):
        try:
            with open(LEADS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        except Exception:
            data = []

    data.append(entry)
    try:
        with open(LEADS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info("Lead saved to %s", LEADS_FILE)
    except Exception as e:
        logger.exception("Failed to save lead: %s", e)
        return "Sorry — I couldn't save your information right now. Please try again later."

    # Friendly wrap-up message
    name = profile.name or "there"
    email = profile.email or "your email"
    use_case = profile.use_case or "your use case"
    return f"Thanks {name}! I have saved your details about {use_case}. We'll reach out at {email} with next steps. Goodbye!"

# ------------------------
# SDR Agent definition
# ------------------------
class SDRAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""
You are Sarah, a professional and friendly Sales Development Representative for VipuXAi.
Use the FAQ below to answer questions concisely and then qualify the lead conversationally.

FAQ:
{STORE_FAQ_TEXT}

Goals:
1. Answer user questions using the FAQ.
2. Qualify leads by collecting: name, email, company/role, use case, team size, timeline.
3. When the user provides information, call update_lead_profile with the fields provided.
4. When the user is ready to finish, call submit_lead_and_end.

Behavior rules:
- Always be polite and concise.
- Do not invent pricing or guarantees; if unsure, say "I'll confirm and email you".
- Prefer to capture at least Name + Email + Use Case before ending the call.
""",
            tools=[update_lead_profile, submit_lead_and_end],
        )

# ------------------------
# Safe TTS creation with fallback
# ------------------------
def create_tts_with_fallback():
    """
    Try to create Murf TTS with a safe voice name; if it fails, return a Silero TTS fallback.
    Use minimal params to avoid API errors.
    """
    murf_api_key = os.getenv("MURF_API_KEY")
    # Preferred simple voice names (Murf plugin variants may differ)
    preferred_voices = ["natalie", "matthew", "alicia", "ken"]
    if murf_api_key:
        for v in preferred_voices:
            try:
                t = murf.TTS(api_key=murf_api_key, voice=v)
                # Optionally test a small synthesis? (not doing network test here)
                logger.info("Using Murf TTS voice '%s'", v)
                return t
            except Exception as e:
                logger.warning("Murf TTS voice '%s' failed: %s", v, e)
    # Fallback to Silero TTS (local) if Murf is not available
    try:
        logger.info("Using Silero TTS fallback.")
        return silero.TTS()
    except Exception as e:
        logger.exception("Silero TTS also failed: %s", e)
        # As final fallback, create a dummy object with expected interface to avoid crashes
        class DummyTTS:
            async def synthesize(self, text):
                return b""  # silent
        return DummyTTS()

# ------------------------
# Entrypoint & session
# ------------------------
def prewarm(proc: JobProcess):
    # Load VAD if available for noise gating
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        proc.userdata["vad"] = None

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("Starting VipuXAi SDR session")

    userdata = Userdata(lead_profile=LeadProfile())

    # Build TTS with fallback
    tts_engine = create_tts_with_fallback()

    # Create STT and LLM objects with API keys
    deepgram_api = os.getenv("DEEPGRAM_API_KEY")
    google_api = os.getenv("GOOGLE_API_KEY")

    # Use minimal init forms that supply API keys if available
    stt_obj = deepgram.STT(model="nova-3", api_key=deepgram_api)
    llm_obj = google.LLM(model="gemini-2.5-flash", api_key=google_api)

    # Make AgentSession
    session = AgentSession(
        stt=stt_obj,
        llm=llm_obj,
        tts=tts_engine,
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )

    # Attach session so tools can access it if needed
    userdata.lead_profile  # just to ensure dataclass exists
    # store session in userdata if tools need access later
    try:
        # Some versions expect session reference available via userdata
        session_userdata = getattr(session, "userdata", None)
    except Exception:
        session_userdata = None

    # Start the agent; guard against TTS failures (try fallback)
    try:
        await session.start(
            agent=SDRAgent(),
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC()
            ),
        )
    except Exception as e:
        logger.exception("AgentSession.start failed: %s", e)
        # Try fallback: replace tts with Silero and restart
        try:
            fallback_tts = silero.TTS()
            session.tts = fallback_tts
            logger.info("Restarting session with Silero TTS fallback...")
            await session.start(
                agent=SDRAgent(),
                room=ctx.room,
                room_input_options=RoomInputOptions()
            )
        except Exception as e2:
            logger.exception("Restart with fallback TTS failed: %s", e2)
            # If restart fails, raise so process supervisor can handle it
            raise

    # connect context if required by environment
    try:
        await ctx.connect()
    except Exception:
        logger.debug("ctx.connect() may not be required in this environment.")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
