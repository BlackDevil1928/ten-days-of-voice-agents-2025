#!/usr/bin/env python3
import logging
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Annotated, List, Optional

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

from livekit.plugins import murf, silero, google, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Basic config
# -------------------------
logger = logging.getLogger("wellness_agent")
load_dotenv(".env.local")

# -------------------------
# Data models
# -------------------------
@dataclass
class CheckInState:
    mood: Optional[str] = None           # free-text mood, or numeric string like "3/5"
    energy: Optional[str] = None         # free-text energy
    objectives: List[str] = field(default_factory=list)
    advice_given: Optional[str] = None

    def is_complete(self) -> bool:
        return bool(self.mood and self.energy and len(self.objectives) > 0)

    def to_dict(self):
        return asdict(self)


@dataclass
class Userdata:
    current_checkin: CheckInState
    history_summary: str
    session_start: datetime = field(default_factory=datetime.now)


# -------------------------
# Persistence (JSON)
# -------------------------
LOG_FILE = "wellness_log.json"

def get_log_path() -> str:
    # store in current working directory of the backend (safe)
    return os.path.join(os.getcwd(), LOG_FILE)

def load_history() -> List[dict]:
    path = get_log_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning("Could not read history file: %s", e)
        return []

def save_checkin_entry(entry: CheckInState) -> None:
    path = get_log_path()
    history = load_history()
    record = {
        "timestamp": datetime.now().isoformat(),
        "mood": entry.mood,
        "energy": entry.energy,
        "objectives": entry.objectives,
        "summary": entry.advice_given,
    }
    history.append(record)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
        logger.info("Saved check-in to %s", path)
    except Exception as e:
        logger.exception("Failed to save check-in: %s", e)


# -------------------------
# Helper: parse numeric mood (optional)
# -------------------------
def parse_mood_numeric(mood_str: Optional[str]) -> Optional[float]:
    """
    Try to parse mood like '3/5' or '4' into float 0-5 scale.
    Returns float or None if not parseable.
    """
    if not mood_str:
        return None
    s = mood_str.strip()
    # formats: "3/5", "4", "4.0", "5/10" (normalize to 0-5)
    try:
        if "/" in s:
            num, den = s.split("/", 1)
            num = float(num.strip())
            den = float(den.strip())
            if den == 0:
                return None
            # convert to 0-5 scale
            return float(num) * 5.0 / float(den)
        else:
            # assume 0-5 or 0-10 — clamp if needed
            val = float(s)
            if 0 <= val <= 5:
                return val
            if 0 <= val <= 10:
                return val * 0.5
            # otherwise normalize roughly
            return None
    except Exception:
        return None


# -------------------------
# Tool functions (exposed to the LLM)
# -------------------------
@function_tool
async def record_mood_and_energy(
    ctx: RunContext[Userdata],
    mood: Annotated[str, Field(description="User mood (text or simple scale, e.g. '3/5')")],
    energy: Annotated[str, Field(description="User energy level (text)")]
) -> str:
    ctx.userdata.current_checkin.mood = mood
    ctx.userdata.current_checkin.energy = energy
    logger.info("Recorded mood=%s energy=%s", mood, energy)
    return f"Got it — mood recorded as '{mood}' and energy as '{energy}'."

@function_tool
async def record_objectives(
    ctx: RunContext[Userdata],
    objectives: Annotated[List[str], Field(description="1-3 objectives for the day")]
) -> str:
    # limit to 1-3 items (truncate if more)
    cleaned = [o.strip() for o in objectives if o and o.strip()]
    ctx.userdata.current_checkin.objectives = cleaned[:3]
    logger.info("Recorded objectives: %s", ctx.userdata.current_checkin.objectives)
    return f"I've saved {len(ctx.userdata.current_checkin.objectives)} objectives."

@function_tool
async def complete_checkin(
    ctx: RunContext[Userdata],
    final_advice_summary: Annotated[str, Field(description="Short one-sentence summary/advice")]
) -> str:
    state = ctx.userdata.current_checkin
    state.advice_given = final_advice_summary

    if not state.is_complete():
        return "I can't finish the check-in yet. I still need your mood, energy, or at least one goal."

    save_checkin_entry(state)

    recap = (
        f"Here's your recap: You are feeling {state.mood} and your energy is {state.energy}. "
        f"Your goals are: {', '.join(state.objectives)}. "
        f"Remember: {final_advice_summary}"
    )
    logger.info("Check-in complete.")
    return recap

@function_tool
async def get_weekly_summary(
    ctx: RunContext[Userdata],
    days: Annotated[int, Field(description="Number of past days to include (default 7)")] = 7
) -> str:
    """
    Compute simple aggregates over the last `days` days.
    Looks for numeric mood entries (3/5, 4/5, 4, etc.) and counts days with objectives.
    """
    try:
        days = int(days)
    except Exception:
        days = 7
    history = load_history()
    if not history:
        return "No history yet to compute a weekly summary."

    cutoff = datetime.now() - timedelta(days=days)
    recent = []
    for entry in reversed(history):  # newest first
        try:
            ts = datetime.fromisoformat(entry.get("timestamp"))
        except Exception:
            continue
        if ts >= cutoff:
            recent.append(entry)
        else:
            break

    if not recent:
        return f"No check-ins in the last {days} days."

    numeric_moods = []
    goal_days = 0
    for e in recent:
        m = parse_mood_numeric(e.get("mood"))
        if m is not None:
            numeric_moods.append(m)
        if e.get("objectives"):
            if len(e.get("objectives")) > 0:
                goal_days += 1

    avg_mood = round(sum(numeric_moods) / len(numeric_moods), 2) if numeric_moods else None
    total = len(recent)
    summary_parts = []
    if avg_mood is not None:
        summary_parts.append(f"Average mood (converted to 0-5) over last {len(numeric_moods)} entries: {avg_mood}/5")
    else:
        summary_parts.append("No numeric mood entries to compute an average.")
    summary_parts.append(f"{goal_days} of {total} days had at least one objective.")
    return " ".join(summary_parts)


# -------------------------
# Agent definition
# -------------------------
class WellnessAgent(Agent):
    def __init__(self, history_context: str):
        super().__init__(
            instructions=f"""
You are a calm, supportive, non-medical wellness companion that conducts a short daily check-in.

Goals:
1) Ask how the user is feeling (mood) and their energy level.
2) Ask for 1-3 objectives for the day.
3) Give small, grounded, non-medical advice (e.g., take a 5-minute walk, break large tasks).
4) Recap the mood and objectives and ask "Does this sound right?".
5) Use the JSON history to reference past sessions when relevant.

If the user asks for a weekly summary or trends, call get_weekly_summary.

DO NOT give medical advice or diagnoses. If the user indicates crisis/self-harm, advise contacting a professional immediately.

Context from previous sessions (brief):
{history_context}
""",
            tools=[record_mood_and_energy, record_objectives, complete_checkin, get_weekly_summary]
        )


# -------------------------
# Entrypoint & initialization
# -------------------------
def prewarm(proc: JobProcess):
    # load silero VAD to speed up session start
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # build a short history summary to pass into the agent prompt
    history = load_history()
    if history:
        last = history[-1]
        last_time = last.get("timestamp", "unknown time")
        last_mood = last.get("mood", "unknown mood")
        last_energy = last.get("energy", "unknown energy")
        last_goals = ", ".join(last.get("objectives", [])) or "no goals recorded"
        history_context = f"Last check-in on {last_time}: mood={last_mood}, energy={last_energy}, goals={last_goals}."
        logger.info("Loaded history context: %s", history_context)
    else:
        history_context = "No previous check-ins."

    userdata = Userdata(current_checkin=CheckInState(), history_summary=history_context)

    # Create the session — ensure your environment has the required API keys
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", api_key=os.getenv("DEEPGRAM_API_KEY")),
        llm=google.LLM(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY")),
        tts=murf.TTS(
            api_key=os.getenv("MURF_API_KEY"),
            voice="en-US-natalie",
            style="narration",
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata
    )

    # Start the agent + room loop
    await session.start(
        agent=WellnessAgent(history_context=history_context),
        room=ctx.room,
        room_input_options=RoomInputOptions()
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
