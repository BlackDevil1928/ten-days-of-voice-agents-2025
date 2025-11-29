# agent.py
"""
Day 8 – Voice Game Master (Cyberpunk City)
- Cyberpunk universe: neon streets, megacorps, rogue AI.
- Voice-first D&D-style GM.
- In-memory JSON world state, character sheet, inventory, d20 checks.
- Tools: start_adventure, get_scene, player_action, show_journal, restart_adventure, save_game, load_game, roll_d20_tool
- Uses LiveKit Agents plumbing (Deepgram STT, Murf TTS, Google GenAI LLM), Multilingual turn detector and Silero VAD prewarm.
"""

import os
import json
import uuid
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Annotated

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
from livekit.agents._exceptions import APIStatusError

# Plugin imports (change if you use different providers)
from livekit.plugins import deepgram, murf, google, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("cyberpunk_gamemaster")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

load_dotenv(".env.local")

# -------------------------
# LLM model defaults (env override)
# -------------------------
PRIMARY_GEMINI = os.environ.get("GEMINI_PRIMARY_MODEL", "gemini-2.5-flash")
FALLBACK_GEMINI = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-1.5-flash")

# -------------------------
# System prompt (Cyberpunk GM persona)
# -------------------------
SYSTEM_PROMPT = """
You are 'Nyx', a Game Master running a cyberpunk city adventure called 'Neon Debt'.
Universe: A rain-slicked megacity of neon, corporate towers, black-market arcologies and rogue AIs.
Tone: gritty, cinematic, slightly sardonic, but empathetic to the player.
Role: You are the GM. You describe scenes vividly, drive the story, and always end each descriptive message with: "What do you do?"
Rules:
- Keep lines concise and voice-friendly for spoken delivery.
- Remember player decisions, named NPCs, and visited locations using session state.
- Offer short choices when helpful, but accept free-text spoken actions.
- Use simple mechanics: when an action is risky, perform a d20 check with attribute modifiers and narrate outcome.
- Aim for a short mini-arc (8–15 meaningful exchanges) that reaches a resolution (discover, escape, or solve a mystery).
"""

# -------------------------
# Base Cyberpunk world template
# -------------------------
WORLD_TEMPLATE = {
    "locations": {
        "neon_alley": {
            "key": "neon_alley",
            "name": "Neon Alley",
            "desc": "A narrow alley glowing with neon signs and steaming vents. Holograms flicker; distant synth music throbs.",
            "paths": ["arcology_entrance", "backstreet_market"]
        },
        "arcology_entrance": {
            "key": "arcology_entrance",
            "name": "Arcology Entrance",
            "desc": "A massive corporate arcology gate with biometric scanners and a security drone hovering above.",
            "paths": ["neon_alley"]
        },
        "market": {
            "key": "market",
            "name": "Backstreet Market",
            "desc": "Stacks of stalls selling hacked implants and bootleg software. Faces half-hidden behind smartglasses.",
            "paths": ["neon_alley"]
        }
    },
    "npcs": {
        "fixer": {"key": "fixer", "name": "Maya the Fixer", "role": "fixer", "alive": True, "attitude": "neutral"},
        "drone": {"key": "drone", "name": "Security Drone", "role": "drone", "alive": True, "attitude": "hostile"}
    },
    "events": [],
    "quests": []
}

# -------------------------
# Per-session Userdata + character sheet
# -------------------------
@dataclass
class Userdata:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    player_name: Optional[str] = None
    current_location: str = "neon_alley"
    world: Dict[str, Any] = field(default_factory=lambda: json.loads(json.dumps(WORLD_TEMPLATE)))
    character: Dict[str, Any] = field(default_factory=lambda: {
        "name": None,
        "class": "Runner",
        "hp": 12,
        "max_hp": 12,
        "strength": 2,
        "agility": 3,
        "intelligence": 3,
        "luck": 1,
        "inventory": ["cyberdeck (basic)", "wallet"]
    })
    history: List[Dict[str, Any]] = field(default_factory=list)
    journal: List[str] = field(default_factory=list)
    turn_count: int = 0
    session_phase: str = "intro"  # intro -> playing -> finished

# -------------------------
# Utilities: world, history, rolls
# -------------------------
def record_history(userdata: Userdata, role: str, text: str):
    entry = {"time": datetime.utcnow().isoformat() + "Z", "role": role, "text": text}
    userdata.history.append(entry)
    # keep history trimmed
    if len(userdata.history) > 200:
        userdata.history = userdata.history[-200:]

def scene_text(userdata: Userdata) -> str:
    loc_key = userdata.current_location
    loc = userdata.world["locations"].get(loc_key)
    if not loc:
        return "You stand in a featureless stretch of city. What do you do?"
    desc = loc["desc"]
    # mention an NPC if present nearby (simple)
    nearby = []
    for npc in userdata.world.get("npcs", {}).values():
        if npc.get("alive", True):
            nearby.append(npc["name"])
            break
    extras = f"\nYou notice {nearby[0]} here." if nearby else ""
    return f"{desc}{extras}\n\nWhat do you do?"

def apply_event(userdata: Userdata, desc: str):
    ev = {"time": datetime.utcnow().isoformat() + "Z", "desc": desc}
    userdata.world.setdefault("events", []).append(ev)
    userdata.journal.append(desc)

def roll_d20(modifier: int = 0) -> Dict[str, int]:
    r = random.randint(1, 20)
    total = r + modifier
    return {"roll": r, "modifier": modifier, "total": total}

# -------------------------
# Tools exposed to LLM/UI
# -------------------------
@function_tool
async def start_adventure(ctx: RunContext[Userdata], player_name: Annotated[Optional[str], Field(default=None)] = None) -> str:
    userdata = ctx.userdata
    if player_name:
        userdata.player_name = player_name
        userdata.character["name"] = player_name
    userdata.session_phase = "playing"
    userdata.current_location = "neon_alley"
    userdata.turn_count = 0
    userdata.history = []
    userdata.journal = []
    record_history(userdata, "gm", "Adventure started")
    greeting = (
        f"Welcome to Neon Debt, {userdata.player_name or 'runner'}. This city never sleeps.\n\n"
        "Quick rules: Speak naturally. I will describe scenes and end each line with 'What do you do?'. "
        "You can ask about your inventory (say 'what's in my bag'), check health ('what is my health'), "
        "or attempt risky actions (I will roll a d20 for checks). Now, begin.\n\n"
        + scene_text(userdata)
    )
    return greeting

@function_tool
async def get_scene(ctx: RunContext[Userdata]) -> str:
    userdata = ctx.userdata
    return scene_text(userdata)

@function_tool
async def player_action(ctx: RunContext[Userdata], action: Annotated[str, Field(description="Player spoken action")]= "") -> str:
    userdata = ctx.userdata
    action_text = (action or "").strip()
    userdata.turn_count += 1
    record_history(userdata, "player", action_text)

    # quick command checks
    lower = action_text.lower()
    if lower in ["what's in my bag", "what is in my bag", "inventory", "what's in my inventory"]:
        inv = userdata.character.get("inventory", [])
        return f"You have: {', '.join(inv) or 'nothing'}. What do you do?"

    if lower in ["what is my health", "how much health do i have", "hp"]:
        hp = userdata.character.get("hp", 0)
        maxhp = userdata.character.get("max_hp", 0)
        return f"HP: {hp}/{maxhp}. What do you do?"

    # movement: left/right/enter/cave/market
    if any(w in lower for w in ["go to", "go", "walk", "enter", "head"]):
        # naive parsing of destination
        if "arcology" in lower or "gate" in lower:
            userdata.current_location = "arcology_entrance"
            apply_event(userdata, "Player moved to Arcology Entrance")
            record_history(userdata, "gm", "Moved to arcology entrance")
            return scene_text(userdata)
        if "market" in lower or "stall" in lower:
            userdata.current_location = "market"
            apply_event(userdata, "Player moved to Backstreet Market")
            record_history(userdata, "gm", "Moved to market")
            return scene_text(userdata)
        # default movement returns scene hint
        return "I didn't catch where you want to go. Try 'go to arcology' or 'walk to the market'. What do you do?"

    # risky action: disable drone, pick lock, hack scanner
    if "open hatch" in lower or "pick lock" in lower or "pick the lock" in lower or "hack" in lower or "disable drone" in lower:
        # map verbs to attribute for modifier
        if "hack" in lower:
            mod = userdata.character.get("intelligence", 0)
            skill = "intelligence"
        elif "pick" in lower or "open" in lower:
            mod = userdata.character.get("agility", 0)
            skill = "agility"
        else:
            mod = userdata.character.get("strength", 0)
            skill = "strength"

        result = roll_d20(mod)
        total = result["total"]
        # interpret roll thresholds
        if total >= 14:
            # success
            apply_event(userdata, f"Succeeded at {action_text} (roll {result['roll']} + {mod} = {total})")
            record_history(userdata, "gm", f"Success on {action_text} with roll {total}")
            # success consequences: location change or item reveal
            if "hack" in lower:
                userdata.character["inventory"].append("access token (arcology)")
                return f"Roll: {result['roll']} + {mod} = {total}. You hack the console and retrieve an access token. What do you do?"
            if "pick" in lower or "open" in lower:
                userdata.character["inventory"].append("sealed datachip")
                return f"Roll: {result['roll']} + {mod} = {total}. The lock yields; inside you find a sealed datachip. What do you do?"
            if "disable drone" in lower:
                userdata.world["npcs"]["drone"]["alive"] = False
                return f"Roll: {result['roll']} + {mod} = {total}. The drone collapses into a heap of sparking servos. What do you do?"
            # generic success
            return f"Roll: {result['roll']} + {mod} = {total}. You succeed. What do you do?"
        elif total >= 8:
            # partial success
            apply_event(userdata, f"Partial success at {action_text} (roll {result['roll']} + {mod} = {total})")
            record_history(userdata, "gm", f"Partial success on {action_text} with roll {total}")
            return f"Roll: {result['roll']} + {mod} = {total}. You make progress but suffer a drawback (alarm raised). What do you do?"
        else:
            # failure
            apply_event(userdata, f"Failed at {action_text} (roll {result['roll']} + {mod} = {total})")
            record_history(userdata, "gm", f"Failed {action_text} with roll {total}")
            # penalty: lose a small HP
            userdata.character["hp"] = max(0, userdata.character.get("hp", 0) - 2)
            return f"Roll: {result['roll']} + {mod} = {total}. You fail and take a small hit. HP is now {userdata.character['hp']}. What do you do?"

    # short-help commands
    if lower in ["help", "rules"]:
        return ("You can move (say 'go to arcology' / 'walk to market'), inspect (say 'look' or 'examine'), "
                "interact (say 'hack console', 'pick lock'), check inventory ('what's in my bag'), or check HP ('what is my health'). What do you do?")

    # look/examine
    if any(w in lower for w in ["look", "examine", "inspect"]):
        return scene_text(userdata)

    # If not matched by quick rules, forward to LLM for richer narration
    # Build a concise context for the LLM
    # We produce a short instruction + recent history to keep responses fast.
    recent = " ".join([h["text"] for h in userdata.history[-6:]])
    prompt = (
        SYSTEM_PROMPT
        + f"\n\nCurrent location: {userdata.current_location}\nCharacter: {json.dumps(userdata.character)}"
        + f"\nRecent history: {recent}\nPlayer action: {action_text}\n\n"
        "Respond as the GM in 1-3 short spoken sentences, evocative and ending with 'What do you do?'"
    )
    record_history(userdata, "gm", f"Forwarding to LLM: {action_text}")
    # Return the prompt to the AgentSession's LLM runner: session.llm will use it to produce the spoken reply.
    # Tools must return strings; the session will treat returned string as either direct GM text or LLM prompt.
    return prompt

@function_tool
async def show_journal(ctx: RunContext[Userdata]) -> str:
    userdata = ctx.userdata
    lines = [f"Session {userdata.session_id} | Started {userdata.started_at}"]
    if userdata.player_name:
        lines.append(f"Player: {userdata.player_name}")
    lines.append(f"Location: {userdata.current_location}")
    lines.append("\nCharacter:")
    for k, v in userdata.character.items():
        if k == "inventory":
            lines.append(f"- inventory: {', '.join(v)}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("\nJournal:")
    if userdata.journal:
        lines.extend([f"- {j}" for j in userdata.journal])
    else:
        lines.append("- (none yet)")
    lines.append("\nRecent history:")
    for h in userdata.history[-6:]:
        lines.append(f"- {h['time']} | {h['role']}: {h['text']}")
    lines.append("\nWhat do you do?")
    return "\n".join(lines)

@function_tool
async def restart_adventure(ctx: RunContext[Userdata]) -> str:
    userdata = ctx.userdata
    userdata.world = json.loads(json.dumps(WORLD_TEMPLATE))
    userdata.current_location = "neon_alley"
    userdata.character = {
        "name": userdata.character.get("name"),
        "class": "Runner",
        "hp": 12,
        "max_hp": 12,
        "strength": 2,
        "agility": 3,
        "intelligence": 3,
        "luck": 1,
        "inventory": ["cyberdeck (basic)", "wallet"]
    }
    userdata.history = []
    userdata.journal = []
    userdata.turn_count = 0
    userdata.session_phase = "intro"
    return "Adventure reset. Say 'start_adventure' to begin a new run."

@function_tool
async def save_game(ctx: RunContext[Userdata]) -> str:
    userdata = ctx.userdata
    dump = json.dumps({
        "session_id": userdata.session_id,
        "started_at": userdata.started_at,
        "player_name": userdata.player_name,
        "current_location": userdata.current_location,
        "world": userdata.world,
        "character": userdata.character,
        "history": userdata.history,
        "journal": userdata.journal,
        "turn_count": userdata.turn_count,
        "session_phase": userdata.session_phase
    }, indent=2)
    return dump

@function_tool
async def load_game(ctx: RunContext[Userdata], saved_json: Annotated[str, Field(description="Saved JSON string")] = "") -> str:
    userdata = ctx.userdata
    if not saved_json:
        return "No JSON provided to load."
    try:
        data = json.loads(saved_json)
    except Exception as e:
        return f"Failed to parse JSON: {e}"
    userdata.player_name = data.get("player_name", userdata.player_name)
    userdata.current_location = data.get("current_location", userdata.current_location)
    userdata.world = data.get("world", userdata.world)
    userdata.character = data.get("character", userdata.character)
    userdata.history = data.get("history", userdata.history)
    userdata.journal = data.get("journal", userdata.journal)
    userdata.turn_count = data.get("turn_count", userdata.turn_count)
    userdata.session_phase = data.get("session_phase", userdata.session_phase)
    return "Save loaded. Continue your adventure. What do you do?"

@function_tool
async def roll_d20_tool(ctx: RunContext[Userdata], modifier: Annotated[int, Field(default=0)] = 0) -> str:
    r = roll_d20(modifier)
    return f"Roll: {r['roll']} + {r['modifier']} = {r['total']}"

# -------------------------
# Agent class
# -------------------------
class GameMasterAgent(Agent):
    def __init__(self):
        instructions = SYSTEM_PROMPT + "\nAvailable tools: start_adventure, get_scene, player_action, show_journal, restart_adventure, save_game, load_game, roll_d20_tool"
        super().__init__(
            instructions=instructions,
            tools=[
                start_adventure, get_scene, player_action, show_journal,
                restart_adventure, save_game, load_game, roll_d20_tool
            ],
        )

# -------------------------
# LLM init helper (safe)
# -------------------------
def safe_google_llm(model_name: str, **kwargs):
    logger.info("Initializing Google LLM: %s", model_name)
    return google.LLM(model=model_name, **kwargs)

# -------------------------
# Prewarm & Entrypoint
# -------------------------
def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("VAD prewarmed")
    except Exception:
        logger.warning("VAD prewarm failed; continuing without VAD.")
    _ = json.dumps(WORLD_TEMPLATE)  # tiny cache

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("Starting Cyberpunk City Game Master in room %s", ctx.room.name)

    userdata = Userdata()

    try:
        session = AgentSession(
            stt=deepgram.STT(model="nova-3"),
            llm=safe_google_llm(PRIMARY_GEMINI, temperature=0.7, max_output_tokens=512),
            tts=murf.TTS(voice="en-US-marcus", style="Conversational", text_pacing=True),
            turn_detection=MultilingualModel(),
            vad=ctx.proc.userdata.get("vad"),
            userdata=userdata,
        )

        await session.start(
            agent=GameMasterAgent(),
            room=ctx.room,
            room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
        )

    except APIStatusError as e:
        logger.warning("Primary LLM model '%s' failed: %s. Trying fallback: %s", PRIMARY_GEMINI, str(e), FALLBACK_GEMINI)
        try:
            session = AgentSession(
                stt=deepgram.STT(model="nova-3"),
                llm=safe_google_llm(FALLBACK_GEMINI, temperature=0.7, max_output_tokens=512),
                tts=murf.TTS(voice="en-US-marcus", style="Conversational", text_pacing=True),
                turn_detection=MultilingualModel(),
                vad=ctx.proc.userdata.get("vad"),
                userdata=userdata,
            )
            await session.start(
                agent=GameMasterAgent(),
                room=ctx.room,
                room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
            )
        except Exception as e2:
            logger.exception("Fallback LLM failed as well. Aborting. Check credentials and model names.")
            raise

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
