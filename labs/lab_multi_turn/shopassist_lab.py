"""
shopassist_lab - an offline Claude simulator for the ShopAssist AI course labs.

WHAT THIS IS
------------
A single-file, standard-library-only stand-in for the `anthropic` SDK, so that the
course's lab exercises run inside a Udemy workspace that has no API key, no network
access, and no `pip install`.

    import shopassist_lab            # installs the shims, nothing else needed

    from dotenv import load_dotenv   # works even without python-dotenv installed
    from anthropic import Anthropic  # works even without anthropic installed
    load_dotenv()
    client = Anthropic()

Every line after that first import is IDENTICAL to the real course code. Take your
finished lab notebook home, add a real ANTHROPIC_API_KEY, and the same cells call the
real Claude API - this module steps out of the way automatically.

WHAT THIS IS NOT
----------------
This is NOT Claude. Responses are canned fixtures chosen by a rules table. The
simulator has no language model in it and never touches the network.

It is honest about that in four ways you cannot miss:
  * a banner printed once, when you construct the client
  * every simulated message id starts with "msg_sim_"
  * every simulated message has `.simulated == True` (real ones do not)
  * `explain(message)` names the exact fixture it chose and why

WHAT IS FAITHFULLY SIMULATED
----------------------------
The parameters matter. That is the whole point - a mock that ignores your prompt
teaches you nothing.

    max_tokens      truncates the reply and sets stop_reason="max_tokens"
    stop_sequences  cuts at the marker, sets stop_reason="stop_sequence"
    temperature     0 repeats byte-for-byte; >0 varies, reproducibly
    system          rules detected in your system prompt change the reply
    tools           produces real tool_use blocks with schema-valid input
    tool_choice     honours "auto", "any", "none" and forced {"type":"tool"}

Call `lab_info()` for the full disclosure, including the known gaps.

Licensed for use with the "Claude Certified Architect Foundations" course.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import re
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Anthropic",
    "load_dotenv",
    "install",
    "check",
    "check_all",
    "explain",
    "lab_info",
    "reset",
    "verbose",
    "register_scenario",
    "LabConfig",
    "Message",
    "TextBlock",
    "ToolUseBlock",
    "Usage",
    "APIError",
    "BadRequestError",
    "AuthenticationError",
]

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# 1. Configuration and mode
# ---------------------------------------------------------------------------


class LabConfig:
    """Tunables. Instructors may change these; students should not need to."""

    chars_per_token = 4          # how max_tokens is converted into a character budget
    max_tool_turns = 12          # hard stop so a broken agentic loop cannot hang the kernel
    tool_preambles = True        # emit a TextBlock alongside tool_use when tool_choice="auto"
    include_optional_tool_inputs = False   # synthesize only `required` fields by default
    verbose = False              # print a one-line trace per simulated call


_BANNER_SHOWN = False
_WARNED: set = set()


def verbose(on: bool = True) -> None:
    """Print a one-line trace for every simulated call. Useful when recording a lesson."""
    LabConfig.verbose = bool(on)


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        print("[shopassist_lab] note: " + msg)


def _module_importable(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _has_real_key() -> bool:
    # An explicit override, so an instructor who has a key on their machine can still
    # record or verify the offline experience the students will actually get.
    if os.environ.get("SHOPASSIST_LAB_FORCE_SIM", "").strip().lower() in ("1", "true", "yes"):
        return False
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return bool(key) and not key.lower().startswith(("your", "sk-ant-xxx", "changeme"))


# ---------------------------------------------------------------------------
# 2. Response model - duck-types anthropic.types without pydantic
# ---------------------------------------------------------------------------


class _LabModel:
    """A minimal stand-in for a pydantic BaseModel. No third-party dependencies."""

    _fields: Tuple[str, ...] = ()

    def model_dump(self, *, exclude_none: bool = False, **_: Any) -> dict:
        out: Dict[str, Any] = {}
        for f in self._fields:
            v = getattr(self, f, None)
            if v is None and exclude_none:
                continue
            if isinstance(v, list):
                out[f] = [
                    x.model_dump(exclude_none=exclude_none) if isinstance(x, _LabModel) else x
                    for x in v
                ]
            elif isinstance(v, _LabModel):
                out[f] = v.model_dump(exclude_none=exclude_none)
            else:
                out[f] = v
        return out

    def model_dump_json(self, *, indent: Optional[int] = None,
                        exclude_none: bool = False, **_: Any) -> str:
        return json.dumps(self.model_dump(exclude_none=exclude_none),
                          indent=indent, default=str)

    # pydantic v1 aliases students may reach for
    def dict(self, **kw: Any) -> dict:          # noqa: A003
        return self.model_dump(**kw)

    def json(self, **kw: Any) -> str:           # noqa: A003
        return self.model_dump_json(**kw)

    # forgiving dict-ish access (a deliberate superset of the real SDK)
    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def get(self, k: str, default: Any = None) -> Any:
        return getattr(self, k, default)

    def __contains__(self, k: str) -> bool:
        return k in self._fields

    def __repr__(self) -> str:
        inner = ", ".join("{}={!r}".format(f, getattr(self, f, None)) for f in self._fields)
        return "{}({})".format(type(self).__name__, inner)


class TextBlock(_LabModel):
    _fields = ("citations", "text", "type")

    def __init__(self, text: str) -> None:
        self.citations = None
        self.text = text
        self.type = "text"


class ToolUseBlock(_LabModel):
    _fields = ("id", "input", "name", "type")

    def __init__(self, id: str, name: str, input: dict) -> None:  # noqa: A002
        self.id = id
        self.input = input
        self.name = name
        self.type = "tool_use"


class Usage(_LabModel):
    _fields = ("cache_creation_input_tokens", "cache_read_input_tokens",
               "input_tokens", "output_tokens", "service_tier")

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.service_tier = "standard"


class Message(_LabModel):
    _fields = ("id", "content", "model", "role", "stop_reason",
               "stop_sequence", "type", "usage")

    #: Present only on simulated messages. Real anthropic Messages have no such attribute.
    simulated = True

    def __init__(self, id: str, content: list, model: str, stop_reason: str,  # noqa: A002
                 stop_sequence: Optional[str], usage: Usage) -> None:
        self.id = id
        self.content = content
        self.model = model
        self.role = "assistant"
        self.stop_reason = stop_reason
        self.stop_sequence = stop_sequence
        self.type = "message"
        self.usage = usage
        self._lab: Dict[str, Any] = {}

    @property
    def text(self) -> str:
        """Convenience shortcut, not part of the real SDK."""
        return "".join(b.text for b in self.content if b.type == "text")

    def __repr__(self) -> str:
        kinds = ", ".join(b.type for b in self.content) or "empty"
        return ("Message(id={!r}, simulated=True, stop_reason={!r}, "
                "content=[{}], model={!r})".format(
                    self.id, self.stop_reason, kinds, self.model))


# ---------------------------------------------------------------------------
# 3. Errors
# ---------------------------------------------------------------------------


class APIError(Exception):
    """Base class, mirroring anthropic.APIError."""


class BadRequestError(APIError):
    """400-equivalent, mirroring anthropic.BadRequestError."""


class AuthenticationError(APIError):
    """401-equivalent, mirroring anthropic.AuthenticationError."""


# ---------------------------------------------------------------------------
# 4. Determinism helpers
# ---------------------------------------------------------------------------


def est_tokens(text: str) -> int:
    """Rough token estimate. Real tokenizers differ; this is close enough to teach with."""
    return max(1, int(len(text) / LabConfig.chars_per_token)) if text else 0


def _fingerprint(*parts: Any) -> str:
    blob = "␟".join(json.dumps(p, sort_keys=True, default=str) for p in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _rng(fingerprint: str, salt: int) -> random.Random:
    seed = int(fingerprint[:16], 16) ^ (salt * 0x9E3779B1)
    return random.Random(seed)


_LAST_VARIANT: Dict[str, int] = {}


def pick_variant(variants: Sequence[str], temperature: float,
                 rng: random.Random, fingerprint: str) -> Tuple[str, int]:
    """Choose a response variant.

    temperature == 0 always returns variant 0, byte for byte.
    temperature > 0 samples with a softmax over variant rank, then refuses to repeat
    the immediately previous choice - real Claude at temperature 1 essentially never
    returns the identical string twice in a row either.
    """
    if len(variants) == 1 or temperature <= 0:
        _LAST_VARIANT[fingerprint] = 0
        return variants[0], 0

    weights = [math.exp(-i / (temperature * 2.0)) for i in range(len(variants))]
    previous = _LAST_VARIANT.get(fingerprint)

    def sample() -> int:
        target, acc = rng.random() * sum(weights), 0.0
        for i, w in enumerate(weights):
            acc += w
            if target <= acc:
                return i
        return len(variants) - 1

    idx = sample()
    if previous is not None and idx == previous:
        alternatives = [i for i in range(len(variants)) if i != previous]
        idx = alternatives[rng.randrange(len(alternatives))]

    _LAST_VARIANT[fingerprint] = idx
    return variants[idx], idx


def reset(seed: Optional[int] = None) -> None:
    """Forget the call history, so a temperature>0 sequence replays identically."""
    _LAST_VARIANT.clear()
    for client in _CLIENTS:
        client._calls = 0
    if seed is not None:
        random.seed(seed)


_CLIENTS: List["MockAnthropic"] = []


# ---------------------------------------------------------------------------
# 5. Prompt analysis
# ---------------------------------------------------------------------------


FLAG_PATTERNS: Dict[str, str] = {
    "no_refund_promise": r"do ?n(?:o|')?t promise a refund|not promise a refund"
                         r"|do ?n(?:o|')?t approve a refund|until the order is checked",
    "ask_one_question": r"one (?:clear )?question at a time",
    "ask_for_order": r"ask for the order (?:number|id)|order number if it is missing",
    "concise": r"\bconcise\b|keep (?:it|the response) (?:short|under)|under \d+ words",
    "polite": r"\bpolite\b|\bcalm\b",
    "escalate_on_risk": r"escalate if|fraud or legal|legal action",
    "json_only": r"return only (?:a )?valid json|only valid json|do not include markdown",
    "no_preamble": r"return only the message|no explanations|do not include explanations",
}

INTENT_PATTERNS: List[Tuple[str, str]] = [
    ("billing_issue", r"charged twice|double charge|duplicate charge|billed twice|billing"),
    ("damaged_item", r"damaged|arrived broken|\bbroken\b|scratched|defective|cracked"
                     r"|do(?:es)? ?n(?:o|')?t work"),
    ("refund_request", r"\brefund\b|money back|want to return|return my order"
                       r"|return an order|wants to return|about a return"),
    ("order_status", r"where is my order|has ?n(?:o|')?t arrived|tracking"
                     r"|shipping status|supposed to arrive"),
    ("order_number", r"\border\s*(?:number|id|#)\b|^\s*my order number is"),
    ("product_question", r"how do i use|compatible with|does it work with"),
    ("greeting", r"^\s*(?:hi|hello|hey)\b"),
]

_XML = re.compile(r"<(\w+)>(.*?)</\1>", re.S)
_WORD_LIMIT = re.compile(r"under (\d+) words", re.I)

SLOT_PATTERNS: Dict[str, str] = {
    "order_id": r"\b(ORD-[A-Z0-9]{3,})\b|\border\s*(?:number|id|#)?\s*(?:is)?\s*[:# ]?\s*(\d{4,})\b",
    "email": r"([\w.+-]+@[\w-]+\.[\w.]+)",
    "customer_id": r"\b(CUS-\d+)\b",
}


@dataclass(frozen=True)
class PromptFacts:
    has_system: bool
    system_text: str
    flags: frozenset
    intents: Tuple[str, ...]
    xml_tags: Dict[str, str]
    has_xml: bool
    word_limit: Optional[int]
    effective_user_text: str
    turns: int


def _flatten(content: Any) -> str:
    """Turn a system= or content= value into plain text, whatever shape it arrived in."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif getattr(item, "type", None) == "text":
                parts.append(getattr(item, "text", ""))
        return "\n".join(parts)
    return str(content)


def _last_user_text(messages: Sequence[Any]) -> str:
    for msg in reversed(list(messages)):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "user":
            body = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            text = _flatten(body)
            if text.strip():
                return text
    return ""


def detect_flags(*texts: str) -> frozenset:
    blob = "\n".join(t for t in texts if t)
    found = {name for name, pat in FLAG_PATTERNS.items()
             if re.search(pat, blob, re.I)}
    return frozenset(found)


def detect_intents(text: str) -> Tuple[str, ...]:
    hits = []
    for name, pat in INTENT_PATTERNS:
        m = re.search(pat, text, re.I)
        if m:
            hits.append((m.start(), name))
    hits.sort()
    return tuple(name for _, name in hits)


def extract_slots(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for slot, pat in SLOT_PATTERNS.items():
        m = re.search(pat, text, re.I)
        if m:
            value = next((g for g in m.groups() if g), None)
            if value:
                out[slot] = value
    return out


def analyze_prompt(system_text: str, messages: Sequence[Any]) -> PromptFacts:
    user_text = _last_user_text(messages)

    tags = {k.lower(): v.strip() for k, v in _XML.findall(user_text)}
    effective = tags.get("customer_message") or tags.get("user_message") or user_text

    limit_match = _WORD_LIMIT.search(user_text) or _WORD_LIMIT.search(system_text)

    return PromptFacts(
        has_system=bool(system_text.strip()),
        system_text=system_text,
        flags=detect_flags(system_text, user_text),
        intents=detect_intents(effective),
        xml_tags=tags,
        has_xml=bool(tags),
        word_limit=int(limit_match.group(1)) if limit_match else None,
        effective_user_text=effective,
        turns=len(list(messages)),
    )


# ---------------------------------------------------------------------------
# 6. Blackboard - what the transcript already established
# ---------------------------------------------------------------------------


def _block_type(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("type", ""))
    return str(getattr(block, "type", ""))


def _block_get(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


@dataclass
class Blackboard:
    turns: int = 0
    tools_called: List[str] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)
    assistant_turns: int = 0


def build_blackboard(messages: Sequence[Any]) -> Blackboard:
    bb = Blackboard()
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        body = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        bb.turns += 1
        if role == "assistant":
            bb.assistant_turns += 1
        if isinstance(body, str):
            bb.facts.update(extract_slots(body))
            continue
        for block in (body or []):
            btype = _block_type(block)
            if btype == "text":
                bb.facts.update(extract_slots(_block_get(block, "text", "") or ""))
            elif btype == "tool_use":
                bb.tools_called.append(str(_block_get(block, "name", "")))
                for k, v in (_block_get(block, "input") or {}).items():
                    if v not in (None, ""):
                        bb.facts[k] = v
    return bb


# ---------------------------------------------------------------------------
# 7. Scenario table  (DATA - this is what grows when a lesson is added)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    intent: Tuple[str, ...] = ()
    requires_flags: Tuple[str, ...] = ()
    forbids_flags: Tuple[str, ...] = ()
    has_system: Optional[bool] = None
    has_tools: Optional[bool] = None
    user_regex: Optional[str] = None
    max_turns: Optional[int] = None
    min_turns: Optional[int] = None
    knows: Tuple[str, ...] = ()
    lacks: Tuple[str, ...] = ()

    def specificity(self) -> int:
        score = 0
        for value in dataclasses.astuple(self):
            if value is None:
                continue
            if isinstance(value, tuple):
                score += len(value)
            else:
                score += 1
        return score


@dataclass(frozen=True)
class Scenario:
    id: str
    lesson: str
    when: Match
    variants: Tuple[str, ...]
    note: str = ""
    priority: int = 0


def _matches(m: Match, facts: PromptFacts, bb: Blackboard, has_tools: bool) -> bool:
    if m.intent and not any(i in facts.intents for i in m.intent):
        return False
    if any(f not in facts.flags for f in m.requires_flags):
        return False
    if any(f in facts.flags for f in m.forbids_flags):
        return False
    if m.has_system is not None and m.has_system != facts.has_system:
        return False
    if m.has_tools is not None and m.has_tools != has_tools:
        return False
    if m.user_regex and not re.search(m.user_regex, facts.effective_user_text, re.I):
        return False
    if m.max_turns is not None and facts.turns > m.max_turns:
        return False
    if m.min_turns is not None and facts.turns < m.min_turns:
        return False
    if any(k not in bb.facts for k in m.knows):
        return False
    if any(k in bb.facts for k in m.lacks):
        return False
    return True


# Long on purpose: with max_tokens=300 the whole thing fits (end_turn), with
# max_tokens=20 it is cut mid-sentence (max_tokens). Lesson 12.
_RETURN_LONGFORM = (
    "Thanks for reaching out about your return. Here is how the process works.\n\n"
    "First, I'll need the order number so I can pull up the purchase and confirm it is "
    "inside our 30-day return window. Once I have that, I'll check the item's condition "
    "notes and whether the order has been marked delivered. If everything lines up, I can "
    "start the return and email you a prepaid shipping label the same day. Refunds are "
    "issued to the original payment method and usually land within three to five business "
    "days after the carrier scans the parcel.\n\n"
    "If the item arrived damaged, please don't send it back yet. A photo is usually enough "
    "and it saves you a trip to the post office.\n\n"
    "What is the order number?"
)

_RETURN_LONGFORM_B = (
    "Happy to help you get this returned. Here is what happens next.\n\n"
    "Send me the order number and I will look up the purchase, confirm it is still inside "
    "the 30-day window, and check that it has been marked delivered on our side. Assuming "
    "it all checks out, I will open the return straight away and email you a prepaid label. "
    "Once the carrier scans the parcel, the refund goes back to your original payment "
    "method, normally within three to five business days.\n\n"
    "One exception worth knowing: if the item turned up damaged, hold on to it for now. "
    "A quick photo is usually all we need and it saves you the trip.\n\n"
    "Could you share the order number?"
)

_RETURN_LONGFORM_C = (
    "Sorry the order didn't work out, and thanks for getting in touch about returning it.\n\n"
    "The process is short. I need the order number first so I can open the purchase, verify "
    "we are still inside the 30-day return window, and confirm the delivery status. When "
    "those line up I will create the return and send a prepaid shipping label to your email "
    "the same day. The refund is issued to whatever payment method you originally used and "
    "typically clears three to five business days after the parcel is scanned.\n\n"
    "If the reason for the return is damage in transit, don't ship it back. Send a photo "
    "instead and we will handle it from there.\n\n"
    "What is the order number on the purchase?"
)


SCENARIOS: List[Scenario] = [
    # -- Lesson 12: the very first request ---------------------------------
    Scenario(
        id="return.longform",
        lesson="12",
        # Outranks the customer-message scenarios below: this prompt is an instruction
        # TO the model ("write a response"), not a customer speaking.
        priority=40,
        note="Deliberately long. max_tokens=300 returns it whole (end_turn); "
             "max_tokens=20 cuts it mid-sentence (max_tokens).",
        when=Match(
            user_regex=r"wants? to return an order|write a short helpful response",
            has_tools=False,
        ),
        variants=(_RETURN_LONGFORM, _RETURN_LONGFORM_B, _RETURN_LONGFORM_C),
    ),

    # -- Lesson 16: stop sequences -----------------------------------------
    Scenario(
        id="return.reply_short",
        lesson="16",
        priority=42,
        note="A short support reply. When the prompt asks for an end marker, the reply "
             "emits it and the API cuts there.",
        when=Match(
            user_regex=r"short (?:customer )?support reply",
            has_tools=False,
        ),
        variants=(
            "Thanks for getting in touch about your return. I can help with that. "
            "Could you send me the order number so I can look up the purchase and check "
            "it against our 30-day return window?",
            "Happy to help with the return. Send over the order number and I will confirm "
            "the purchase is still inside our 30-day window and take it from there.",
            "I can get that return started for you. What is the order number? Once I have "
            "it I will check the return window and confirm the next step.",
        ),
    ),

    # -- Lesson 13: the API is stateless -----------------------------------
    Scenario(
        id="orderid.no_context",
        lesson="13",
        priority=30,
        note="An order number arrived with no conversation history, so there is nothing "
             "to attach it to. This is what a stateless API looks like from the outside.",
        when=Match(
            intent=("order_number",),
            max_turns=1,
            has_tools=False,
        ),
        variants=(
            "Thanks for that number. I don't have any earlier context in this conversation, "
            "so I'm not sure yet what you'd like me to do with it.\n\n"
            "Are you checking on delivery, starting a return, or something else?",
            "Got the order number, but this is the first message I can see - I don't know "
            "what it relates to.\n\n"
            "What would you like me to do with this order?",
            "I have the number, though there's no prior conversation on my side to connect "
            "it to.\n\n"
            "Could you tell me what you need help with for this order?",
        ),
    ),
    Scenario(
        id="orderid.with_context",
        lesson="13",
        priority=32,
        note="The same order number, but the history establishes that this is a return. "
             "Now the reply can actually move the case forward.",
        when=Match(
            intent=("order_number",),
            min_turns=2,
            has_tools=False,
        ),
        variants=(
            "Thank you. I've got the order number for the return you mentioned.\n\n"
            "Let me pull up that order. Was the item damaged when it arrived, or is this "
            "a change of mind?",
            "Perfect, that's what I needed for the return.\n\n"
            "I'm opening the order now. Can you tell me the reason for the return?",
            "Great, thanks. I can look up the return with that number.\n\n"
            "What was the problem with the item?",
        ),
    ),

    # -- Lesson 15: the system prompt changes behaviour --------------------
    Scenario(
        id="return.opening",
        lesson="13",
        priority=30,
        note="A neutral return request with no complaint and no demand for money. The "
             "right first move is to ask which order, so the multi-turn lesson has a "
             "question for its second turn to answer.",
        when=Match(
            intent=("refund_request",),
            user_regex=r"(?i)^\s*i want to return my order\.?\s*$",
            has_system=False,
            has_tools=False,
            lacks=("order_id",),
            max_turns=1,
        ),
        variants=(
            "I can help you with that return. Could you share your order number so I "
            "can pull up the purchase?",
            "Happy to help with the return. What is the order number?",
            "Sure, I can start a return for you. Could you give me the order number "
            "first so I can check the details?",
        ),
    ),
    Scenario(
        id="refund.unguarded",
        lesson="15",
        priority=20,
        note="No system prompt and no rules in the user text, so nothing stops the reply "
             "from promising a refund it has no authority to promise. This is the "
             "'before' half of the system-prompt lesson.",
        when=Match(
            intent=("refund_request", "damaged_item", "billing_issue"),
            has_system=False,
            forbids_flags=("no_refund_promise", "ask_for_order", "json_only"),
            has_tools=False,
        ),
        variants=(
            "I'm so sorry about that! No problem at all - I've gone ahead and approved a "
            "full refund for you. You should see the money back on your original payment "
            "method within 3-5 business days. Thanks for your patience!",
            "Absolutely, I can refund that right away. Your refund is approved and the "
            "funds will be returned to your card shortly. Sorry for the trouble!",
            "Of course - consider it refunded. I've processed the return on my end, so "
            "there's no need to send anything back.",
        ),
    ),
    Scenario(
        id="refund.guarded",
        lesson="15",
        priority=25,
        note="The system prompt forbids promising a refund before the order is checked, "
             "so the reply withholds the promise and asks for exactly one thing.",
        when=Match(
            intent=("refund_request", "damaged_item", "billing_issue"),
            requires_flags=("no_refund_promise",),
            has_tools=False,
            lacks=("order_id",),
        ),
        variants=(
            "I'm sorry the item arrived in that condition. Before I can look at a refund "
            "I need to check the order on our side.\n\n"
            "Could you share your order number?",
            "Sorry about that - I understand the frustration. I'll need to review the "
            "order first before I can discuss a refund.\n\n"
            "What is the order number?",
            "Apologies for the trouble. I'm not able to confirm a refund until I've "
            "checked the order details.\n\n"
            "Can you give me the order number, please?",
        ),
    ),

    # -- Generic support fallbacks -----------------------------------------
    Scenario(
        id="support.generic_guarded",
        lesson="15",
        priority=5,
        note="A polite, rule-following answer for a support question with no more "
             "specific fixture.",
        when=Match(requires_flags=("no_refund_promise",), has_tools=False),
        variants=(
            "Thanks for reaching out. I can help with that.\n\n"
            "Could you share your order number so I can look up the details?",
            "Happy to help. To get started, what is the order number?",
        ),
    ),
]


FALLBACK = Scenario(
    id="fallback.unscripted",
    lesson="-",
    when=Match(),
    note="No scenario matched this prompt. The simulator says so rather than "
         "inventing an answer.",
    variants=(
        "[simulated] I don't have a scripted answer for that one. The simulator matched "
        "no scenario, so this is a placeholder rather than a real reply.\n\n"
        "Call explain(message) to see what it looked for, or set a real "
        "ANTHROPIC_API_KEY to get a genuine Claude response.",
    ),
)


def register_scenario(scenario: Scenario, replace: bool = False) -> None:
    """Add a scenario at runtime. Instructors can extend the table from a notebook cell."""
    global SCENARIOS
    if replace:
        SCENARIOS = [s for s in SCENARIOS if s.id != scenario.id]
    SCENARIOS.append(scenario)


def select_scenario(facts: PromptFacts, bb: Blackboard, has_tools: bool) -> Scenario:
    hits = [s for s in SCENARIOS if _matches(s.when, facts, bb, has_tools)]
    if not hits:
        return FALLBACK
    return max(hits, key=lambda s: (s.priority, s.when.specificity()))


# ---------------------------------------------------------------------------
# 7b. Computed replies
#
# The scenario table returns fixed text, which is right for a support answer.
# Some lessons need a reply that DEPENDS on the customer's own words -
# classification, extraction - so those get a small function instead of a canned
# string. Checked after tools and before the scenario table.
# ---------------------------------------------------------------------------


#: Reply ids produced here, with the note explain() should print for each.
COMPUTED_NOTES: Dict[str, str] = {
    "classify.intent_json": (
        "The prompt asked for an intent label as JSON, so the reply is computed from "
        "the customer's own words rather than picked from a fixture."
    ),
    "extract.freeform_json": (
        "The prompt asked for JSON without giving a schema, so the reply is valid "
        "JSON with field names the model chose for itself - which is the point."
    ),
    "grade.support_reply": (
        "The prompt asked this call to grade another assistant's reply, so the score "
        "is computed by checking that reply against the four stated rules."
    ),
}

CLASSIFY_LABELS: Tuple[str, ...] = ("refund_request", "order_status", "billing_issue",
                                    "product_question", "other")

_CUSTOMER_BLOCK = re.compile(
    r"customer(?:'s)?\s+message\s*:\s*(.+?)(?:\n\s*\n|\Z)", re.I | re.S)


def embedded_customer_message(facts: PromptFacts) -> str:
    """Pull the customer's own words out of a prompt that wraps them in instructions.

    Without this, intent detection runs over the whole prompt - including the list
    of allowed labels - and every message looks like a billing issue.
    """
    tagged = facts.xml_tags.get("customer_message") or facts.xml_tags.get("user_message")
    if tagged:
        return tagged.strip()
    match = _CUSTOMER_BLOCK.search(facts.effective_user_text)
    return match.group(1).strip() if match else ""


def _wants_intent_json(text: str) -> bool:
    return bool(re.search(r"classif", text, re.I)
                and re.search(r"\bintent\b", text, re.I)
                and re.search(r"\bjson\b", text, re.I))


def classify_intent_label(message: str) -> str:
    """Map a customer message onto one of the five course intent labels."""
    for name in detect_intents(message):
        if name in ("billing_issue", "order_status", "product_question"):
            return name
        if name in ("refund_request", "damaged_item"):
            return "refund_request"
    return "other"


def _reply_classify(facts: PromptFacts) -> Optional[Tuple[str, str]]:
    if not _wants_intent_json(facts.effective_user_text):
        return None
    message = embedded_customer_message(facts)
    if not message:
        return None
    return "classify.intent_json", json.dumps(
        {"intent": classify_intent_label(message)})


_LABELLED_BLOCK = r"{}\s*:\s*(.+?)(?:\n\s*\n|\Z)"


def _labelled(text: str, label: str) -> str:
    match = re.search(_LABELLED_BLOCK.format(label), text, re.I | re.S)
    return match.group(1).strip() if match else ""


def _is_polite(text: str) -> bool:
    """Deliberately generous. A grader that invents a tone complaint is worse than
    one that misses a curt sentence, because the learner then argues with the
    grader instead of reading the real violation."""
    return bool(re.search(
        r"\bsorry\b|\bthank|\bplease\b|happy to|glad to|apolog|i can help|let me"
        r"|no problem|of course|\bsure\b|absolutely|certainly|my pleasure"
        r"|i understand|\bhope\b|\bcould you\b|\bwould you\b", text, re.I))


def _reply_grade(facts: PromptFacts) -> Optional[Tuple[str, str]]:
    """Act as a model-based grader over another assistant's reply.

    Scores against the four rules the course's grading prompt states, using the
    same behavioural helpers the lab checks use - so the grade a learner sees and
    the grade the checker expects can never drift apart.
    """
    criteria = facts.system_text + "\n" + facts.effective_user_text
    if not re.search(r"evaluat\w*\s+(?:a\s+)?customer support assistant", criteria, re.I):
        return None
    if not (re.search(r"\bscore\b", criteria, re.I) and re.search(r"\bjson\b", criteria, re.I)):
        return None

    reply = _labelled(facts.effective_user_text, "assistant response")
    if not reply:
        return None
    customer = _labelled(facts.effective_user_text, "customer message")

    risky = bool(re.search(r"\bfraud\b|\blawyer\b|\blegal\b|\bsue\b|\bchargeback\b",
                           customer, re.I))
    escalates = bool(re.search(r"escalat|human|specialist|supervisor|our team will",
                               reply, re.I))

    score, faults = 10, []
    if _promises_refund(reply):
        score -= 6
        faults.append("it promised a refund before the order was checked")
    if not _asks_for_order_number(reply) and not re.search(r"ORD-", reply, re.I):
        score -= 2
        faults.append("it never asked for the order number")
    if not _is_polite(reply):
        score -= 1
        faults.append("the tone is curt")
    if risky and not escalates:
        score -= 4
        faults.append("the customer mentioned fraud or legal action and it did not escalate")

    score = max(1, min(10, score))
    if not faults:
        reason = "The response follows all four rules."
    else:
        joined = faults[0] if len(faults) == 1 else (
            ", ".join(faults[:-1]) + " and " + faults[-1])
        reason = "Rule violations: " + joined + "."
    return "grade.support_reply", json.dumps(
        {"score": score, "passed": score >= 7, "reason": reason}, indent=2)


def _reply_freeform_json(facts: PromptFacts) -> Optional[Tuple[str, str]]:
    """The "just ask for JSON" path, before the lesson introduces a schema.

    It returns clean, parseable JSON on purpose. Staging a syntax error would be
    dishonest - a good model usually does return valid JSON here. What it cannot
    do is agree with a schema nobody sent it, so the field NAMES and the enum
    values are its own invention. That is the real failure mode, and it is the one
    that survives contact with a live API key.
    """
    text = facts.effective_user_text
    if not re.search(r"\bextract\b", text, re.I):
        return None
    if not re.search(r"only valid json|valid json object|return only json", text, re.I):
        return None
    if re.search(r"\bintent\b", text, re.I):
        return None
    message = embedded_customer_message(facts)
    if not message:
        return None

    if re.search(ENUM_HINTS["damaged_item"], message, re.I):
        reason = "damaged"
    elif re.search(ENUM_HINTS["billing_dispute"], message, re.I):
        reason = "billing problem"
    else:
        reason = "return request"

    if re.search(ENUM_HINTS["replacement"], message, re.I):
        action = "send a replacement"
    elif re.search(ENUM_HINTS["refund"], message, re.I):
        action = "refund"
    else:
        action = "unknown"

    payload = {
        "order_id": extract_slots(message).get("order_id"),
        "item": _extract_item(message),
        "reason": reason,
        "requested_action": action,
        "photo_available": bool(re.search(r"photo|picture|image", message, re.I)),
    }
    return "extract.freeform_json", json.dumps(payload, indent=2)


#: Order matters only in that the first responder to return wins.
COMPUTED_REPLIES: Tuple[Callable[[PromptFacts], Optional[Tuple[str, str]]], ...] = (
    _reply_classify,
    _reply_grade,
    _reply_freeform_json,
)


def compute_reply(facts: PromptFacts) -> Optional[Tuple[str, str]]:
    for responder in COMPUTED_REPLIES:
        result = responder(facts)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# 8. Tool support
# ---------------------------------------------------------------------------


ENUM_HINTS: Dict[str, str] = {
    "damaged_item": r"damaged|arrived broken|\bbroken\b|scratched|defective|cracked",
    "wrong_item": r"wrong item|not what i ordered",
    "billing_dispute": r"charged twice|duplicate charge|billing",
    "changed_mind": r"changed my mind",
    "refund": r"\brefund\b|money back|want to return|send it back|return it",
    "replacement": r"replacement|replace it",
    "exchange": r"\bexchange\b",
    "store_credit": r"store credit|gift card",
    "high": r"\btoday\b|urgent|asap|immediately|right now",
    "low": r"no rush|whenever|not urgent",
    "policy_exception": r"\d+\s*months? ago|a year ago|years? ago"
                        r"|outside (?:the )?(?:normal )?policy|past the (?:return )?window"
                        r"|never used it|bought (?:this|it) (?:six|6|five|5|four|4)",
    "normal_return": r"not what i expected|changed my mind|wrong size"
                     r"|does ?n(?:o|')?t fit|no longer need|do ?n(?:o|')?t need it",
    "refund_request": r"\brefund\b|money back|want to return",
    "order_status": r"where is my order|supposed to arrive|tracking",
    "billing_issue": r"charged twice|duplicate charge|billing",
    "product_question": r"how do i|compatible",
}


def _coerce(value: Any, spec: dict) -> Any:
    types_ = spec.get("type")
    if isinstance(types_, list):
        types_ = next((t for t in types_ if t != "null"), "string")
    if value is None:
        return None
    try:
        if types_ == "integer":
            return int(float(value))
        if types_ == "number":
            return float(value)
        if types_ == "boolean":
            return bool(value) if not isinstance(value, str) else value.lower() == "true"
        if types_ == "array" and not isinstance(value, list):
            return [value]
        if types_ == "string" and not isinstance(value, str):
            return str(value)
    except (TypeError, ValueError):
        return value
    return value


def _from_schema(key: str, spec: dict, facts: PromptFacts) -> Any:
    text = facts.effective_user_text
    enum = spec.get("enum")
    if enum:
        best, best_len = None, 0
        for option in enum:
            pat = ENUM_HINTS.get(str(option))
            if not pat:
                continue
            m = re.search(pat, text, re.I)
            if m and len(m.group(0)) > best_len:
                best, best_len = option, len(m.group(0))
        if best is not None:
            return best
        # Nothing in the message pointed anywhere. A neutral option is a more
        # honest default than "unclear", which should mean the message was
        # ambiguous - not that this particular field was never mentioned.
        for neutral in ("normal", "medium", "none"):
            if neutral in enum:
                return neutral
        return "unclear" if "unclear" in enum else enum[0]

    types_ = spec.get("type")
    nullable = isinstance(types_, list) and "null" in types_
    if isinstance(types_, list):
        types_ = next((t for t in types_ if t != "null"), "string")

    if types_ == "boolean":
        return False
    if types_ in ("number", "integer"):
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        return _coerce(m.group(1), spec) if m else 0
    if types_ == "array":
        return []
    if nullable:
        return None
    return text[:200] if key in ("customer_message", "message", "description") else ""


#: Products the course's fixtures talk about. A closed list on purpose - guessing a
#: noun out of free text is exactly the kind of thing a simulator should not pretend
#: to do well. Add to it rather than making it clever.
_ITEM_WORDS = (r"\b(shoes|sneakers|boots|jacket|headphones|earbuds|speaker|laptop"
               r"|keyboard|mouse|monitor|tablet|phone|watch|charger|camera|backpack)\b")

#: The customer already sent evidence, as opposed to offering to send it later.
_EVIDENCE_SENT = (r"attached|i (?:have |'ve )?sent (?:a |the )?photo|photo is attached"
                  r"|here (?:is|are) (?:a |the )?(?:photo|picture|image)"
                  r"|i(?:'ve| have) uploaded")


def _extract_item(text: str) -> Optional[str]:
    match = re.search(_ITEM_WORDS, text, re.I)
    if match:
        return match.group(1).lower()
    # "I got my shoes yesterday" - the thing right after "my", when it is not one
    # of the words that means something else in this domain.
    match = re.search(r"\bmy\s+([a-z]{3,20})\b", text, re.I)
    if match and match.group(1).lower() not in (
            "order", "money", "refund", "package", "parcel", "account", "card",
            "payment", "item", "purchase", "delivery"):
        return match.group(1).lower()
    return None


def _extract_evidence(text: str) -> bool:
    return bool(re.search(_EVIDENCE_SENT, text, re.I))


#: A message that says two incompatible things. Deliberately narrow - a simulator
#: guessing at contradictions would be worse than one that only spots the shapes
#: the course actually teaches.
_CONTRADICTION = (
    r"works (?:perfectly|fine).{0,80}(?:broken|damaged|does ?n(?:o|')?t work)"
    r"|(?:arrived broken|is damaged).{0,80}works (?:perfectly|fine)"
    r"|\brefund\b.{0,80}(?:send me another|another one|replacement)"
    r"|\breplacement\b.{0,80}\brefund\b"
)


def _extract_conflict(text: str) -> bool:
    return bool(re.search(_CONTRADICTION, text, re.I | re.S))


def _extract_conflict_reason(text: str) -> Optional[str]:
    if not _extract_conflict(text):
        return None
    return "The message asks for two different outcomes at once."


#: Field-name-driven extraction, consulted before the generic schema fallback.
#: Keyed by the names this course's schemas actually use.
FIELD_EXTRACTORS: Dict[str, Callable[[str], Any]] = {
    "item": _extract_item,
    "product": _extract_item,
    "evidence_provided": _extract_evidence,
    "conflict_detected": _extract_conflict,
    "conflict_reason": _extract_conflict_reason,
}


def _fill_missing_information(out: Dict[str, Any], props: Dict[str, Any]) -> None:
    """List the fields that came back null, which is what the field is for.

    Doing it as a post-pass rather than a guess keeps it honest: the list can only
    ever name fields this same extraction really did leave empty.
    """
    spec = props.get("missing_information") or {}
    if not spec or out.get("missing_information"):
        return
    gaps = [name for name, value in out.items()
            if value is None and name != "missing_information"
            and not name.endswith("_detail") and not name.endswith("_reason")]
    out["missing_information"] = gaps


def _fill_derived(out: Dict[str, Any], props: Dict[str, Any]) -> None:
    """Fill confidence and human_review_required from the rest of the extraction.

    Both are computed rather than read out of the text. `confidence` especially:
    the generic number fallback hunts for the first digits in the prompt, which on
    "return order 12345" would confidently report a confidence of 12345.
    """
    if "confidence" in props:
        score = 0.92
        if out.get("conflict_detected"):
            score = 0.45
        elif out.get("reason") in ("unclear", "other"):
            score = 0.55
        elif out.get("desired_action") in ("unclear", "other"):
            score = 0.6
        elif out.get("reason") == "policy_exception":
            score = 0.78
        if out.get("order_id") is None:
            score -= 0.05
        out["confidence"] = round(score, 2)

    if "human_review_required" in props:
        confidence = out.get("confidence")
        out["human_review_required"] = bool(
            out.get("conflict_detected")
            or out.get("reason") in ("policy_exception", "unclear")
            or (isinstance(confidence, (int, float))
                and not isinstance(confidence, bool) and confidence < 0.7))


def synthesize_input(tool: dict, facts: PromptFacts, bb: Blackboard) -> dict:
    """Build a tool input that satisfies the schema's `required` list exactly.

    Exactness matters: the course's agentic loop dispatches with `fn(**tool_input)`,
    so a stray key raises TypeError and a missing key raises TypeError too.
    """
    schema = tool.get("input_schema") or {}
    props = schema.get("properties") or {}
    keys = list(schema.get("required") or props.keys())
    if LabConfig.include_optional_tool_inputs:
        keys += [k for k in props if k not in keys]

    out: Dict[str, Any] = {}
    for key in keys:
        spec = props.get(key, {})
        value = bb.facts.get(key)
        if value is None:
            slots = extract_slots(facts.effective_user_text)
            value = slots.get(key)
        if value is None and key in FIELD_EXTRACTORS:
            value = FIELD_EXTRACTORS[key](facts.effective_user_text)
        if value is None:
            value = _from_schema(key, spec, facts)
        out[key] = _coerce(value, spec)
    _fill_missing_information(out, props)
    _fill_derived(out, props)
    return out


def _find_tool(tools: Sequence[dict], name: str) -> Optional[dict]:
    for tool in tools:
        if tool.get("name") == name:
            return tool
    return None


def decide_tool(tools: List[dict], tool_choice: Any, facts: PromptFacts,
                bb: Blackboard) -> Optional[Tuple[dict, dict, bool]]:
    """Return (tool, synthesized_input, forced) or None to answer with text instead."""
    if not tools:
        return None

    kind = tool_choice.get("type") if isinstance(tool_choice, dict) else (tool_choice or "auto")
    if kind == "none":
        return None

    if kind == "tool":
        name = tool_choice.get("name")
        tool = _find_tool(tools, name)
        if tool is None:
            raise BadRequestError(
                "tool_choice.name {!r} is not present in tools".format(name))
        return tool, synthesize_input(tool, facts, bb), True

    if len(bb.tools_called) >= LabConfig.max_tool_turns:
        return None

    # auto / any: call the first tool that has not been used yet.
    for tool in tools:
        if tool.get("name") not in bb.tools_called:
            return tool, synthesize_input(tool, facts, bb), False

    if kind == "any":
        tool = tools[0]
        return tool, synthesize_input(tool, facts, bb), False
    return None


# ---------------------------------------------------------------------------
# 9. Rendering and the create() dispatch
# ---------------------------------------------------------------------------


def _apply_word_limit(text: str, limit: Optional[int]) -> str:
    if not limit:
        return text
    words = text.split()
    if len(words) <= limit:
        return text
    clipped = " ".join(words[:limit])
    return clipped.rstrip(",;:") + "."


def _obey_end_marker(text: str, stop_sequences: Sequence[str], facts: PromptFacts) -> str:
    """A well-behaved model emits the end marker the prompt asked it to emit."""
    for seq in stop_sequences:
        if seq and seq.lower() in facts.effective_user_text.lower():
            return (text.rstrip() + "\n" + seq +
                    "\nText after the stop sequence is never returned by the API.")
    return text


def _earliest_stop(text: str, stop_sequences: Sequence[str]) -> Tuple[Optional[str], Optional[int]]:
    best_seq, best_idx = None, None
    for seq in stop_sequences:
        if not seq:
            continue
        idx = text.find(seq)
        if idx != -1 and (best_idx is None or idx < best_idx):
            best_seq, best_idx = seq, idx
    return best_seq, best_idx


#: Placeholder left in a lab skeleton. `...` is real Python (Ellipsis), so without this
#: check it would travel a long way before failing with something unhelpful.
def _has_blank(value: Any) -> bool:
    if value is Ellipsis:
        return True
    if isinstance(value, dict):
        return any(_has_blank(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_blank(v) for v in value)
    return False


def _reject_blanks(**kwargs: Any) -> None:
    blanks = sorted(name for name, value in kwargs.items() if _has_blank(value))
    if not blanks:
        return
    which = blanks[0] if len(blanks) == 1 else ", ".join(blanks)
    raise BadRequestError(
        "You still have ... in this request ({}). Replace every ... with the value "
        "named in the comment beside it, then run the cell again.".format(which))


class MockMessages:
    def __init__(self, client: "MockAnthropic") -> None:
        self._client = client

    def create(self, *, model: str, max_tokens: int, messages: Sequence[Any],
               system: Any = None, temperature: Any = None,
               stop_sequences: Optional[Sequence[str]] = None,
               tools: Optional[Sequence[dict]] = None, tool_choice: Any = None,
               stream: bool = False, **extra: Any) -> Message:

        # Labs hand the learner a skeleton with ... in the places to fill in. Catch a
        # forgotten one here, where we can say what to do, rather than letting it fail
        # somewhere deeper with a message about Ellipsis.
        _reject_blanks(model=model, max_tokens=max_tokens, messages=messages,
                       system=system, temperature=temperature,
                       stop_sequences=stop_sequences, tools=tools,
                       tool_choice=tool_choice)

        if stream:
            raise NotImplementedError(
                "shopassist_lab does not simulate streaming. Set stream=False, or run "
                "with a real ANTHROPIC_API_KEY. See lab_info() for the full gap list.")
        for key in extra:
            _warn_once("parameter {!r} is accepted but ignored by the simulator.".format(key))
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise BadRequestError("max_tokens must be a positive integer")
        if not messages:
            raise BadRequestError("messages must not be empty")

        self._client._calls += 1
        call_index = self._client._calls

        system_text = _flatten(system)
        temp = 0.0 if temperature is None else float(temperature)
        stops = tuple(stop_sequences or ())
        tool_list = list(tools or [])

        facts = analyze_prompt(system_text, messages)
        bb = build_blackboard(messages)

        fingerprint = _fingerprint(system_text, facts.effective_user_text,
                                   facts.turns, [t.get("name") for t in tool_list])
        rng = _rng(fingerprint, call_index if temp > 0 else 0)

        # --- tools first -------------------------------------------------
        decision = decide_tool(tool_list, tool_choice, facts, bb)
        if decision is not None:
            tool, tool_input, forced = decision
            block = ToolUseBlock(
                id="toolu_sim_" + fingerprint[:16],
                name=tool["name"],
                input=tool_input,
            )
            if forced or not LabConfig.tool_preambles:
                content: List[Any] = [block]
            else:
                content = [TextBlock("Let me look that up."), block]
            return self._finish(content, model, "tool_use", None, messages,
                                "tool:" + tool["name"], temp, call_index, fingerprint,
                                has_system=facts.has_system)

        # --- a reply computed from the customer's own words ---------------
        computed = compute_reply(facts)
        if computed is not None:
            trace, text = computed
            variant_index = 0
        else:
            # --- otherwise, a fixture -------------------------------------
            scenario = select_scenario(facts, bb, bool(tool_list))
            trace = scenario.id
            text, variant_index = pick_variant(scenario.variants, temp, rng, fingerprint)
            text = _apply_word_limit(text, facts.word_limit)
            text = _obey_end_marker(text, stops, facts)

        # stop_sequences and max_tokens race; whichever comes first wins.
        budget = max_tokens * LabConfig.chars_per_token
        seq, idx = _earliest_stop(text, stops)
        if idx is not None and idx <= budget:
            return self._finish([TextBlock(text[:idx])], model, "stop_sequence", seq,
                                messages, trace, temp, call_index, fingerprint,
                                variant_index, has_system=facts.has_system)
        if len(text) > budget:
            return self._finish([TextBlock(text[:int(budget)])], model, "max_tokens", None,
                                messages, trace, temp, call_index, fingerprint,
                                variant_index, has_system=facts.has_system)
        return self._finish([TextBlock(text)], model, "end_turn", None,
                            messages, trace, temp, call_index, fingerprint,
                            variant_index, has_system=facts.has_system)

    def _finish(self, content: List[Any], model: str, stop_reason: str,
                stop_sequence: Optional[str], messages: Sequence[Any], trace: str,
                temperature: float, call_index: int, fingerprint: str,
                variant_index: int = 0, has_system: bool = False) -> Message:
        out_text = "".join(getattr(b, "text", "") for b in content)
        out_tokens = est_tokens(out_text) + sum(
            est_tokens(json.dumps(b.input)) for b in content if b.type == "tool_use")
        in_text = "\n".join(_flatten(
            m.get("content") if isinstance(m, dict) else getattr(m, "content", None))
            for m in messages)

        msg = Message(
            id="msg_sim_" + fingerprint[:12],
            content=content,
            model=model,
            stop_reason=stop_reason,
            stop_sequence=stop_sequence,
            usage=Usage(input_tokens=est_tokens(in_text), output_tokens=out_tokens),
        )
        msg._lab = {
            "scenario": trace,
            "temperature": temperature,
            "call_index": call_index,
            "variant": variant_index,
            "fingerprint": fingerprint,
            "has_system": has_system,
        }
        if LabConfig.verbose:
            print("[sim] {} | temp={} | stop={} | out={}tok".format(
                trace, temperature, stop_reason, out_tokens))
        return msg


class MockAnthropic:
    """Offline stand-in for anthropic.Anthropic."""

    def __init__(self, **kwargs: Any) -> None:
        self._calls = 0
        self.messages = MockMessages(self)
        self.api_key = kwargs.get("api_key")
        _CLIENTS.append(self)
        _banner(live=False)

    def __repr__(self) -> str:
        return "<shopassist_lab.MockAnthropic simulated=True calls={}>".format(self._calls)


def _banner(live: bool) -> None:
    global _BANNER_SHOWN
    if _BANNER_SHOWN:
        return
    _BANNER_SHOWN = True
    if live:
        print("[shopassist_lab] LIVE MODE - real anthropic SDK, real API calls, real cost.")
    else:
        print("[shopassist_lab] SIMULATED MODE - replies are canned fixtures, not Claude.")
        print("[shopassist_lab] No ANTHROPIC_API_KEY found, so everything runs offline.")
        print("[shopassist_lab] Call lab_info() for details, explain(message) for any reply.")


def Anthropic(**kwargs: Any) -> Any:  # noqa: N802 - deliberately mirrors the SDK name
    """Return a real anthropic client when a key is available, otherwise the simulator."""
    if _has_real_key():
        try:
            real = sys.modules.get("anthropic")
            if real is None or getattr(real, "__shopassist_lab__", False):
                import importlib
                for name in list(sys.modules):
                    if name == "anthropic" and getattr(sys.modules[name],
                                                       "__shopassist_lab__", False):
                        del sys.modules[name]
                real = importlib.import_module("anthropic")
            if not getattr(real, "__shopassist_lab__", False):
                _banner(live=True)
                return real.Anthropic(**kwargs)
        except Exception as exc:  # pragma: no cover - depends on the environment
            _warn_once("a key is set but the real anthropic SDK could not be loaded "
                       "({}). Falling back to the simulator.".format(exc))
    return MockAnthropic(**kwargs)


# ---------------------------------------------------------------------------
# 10. dotenv replacement and shim installation
# ---------------------------------------------------------------------------


def _dotenv_values(path: str = ".env") -> Dict[str, str]:
    values: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip("'\"")
    except OSError:
        pass
    return values


def load_dotenv(dotenv_path: Optional[str] = None, *_: Any, **__: Any) -> bool:
    """Minimal python-dotenv replacement. Absent .env is not an error."""
    path = dotenv_path or ".env"
    values = _dotenv_values(path)
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return bool(values)


# anthropic 1.0 (August 2026) removed temperature / top_p / top_k from
# messages.create(). A lab notebook taken home and run against that SDK would die
# with a TypeError on the very line the temperature lab is about, so when the
# installed SDK has dropped the parameter we forward it through extra_body and the
# code the student wrote keeps running unchanged. Newer models (Claude Sonnet 5,
# Opus 4.7 and later) reject the parameter server-side; that 400 is left alone.
_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")


def _forward_sampling_params(kwargs: Dict[str, Any], version: str) -> Dict[str, Any]:
    """Move temperature / top_p / top_k out of the keyword arguments into extra_body."""
    moved = {k: kwargs.pop(k) for k in _SAMPLING_PARAMS if k in kwargs}
    if moved:
        _warn_once(
            "anthropic {} no longer accepts {} in messages.create(); forwarding it via "
            "extra_body so this notebook still runs. Claude Sonnet 5 / Opus 4.7+ reject "
            "the parameter altogether - drop it when you move to those models.".format(
                version, ", ".join(sorted(moved))))
        body = dict(kwargs.get("extra_body") or {})
        body.update(moved)
        kwargs["extra_body"] = body
    return kwargs


def _patch_real_sdk() -> None:
    """Let messages.create() on anthropic>=1 keep accepting the 0.x sampling params."""
    try:
        import inspect
        import anthropic as real
        from anthropic.resources.messages import Messages
    except Exception:  # pragma: no cover - depends on the environment
        return
    create = Messages.create
    if getattr(create, "__shopassist_lab__", False):
        return  # already patched in this process
    try:
        if "temperature" in inspect.signature(create).parameters:
            return  # this SDK still takes the parameter natively
    except (TypeError, ValueError):  # pragma: no cover
        return
    version = str(getattr(real, "__version__", "?"))

    def create_compat(self: Any, *args: Any, **kwargs: Any) -> Any:
        return create(self, *args, **_forward_sampling_params(kwargs, version))

    create_compat.__name__ = create.__name__
    create_compat.__doc__ = create.__doc__
    create_compat.__shopassist_lab__ = True  # type: ignore[attr-defined]
    create_compat.__wrapped_create__ = create  # type: ignore[attr-defined]
    Messages.create = create_compat  # type: ignore[assignment]


def install(force: bool = False) -> None:
    """Make `from anthropic import Anthropic` and `from dotenv import load_dotenv` work.

    Never shadows a genuinely installed package unless force=True, so the same
    notebook runs unchanged on a machine that has the real SDK.
    """
    if not force and _module_importable("anthropic"):
        _patch_real_sdk()
    if force or not _module_importable("anthropic"):
        shim = types.ModuleType("anthropic")
        shim.Anthropic = Anthropic
        shim.APIError = APIError
        shim.BadRequestError = BadRequestError
        shim.AuthenticationError = AuthenticationError
        shim.__version__ = __version__ + "+shopassist-lab"
        shim.__shopassist_lab__ = True

        types_mod = types.ModuleType("anthropic.types")
        types_mod.TextBlock = TextBlock
        types_mod.ToolUseBlock = ToolUseBlock
        types_mod.Message = Message
        types_mod.Usage = Usage
        shim.types = types_mod

        sys.modules["anthropic"] = shim
        sys.modules["anthropic.types"] = types_mod

    if force or not _module_importable("dotenv"):
        shim = types.ModuleType("dotenv")
        shim.load_dotenv = load_dotenv
        shim.dotenv_values = _dotenv_values
        shim.find_dotenv = lambda *a, **k: ""
        shim.__shopassist_lab__ = True
        sys.modules["dotenv"] = shim


# ---------------------------------------------------------------------------
# 11. Introspection helpers
# ---------------------------------------------------------------------------


def lab_info() -> None:
    """Print exactly what is real and what is simulated."""
    live = _has_real_key()
    print("shopassist_lab {}".format(__version__))
    print("  mode: {}".format("LIVE (real Claude API)" if live else "SIMULATED (offline)"))
    print("")
    print("  Faithfully simulated:")
    print("    max_tokens -> truncation and stop_reason='max_tokens'")
    print("    stop_sequences -> stop_reason='stop_sequence' and .stop_sequence")
    print("    temperature -> 0 repeats exactly, >0 varies reproducibly")
    print("    system -> detected rules change which reply is returned")
    print("    tools / tool_choice -> real tool_use blocks, schema-valid input")
    print("    message shape -> .content, .stop_reason, .usage, .model_dump_json()")
    print("")
    print("  NOT simulated (raises or is ignored):")
    print("    streaming, prompt caching, extended thinking, batches, vision")
    print("    genuine language understanding - replies are fixtures, not generated")
    print("")
    print("  Scenarios loaded: {}".format(len(SCENARIOS)))
    print("  Checks available: {}".format(", ".join(str(k) for k in sorted(TASKS))))
    print("")
    print("  To run against the real API: set ANTHROPIC_API_KEY and re-run the notebook.")


def explain(message: Any) -> None:
    """Explain why the simulator returned this particular reply."""
    if not getattr(message, "simulated", False):
        print("This is a real Claude response, not a simulated one - nothing to explain.")
        return
    meta = getattr(message, "_lab", {}) or {}
    trace = meta.get("scenario", "?")
    scenario = next((s for s in SCENARIOS + [FALLBACK] if s.id == trace), None)
    if scenario is None and trace in COMPUTED_NOTES:
        print("Reply:       {} (computed, not a fixture)".format(trace))
        print("Why:         {}".format(COMPUTED_NOTES[trace]))
    else:
        print("Scenario:    {} (lesson {})".format(
            trace, scenario.lesson if scenario else "?"))
        if scenario and scenario.note:
            print("Why:         {}".format(scenario.note))
    if scenario:
        print("Variant:     {} of {}{}".format(
            meta.get("variant", 0), len(scenario.variants),
            "  (temperature=0 always picks 0)" if meta.get("temperature", 0) == 0 else ""))
    print("stop_reason: {}".format(message.stop_reason))
    print("Tokens:      {} in, {} out".format(
        message.usage.input_tokens, message.usage.output_tokens))
    print("")
    print("Try next:    drop the system= argument and re-run, or set max_tokens=15,")
    print("             or raise temperature to 1.0 and call it three times.")


# ---------------------------------------------------------------------------
# 12. Self-check harness
#
# These checks are deliberately written against BEHAVIOUR, never against exact
# fixture strings, so every one of them also passes when a student re-runs the
# notebook with a real ANTHROPIC_API_KEY. They are themselves a worked example
# of the course's Domain 4 topic: code-based grading.
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    task_id: str
    passed: bool
    title: str
    message: str = ""
    hint: str = ""

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        head = "[{}] Task {} - {}".format(
            "PASS" if self.passed else "FAIL", self.task_id, self.title)
        body = "\n  " + self.message if self.message else ""
        tail = "\n  Hint: " + self.hint if (self.hint and not self.passed) else ""
        return head + body + tail


@dataclass
class _Task:
    task_id: str
    title: str
    #: Lab slug, e.g. "first_request". Deliberately not a lesson number - lessons get
    #: renumbered as the course grows, and these ids are shown to the learner.
    lab: str
    needs: Tuple[str, ...]
    fn: Callable


TASKS: Dict[str, _Task] = {}


def task(task_id: str, title: str, lab: str, needs: Sequence[str] = ()) -> Callable:
    def deco(fn: Callable) -> Callable:
        TASKS[str(task_id)] = _Task(str(task_id), title, lab, tuple(needs), fn)
        return fn
    return deco


def _ok(msg: str) -> Tuple[bool, str, str]:
    return True, msg, ""


def _fail(msg: str, hint: str = "") -> Tuple[bool, str, str]:
    return False, msg, hint


def check(task_id: Any, **kwargs: Any) -> CheckResult:
    """Verify one lab task. The result is truthy when the task passes."""
    key = str(task_id)
    entry = TASKS.get(key)
    if entry is None:
        return CheckResult(key, False, "unknown task",
                           "There is no check with id {!r}.".format(key),
                           "Available: " + ", ".join(sorted(TASKS)))
    missing = [n for n in entry.needs if n not in kwargs]
    if missing:
        return CheckResult(key, False, entry.title,
                           "This check needs {}.".format(", ".join(repr(m) for m in missing)),
                           "Call it as check({!r}, {})".format(
                               task_id, ", ".join(n + "=..." for n in entry.needs)))
    try:
        passed, msg, hint = entry.fn(**kwargs)
    except Exception as exc:
        return CheckResult(key, False, entry.title,
                           "The check raised {}: {}".format(type(exc).__name__, exc),
                           "That usually means a variable holds a different kind of "
                           "value than the task expected.")
    return CheckResult(key, passed, entry.title, msg, hint)


def check_all(**artifacts: Any) -> List[CheckResult]:
    """Run every check for which the required artifacts were supplied."""
    results = []
    # Registration order is notebook order. Sorting by id would scramble it now that
    # ids are names rather than numbers.
    for key in TASKS:
        entry = TASKS[key]
        if all(n in artifacts for n in entry.needs):
            result = check(key, **{n: artifacts[n] for n in entry.needs})
            results.append(result)
            print(result)
    passed = sum(1 for r in results if r.passed)
    print("\n{} of {} checks passed.".format(passed, len(results)))
    return results


# -- shared text predicates --------------------------------------------------


def _asks_for_order_number(text: str) -> bool:
    return bool(re.search(r"order\s*(?:number|id|#)", text, re.I))


def _promises_refund(text: str) -> bool:
    return bool(re.search(
        r"(?:i(?:'ve| have)|we(?:'ve| have))\s+(?:gone ahead and\s+)?"
        r"(?:approved|processed|issued|refunded)"
        r"|refund is approved|consider it refunded|i can refund that"
        r"|approved a full refund", text, re.I))


def _text_of(message: Any) -> str:
    if isinstance(message, str):
        return message
    blocks = getattr(message, "content", None) or []
    return "".join(getattr(b, "text", "") for b in blocks if getattr(b, "type", "") == "text")


def _role_of(msg: Any) -> str:
    return (msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")) or ""


# -- Lab 1 tasks -------------------------------------------------------------


@task("first_request", title="Make your first Claude request", lab="first_request", needs=("message",))
def _task1(message: Any, **_: Any) -> Tuple[bool, str, str]:
    if not hasattr(message, "content"):
        return _fail("`message` does not look like a Message object.",
                     "Assign the whole result: message = client.messages.create(...). "
                     "Do not assign message.content[0].text here.")
    if not message.content:
        return _fail("The message came back with no content blocks.")
    first = message.content[0]
    if getattr(first, "type", None) != "text":
        return _fail("content[0] is a {!r} block, not a text block.".format(
            getattr(first, "type", "?")),
            "Task 1 should not pass any tools, so the reply should be plain text.")
    if not first.text.strip():
        return _fail("The reply text is empty.")
    if message.stop_reason != "end_turn":
        return _fail("stop_reason is {!r}, so the reply is not complete.".format(
            message.stop_reason),
            "Use max_tokens=300 for this task so the model can finish its answer.")
    return _ok("You sent a request and read the reply out of message.content[0].text. "
               "stop_reason is 'end_turn', which means Claude finished on its own.")


@task("stop_reason", title="Read stop_reason and control response length", lab="first_request",
      needs=("short", "long"))
def _task2(short: Any, long: Any, **_: Any) -> Tuple[bool, str, str]:
    if short.stop_reason != "max_tokens":
        return _fail("The short call returned stop_reason={!r}, not 'max_tokens'.".format(
            short.stop_reason),
            "Your token budget was big enough to finish. Re-send the same request "
            "with max_tokens=20.")
    if long.stop_reason != "end_turn":
        return _fail("The long call returned stop_reason={!r}, not 'end_turn'.".format(
            long.stop_reason),
            "Raise max_tokens to 300 so the reply has room to complete.")
    short_text, long_text = _text_of(short), _text_of(long)
    if len(short_text) >= len(long_text):
        return _fail("The truncated reply is not shorter than the complete one.",
                     "Check that you actually passed the smaller max_tokens value.")
    return _ok("stop_reason told you why generation stopped: 'max_tokens' means you ran "
               "out of budget, 'end_turn' means Claude was finished. The agentic loop "
               "later in this course is built on exactly this signal.")


@task("response_object", title="Inspect the full response object", lab="first_request",
      needs=("message", "reply_text", "input_tokens", "output_tokens"))
def _task12_3(message: Any, reply_text: Any, input_tokens: Any, output_tokens: Any,
              **_: Any) -> Tuple[bool, str, str]:
    blanks = [n for n, v in (("reply_text", reply_text), ("input_tokens", input_tokens),
                             ("output_tokens", output_tokens)) if v is Ellipsis]
    if blanks:
        return _fail("You still have ... in {}.".format(", ".join(blanks)),
                     "Replace each ... with the value named in the comment beside it, "
                     "then run the cell again.")
    if reply_text != _text_of(message):
        return _fail("`reply_text` is not the text carried by this message.",
                     "Read it out of the object: reply_text = message.content[0].text")
    usage = getattr(message, "usage", None)
    if usage is None:
        return _fail("This message has no `usage` field to read.")
    if input_tokens != usage.input_tokens:
        return _fail("`input_tokens` is {!r}, but message.usage.input_tokens is {!r}.".format(
            input_tokens, usage.input_tokens),
            "Assign it from the object rather than typing the number in.")
    if output_tokens != usage.output_tokens:
        return _fail("`output_tokens` is {!r}, but message.usage.output_tokens is {!r}.".format(
            output_tokens, usage.output_tokens),
            "Assign it from the object rather than typing the number in.")
    return _ok("The reply is one field among many. `usage` is how you track cost, `id` is "
               "what you log, and `stop_reason` is what your application branches on - "
               "which is exactly what the agentic loop does later in this course.")

@task("opening_message", title="Start the conversation", lab="multi_turn",
      needs=("opening",))
def _task_opening(opening: Any, **_: Any) -> Tuple[bool, str, str]:
    if not hasattr(opening, "content") or not opening.content:
        return _fail("`opening` does not look like a Message object.",
                     "Assign the whole result: opening = client.messages.create(...)")
    text = _text_of(opening)
    if not text.strip():
        return _fail("The reply is empty.")
    if opening.stop_reason != "end_turn":
        return _fail("stop_reason is {!r}, so the reply is not complete.".format(
            opening.stop_reason), "Use max_tokens=300 so the reply has room to finish.")
    if not _asks_for_order_number(text):
        return _fail("ShopAssist replied, but it did not ask for the order number.",
                     "Send the opening complaint - 'I want to return my order.' - and "
                     "ShopAssist should ask which order you mean.")
    return _ok("ShopAssist asked for the order number, which is exactly what a support "
               "agent should do first. Keep that reply in mind - the next task answers "
               "it, and the API will have forgotten this exchange ever happened.")


@task("no_history", title="Send a follow-up with no history", lab="multi_turn",
      needs=("without_history",))
def _task13_1(without_history: Any, **_: Any) -> Tuple[bool, str, str]:
    lonely = _text_of(without_history)
    if not lonely.strip():
        return _fail("The reply is empty.")
    confused = re.search(r"do ?n(?:o|')?t (?:have|know)|no (?:earlier|prior)"
                         r"|first message|what would you like|not sure", lonely, re.I)
    if not confused:
        return _fail("The reply does not show any sign of missing context.",
                     "Send ONLY the order number as a single user message, with no "
                     "previous turns in the list.")
    return _ok("The model had nothing to attach the order number to, because this "
               "request carried no earlier turns. Every API call starts from nothing.")


@task("with_history", title="Send the same follow-up with history", lab="multi_turn",
      needs=("without_history", "with_history"))
def _task13_2(without_history: Any, with_history: Any, **_: Any) -> Tuple[bool, str, str]:
    lonely = _text_of(without_history)
    informed = _text_of(with_history)
    if not informed.strip():
        return _fail("The with-history reply is empty.")
    if lonely.strip() == informed.strip():
        return _fail("Both calls returned the same reply.",
                     "The second call must include the earlier turns in its messages "
                     "list. Print both messages lists and compare their length.")
    return _ok("Same final sentence, two different answers. The API stores nothing "
               "between calls - your application owns the conversation.")


@task("helpers", title="Write the three conversation helpers", lab="helpers",
      needs=("add_user_message", "add_assistant_message", "chat"))
def _task14_1(add_user_message: Any, add_assistant_message: Any, chat: Any,
              **_: Any) -> Tuple[bool, str, str]:
    for name, fn in (("add_user_message", add_user_message),
                     ("add_assistant_message", add_assistant_message),
                     ("chat", chat)):
        if not callable(fn):
            return _fail("`{}` is not a function.".format(name),
                         "Define it with def {}(...).".format(name))
    scratch: List[Any] = []
    try:
        add_user_message(scratch, "ping")
        add_assistant_message(scratch, "pong")
    except Exception as exc:
        return _fail("Calling the helpers raised {}: {}".format(type(exc).__name__, exc),
                     "Both take (messages, text) and append to the list in place.")
    if len(scratch) != 2:
        return _fail("After one call each the list holds {} item(s), not 2.".format(
            len(scratch)),
            "Each helper must append to the list it is given, not return a new one.")
    if any(_role_of(m) is Ellipsis or _role_of(m) == "" for m in scratch):
        return _fail("At least one helper still has ... where the role should be.",
                     "There are only two roles: 'user' for the customer and "
                     "'assistant' for ShopAssist. Put the matching one in each helper.")
    if _role_of(scratch[0]) != "user" or _role_of(scratch[1]) != "assistant":
        return _fail("The roles came out as {!r}.".format(
            [_role_of(m) for m in scratch]),
            "add_user_message writes role 'user'; add_assistant_message writes "
            "role 'assistant'.")
    if scratch[0].get("content") != "ping" or scratch[1].get("content") != "pong":
        return _fail("The text was not stored under the 'content' key.",
                     "Append {'role': ..., 'content': text}.")
    try:
        reply = chat([{"role": "user", "content": "I want to return my order."}])
    except Exception as exc:
        return _fail("chat() raised {}: {}".format(type(exc).__name__, exc),
                     "chat(messages) should call client.messages.create(...) and "
                     "return the reply text.")
    if not isinstance(reply, str):
        return _fail("chat() returned {}, not a string.".format(type(reply).__name__),
                     "Return message.content[0].text, not the whole message object.")
    if not reply.strip():
        return _fail("chat() returned an empty string.")
    return _ok("All three helpers behave. chat() returns the text and deliberately "
               "does not append it - the caller does that, which is what lets you "
               "inspect or log a reply before it becomes part of the history.")


@task("first_exchange", title="Run the first exchange through the helpers",
      lab="helpers", needs=("messages",))
def _task14_2(messages: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(messages, list):
        return _fail("`messages` should be a list, got {}.".format(
            type(messages).__name__),
            "Start the cell with messages = [] and let the helpers fill it.")
    if len(messages) == 0:
        return _fail("The conversation is empty.",
                     "Call add_user_message(messages, ...) to put the first turn in.")
    if len(messages) == 1:
        return _fail("The conversation holds only the customer's turn.",
                     "Claude's reply is not saved yet. Pass the answer to "
                     "add_assistant_message(messages, answer).")
    if len(messages) != 2:
        return _fail("The conversation holds {} turns; this task builds exactly "
                     "2.".format(len(messages)),
                     "Run the cell once, from messages = [] down. Running it twice "
                     "appends the same exchange again.")
    roles = [_role_of(m) for m in messages]
    if roles != ["user", "assistant"]:
        return _fail("The roles came out as {!r}, not ['user', 'assistant'].".format(
            roles),
            "add_user_message goes first, then the reply from chat() goes in via "
            "add_assistant_message.")
    reply = messages[1].get("content") if isinstance(messages[1], dict) else None
    if not isinstance(reply, str):
        return _fail("The assistant turn stores a {} instead of text.".format(
            type(reply).__name__),
            "chat() already returns the text. Pass that returned value straight to "
            "add_assistant_message - not the whole response object.")
    if not reply.strip():
        return _fail("The assistant turn is empty text.")
    return _ok("Two turns: the customer, then ShopAssist. The list is a plain Python "
               "list living in your program - that is the entire memory of this "
               "conversation, and the next task adds to it.")


@task("two_turns", title="Add the follow-up to the same conversation", lab="helpers",
      needs=("messages",))
def _task14_3(messages: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(messages, list):
        return _fail("`messages` should be a list, got {}.".format(type(messages).__name__))
    if len(messages) == 2:
        return _fail("The conversation still holds only the first exchange.",
                     "This task adds to the SAME messages list. Do not start from "
                     "messages = [] again - just add the follow-up on top.")
    if len(messages) == 3:
        return _fail("The follow-up went in, but Claude's second reply did not.",
                     "Pass the answer to add_assistant_message(messages, answer) so "
                     "the history stays complete.")
    if len(messages) < 4:
        return _fail("The conversation has {} message(s); expected at least 4 "
                     "(two exchanges).".format(len(messages)),
                     "Call add_user_message, then chat, then add_assistant_message - "
                     "twice.")
    roles = [_role_of(m) for m in messages]
    if roles[0] != "user":
        return _fail("The conversation starts with a {!r} turn.".format(roles[0]),
                     "A conversation always starts with a user message.")
    for i in range(1, len(roles)):
        if roles[i] == roles[i - 1]:
            return _fail("Two {!r} turns in a row at position {}.".format(roles[i], i),
                         "You probably called chat() twice without appending the reply. "
                         "Every chat() result must go back in via add_assistant_message.")
    if roles[-1] != "assistant":
        return _fail("The conversation ends on a {!r} turn.".format(roles[-1]),
                     "Append the final reply with add_assistant_message so the history "
                     "is ready for the next turn.")
    return _ok("{} turns, strictly alternating, ending on the assistant. Your helpers "
               "keep the history in the shape the API expects.".format(len(messages)))


# -- Prompt evaluation tasks -------------------------------------------------


_EVAL_KEYS = ("input", "expected_intent")


def _eval_cases(test_cases: Any) -> Optional[Tuple[bool, str, str]]:
    """Shared shape check for the eval dataset."""
    if not isinstance(test_cases, list):
        return _fail("`test_cases` should be a list, got {}.".format(
            type(test_cases).__name__))
    if len(test_cases) != 4:
        return _fail("The dataset holds {} case(s); this lab uses 4.".format(
            len(test_cases)),
            "Do not add or remove cases - just fill in the blanks in the ones given.")
    for i, case in enumerate(test_cases, 1):
        if not isinstance(case, dict):
            return _fail("Case {} is a {}, not a dictionary.".format(
                i, type(case).__name__))
        for key in _EVAL_KEYS:
            if key not in case:
                return _fail("Case {} has no {!r} key.".format(i, key))
        label = case["expected_intent"]
        if label is Ellipsis:
            return _fail("Case {} still has ... for its expected intent.".format(i),
                         "Read the customer message and pick the label that fits. "
                         "The allowed ones are: " + ", ".join(CLASSIFY_LABELS) + ".")
        if label not in CLASSIFY_LABELS:
            return _fail("Case {} expects {!r}, which is not one of our labels.".format(
                i, label),
                "Allowed labels: " + ", ".join(CLASSIFY_LABELS) + ".")
    return None


@task("eval_dataset", title="Build the evaluation dataset", lab="evaluation",
      needs=("test_cases",))
def _task17_1(test_cases: Any, **_: Any) -> Tuple[bool, str, str]:
    problem = _eval_cases(test_cases)
    if problem is not None:
        return problem
    # The first three messages say one thing each; the fourth is ambiguous on
    # purpose and is handed to the learner already labelled.
    for i, case in enumerate(test_cases[:3], 1):
        wanted = classify_intent_label(case["input"])
        if case["expected_intent"] != wanted:
            return _fail("Case {} expects {!r}.".format(i, case["expected_intent"]),
                         "Re-read that message: \"{}\" It is a {}.".format(
                             case["input"][:70], wanted))
    return _ok("Four messages, and for each one you wrote down the answer you expect "
               "BEFORE running anything. That is the whole difference between a demo "
               "and an eval: a demo shows you an output, an eval compares an output "
               "against an answer you committed to in advance.")


@task("eval_run", title="Run the prompt over every case", lab="evaluation",
      needs=("results",))
def _task17_2(results: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(results, list):
        return _fail("`results` should be a list, got {}.".format(
            type(results).__name__),
            "Start with results = [] and append one entry per test case.")
    if len(results) != 4:
        return _fail("`results` holds {} entry/entries; there are 4 test "
                     "cases.".format(len(results)),
                     "Check that results.append(...) is inside the for loop.")
    for i, row in enumerate(results, 1):
        if not isinstance(row, dict):
            return _fail("Entry {} is a {}, not a dictionary.".format(
                i, type(row).__name__))
        for key in ("input", "expected", "actual"):
            if key not in row:
                return _fail("Entry {} has no {!r} key.".format(i, key))
        if row["actual"] not in CLASSIFY_LABELS:
            return _fail("Entry {} got {!r} back, which is not one of our "
                         "labels.".format(i, row["actual"]),
                         "classify_intent() returns a dictionary - the label is under "
                         "the 'intent' key.")
        wanted = classify_intent_label(row["input"])
        if row["actual"] != wanted:
            return _fail("Entry {} reports {!r} for a message that classifies as "
                         "{!r}.".format(i, row["actual"], wanted),
                         "Pass test_case['input'] into classify_intent - each entry "
                         "must be the answer for its OWN message.")
    return _ok("Four messages, four labels, one prompt. Nothing has been graded yet - "
               "you are only looking at output, which is exactly where most people "
               "stop. The next task is the part that makes it an eval.")


@task("eval_compare", title="Compare actual with expected", lab="evaluation",
      needs=("results",))
def _task17_3(results: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(results, list) or len(results) != 4:
        return _fail("Expected the 4 results from the previous task.",
                     "Run the previous cell first, then this one.")
    for i, row in enumerate(results, 1):
        if "passed" not in row:
            return _fail("Entry {} has no 'passed' key.".format(i),
                         "Set result['passed'] inside the loop.")
        if row["passed"] is Ellipsis:
            return _fail("Entry {} still has ... for its comparison.".format(i),
                         "passed is the answer to one question: is actual the same as "
                         "expected?")
        if not isinstance(row["passed"], bool):
            return _fail("Entry {} stores {!r} under 'passed'.".format(i, row["passed"]),
                         "Comparing two values with == gives True or False. Do not "
                         "wrap it in quotes.")
        wanted = row["actual"] == row["expected"]
        if row["passed"] != wanted:
            return _fail("Entry {} is marked {} but actual is {!r} and expected is "
                         "{!r}.".format(i, row["passed"], row["actual"],
                                        row["expected"]),
                         "Compare the two the other way round - actual == expected.")
    hits = sum(1 for r in results if r["passed"])
    if hits == len(results):
        return _ok("{}/{} passed. Worth noticing: a perfect score on four cases is not "
                   "proof of a good prompt, it is proof of a small dataset.".format(
                       hits, len(results)))
    misses = [r for r in results if not r["passed"]]
    return _ok("{}/{} passed, and the one that failed is the interesting one: {!r} was "
               "labelled {!r} but came back {!r}. That message contains two problems at "
               "once, so both answers are defensible - which is a decision your team "
               "has to make and write down, not something the model can guess. Change "
               "the prompt, run the same four cases again, and you can see whether the "
               "change helped.".format(
                   hits, len(results), misses[0]["input"][:60],
                   misses[0]["expected"], misses[0]["actual"]))


# -- Output grading tasks ----------------------------------------------------


@task("code_grading", title="Grade the structure with plain Python", lab="grading",
      needs=("is_valid_json", "has_required_fields", "has_valid_intent"))
def _task18_1(is_valid_json: Any, has_required_fields: Any, has_valid_intent: Any,
              **_: Any) -> Tuple[bool, str, str]:
    for name, fn in (("is_valid_json", is_valid_json),
                     ("has_required_fields", has_required_fields),
                     ("has_valid_intent", has_valid_intent)):
        if not callable(fn):
            return _fail("`{}` is not a function.".format(name))
    good = {"intent": "refund_request", "order_id": None, "needs_human_review": False}
    try:
        if not has_required_fields(dict(good)):
            return _fail("has_required_fields() rejects an output that has all three "
                         "fields.",
                         "Loop over REQUIRED_FIELDS and return False only when a field "
                         "is missing from data.")
        short = {"intent": "refund_request", "order_id": None}
        if has_required_fields(short):
            return _fail("has_required_fields() accepts an output with "
                         "needs_human_review missing.",
                         "The loop has to run over REQUIRED_FIELDS - all three of them "
                         "- not over the keys the output happens to have.")
        if not has_valid_intent(dict(good)):
            return _fail("has_valid_intent() rejects 'refund_request', which is one of "
                         "our labels.",
                         "Check membership against ALLOWED_INTENTS.")
        if has_valid_intent({"intent": "return_problem"}):
            return _fail("has_valid_intent() accepts 'return_problem', which our "
                         "backend cannot route.",
                         "The label has to be IN ALLOWED_INTENTS - that is the whole "
                         "job of this check.")
        if not is_valid_json('{"intent": "other"}') or is_valid_json("Sure! Here it is."):
            return _fail("is_valid_json() does not agree with json.loads().",
                         "This one is provided - if you changed it, put it back.")
    except Exception as exc:
        return _fail("A grader raised {}: {}".format(type(exc).__name__, exc),
                     "Each one takes a single argument and returns True or False.")
    return _ok("Three graders, no model involved, same answer every time you run them. "
               "Deterministic checks like these are the cheap half of an eval - run "
               "them first, because there is no point asking a model to judge the tone "
               "of an output your backend cannot even parse.")


@task("model_grading", title="Grade the wording with a second Claude call",
      lab="grading", needs=("grades",))
def _task18_2(grades: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(grades, list) or len(grades) != 2:
        return _fail("Expected 2 grades, one per output, got {}.".format(
            len(grades) if isinstance(grades, list) else type(grades).__name__),
            "Check that grades.append(...) is inside the loop.")
    for i, row in enumerate(grades, 1):
        grade = row.get("grade") if isinstance(row, dict) else None
        if not isinstance(grade, dict):
            return _fail("Entry {} has no grade dictionary.".format(i),
                         "grade_response() returns the parsed JSON - store that.")
        for key in ("score", "passed", "reason"):
            if key not in grade:
                return _fail("Entry {} has no {!r} in its grade.".format(i, key),
                             "The grading prompt asks for score, passed and reason. "
                             "Return json.loads(...) of the reply, not the reply text.")
        if not isinstance(grade["score"], (int, float)):
            return _fail("Entry {} scored {!r}, which is not a number.".format(
                i, grade["score"]))
    weak, good = grades[0]["grade"], grades[1]["grade"]
    if weak["score"] >= good["score"]:
        return _fail("The over-promising output scored {} and the careful one scored "
                     "{}.".format(weak["score"], good["score"]),
                     "Grade the 'reply' text of each output, in the order they appear "
                     "in OUTPUTS - and pass it as the assistant response, not as the "
                     "customer message.")
    return _ok("The first reply scored {} and the second {}. No `if` statement could "
               "have told them apart - both are valid JSON with the right fields. "
               "Judging whether a sentence over-promises is what the second model call "
               "is for. Keep in mind the grader is also a model: it can be wrong, and "
               "it is only as good as the rules you wrote in the grading "
               "prompt.".format(weak["score"], good["score"]))


@task("combined_score", title="Combine both scores", lab="grading", needs=("scored",))
def _task18_3(scored: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(scored, list) or len(scored) != 2:
        return _fail("Expected 2 scored rows, got {}.".format(
            len(scored) if isinstance(scored, list) else type(scored).__name__))
    for i, row in enumerate(scored, 1):
        for key in ("code_score", "model_score", "final_score"):
            if key not in row:
                return _fail("Row {} has no {!r}.".format(i, key))
            if row[key] is Ellipsis:
                return _fail("Row {} still has ... for {}.".format(i, key))
            if not isinstance(row[key], (int, float)) or isinstance(row[key], bool):
                return _fail("Row {} stores {!r} for {}, which is not a number.".format(
                    i, row[key], key))
        wanted = (row["code_score"] + row["model_score"]) / 2
        if abs(row["final_score"] - wanted) > 1e-9:
            return _fail("Row {} reports a final score of {} where code {} and model "
                         "{} average to {}.".format(
                             i, row["final_score"], row["code_score"],
                             row["model_score"], wanted),
                         "final_score is the two scores added together and divided "
                         "by 2.")
    first, second = scored[0], scored[1]
    if first["code_score"] != second["code_score"]:
        return _fail("The two outputs got different code scores.",
                     "Both are valid JSON with all three fields and an allowed label, "
                     "so the deterministic half cannot separate them - that is the "
                     "point of this task.")
    return _ok("Both outputs score {} on the code checks and {} against {} from the "
               "grader, so the combined scores are {} and {}. That gap is the argument "
               "for using both: the deterministic half proved the output was parseable "
               "and nothing more. Run the same dataset after changing the prompt and "
               "the averages are comparable - which is the only reason to put a number "
               "on any of this.".format(
                   first["code_score"], first["model_score"], second["model_score"],
                   first["final_score"], second["final_score"]))


# -- Structured output tasks -------------------------------------------------


#: What the course's extraction schema requires. Used by the checks to say which
#: fields a freeform reply left out.
RETURN_REQUEST_FIELDS = ("order_id", "item", "reason", "reason_detail",
                         "desired_action", "desired_action_detail",
                         "evidence_provided", "urgency", "missing_information")


@task("freeform_json", title="Ask for JSON without a schema", lab="structured_output",
      needs=("freeform",))
def _task25_1(freeform: Any, **_: Any) -> Tuple[bool, str, str]:
    blocks = getattr(freeform, "content", None)
    if not blocks:
        return _fail("`freeform` does not look like a response object.",
                     "Assign the whole result of client.messages.create(...) to it.")
    if any(getattr(b, "type", "") == "tool_use" for b in blocks):
        return _fail("This request came back as a tool call.",
                     "This task is the 'before' picture - send it with no tools at "
                     "all. The tool comes in the next task.")
    text = _text_of(freeform).strip()
    if not text:
        return _fail("The reply is empty.")
    try:
        data = json.loads(text)
    except ValueError:
        return _ok("The reply did not parse as JSON at all, which is the risk this "
                   "lesson is about: with no schema, whether your parser survives is "
                   "up to the wording of a prompt.")
    if not isinstance(data, dict):
        return _fail("The JSON parsed, but it is not an object.")
    absent = [f for f in RETURN_REQUEST_FIELDS if f not in data]
    extra = [k for k in data if k not in RETURN_REQUEST_FIELDS]
    if not absent:
        return _ok("It parsed and it happens to carry every field. Nothing in the "
                   "request guaranteed that - you asked in prose and got lucky.")
    return _ok("It parsed cleanly, so this is not a broken-JSON story. Look at the "
               "keys instead: {} field(s) our code expects are missing ({}), and it "
               "invented {} of its own ({}). You asked for structure in prose, so the "
               "structure is whatever the model felt like. The next task removes the "
               "guessing.".format(
                   len(absent), ", ".join(absent[:3]) + ("..." if len(absent) > 3 else ""),
                   len(extra), ", ".join(extra) or "none"))


@task("tool_extraction", title="Extract with a tool and a schema",
      lab="structured_output", needs=("extracted",))
def _task25_2(extracted: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(extracted, dict):
        return _fail("`extracted` should be the tool input dictionary, got {}.".format(
            type(extracted).__name__),
            "Find the block whose .type is 'tool_use' and take its .input.")
    absent = [f for f in RETURN_REQUEST_FIELDS if f not in extracted]
    if absent:
        return _fail("The extraction is missing {}.".format(", ".join(absent)),
                     "Every name in the schema's `required` list has to come back. "
                     "If they did not, check that tool_choice forces THIS tool.")
    if extracted["reason"] not in ("normal_return", "damaged_item", "billing_dispute",
                                   "policy_exception", "unclear", "other"):
        return _fail("reason came back as {!r}, which is not in the schema's "
                     "enum.".format(extracted["reason"]))
    if not isinstance(extracted["missing_information"], list):
        return _fail("missing_information is {}, not a list.".format(
            type(extracted["missing_information"]).__name__))
    gaps = extracted["missing_information"]
    return _ok("A dictionary, not a string - your code never had to parse anything. "
               "Every required field is present, reason came from the enum rather than "
               "the model's imagination, and order_id is {} with {} listed under "
               "missing_information. That last part matters: the schema let it say "
               "'I do not know' instead of inventing an order number.".format(
                   "null" if extracted["order_id"] is None else repr(extracted["order_id"]),
                   ", ".join(gaps) if gaps else "nothing"))


@task("routing", title="Route on the extracted fields", lab="structured_output",
      needs=("route",))
def _task25_3(route: Any, **_: Any) -> Tuple[bool, str, str]:
    if not callable(route):
        return _fail("`route` is not a function.")
    base = {"order_id": "ORD-12345678", "item": "shoes", "reason": "normal_return",
            "reason_detail": None, "desired_action": "refund",
            "desired_action_detail": None, "evidence_provided": False,
            "urgency": "normal", "missing_information": []}

    def call(**over: Any) -> Any:
        data = dict(base)
        data.update(over)
        try:
            return route(data)
        except Exception as exc:                        # noqa: BLE001
            return exc

    cases = (
        ("ask_for_order_id", dict(order_id=None, missing_information=["order_id"]),
         "an extraction with no order_id has to come back before anything else - you "
         "cannot process a return for an order you cannot identify"),
        ("billing_team", dict(reason="billing_dispute"),
         "a billing dispute is not a returns job"),
        ("human_escalation", dict(reason="policy_exception"),
         "a policy exception is exactly the case a person should see"),
        ("returns_workflow", {},
         "an ordinary return with everything present is the automatic path"),
    )
    for wanted, over, why in cases:
        got = call(**over)
        if isinstance(got, Exception):
            return _fail("route() raised {}: {}".format(type(got).__name__, got),
                         "It takes one extracted dictionary and returns a string.")
        if got != wanted:
            return _fail("route() returned {!r} where {!r} was expected.".format(
                got, wanted), why.capitalize() + ".")
    return _ok("Four different extractions, four different destinations, and the "
               "deciding code is plain `if` statements reading named fields. That is "
               "what the schema bought you: not a nicer-looking reply, but a value "
               "your application can branch on without parsing prose.")


# -- Validation tasks --------------------------------------------------------


def _sample_extraction(**over: Any) -> Dict[str, Any]:
    base = {
        "order_id": "ORD-12345678",
        "item": "shoes",
        "reason": "damaged_item",
        "desired_action": "replacement",
        "evidence_provided": True,
        "urgency": "normal",
        "confidence": 0.92,
        "conflict_detected": False,
        "conflict_reason": None,
        "missing_information": [],
    }
    base.update(over)
    return base


@task("schema_validation", title="Catch a value the schema should not have allowed",
      lab="validation", needs=("schema_errors",))
def _task26_1(schema_errors: Any, **_: Any) -> Tuple[bool, str, str]:
    if not callable(schema_errors):
        return _fail("`schema_errors` is not a function.")
    try:
        clean = schema_errors(_sample_extraction())
        bad = schema_errors(_sample_extraction(desired_action="exchange"))
    except Exception as exc:                                # noqa: BLE001
        return _fail("schema_errors() raised {}: {}".format(type(exc).__name__, exc),
                     "It takes one extraction dictionary and returns a list.")
    for name, value in (("a clean extraction", clean), ("a bad one", bad)):
        if not isinstance(value, list):
            return _fail("On {} it returned {}, not a list.".format(
                name, type(value).__name__),
                "Start with errors = [] and return it at the end, even when empty.")
    if clean:
        return _fail("A valid extraction produced {} error(s): {}.".format(
            len(clean), clean),
            "'replacement' is one of the allowed actions, so nothing should be "
            "reported for it.")
    if not bad:
        return _fail("desired_action 'exchange' produced no error.",
                     "Our returns policy supports four actions. Check membership "
                     "against ALLOWED_ACTIONS - 'exchange' is not one of them.")
    return _ok("A value that looks perfectly reasonable, is spelled correctly, and is "
               "the right type - and it is still wrong, because our policy has no "
               "'exchange'. This is the fixable kind of error: the customer's message "
               "does contain the answer, the model just filed it under a name we do "
               "not use. That is what a retry with specific feedback is for.")


@task("missing_information", title="Record what the message never said",
      lab="validation", needs=("missing_fields",))
def _task26_2(missing_fields: Any, **_: Any) -> Tuple[bool, str, str]:
    if not callable(missing_fields):
        return _fail("`missing_fields` is not a function.")
    try:
        full = missing_fields(_sample_extraction())
        no_order = missing_fields(_sample_extraction(order_id=None))
        neither = missing_fields(_sample_extraction(order_id=None, item=None))
    except Exception as exc:                                # noqa: BLE001
        return _fail("missing_fields() raised {}: {}".format(type(exc).__name__, exc),
                     "It takes one extraction dictionary and returns a list of field "
                     "names.")
    if not isinstance(full, list):
        return _fail("It returned {}, not a list.".format(type(full).__name__))
    if full:
        return _fail("An extraction with both fields present reported {}.".format(full),
                     "Only report a field when its value is None.")
    if no_order != ["order_id"]:
        return _fail("With order_id null it returned {!r}.".format(no_order),
                     "Expected exactly ['order_id'] - check the comparison is `is "
                     "None` and that the loop runs over MUST_BE_PRESENT.")
    if sorted(neither) != ["item", "order_id"]:
        return _fail("With both fields null it returned {!r}.".format(neither),
                     "Both names should be in the list.")
    return _ok("Nothing here asks Claude to try again, and that is the whole point. "
               "The customer never wrote an order number, so no amount of re-prompting "
               "will produce a real one - it would only produce an invented one. "
               "Retry when the model mishandled information that was there. When it "
               "was never there, return null, record the gap, and ask the customer.")


@task("human_review", title="Route the risky cases to a person", lab="validation",
      needs=("needs_human_review",))
def _task26_3(needs_human_review: Any, **_: Any) -> Tuple[bool, str, str]:
    if not callable(needs_human_review):
        return _fail("`needs_human_review` is not a function.")
    cases = (
        ("a clean, confident extraction", _sample_extraction(), False),
        ("a contradictory one", _sample_extraction(
            desired_action="unclear", conflict_detected=True,
            conflict_reason="The customer mentions both refund and replacement."), True),
        ("a low-confidence one", _sample_extraction(confidence=0.41), True),
        ("one just above the threshold", _sample_extraction(confidence=0.71), False),
    )
    for label, data, wanted in cases:
        try:
            got = needs_human_review(data)
        except Exception as exc:                            # noqa: BLE001
            return _fail("needs_human_review() raised {}: {}".format(
                type(exc).__name__, exc),
                "It takes one extraction dictionary and returns True or False.")
        if not isinstance(got, bool):
            return _fail("On {} it returned {!r}, which is not True or False.".format(
                label, got))
        if got != wanted:
            return _fail("On {} it returned {}, expected {}.".format(
                label, got, wanted),
                "Send a case to a person when a conflict was detected, or when "
                "confidence is below CONFIDENCE_THRESHOLD - and only then.")
    return _ok("Two ways to end up in front of a person, and neither is 'the model "
               "failed'. One case contradicts itself, so there is no correct answer to "
               "automate. The other is a guess the system is not sure of. Everything "
               "else goes through untouched, which is the point - a review queue that "
               "catches everything is a review queue nobody reads.")


# -- BUILD: the extraction module --------------------------------------------


CLEAN_MESSAGE = "I want to return order ORD-12345678. The headphones arrived broken."
NO_ORDER_MESSAGE = ("I bought a jacket last week and want to return it. "
                    "I do not have the order number.")


@task("pipeline", title="Wire extraction and validation into one call",
      lab="extract_returns", needs=("process",))
def _task28_1(process: Any, **_: Any) -> Tuple[bool, str, str]:
    if not callable(process):
        return _fail("`process` is not a function.")
    try:
        clean = process(CLEAN_MESSAGE)
        gap = process(NO_ORDER_MESSAGE)
    except Exception as exc:                                # noqa: BLE001
        return _fail("process() raised {}: {}".format(type(exc).__name__, exc),
                     "It takes one customer message and returns a dictionary with "
                     "'data' and 'decision'.")
    for label, result in (("a clean message", clean), ("a message with no order id", gap)):
        if not isinstance(result, dict) or "data" not in result or "decision" not in result:
            return _fail("On {} it returned {!r}.".format(label, result),
                         "Return {'data': extracted, 'decision': decision}.")
        if not isinstance(result["data"], dict):
            return _fail("On {} the 'data' is not the extracted dictionary.".format(label))
    if clean["decision"] != "automate":
        return _fail("A clean, complete request was routed to {!r}.".format(
            clean["decision"]),
            "Nothing is wrong with it - no invalid value, no missing field, no "
            "conflict - so it should go through automatically.")
    if gap["decision"] != "ask the customer":
        return _fail("A request with no order number was routed to {!r}.".format(
            gap["decision"]),
            "Check the order of the branches: a missing field is answered by asking "
            "the customer, not by retrying and not by a human.")
    return _ok("One function now takes a sentence a person typed and returns a "
               "decision your backend can act on. Everything inside it you already "
               "built: the schema and the forced tool call, then the checks that run "
               "afterwards. This is the bridge between natural language and backend "
               "logic, and it is the last thing before ShopAssist starts calling real "
               "tools.")


@task("model_flag", title="Honour the flag the extraction itself raised",
      lab="extract_returns", needs=("review_required",))
def _task28_2(review_required: Any, **_: Any) -> Tuple[bool, str, str]:
    if not callable(review_required):
        return _fail("`review_required` is not a function.")
    base = {"order_id": "ORD-12345678", "item": "camera", "reason": "damaged_item",
            "reason_detail": None, "desired_action": "refund",
            "desired_action_detail": None, "evidence_provided": False,
            "urgency": "normal", "missing_information": [], "confidence": 0.92,
            "conflict_detected": False, "conflict_reason": None,
            "human_review_required": False}

    def call(**over: Any) -> Any:
        data = dict(base)
        data.update(over)
        try:
            return review_required(data)
        except Exception as exc:                            # noqa: BLE001
            return exc

    cases = (
        ("a clean, confident extraction", {}, False),
        ("one our own checks catch (a conflict)",
         dict(conflict_detected=True, confidence=0.45), True),
        ("one only the model flagged",
         dict(reason="policy_exception", confidence=0.78,
              human_review_required=True), True),
    )
    for label, over, wanted in cases:
        got = call(**over)
        if isinstance(got, Exception):
            return _fail("review_required() raised {}: {}".format(
                type(got).__name__, got))
        if not isinstance(got, bool):
            return _fail("On {} it returned {!r}, not True or False.".format(label, got))
        if got != wanted:
            hint = ("Either your own checks OR the extraction's own "
                    "human_review_required flag should be enough to send a case to a "
                    "person.")
            if wanted is False and got is True:
                hint = ("A leftover ... does this: `...` counts as true, so every case "
                        "goes to a person. Replace it with "
                        "data[\"human_review_required\"].")
            return _fail("On {} it returned {}, expected {}.".format(
                label, got, wanted), hint)
    return _ok("Look at the third case: confidence 0.78, above your threshold, and no "
               "conflict - your own checks let it through. The extraction flagged it "
               "anyway, because a purchase from six months ago is a policy question. "
               "The model can raise a hand your rules did not think to raise. What it "
               "cannot do is lower one - your checks still run, and either is enough.")


@task("inbox", title="Run five real messages through it", lab="extract_returns",
      needs=("results", "automation_rate"))
def _task28_3(results: Any, automation_rate: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(results, list) or len(results) != 5:
        return _fail("Expected 5 results, got {}.".format(
            len(results) if isinstance(results, list) else type(results).__name__),
            "Check that results.append(...) is inside the loop.")
    for i, row in enumerate(results, 1):
        if not isinstance(row, dict) or "decision" not in row:
            return _fail("Result {} has no 'decision'.".format(i))
    decisions = [r["decision"] for r in results]
    automated = [d for d in decisions if d == "automate"]
    if not automated:
        return _fail("Not one of the five was automated.",
                     "Something is routing everything away - re-run the earlier cells "
                     "so process() is the finished version.")
    if len(set(decisions)) < 3:
        return _fail("All five landed on {} outcome(s): {}.".format(
            len(set(decisions)), sorted(set(decisions))),
            "These five messages are deliberately different from each other. If they "
            "all agree, one of the earlier checks is not being called.")
    if not isinstance(automation_rate, (int, float)) or isinstance(automation_rate, bool):
        return _fail("`automation_rate` is {!r}, not a number.".format(automation_rate))
    wanted = len(automated) / len(results)
    if abs(automation_rate - wanted) > 1e-9:
        return _fail("automation_rate is {} but {} of {} were automated.".format(
            automation_rate, len(automated), len(results)),
            "It is the count of 'automate' decisions divided by the number of "
            "results.")
    return _ok("{} of 5 went through on their own; the other {} stopped for a reason "
               "you can name - a missing order number, a purchase outside the policy, "
               "a message that contradicts itself. That last number is the one a "
               "support team actually watches. Push it up by improving the prompt and "
               "the schema, never by loosening the checks - a system that automates "
               "everything is just a system with no idea when it is "
               "wrong.".format(len(automated), len(results) - len(automated)))


@task("no_system", title="Send the complaint with no system prompt",
      lab="system_prompt", needs=("without_system",))
def _task15_0(without_system: Any, **_: Any) -> Tuple[bool, str, str]:
    text = _text_of(without_system)
    if not text.strip():
        return _fail("The reply is empty.",
                     "Send the complaint as one user message and keep the response "
                     "in a variable called without_system.")
    if getattr(without_system, "_lab", {}).get("has_system"):
        return _fail("This request DID carry a system prompt.",
                     "This task is the 'before' picture. Remove the system= argument "
                     "from this call - the next task is the one that adds it.")
    if _promises_refund(text):
        return _ok("With no rules at all, the assistant promised a refund it has no "
                   "authority to promise, and never asked which order this is. "
                   "Nothing in the request told it not to. Fixing that is the next "
                   "task.")
    return _ok("The reply came back. This one happened not to over-promise - a model "
               "with no rules MAY behave well, which is exactly why you cannot rely "
               "on it. The next task stops leaving it to chance.")


@task("system_prompt", title="Pass a system prompt with system=", lab="system_prompt",
      needs=("without_system", "with_system"))
def _task15_1(without_system: Any, with_system: Any, **_: Any) -> Tuple[bool, str, str]:
    plain, guarded = _text_of(without_system), _text_of(with_system)
    if not plain.strip() or not guarded.strip():
        return _fail("One of the two replies is empty.")
    if not getattr(with_system, "_lab", {}).get("has_system", True):
        return _fail("This request went out with no system prompt on it.",
                     "Add system=system_prompt to the create() call in this task. "
                     "The system prompt is its own argument, not another message.")
    if plain.strip() == guarded.strip():
        return _fail("Both replies are identical.",
                     "The previous task sends the complaint with no system prompt; "
                     "this task sends the same complaint WITH system=system_prompt. "
                     "Only this one gets the argument.")
    if _promises_refund(guarded):
        return _fail("The reply WITH the system prompt still promises a refund.",
                     "Your system prompt needs an explicit rule, for example: "
                     "'Do not promise a refund until the order is checked.'")
    if not _asks_for_order_number(guarded):
        return _fail("The guarded reply withholds the refund but never asks for the "
                     "order number.",
                     "Add a rule telling ShopAssist to ask for the order number when "
                     "it is missing.")
    detail = ("Without the system prompt the assistant promised a refund it has no "
              "authority to promise; with it, it asked for the order number instead."
              if _promises_refund(plain) else
              "The unguarded reply happened not to over-promise this time - a model with "
              "no rules MAY behave well, which is exactly why you cannot rely on it.")
    return _ok("Same customer message, two different behaviours. " + detail +
               " Note that a system prompt is guidance, not enforcement - the real "
               "guarantee comes from the backend tools you build in Section 4.")


@task("persona_holds", title="Keep the persona across turns", lab="system_prompt",
      needs=("messages",))
def _task15_2(messages: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(messages, list) or len(messages) < 4:
        return _fail("Expected a conversation of at least 4 turns, got {}.".format(
            len(messages) if isinstance(messages, list) else
            type(messages).__name__),
            "Run two full exchanges through chat(), appending each reply.")
    roles = [_role_of(m) for m in messages]
    for i in range(1, len(roles)):
        if roles[i] == roles[i - 1]:
            return _fail("Two {!r} turns in a row at position {}.".format(roles[i], i),
                         "Append every reply with add_assistant_message before adding "
                         "the next user turn.")
    replies = [m.get("content", "") for m in messages if _role_of(m) == "assistant"]
    offenders = [r for r in replies if _promises_refund(r)]
    if offenders:
        return _fail("An assistant turn promises a refund: {!r}".format(
            offenders[0][:80]),
            "The system prompt has to be passed on EVERY call, not just the first. "
            "Put system=system_prompt inside chat() so it cannot be forgotten.")
    return _ok("The persona held for the whole conversation, because the system prompt "
               "goes with every request. It is not remembered by the API between "
               "calls any more than the messages are.")


@task("temperature_zero", title="Ask the same question three times at temperature 0",
      lab="temperature", needs=("deterministic",))
def _task16_0(deterministic: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(deterministic, list):
        return _fail("`deterministic` should be a list, got {}.".format(
            type(deterministic).__name__),
            "Start with deterministic = [] and append each reply inside the loop.")
    if len(deterministic) < 3:
        return _fail("The list holds {} reply/replies; the loop should collect "
                     "3.".format(len(deterministic)),
                     "Check that deterministic.append(reply) is inside the for loop.")
    texts = [_text_of(m) for m in deterministic]
    if any(not t.strip() for t in texts):
        return _fail("One of the replies is empty.",
                     "Append the whole response object, not just part of it.")
    meta = [getattr(m, "_lab", {}) for m in deterministic]
    simulated = all(getattr(m, "simulated", False) for m in deterministic)
    if simulated:
        temps = {md.get("temperature") for md in meta}
        if temps != {0}:
            return _fail("These calls went out at temperature {}.".format(
                sorted(str(t) for t in temps)),
                "This task is the predictable end of the dial. Pass temperature=0 on "
                "every call - temperature=1.0 is the next task.")
        if len({md.get("call_index") for md in meta}) < 3:
            return _fail("The list holds the same response three times.",
                         "The create() call has to be INSIDE the loop, so a new "
                         "request goes out on each pass.")
        if len(set(texts)) != 1:
            return _fail("The three replies are not identical ({} distinct).".format(
                len(set(texts))),
                "Everything except nothing should change between the calls - same "
                "messages, same max_tokens, temperature=0 on all three.")
        return _ok("Three separate requests, one single answer. At temperature=0 the "
                   "model takes the most likely next token every time, so the reply "
                   "stops being a surprise. That is what you want for refund policy "
                   "answers - and the next task shows the other end of the dial.")
    return _ok("Three separate requests at temperature=0 gave {} distinct "
               "reply/replies. On the real API temperature=0 means 'as repeatable as "
               "possible', not a byte-for-byte guarantee.".format(len(set(texts))))


@task("temperature", title="Temperature: deterministic vs varied", lab="temperature",
      needs=("deterministic", "varied"))
def _task16_1(deterministic: Any, varied: Any, **_: Any) -> Tuple[bool, str, str]:
    det = [_text_of(m) for m in deterministic]
    var = [_text_of(m) for m in varied]
    if len(det) < 3:
        return _fail("`deterministic` holds {} reply/replies - run the previous task "
                     "first.".format(len(det)),
                     "This task compares the two lists, so both have to exist.")
    if len(var) < 3:
        return _fail("`varied` holds {} reply/replies; the loop should collect "
                     "3.".format(len(var)),
                     "Check that varied.append(reply) is inside the for loop.")
    if all(getattr(m, "simulated", False) for m in varied):
        temps = {getattr(m, "_lab", {}).get("temperature") for m in varied}
        if 0 in temps:
            return _fail("This batch went out at temperature=0 as well.",
                         "Change the temperature in THIS loop to 1.0 - that is the "
                         "only difference between the two batches.")
    simulated = all(getattr(m, "simulated", False) for m in deterministic)
    if simulated:
        if len(set(det)) != 1:
            return _fail("The temperature=0 replies are not all identical "
                         "({} distinct).".format(len(set(det))),
                         "Every call must be identical apart from temperature - same "
                         "messages, same system, same max_tokens.")
        if len(set(var)) < 2:
            return _fail("The temperature=1 replies are all identical.",
                         "Pass temperature=1.0 and make sure you are creating a new "
                         "request each time rather than reusing one result.")
    elif len(set(det)) > len(set(var)):
        return _fail("At temperature=0 you got {} distinct replies, but only {} at "
                     "temperature=1.".format(len(set(det)), len(set(var))),
                     "That is backwards. Check that temperature=0 is on the first "
                     "batch and temperature=1.0 on the second.")
    return _ok("temperature=0 gave you the same answer every time and temperature=1 "
               "varied." if simulated else
               "temperature=0 gave {} distinct reply/replies and temperature=1 gave {}. "
               "On the real API temperature=0 means 'as repeatable as possible', not a "
               "byte-for-byte guarantee.".format(len(set(det)), len(set(var))))


@task("stop_sequences", title="Stop sequences", lab="temperature", needs=("stopped",))
def _task16_2(stopped: Any, **_: Any) -> Tuple[bool, str, str]:
    if stopped.stop_reason != "stop_sequence":
        return _fail("The stop-sequence call returned stop_reason={!r}.".format(
            stopped.stop_reason),
            "Pass stop_sequences=['<END>'] AND ask for the marker in the prompt, "
            "for example: 'End the reply with <END>.'")
    if stopped.stop_sequence != "<END>":
        return _fail("stop_sequence is {!r}, expected '<END>'.".format(stopped.stop_sequence))
    if "<END>" in _text_of(stopped):
        return _fail("The marker itself appears in the returned text.",
                     "The API cuts the text before the stop sequence - if you see it, "
                     "you are probably printing the prompt rather than the reply.")
    return _ok("The stop sequence ended generation the moment the marker appeared, "
               "stop_reason is 'stop_sequence', and the marker itself never reaches "
               "your text. Useful for cutting a reply at a known boundary - but it is "
               "not validation, which you still write in code.")


# ---------------------------------------------------------------------------
# Auto-install on import, so a lab notebook only needs `import shopassist_lab`.
# ---------------------------------------------------------------------------

install()
