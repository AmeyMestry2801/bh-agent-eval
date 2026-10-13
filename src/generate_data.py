"""
Synthetic intake-conversation generator.

ALL DATA IS SYNTHETIC. No real patient data is used or imitated.
Every rate, mix and effect size below is an assumption (docs/assumptions.md).

Run from the project root:
    python -m src.generate_data
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from src.config import DATA_RAW, SEED, ensure_dirs, set_seed

# ---------------------------------------------------------------------------
# Parameters (all assumptions)
# ---------------------------------------------------------------------------
N_CONV = 6000
N_WEEKS = 26
DRIFT_START_WEEK = 17  # weeks 1-16 = reference regime, 17-26 = drift regime

CLASS_MIX = {"no_risk": 0.70, "low": 0.17, "moderate": 0.09, "crisis": 0.04}
VOICE_SHARE = {"pre": 0.35, "post": 0.55}
SPANISH_SHARE = 0.10                      # share of conversations with Spanish phrases
SPANISH_FULL_SENTENCE_PROB = 0.60         # else English core + Spanish interjections
WER_RANGE = {"pre": (0.05, 0.15), "post": (0.10, 0.22)}  # per-conversation word error rate
TYPO_RATE = 0.03                          # per-word typo prob, text channel only
LABEL_NOISE = 0.025                       # prob. the recorded label is an adjacent class
POST_ERA_FAMILY_SHARE = 0.35              # share of weeks 17-26 drawn from new-slang families
HEDGE_PROB = {"crisis": 0.25, "moderate": 0.25, "low": 0.20, "no_risk": 0.10}
AGE_BANDS = ["18-24", "25-34", "35-49", "50-64", "65+"]
AGE_PROBS = [0.18, 0.27, 0.27, 0.18, 0.10]
DELAY_PROBS = {  # label_delay_weeks in {1,2,3,4}
    "crisis": [0.50, 0.30, 0.15, 0.05],
    "other": [0.20, 0.30, 0.30, 0.20],
}
N_FAMILIES = {
    "base": {"crisis": 50, "moderate": 60, "low": 80, "no_risk": 150},
    "post": {"crisis": 10, "moderate": 10, "low": 15, "no_risk": 25},
}
MODE_MIX = {
    "crisis": (["explicit", "indirect"], [0.70, 0.30]),
    "moderate": (["explicit", "indirect"], [0.65, 0.35]),
    "low": (["benign", "hardneg"], [0.60, 0.40]),
    "no_risk": (["admin", "hardneg"], [0.70, 0.30]),
}
ADJACENT = {
    "no_risk": ["low"],
    "low": ["no_risk", "moderate"],
    "moderate": ["low", "crisis"],
    "crisis": ["moderate"],
}

# ---------------------------------------------------------------------------
# Text pools. "{a|b|c}" is a slot: one option is chosen per conversation.
# ---------------------------------------------------------------------------
OPENERS = [
    "Hi.", "Hello, thanks for picking up.", "Hey.", "Good morning.",
    "Hi, I'm not sure where to start.", "Thanks for taking my call.",
    "I filled out the form but wanted to explain.",
    "I was told to talk to someone here.", "So, um, yeah.",
    "I don't really do this kind of thing.", "My doctor suggested I reach out.", "",
]
CONTEXTS = [
    "It's been going on for a few weeks.", "This started after I lost my job.",
    "My partner and I have been fighting a lot.", "I've been staying at my sister's place.",
    "I work nights at a warehouse.", "I'm a student and finals are coming up.",
    "I moved here about six months ago.", "My kids are at their dad's this week.",
    "I haven't talked to anyone about this.", "I tried a counselor once but it didn't stick.",
    "The weather has been awful lately.", "I've been busy with work.",
    "My insurance changed recently.", "I take medication for my blood pressure.",
    "It's been a rough month overall.",
]
CLOSERS = [
    "Can someone help me?", "I don't know what to do.", "What happens next?",
    "Please let me know what the next step is.", "Thanks.", "I just wanted to tell someone.",
    "Is there a nurse I can speak to?", "Sorry if this is a lot.", "That's all.",
    "Okay, go ahead and ask your questions.",
]

POOLS: dict[tuple[str, str], list[str]] = {
    # ---------------- crisis ----------------
    ("crisis", "explicit"): [
        "I {have been|keep|am} thinking about {killing myself|ending my life|suicide} and I {have a plan|know how I would do it|have decided how}.",
        "I {am going to|will|plan to} {kill myself|end my life|end it} {tonight|today|this weekend|as soon as I hang up}.",
        "I {took|swallowed} {a bunch of|too many|all of my} {pills|sleeping pills|meds} and I {don't want to|do not want to} {wake up|be saved|get help}.",
        "I {wrote|left} a {goodbye note|note for my family} and I {am ready to go|have everything ready|have the means with me}.",
        "I {don't|do not} {want to|wanna} be alive {anymore|any more} and I {have the pills|have a gun|have a rope|have the means} {right here|ready|within reach}.",
        "{Tonight|Today} is the {night|day} I {end|finish} {it|everything}, {I have decided|I'm done|I can't do this anymore}.",
        "I {keep|can't stop} {thinking about|picturing} {hurting myself|cutting deep|overdosing}, and {this time|today} I {might|think I will|probably will} {actually do it|go through with it}.",
        "I {am|feel} {going to hurt|about to hurt} myself and I {can't|cannot} {stop|promise I will be safe}.",
    ],
    ("crisis", "indirect"): [
        "I {gave away|have given away} {my dog|my guitar|most of my things|my savings} and {said my goodbyes|called everyone to say sorry|sorted out my will}, {it's all sorted now|everything is in order|there's nothing left to do}.",
        "{Everyone|My family} {would be|will be} {better off|fine} without me and I {have figured out|worked out|know} {how|when|the way}.",
        "I {don't see|can't see} {a reason|any point} to {see|wake up to|get through} {tomorrow|another day|next week}, and {I know how|I have it planned|the plan is set}.",
        "I {keep looking at|counted|lined up} {my meds|the pills in the cabinet|the bridge on my commute} {every night|again today|a lot}, {it feels close|it won't be long|I think tonight}.",
        "{I'm done|I'm finished|I'm at the end}. {I won't be around|You won't hear from me|I won't be here} {after tonight|this weekend|much longer}.",
        "I {just want|only want} the {pain|noise|thoughts} to {stop for good|be over for good|end permanently} and {I have a way|I found a way|I know a way}.",
    ],
    ("crisis", "slang"): [
        "ngl I'm gonna {unalive|off|kms} {myself|tonight}, {I've got the pills|it's planned|I'm done fr}.",
        "{kms|unaliving myself} {tonight|this weekend} {no cap|fr fr|for real}, {pills are ready|I have a way|my plan is set}.",
        "I'm {about to|gonna} {unalive myself|check out for good}, {bye|tell my mom sorry|it's over}.",
        "{lowkey|literally} {ready to unalive|done with existing}, {got everything I need|plan's set|tonight's the night} {fr|no cap}.",
    ],
    # ---------------- moderate ----------------
    ("moderate", "explicit"): [
        "{Sometimes|Some nights|Lately} I {wish|think about wishing} I {could just disappear|wouldn't wake up|was dead}, {but I would never act on it|but I'm not going to do anything|though I have no plan}.",
        "I {have been|keep} {cutting|scratching myself|hurting myself} {again|to cope|when it gets bad}, {nothing too serious|it's not deep|not like before}.",
        "I {haven't slept|barely slept|slept maybe two hours} in {days|three days|a week} and I feel {hopeless|completely empty|like a burden}.",
        "I {feel|think} like a {burden|failure} and {things will never get better|I can't see a way forward}, {but I don't want to die|though I am not going to hurt myself}.",
        "I {have thoughts of death|think about dying} {sometimes|now and then|at night} {but no plan|without any intention|but I'd never do it}.",
        "I've been {drinking|using|taking} {way more|a lot more|too much} {than I should|lately|to numb it} and I {feel out of control|can't keep going like this|am scared of myself}.",
        "{I had a panic attack|I've been panicking|I can't stop shaking} {at work|every day|most nights} and {I think I'm falling apart|I feel like I'm losing it|I can't function}.",
    ],
    ("moderate", "indirect"): [
        "{Nothing|Nobody} {matters|cares} {anymore|to me} and {I have stopped|I quit} {eating|answering calls|going to work}.",
        "I {just want to|keep wanting to} {sleep forever|vanish|not exist} {lately|these days|most days}.",
        "I {don't|do not} {feel like|have the energy for} {living like this|trying anymore|any of it}.",
    ],
    ("moderate", "slang"): [
        "{ngl|lowkey|tbh} {been wanting to unalive but I won't|been thinking about kms but not really serious|wishing I could just not exist}, {no plan|no cap}.",
        "{ngl|tbh} I'm {cooked|so done|not ok}, {haven't slept|can't eat}, {feels hopeless|everything is mid} {fr|lately}.",
    ],
    # ---------------- low ----------------
    ("low", "benign"): [
        "I {have been feeling|feel} {down|sad|flat|anxious} {lately|most days|since the move}, {and my sleep is off|and I have trouble focusing|and I snap at people}.",
        "{Work|School|Money} stress is {making me anxious|wearing me down|keeping me up at night}.",
        "I {get nervous|get really tense|worry a lot} {before meetings|in crowds|about everything} and {would like to|want to|should} talk to someone.",
        "I {think I might have|was wondering if I have} {mild depression|anxiety|burnout}, {nothing urgent|it isn't an emergency|I'm safe}.",
        "I {lost my appetite|can't enjoy things|feel unmotivated} {a bit|lately|this month} but {I'm coping|I'm managing|I'm okay}.",
    ],
    ("low", "hardneg"): [
        "I {am not|am definitely not|have never been} thinking {of|about} {hurting myself|suicide|ending my life}, {I'm just|I'm only|it's just} {stressed|exhausted|overwhelmed} {about work|about exams|about money}.",
        "My {job|boss|commute|landlord} is {killing me|driving me insane|going to be the death of me}, {I just need|I could use|I'd like} {some coping tips|to talk to someone|help managing stress}.",
        "I {watched|read} a {movie|show|book} where a character {died by suicide|killed himself|took her own life} and it {stuck with me|made me sad|left me anxious} {for days|all week|a bit}.",
        "I {used to|once|years ago} {have|had} {dark thoughts|thoughts of ending it}, {but that was in the past|but not now|but I'm fine now}, {and I'm here for anxiety|I just want a check-in|I'm looking for therapy}.",
    ],
    ("low", "slang"): [
        "ngl I've been {lowkey|kinda} {depressed|stressed|cooked} {fr|lately|no cap}, {sleep is trash|can't focus|everything feels mid}.",
        "{tbh|ngl} {school|work} has me {stressed|anxious} {af|fr}, {want to talk to someone|need some tips}.",
    ],
    # ---------------- no_risk ----------------
    ("no_risk", "admin"): [
        "I {need to|want to|would like to} {reschedule|cancel|confirm|book} my {appointment|intake visit|follow-up} {for next week|on Thursday|for the morning}.",
        "{Can you tell me|I wanted to ask|Quick question,} {whether you take my insurance|what the copay is|if I need a referral}.",
        "I {need|am requesting} a {refill|renewal} of my {prescription|medication} {before the weekend|this week|today}.",
        "I {am calling|am writing} to {update|change} my {phone number|address|insurance details|emergency contact}.",
        "I {haven't received|never got} the {intake forms|confirmation email|portal login} {yet|you said you sent}.",
        "{Where|How} do I {find|download|upload} my {lab results|visit summary|forms} {in the portal|online}?",
    ],
    ("no_risk", "hardneg"): [
        "My {job|boss|commute|inbox|landlord} is {killing me|driving me crazy|going to be the death of me} {this week|lately|right now}, anyway I just need to {reschedule|confirm|move} my {appointment|visit}.",
        "I'm {not|definitely not|in no way} {thinking about|going to} {hurting myself|suicide|ending my life}, {I'm just|I just} {calling to|here to|trying to} {check|confirm|ask about} {my appointment time|the parking|my paperwork}.",
        "{Oh great|Wonderful|Fantastic}, {another|yet another} {Monday|meeting|deadline}, {kill me now|just shoot me|I'm dying here}, anyway {can I|I need to} {change|confirm} my {appointment|time slot}.",
        "I'm {dying|dead} {to see|to know} {the new season|what happens next|if the results came in}, {can you tell me|do you know} {when my results are ready|if my results are in}.",
        "In the {movie|show|novel} {I watched|we saw} the {main character|father|detective} {dies|kills himself|ends his life} and my {friend|book club|class} wants to {discuss|talk about} it, unrelated, {I need|can you help me with} {a refill|my appointment|my forms}.",
        "My {grandfather|neighbor|coworker} {passed away|died} {last year|last month|a while ago} and {I'm handling the paperwork|I'm helping the family|we're sorting out his things}, {I just need|can you help with} {a copy of the form|the records|the portal login}.",
    ],
    ("no_risk", "slang"): [
        "bruh my {boss|exam|group project} is {literally killing me|unaliving me|so dead} {lol|lmao|fr}, anyway {can I move my appointment|where do I find the forms}.",
        "I'm {dead|dying} {lmao|lol} {that meme|this video} {killed me|slayed}, {unrelated|anyway}, {need a refill|need to confirm my slot}.",
        "{ngl|tbh} {need to|gotta} {reschedule|confirm} my {appointment|visit} {fr|asap|no cap}, {thx|ty}.",
    ],
}

SPANISH = {
    "crisis": [
        "{No puedo más|Ya no puedo|Estoy al límite}, {quiero morir|me voy a matar|me voy a quitar la vida} {esta noche|hoy|ahora mismo} y {tengo un plan|ya tengo las pastillas|ya sé cómo}.",
        "{Tomé|Me tomé} {muchas pastillas|todas mis pastillas} y no quiero {despertar|que me salven|pedir ayuda}.",
        "Ya me despedí de {todos|mi familia|mis hijos}, {ya está todo listo|ya lo decidí|esta es la última vez que hablo}.",
    ],
    "moderate": [
        "{A veces|Últimamente} {quisiera desaparecer|pienso en la muerte|no quiero despertar}, pero {no voy a hacer nada|no tengo ningún plan|nunca lo haría}.",
        "Me {estoy cortando|he vuelto a cortar} {otra vez|para aguantar}, {nada grave|no es profundo}.",
        "{No he dormido|Casi no he dormido} en {días|una semana} y me siento {sin esperanza|como una carga|vacío}.",
    ],
    "low": [
        "Me siento {muy triste|muy estresada|ansioso} {últimamente|casi todos los días}, {y no duermo bien|y no me puedo concentrar}.",
        "El {trabajo|estudio} me tiene {muy ansioso|agotada|estresado} y {quiero hablar con alguien|necesito ayuda}.",
        "{No estoy pensando|Nunca he pensado} en hacerme daño, solo estoy {muy estresado|agotada|abrumado}.",
    ],
    "no_risk": [
        "{Necesito|Quiero} {cambiar|confirmar|cancelar} mi cita {para la próxima semana|el jueves|por la mañana}.",
        "{Quiero saber|Quería preguntar} si {aceptan mi seguro|necesito una referencia|cuánto cuesta la consulta}.",
        "Mi {trabajo|jefe} me está {matando|volviendo loco}, pero {solo quiero confirmar mi cita|necesito renovar mi receta}.",
        "{No estoy pensando|Nunca pienso} en hacerme daño, solo {quiero confirmar mi cita|necesito mis formularios}.",
    ],
}
ES_INTERJECTIONS = ["Pues,", "La verdad,", "Ay,", "Mira,", "No sé,", "Bueno,"]
ES_CLOSERS = ["Por favor, ayúdenme.", "¿Qué hago?", "Gracias.", "No sé qué hacer.", "Es todo, gracias."]

HEDGES = ["maybe", "I guess", "kind of", "sort of", "I'm not sure but",
          "I don't know,", "it's probably nothing but", "I could be wrong but"]

# Voice (ASR) confusions. Several deliberately drop or flip negation and crisis keywords.
HOMOPHONES = {
    "kill": ["hill", "kiln", "will"], "myself": ["my self", "mice elf", "herself"],
    "suicide": ["sue aside", "sweet side"], "die": ["dye", "dime", "by"],
    "died": ["dyed", "tide"], "dead": ["debt", "bed", "dad"],
    "hurting": ["hurtin", "heard in", "hurt in"], "hurt": ["heard", "hurts"],
    "pills": ["bills", "fills", "hills"], "life": ["light", "live", "wife"],
    "end": ["and", "ten", "in"], "plan": ["plant", "clan", "plain"],
    "gun": ["gone", "fun", "done"], "rope": ["robe", "road", "hope"],
    "alive": ["a live", "olive", "arrive"], "wake": ["weight", "way", "make"],
    "goodbye": ["good buy", "good bye"], "cutting": ["cuttin", "cuddling"],
    "overdosing": ["over dosing", "over closing"], "tonight": ["to night", "to knight"],
    "sleep": ["sleet", "slip", "sheep"], "not": ["knot", "nut", "now"],
    "never": ["ever", "lever"], "don't": ["dont", "don", "down"],
    "can't": ["can", "cant", "count"], "appointment": ["a point meant", "apartment"],
    "refill": ["re fill", "refile"], "insurance": ["in shoe rants"],
}
MISHEARD = ["the", "a", "to", "and", "it", "so", "that", "is", "for", "with", "I", "my", "you"]
FILLERS = ["uh", "um", "like", "you know", "so"]
AGE_PREFIX_YOUNG = ["tbh", "ngl", "honestly", "like"]

_SLOT = re.compile(r"\{([^{}]*)\}")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def fill(template: str, rng: np.random.Generator) -> str:
    """Resolve every {a|b|c} slot with one random option."""
    return _SLOT.sub(lambda m: str(rng.choice(m.group(1).split("|"))), template)


def upper_first(s: str) -> str:
    return s[:1].upper() + s[1:]


def lower_first(s: str) -> str:
    if not s or s.startswith(("I ", "I'")):
        return s
    return s[0].lower() + s[1:]


def ensure_end(s: str) -> str:
    return s if not s or s[-1] in ".?!" else s + "."


def apply_hedge(core: str, rng: np.random.Generator) -> str:
    hedge = str(rng.choice(HEDGES))
    body = lower_first(core.rstrip(".!?"))
    tail = " or something" if rng.random() < 0.3 else ""
    return upper_first(f"{hedge} {body}{tail}.")


def add_typos(text: str, rate: float, rng: np.random.Generator) -> str:
    out = []
    for w in text.split(" "):
        if len(w) > 3 and rng.random() < rate:
            i = int(rng.integers(1, len(w) - 1))
            op = int(rng.integers(0, 3))
            if op == 0:    # swap adjacent characters
                w = w[: i - 1] + w[i] + w[i - 1] + w[i + 1:]
            elif op == 1:  # drop a character
                w = w[:i] + w[i + 1:]
            else:          # double a character
                w = w[:i] + w[i] + w[i:]
        out.append(w)
    return " ".join(out)


def asr_noise(text: str, wer: float, rng: np.random.Generator) -> str:
    """Mimic a speech recogniser: lowercase, no punctuation, word-level errors."""
    t = "".join(ch for ch in text.lower() if ord(ch) < 256)
    t = re.sub(r"[^\w\s']", " ", t)
    out = []
    for tok in t.split():
        if rng.random() >= wer:
            out.append(tok)
            continue
        r = rng.random()
        if r < 0.55:  # substitution
            if tok in HOMOPHONES:
                out.append(str(rng.choice(HOMOPHONES[tok])))
            elif len(tok) > 3 and rng.random() < 0.5:
                i = int(rng.integers(0, len(tok)))
                out.append(tok[:i] + str(rng.choice(list("aeioustnrl"))) + tok[i + 1:])
            else:
                out.append(str(rng.choice(MISHEARD)))
        elif r < 0.80:  # deletion
            continue
        else:           # insertion
            out.append(tok)
            out.append(str(rng.choice(FILLERS)))
    return " ".join(out)


# ---------------------------------------------------------------------------
# Template families
# ---------------------------------------------------------------------------
def build_families(rng: np.random.Generator) -> dict[tuple[str, str], list[dict]]:
    """A family = a fixed skeleton (opener, 3 core templates, context, closer).
    Conversations in one family are near-duplicates; split by family later."""
    fams: dict[tuple[str, str], list[dict]] = {}
    seen: set = set()
    for era, counts in N_FAMILIES.items():
        for risk, n in counts.items():
            lst = []
            for i in range(n):
                while True:
                    if era == "post":
                        mode = "slang"
                    else:
                        modes, probs = MODE_MIX[risk]
                        mode = str(rng.choice(modes, p=probs))
                    pool = POOLS[(risk, mode)]
                    idx = tuple(sorted(int(x) for x in rng.choice(
                        len(pool), size=min(3, len(pool)), replace=False)))
                    o = int(rng.integers(len(OPENERS)))
                    c = int(rng.integers(len(CONTEXTS)))
                    cl = int(rng.integers(len(CLOSERS)))
                    key = (risk, era, mode, idx, o, c, cl)
                    if key not in seen:
                        seen.add(key)
                        break
                lst.append({
                    "family_id": f"{'B' if era == 'base' else 'D'}-{risk}-{i:03d}",
                    "risk": risk, "era": era, "mode": mode,
                    "cores": [pool[j] for j in idx],
                    "opener": OPENERS[o], "context": CONTEXTS[c], "closer": CLOSERS[cl],
                })
            fams[(risk, era)] = lst
    return fams


def build_text(fam: dict, risk_clean: str, language: str, channel: str, age_band: str,
               wer: float, rng: np.random.Generator) -> str:
    core = ensure_end(fill(str(rng.choice(fam["cores"])), rng))
    closer = fam["closer"]
    spanish_core = False

    if language == "en_es":
        if rng.random() < SPANISH_FULL_SENTENCE_PROB:
            core = ensure_end(fill(str(rng.choice(SPANISH[risk_clean])), rng))
            spanish_core = True
        else:
            core = f"{rng.choice(ES_INTERJECTIONS)} {lower_first(core)}"
        if rng.random() < 0.5:
            closer = str(rng.choice(ES_CLOSERS))

    if not spanish_core:
        if age_band == "18-24" and rng.random() < 0.30:
            core = upper_first(f"{rng.choice(AGE_PREFIX_YOUNG)} {lower_first(core)}")
        elif age_band == "65+" and rng.random() < 0.30:
            core = f"Well, {lower_first(core)}"
        if rng.random() < HEDGE_PROB[risk_clean]:
            core = apply_hedge(core, rng)

    parts = [fam["opener"], core]
    if rng.random() < 0.75:
        parts.append(fam["context"])
    if rng.random() < 0.80:
        parts.append(closer)
    if rng.random() < 0.15:  # sometimes lead with the core statement
        parts = [core] + [p for p in parts if p is not core]
    text = " ".join(p for p in parts if p).strip()

    if channel == "voice":
        return asr_noise(text, wer, rng)
    return add_typos(text, TYPO_RATE, rng)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def generate(seed: int = SEED, n: int = N_CONV) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fams = build_families(rng)

    counts = {k: int(round(v * n)) for k, v in CLASS_MIX.items()}
    classes = np.array([k for k, c in counts.items() for _ in range(c)])
    assert len(classes) == n, "class counts must sum to n"
    rng.shuffle(classes)
    weeks = (np.arange(n) % N_WEEKS) + 1
    rng.shuffle(weeks)

    rows = []
    for risk_clean, week in zip(classes, weeks):
        period = "pre" if week < DRIFT_START_WEEK else "post"
        channel = "voice" if rng.random() < VOICE_SHARE[period] else "text"
        language = "en_es" if rng.random() < SPANISH_SHARE else "en"
        age_band = str(rng.choice(AGE_BANDS, p=AGE_PROBS))
        era = "post" if (period == "post" and rng.random() < POST_ERA_FAMILY_SHARE) else "base"
        fam = fams[(str(risk_clean), era)][int(rng.integers(len(fams[(str(risk_clean), era)])))]
        wer = float(rng.uniform(*WER_RANGE[period])) if channel == "voice" else np.nan
        text = build_text(fam, str(risk_clean), language, channel, age_band,
                          0.0 if np.isnan(wer) else wer, rng)

        label = str(risk_clean)
        if rng.random() < LABEL_NOISE:
            label = str(rng.choice(ADJACENT[label]))
        delay_key = "crisis" if label == "crisis" else "other"
        delay = int(rng.choice([1, 2, 3, 4], p=DELAY_PROBS[delay_key]))

        rows.append({
            "week": int(week), "channel": channel, "language": language,
            "age_band": age_band, "text": text, "risk_level": label,
            "escalate_needed": int(label == "crisis"), "label_delay_weeks": delay,
            "template_family": fam["family_id"], "risk_level_clean": str(risk_clean),
            "gen_asr_wer": wer,
        })

    df = pd.DataFrame(rows).sort_values("week", kind="stable").reset_index(drop=True)
    df.insert(0, "conversation_id", [f"C{i + 1:05d}" for i in range(len(df))])
    return df


def summarize(df: pd.DataFrame) -> None:
    pre, post = df[df.week < DRIFT_START_WEEK], df[df.week >= DRIFT_START_WEEK]
    print(f"rows: {len(df)} | columns: {len(df.columns)} | weeks: {df.week.min()}-{df.week.max()}")
    print("\nrecorded label mix (after label noise):")
    print(df.risk_level.value_counts(normalize=True).round(4).to_string())
    print("\nlabel noise rate (recorded != clean):",
          round(float((df.risk_level != df.risk_level_clean).mean()), 4))
    print("\nvoice share  weeks 1-16:", round(float((pre.channel == 'voice').mean()), 3),
          "| weeks 17-26:", round(float((post.channel == 'voice').mean()), 3))
    print("mean ASR WER weeks 1-16:", round(float(pre.gen_asr_wer.mean()), 3),
          "| weeks 17-26:", round(float(post.gen_asr_wer.mean()), 3))
    print("spanish-mixed share:", round(float((df.language == 'en_es').mean()), 3))
    print("families used:", df.template_family.nunique())
    print("\nmean text length (words) by recorded class:")
    print(df.assign(n=df.text.str.split().str.len()).groupby("risk_level").n.mean().round(1).to_string())
    print("\ncrisis rows weeks 1-16:", int((pre.risk_level == 'crisis').sum()),
          "| weeks 17-26:", int((post.risk_level == 'crisis').sum()))
    print("\n--- sample (one per class, text channel, weeks 1-16) ---")
    for r in ["no_risk", "low", "moderate", "crisis"]:
        s = pre[(pre.risk_level_clean == r) & (pre.channel == "text")].iloc[0]
        print(f"[{r}] {s.text}")
    print("\n--- sample voice crisis, weeks 17-26 ---")
    s = post[(post.risk_level_clean == "crisis") & (post.channel == "voice")].iloc[0]
    print(f"[crisis/voice/week {s.week}] {s.text}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic intake conversations.")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    set_seed(args.seed)
    ensure_dirs()
    df = generate(seed=args.seed)
    out = DATA_RAW / "conversations.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}\n")
    summarize(df)


if __name__ == "__main__":
    main()
