"""Curated English resources used by detectors and heuristic rewriters.

Frequency ranks are a compact Zipf-like list of common word forms. They are
good enough to (a) estimate unigram surprisal for entropy targeting and
(b) avoid substituting the most frequent function words.
"""

from __future__ import annotations

import math

# Zipf-ish rank: lower number = more frequent. Rank 1 = "the".
_FREQ_HEAD: list[str] = """
the be to of and a in that have i it for not on with he as you do at
this but his by from they we say her she or an will my one all would there
their what so up out if about who get which go me when make can like time
no just him know take people into year your good some could them see other
than then now look only come its over think also back after use two how
our work first well way even new want because any these give day most us
is was are were been being had has did does am
""".split()

_FREQ_BODY: list[str] = """
man find here thing come still should own life against each few those
place same great where help through back little house world old long
while last never become between both high something under last own
need feel seem leave put mean keep let begin might something same
another around however although therefore moreover furthermore indeed
perhaps maybe actually really quite rather almost already enough
important possible different various several many much more less
increase decrease include provide require consider develop produce
create support describe explain suggest indicate remain appear become
continue start stop try ask tell call show follow turn move play run
write read speak listen learn teach change grow open close build
buy sell pay cost spend save live die eat drink walk sit stand
sleep wake smile laugh cry shout whisper think believe know understand
remember forget hope fear love hate like dislike prefer choose decide
agree disagree accept reject allow prevent cause result happen occur
exist contain consist depend relate compare contrast
""".split()

# Additional mid-frequency content words useful for entropy targeting.
_FREQ_TAIL: list[str] = """
system process information data analysis research study report paper
article content text model language token sequence context sample
watermark detector detection score confidence metric quality
rewrite paraphrase synonym structure style human natural
company product service market customer user team project
problem solution approach method technique strategy plan
result effect impact benefit risk issue challenge opportunity
example case instance situation condition factor reason
time period moment today tomorrow yesterday week month year
person people group community society public private
place location area region country city world
number amount value level rate size type kind form
part section chapter page paragraph sentence word
question answer idea thought opinion view point
""".split()

FREQUENCY: dict[str, int] = {}
_rank = 1
for _block in (_FREQ_HEAD, _FREQ_BODY, _FREQ_TAIL):
    for _w in _block:
        key = _w.lower()
        if key not in FREQUENCY:
            FREQUENCY[key] = _rank
            _rank += 1

# Closed-class weights used by stylometry (relative expected frequency).
FUNCTION_WORD_WEIGHTS: dict[str, float] = {
    "the": 0.070,
    "be": 0.025,
    "to": 0.026,
    "of": 0.025,
    "and": 0.027,
    "a": 0.023,
    "in": 0.018,
    "that": 0.012,
    "it": 0.011,
    "for": 0.010,
    "on": 0.007,
    "with": 0.007,
    "as": 0.006,
    "by": 0.005,
    "this": 0.005,
    "but": 0.004,
    "from": 0.004,
    "or": 0.004,
    "an": 0.004,
    "not": 0.006,
}

# Phrase → more human / less detector-bait replacements.
AI_PHRASES: dict[str, list[str]] = {
    "in conclusion": ["so", "all told", "wrapping up"],
    "in summary": ["short version", "to put it briefly", "so"],
    "to summarize": ["briefly", "in short"],
    "it is important to note that": ["worth saying:", "note that", "one catch:"],
    "it's important to note that": ["worth saying:", "note that"],
    "it is worth noting that": ["also", "and"],
    "it is crucial to": ["you need to", "it pays to"],
    "plays a crucial role": ["matters a lot", "does a lot of the work"],
    "plays a vital role": ["is a big part of", "matters here"],
    "plays a significant role": ["matters", "shows up a lot in"],
    "in today's world": ["these days", "right now", "now"],
    "in today's digital age": ["online now", "these days"],
    "in today's rapidly evolving": ["lately, in a fast-changing"],
    "delve into": ["look at", "get into", "unpack"],
    "delve deeper": ["go further", "dig more"],
    "dive into": ["look at", "get into"],
    "a wide range of": ["lots of", "many kinds of", "all sorts of"],
    "a plethora of": ["a pile of", "plenty of", "more than enough"],
    "a myriad of": ["countless", "a lot of"],
    "leverage": ["use", "apply", "draw on"],
    "utilize": ["use", "make use of"],
    "utilizing": ["using"],
    "facilitate": ["help", "ease"],
    "comprehensive": ["full", "thorough"],
    "robust": ["solid", "sturdy", "reliable"],
    "cutting-edge": ["new", "current"],
    "state-of-the-art": ["current", "top-end"],
    "groundbreaking": ["new", "unusual"],
    "innovative": ["new", "fresh"],
    "seamless": ["smooth"],
    "streamline": ["simplify", "tighten"],
    "empower": ["help", "enable"],
    "unlock": ["open", "enable"],
    "harness": ["use", "apply"],
    "foster": ["encourage", "support"],
    "enhance": ["improve", "sharpen"],
    "optimize": ["tune", "improve"],
    "revolutionize": ["change", "shake up"],
    "transformative": ["major", "lasting"],
    "holistic": ["overall", "joined-up"],
    "paradigm": ["model", "way of thinking"],
    "landscape": ["scene", "field", "market"],
    "tapestry": ["mix", "set"],
    "beacon": ["signal", "marker"],
    "testament to": ["sign of", "proof of"],
    "underscores": ["shows", "makes clear"],
    "highlights the importance": ["shows why this matters"],
    "serves as a reminder": ["is a reminder"],
    "it should be noted that": ["note:", "also"],
    "needless to say": ["obviously", "of course"],
    "when it comes to": ["for", "on"],
    "at the end of the day": ["in the end", "finally"],
    "in order to": ["to"],
    "due to the fact that": ["because"],
    "in the event that": ["if"],
    "a significant number of": ["many", "a lot of"],
    "the vast majority of": ["most"],
    "on the other hand": ["then again", "but"],
    "first and foremost": ["first", "mainly"],
    "last but not least": ["finally", "and one more thing"],
    "in this article": ["here", "below"],
    "this article will": ["i'll", "this piece"],
    "this blog post": ["this piece", "what follows"],
    "let's explore": ["look at", "here's"],
    "let us explore": ["look at"],
    "let's dive in": ["here goes", "starting with"],
    "without further ado": ["anyway", "here it is"],
    "moving forward": ["from here", "next"],
    "as previously mentioned": ["as i said", "again"],
    "as mentioned earlier": ["earlier", "as i said"],
    "it goes without saying": ["obviously"],
    "the fact that": ["that"],
    "in terms of": ["for", "on"],
    "with regard to": ["about", "on"],
    "with respect to": ["about"],
    "in light of": ["given"],
    "based on the above": ["from that"],
    "taking into account": ["given"],
    "from a broader perspective": ["more broadly"],
    "it is widely believed": ["plenty of people think"],
    "studies have shown": ["research suggests", "there's evidence"],
    "research suggests that": ["research points to"],
    "experts agree that": ["a lot of people in the field say"],
    "one of the most important": ["a big"],
    "a key aspect of": ["a big part of"],
    "a key component of": ["part of"],
    "the key takeaway": ["the point"],
    "the bottom line is": ["the point is"],
    "overall": ["all in all", "broadly"],
    "additionally": ["also", "and"],
    "furthermore": ["also", "and"],
    "moreover": ["also", "plus"],
    "consequently": ["so", "as a result"],
    "subsequently": ["later", "then"],
    "nevertheless": ["still", "even so"],
    "nonetheless": ["still"],
    "therefore": ["so", "that's why"],
    "thus": ["so"],
    "hence": ["so"],
    "thereby": ["and that"],
    "indeed": ["really", "honestly"],
    "notably": ["worth a look:", "especially"],
    "importantly": ["more usefully"],
    "significantly": ["by a lot", "noticeably"],
    "essentially": ["basically"],
    "ultimately": ["in the end"],
    "particularly": ["especially"],
}

AI_OPENERS: tuple[str, ...] = (
    "in today's",
    "in conclusion",
    "in summary",
    "this article",
    "this comprehensive",
    "in the ever-evolving",
    "in the rapidly",
    "in a world where",
    "when it comes to",
    "it is important",
    "it's important",
    "first and foremost",
    "let's delve",
    "let's dive",
    "imagine a world",
    "picture this",
)

TRANSITION_SWAPS: dict[str, list[str]] = {
    "however": ["but", "still", "even so", "that said"],
    "therefore": ["so", "that's why", "which is why"],
    "furthermore": ["also", "and", "plus"],
    "moreover": ["also", "on top of that"],
    "additionally": ["also", "and"],
    "consequently": ["so", "as a result"],
    "subsequently": ["later", "after that"],
    "nevertheless": ["still", "even so"],
    "nonetheless": ["still"],
    "meanwhile": ["at the same time", "while that happened"],
    "similarly": ["same idea:", "in the same way"],
    "conversely": ["the other way around", "flipped"],
    "accordingly": ["so"],
    "hence": ["so"],
    "thus": ["so"],
    "indeed": ["really"],
    "overall": ["all in all"],
    "finally": ["last", "to close"],
}

STRUCTURE_TRANSITIONS: tuple[str, ...] = (
    "that said,",
    "here's the thing:",
    "zoom out for a second.",
    "two caveats.",
    "practically,",
    "in practice,",
    "on the ground,",
    "worth flagging:",
)

HUMAN_ASIDES: tuple[str, ...] = (
    "at least in the cases i've seen",
    "your mileage will vary",
    "this is the boring but useful part",
    "i keep coming back to this",
    "not a law of nature, just a pattern",
    "take this with some salt",
)

CONTRACTIONS: dict[str, str] = {
    "do not": "don't",
    "does not": "doesn't",
    "did not": "didn't",
    "is not": "isn't",
    "are not": "aren't",
    "was not": "wasn't",
    "were not": "weren't",
    "have not": "haven't",
    "has not": "hasn't",
    "had not": "hadn't",
    "will not": "won't",
    "would not": "wouldn't",
    "could not": "couldn't",
    "should not": "shouldn't",
    "cannot": "can't",
    "can not": "can't",
    "it is": "it's",
    "that is": "that's",
    "there is": "there's",
    "there are": "there're",
    "i am": "i'm",
    "we are": "we're",
    "they are": "they're",
    "you are": "you're",
    "i have": "i've",
    "we have": "we've",
    "they have": "they've",
    "you have": "you've",
    "i will": "i'll",
    "we will": "we'll",
    "they will": "they'll",
    "you will": "you'll",
    "i would": "i'd",
    "we would": "we'd",
    "let us": "let's",
}

EXPAND_CONTRACTIONS: dict[str, str] = {v: k for k, v in CONTRACTIONS.items()}

PUNCTUATION_SWAPS: dict[str, list[str]] = {
    ",": [",", " —", ";"],
    ".": [".", "."],
    "!": [".", "."],
    ":": [":", " —"],
}

# Meaning-preserving, conservative synonym map. Only high-confidence pairs.
SYNONYMS: dict[str, list[str]] = {
    "achieve": ["reach", "hit", "get to"],
    "additional": ["extra", "more", "added"],
    "advantage": ["plus", "upside", "benefit"],
    "allow": ["let", "permit", "enable"],
    "almost": ["nearly", "just about"],
    "analysis": ["look", "read", "breakdown"],
    "analyze": ["look at", "break down", "examine"],
    "appear": ["show up", "seem", "turn up"],
    "approach": ["take", "method", "way in"],
    "approximately": ["about", "roughly"],
    "assist": ["help", "back"],
    "attempt": ["try", "shot"],
    "begin": ["start", "open"],
    "beneficial": ["helpful", "useful"],
    "better": ["stronger", "clearer", "improved"],
    "big": ["large", "huge", "major"],
    "challenge": ["problem", "hurdle", "snag"],
    "change": ["shift", "alter", "tweak"],
    "choose": ["pick", "select"],
    "common": ["usual", "typical", "everyday"],
    "complete": ["finish", "wrap up"],
    "complex": ["messy", "involved", "dense"],
    "component": ["part", "piece"],
    "concept": ["idea", "notion"],
    "conclude": ["finish", "close", "wrap"],
    "consider": ["look at", "weigh", "think about"],
    "construct": ["build", "put together"],
    "contain": ["hold", "include"],
    "continue": ["keep going", "go on"],
    "contribute": ["add", "feed into"],
    "create": ["make", "build", "put together"],
    "currently": ["now", "right now"],
    "decrease": ["drop", "shrink", "cut"],
    "demonstrate": ["show", "make clear"],
    "describe": ["lay out", "spell out"],
    "determine": ["figure out", "work out"],
    "develop": ["build", "grow", "shape"],
    "different": ["other", "separate", "unlike"],
    "difficult": ["hard", "tough"],
    "discover": ["find", "uncover"],
    "discuss": ["talk through", "go over"],
    "distribute": ["spread", "hand out"],
    "effective": ["it works", "solid", "useful"],
    "enable": ["let", "make possible"],
    "encourage": ["nudge", "push"],
    "ensure": ["make sure", "see that"],
    "entire": ["whole", "full"],
    "establish": ["set up", "put in place"],
    "evaluate": ["weigh", "judge", "score"],
    "evidence": ["signs", "proof", "data"],
    "example": ["case", "instance"],
    "existing": ["current", "already there"],
    "explain": ["spell out", "walk through"],
    "explore": ["look at", "poke at"],
    "fast": ["quick", "rapid"],
    "feature": ["trait", "piece"],
    "final": ["last", "closing"],
    "find": ["spot", "come across"],
    "following": ["next", "below"],
    "function": ["job", "role"],
    "fundamental": ["basic", "core"],
    "generate": ["make", "produce"],
    "goal": ["aim", "target"],
    "great": ["strong", "solid", "good"],
    "help": ["aid", "back"],
    "however": ["but", "still"],
    "identify": ["spot", "name", "pin down"],
    "implement": ["put in", "ship", "roll out"],
    "important": ["big", "material", "that matters"],
    "improve": ["make better", "sharpen", "lift"],
    "include": ["take in", "cover"],
    "increase": ["raise", "grow", "bump"],
    "indicate": ["point to", "show"],
    "individual": ["person", "single"],
    "information": ["info", "detail"],
    "initial": ["first", "early"],
    "issue": ["problem", "snag"],
    "large": ["big", "huge"],
    "limit": ["cap", "bound"],
    "maintain": ["keep", "hold"],
    "major": ["big", "main"],
    "make": ["build", "create"],
    "many": ["lots of", "plenty of"],
    "method": ["way", "approach"],
    "modify": ["change", "tweak"],
    "necessary": ["needed", "required"],
    "need": ["require", "call for"],
    "new": ["fresh", "recent"],
    "numerous": ["many", "lots of"],
    "obtain": ["get", "pick up"],
    "occur": ["happen", "show up"],
    "offer": ["give", "put forward"],
    "often": ["a lot", "frequently"],
    "operate": ["run", "work"],
    "opportunity": ["opening", "shot", "chance"],
    "option": ["choice", "route"],
    "outcome": ["result", "ending"],
    "overall": ["broadly", "all in all"],
    "perform": ["do", "run", "carry out"],
    "perhaps": ["maybe", "possibly"],
    "period": ["stretch", "span"],
    "permit": ["let", "allow"],
    "perspective": ["view", "angle"],
    "potential": ["possible", "likely"],
    "previous": ["earlier", "prior"],
    "primary": ["main", "first"],
    "problem": ["issue", "snag"],
    "process": ["steps", "flow"],
    "produce": ["make", "turn out"],
    "provide": ["give", "offer", "hand over"],
    "purpose": ["point", "aim"],
    "quickly": ["fast", "in a hurry"],
    "receive": ["get", "take"],
    "reduce": ["cut", "shrink"],
    "regard": ["see", "treat"],
    "related": ["linked", "tied"],
    "remain": ["stay", "keep"],
    "require": ["need", "call for"],
    "result": ["outcome", "effect"],
    "reveal": ["show", "bring out"],
    "several": ["a few", "multiple"],
    "significant": ["big", "material", "real"],
    "similar": ["alike", "close"],
    "simple": ["plain", "straightforward"],
    "small": ["tiny", "modest"],
    "specific": ["particular", "exact"],
    "start": ["begin", "open"],
    "strong": ["solid", "firm"],
    "structure": ["shape", "layout"],
    "subsequent": ["later", "next"],
    "successful": ["it worked", "it landed"],
    "sufficient": ["enough", "adequate"],
    "suggest": ["point to", "float"],
    "support": ["back", "help"],
    "system": ["setup", "stack"],
    "technique": ["trick", "method"],
    "therefore": ["so", "that's why"],
    "typical": ["usual", "normal"],
    "understand": ["get", "see"],
    "use": ["put to work", "apply"],
    "usually": ["most of the time", "typically"],
    "various": ["different", "assorted"],
    "very": ["really", "pretty"],
    "want": ["need", "look for"],
    "work": ["job", "effort"],
}


def frequency_rank(word: str) -> int | None:
    return FREQUENCY.get(word.lower())


def unigram_logprob(word: str, vocab: int = 50_000) -> float:
    """Smoothed log-unigram using Zipf ranks.

    Unknown words get a low probability (high surprisal) — they are
    exactly the high-entropy positions tournament watermarks prefer.
    """

    rank = FREQUENCY.get(word.lower())
    if rank is None:
        # Tail mass for OOV.
        p = 1.0 / (vocab * math.log(vocab))
    else:
        p = 1.0 / ((rank + 2.0) * math.log(vocab + 1.0))
    return math.log(max(p, 1e-12))


def word_entropy_hint(word: str) -> float:
    """0–1 score: higher means a better watermark-bearing substitution target."""

    w = word.lower()
    if not w.isalpha() or len(w) < 4:
        return 0.05
    rank = FREQUENCY.get(w)
    if rank is None:
        return 0.92
    # Mid-frequency content words are the sweet spot: frequent enough to
    # have synonyms, rare enough that a sampler had a real choice.
    if rank <= 80:
        return 0.08
    if rank <= 250:
        return 0.45
    return 0.78
