#!/usr/bin/env python3
"""
Tate & Liza (Pokemon Ruby/Sapphire) Double Battle Simulator
=============================================================

A from-scratch simulation of the Mossdeep Gym leader double battle against
Tate & Liza, whose team in Ruby/Sapphire is a Level 42 Solrock and a
Level 42 Lunatone (their whole team -- no reinforcements, matching the
real games).

This models core Generation III battle mechanics:
  - Gen 3 damage formula (base power / Atk / Def / level, STAB, type
    effectiveness, weather, screens, critical hits, the 85-100% random roll)
  - Gen 3 stat calculation (IV/EV/nature) and stat-stage multiplier table
  - Status conditions (sleep, paralysis, burn, poison/toxic, freeze, confusion)
  - Light Screen (halves special damage, or only reduces it by 1/3 in a
    double battle with two Pokemon still up on that side -- a real Gen 3/4
    quirk), Sunny Day / weather, Calm Mind, etc.
  - Double-battle targeting: single-target moves need a target chosen;
    Earthquake/Surf/Explosion-type moves hit everyone else on the field;
    Rock Slide/Discharge-type moves hit both opposing Pokemon.
  - Tate & Liza's trainer AI, ported directly from the pokeruby source
    (see the "TRAINER AI" section below for exactly which files/functions).

SOURCE VERIFICATION NOTE
-------------------------
The person supplied the actual pokeruby decompilation (github.com/pret/
pokeruby) as a zip, which was extracted and read directly. The following
are taken verbatim from that source rather than approximated:
  - Tate & Liza's exact party data (src/data/trainer_parties.h /
    trainers_en.h): party order (Lunatone, then Solrock), levels (42/42),
    movesets, the doubleBattle flag, aiFlags = 0x7, and 4 Hyper Potions.
  - Their IVs: the party's raw "iv" byte is 250, which Gen 3 converts via
    fixedIV = iv * 31 / 255 (src/battle_main.c) -> IV 30 in every stat.
  - Their natures: derived by hand-computing the real personality-value
    hash (trainer name + species name, run through the game's charmap.txt
    byte encoding, per src/battle_main.c) -> Lunatone is Bold, Solrock is
    Serious (neutral).
  - Base stats/types (src/data/pokemon/base_stats.h): both Rock/Psychic,
    Levitate; Solrock 70/95/85/55/65/70, Lunatone 70/55/65/95/85/70
    (HP/Atk/Def/SpA/SpD/Spe).
  - The AI's actual scoring scripts for aiFlags=0x7 (AI_CheckBadMove +
    AI_CheckViability + AI_TryToFaint -- notably NOT AI_SetupFirstTurn),
    read out of data/battle_ai_scripts.s and src/battle_ai_script_commands.c,
    and ported opcode-for-opcode for the branches Tate & Liza's 7 moves
    actually reach (see the "TRAINER AI" section for the full citation).
  - The real item-use threshold (src/battle_ai_switch_items.c
    ShouldUseItem): heal at HP < 25% of max, not 50%, with no "skip if I
    can KO" logic in the real code.

CHANGELOG -- SECOND PASS FIXES
--------------------------------
A follow-up round of corrections, once again checked against the source
rather than guessed:
  - The damage-modifier ORDER was wrong before and is now verified against
    src/calculate_base_damage.c and the atk05/atk06/atk07 battle-script
    handlers: base -> burn -> screens -> multi-target halving -> weather
    -> (+2, which really is added LAST inside the base-damage step, not
    right after the power*Atk/Def division like most write-ups of the
    formula show it) -> critical hit -> STAB -> type effectiveness ->
    random 85-100% roll (applied dead last, after STAB/type, not before).
    This was the source of the reported Surf + Sunny Day + Light Screen
    interaction being off.
  - Surf's real target is TARGET_BOTH_ENEMIES, not "all other Pokemon" --
    it does NOT hit your own partner (confirmed in src/data/battle_moves.c).
    Blizzard is also TARGET_BOTH_ENEMIES (fixed to auto-hit both foes
    rather than being player-targeted).
  - The Gen 3 multi-target doubles halving rule is real and now
    implemented exactly: moves whose target is TARGET_BOTH_ENEMIES (Surf,
    Rock Slide, Blizzard, Powder Snow) are halved only while both
    opposing Pokemon are alive; TARGET_ALL_EXCEPT_USER moves (Earthquake,
    Explosion) are NEVER halved -- a genuine, verified asymmetry, not a
    bug (it's why double-Earthquake was a notorious RSE doubles strategy).
    Checked fresh per target hit, so a spread move that faints the first
    target correctly deals full, unhalved damage to the second.
  - Mystic Water (+10% to the raw Atk/SpA for matching-type moves) and
    Torrent/Overgrow/Blaze/Swarm (+50% move power at <=1/3 max HP) are
    implemented from the exact values in src/data/items_en.h and
    src/calculate_base_damage.c.
  - Audited the whole player movedex against Gen 3's TYPE-based physical/
    special split (include/battle.h: IS_TYPE_PHYSICAL/IS_TYPE_SPECIAL)
    and found 12 moves miscategorized using modern (Gen 4+) categories
    instead -- e.g. the elemental punches, Crunch, Bite, Shadow Ball, and
    Waterfall all use the WRONG stat pre-fix. All 12 corrected. Also
    caught and fixed the type chart having Ghost vs. Psychic backwards
    (that 0x interaction was a Gen 1-only bug, corrected industry-wide
    since Gen 2 -- Ghost is 2x on Psychic in Ruby/Sapphire).
  - Removed Discharge and Energy Ball from the movedex -- neither move
    exists in Gen 3 (both are Gen 4 additions); verified against
    include/constants/moves.h (354 moves total, ending at Psycho Boost).
  - Added Dive (two-turn, semi-invulnerable, Special despite "looking"
    physical -- Water is a special-category type here), Strength, Rock
    Smash, and Powder Snow, all with exact Gen 3 data.
  - Healing/status items (Potion...Full Heal) can now target ANY of your
    Pokemon, active or benched; X-items are restricted to whichever
    Pokemon is using them (a benched Pokemon doesn't have battle stat
    stages for an X-item to raise).
  - You can now specify a starting HP (and optionally a starting status
    condition) per Pokemon in the setup wizard.
  - Your party can now be 2-4 Pokemon: the first two are active, the
    rest sit in reserve and get sent in immediately (with a replacement
    prompt) the instant an active slot faints, checked after every
    individual action and again after end-of-turn residual damage.

CHANGELOG -- THIRD PASS (badge boosts, preset team, weather moves)
---------------------------------------------------------------------
  - Gym badge stat boosts implemented: a real (if obscure) Gen 3 mechanic,
    verified in src/calculate_base_damage.c's BADGE_BOOST macro and the
    speed-boost check in src/battle_main.c. Each badge gives the PLAYER's
    Pokemon a flat +10% to exactly one stat: Stone Badge -> Attack, Dynamo
    Badge -> Speed, Balance Badge -> Defense. The source also shows a
    Special Attack/Special Defense boost tied to the Mind Badge -- deliberately
    NOT modeled here, since the Mind Badge is what Tate & Liza award you
    for winning this exact fight, so it's impossible to already have it.
    PLAYER_BADGES defaults to {1,2,3,4,5,6}, i.e. every badge obtainable
    before Mossdeep.
  - Added Growl, Fly, Teleport, and Hail, all with exact Gen 3 data (Growl
    auto-hits both foes; Fly is semi-invulnerable like Dive; Teleport
    correctly does nothing in a trainer battle -- you can't flee one in
    Gen 3; Hail sets weather and deals 1/16 max HP/turn to non-Ice-types,
    per src/battle_script_commands.c's atk96_weatherdamage).
  - Implemented Castform's Forecast ability, ported directly from
    src/battle_util.c's real Castform form-change function: it becomes
    Fire/Water/Ice-typed in sun/rain/hail and Normal otherwise, reverting
    the instant the weather ends. This means a Forecast Castform using its
    own Hail becomes Ice-typed and immune to that same Hail's chip damage,
    exactly like the real games.
  - Added a preset 4-Pokemon party (Swampert/Wingull/Abra/Castform)
    available via the in-app menu or `--preset`. The requested stats were
    given as final, post-IV/EV/nature numbers rather than raw IVs, so the
    IV/EV combination for each stat was solved by brute-force search
    against this simulator's own stat formula until it reproduced the
    exact requested integer (preferring EV 0 wherever any IV 0-31
    achieves it at EV 0). Abilities/items left unspecified default to
    each species' real, only available Gen 3 ability (Torrent, Keen Eye,
    Synchronize, Forecast).

CHANGELOG -- FOURTH PASS (two reported bugs, both source-verified)
---------------------------------------------------------------------
  - FIXED: a spread move that faints its first target no longer loses its
    multi-target damage halving on the second target. Verified in
    src/pokemon_2.c's CountAliveMons (which checks gAbsentBattlerFlags)
    and src/battle_script_commands.c's atk50_openpartyscreen: Gen 3 does
    NOT mark a fainted Pokemon "absent" until a separate replacement step
    that only runs after the WHOLE move (every target a spread move
    hits) finishes -- ATK49_NEXT_TARGET loops back and re-runs the hit
    script for the second target before that replacement step ever runs.
    So if Surf's first hit faints Solrock, the second hit on Lunatone
    still sees "2 alive" and is still halved. The fix threads a snapshot
    of each side's alive count, taken before any of the move's hits are
    resolved, through the whole damage pipeline instead of recomputing it
    fresh per hit.
  - FIXED: the AI's move-scoring wasn't excluding "discouraged powerful"
    move effects (src/battle_ai_script_commands.c's
    sDiscouragedPowerfulMoveEffects -- Explosion, Solar Beam, Hyper Beam,
    Overheat, etc.) from its "is this my most powerful move" comparison.
    In the real game, a discouraged move can still get the +4 "I can KO
    this turn" bonus, but it never earns the -1 penalty for being weaker
    than another option, AND its high raw power doesn't count against
    other moves either. Net effect: Solar Beam now correctly scores a
    flat, unremarkable 100 against a target it can't one-shot (tying with
    Solrock's other options rather than beating them on raw power alone),
    only clearly separating itself when it can secure a KO outright.
  - Also fixed: real Gen 3 clamps a move's score to a minimum of 0 after
    EACH individual scoring script's contribution, not just once at the
    end -- so a step that drives the score deeply negative can't be
    "refunded" by a later positive step. The AI scorer now clamps after
    every step (CheckBadMove, CheckViability, TryToFaint) to match.
  - Confirmed (not a bug): Gen 3's actual tie-breaking for equally-scored
    moves is genuinely `Random() % numOfBestMoves` -- a uniform random
    choice among every move tied for the top score, not a deterministic
    "first" or "last" rule. Verified directly in
    BattleAI_GetAIActionToUse. This means a target so frail that several
    of Solrock's moves can all one-shot it (this preset team's Wingull
    and Abra are Level 5 and 8, respectively, facing a Level 42 Solrock)
    will see a genuine random tie between those moves, including Solar
    Beam -- it is not excluded from being chosen in that case, though it
    also isn't specially favored. Solar Beam's real, verified edge is
    specifically that its 4x Grass weakness against Swampert makes it far
    more likely than the others to reach the outright "+4, unambiguously
    best" bonus rather than a plain tie.

CHANGELOG -- FIFTH PASS (Solar Beam targeting, empirically verified)
---------------------------------------------------------------------
  - Added an explicit rule to ai_try_to_faint: when a charging move like
    Solar Beam could secure a KO this turn, but some OTHER, non-charging
    move in the same moveset could ALSO secure that KO, the charging move
    no longer gets the "+4, clearly the best" bonus -- the immediate
    attack wins outright instead of the two tying. This matches behavior
    reported from hundreds of hands-on reps of this exact fight: Solrock
    reliably uses Solar Beam against a full-HP Swampert (which nothing
    else can threaten to KO), but switches to Psychic the moment
    Swampert's HP drops low enough for Psychic alone to finish it, and
    never uses Solar Beam in that situation.
  - Honesty note on sourcing: unlike the rest of this AI port, this
    specific rule is NOT something a source read turned up. Both a full
    pass through data/battle_ai_scripts.s / battle_ai_script_commands.c
    (is_most_powerful_move, AI_TryToFaint, AI_CV_ChargeUpMove, the
    sDiscouragedPowerfulMoveEffects list) and cross-referencing an
    independently written, structurally-parallel Gen 4 AI reference
    (which documents an equivalent "discouraged move" mechanism with no
    analogous "immediate move beats charging move in a KO tie" rule)
    turned up nothing that would produce this specific preference as a
    guaranteed outcome rather than a 50/50 coin flip. It's implemented
    here on the strength of the reported empirical testing, flagged
    distinctly from the rest of the AI logic, which remains a direct,
    verified port.

REMAINING KNOWN SIMPLIFICATIONS
---------------------------------
A few corners were still simplified, either because they're extremely
low-impact for this specific 2-move-effect-set matchup or because they'd
require restructuring beyond what's reasonable here:
  - AI_CV_ChargeUpMove's "does the target know Protect" check is
    approximated as "is Protect anywhere in the target's moveset" rather
    than "did the target lock in Protect THIS turn" (the real games
    resolve all battlers' chosen actions before executing any of them, so
    the AI script may actually be able to see the turn's already-chosen
    move -- this simulator's turn loop doesn't thread that information
    through).
  - AI_CV_Sleep's minor bonus for putting a Pokemon to sleep when it
    knows Dream Eater/Nightmare is omitted (always 0) since arbitrary
    player movesets make it a rare, low-impact edge case.
  - Type effectiveness across a defender's two types is applied as one
    combined multiply-then-floor rather than two fully separate sequential
    integer truncations (ModulateDmgByType is called once per type in the
    real code); the two can differ by at most 1 damage in rare cases.
  - The player-side movedex is a curated ~75-move subset useful against
    Rock/Psychic, not the full Gen 3 move list; picking an unlisted move
    prompts you to define its stats on the spot.
"""

import random
import math
import sys
from dataclasses import dataclass, field
from enum import Enum, auto


# ============================================================================
# TYPES
# ============================================================================

TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting",
    "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost",
    "Dragon", "Dark", "Steel",
]

# Gen 3 type chart: ATTACK_TYPE -> {DEFEND_TYPE: multiplier}. Omitted entries = 1x.
# (No Fairy type existed yet; Steel resists Ghost & Dark in Gen 3; Dark is
#  immune... no, Psychic is what's immune to nothing special vs Dark; Ghost is
#  immune to Normal & Fighting; Ground is immune to Flying-type... wait it's
#  the reverse -- Flying is immune to Ground. See table below.)
TYPE_CHART = {
    "Normal":  {"Rock": 0.5, "Ghost": 0.0, "Steel": 0.5},
    "Fire":    {"Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ice": 2.0, "Bug": 2.0,
                "Rock": 0.5, "Dragon": 0.5, "Steel": 2.0},
    "Water":   {"Fire": 2.0, "Water": 0.5, "Grass": 0.5, "Ground": 2.0,
                "Rock": 2.0, "Dragon": 0.5},
    "Electric":{"Water": 2.0, "Electric": 0.5, "Grass": 0.5, "Ground": 0.0,
                "Flying": 2.0, "Dragon": 0.5},
    "Grass":   {"Fire": 0.5, "Water": 2.0, "Grass": 0.5, "Poison": 0.5,
                "Ground": 2.0, "Flying": 0.5, "Bug": 0.5, "Rock": 2.0,
                "Dragon": 0.5, "Steel": 0.5},
    "Ice":     {"Water": 0.5, "Grass": 2.0, "Ice": 0.5, "Ground": 2.0,
                "Flying": 2.0, "Dragon": 2.0, "Steel": 0.5, "Fire": 0.5},
    "Fighting":{"Normal": 2.0, "Ice": 2.0, "Poison": 0.5, "Flying": 0.5,
                "Psychic": 0.5, "Bug": 0.5, "Rock": 2.0, "Ghost": 0.0,
                "Dark": 2.0, "Steel": 2.0},
    "Poison":  {"Grass": 2.0, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5,
                "Ghost": 0.5, "Steel": 0.0},
    "Ground":  {"Fire": 2.0, "Electric": 2.0, "Grass": 0.5, "Poison": 2.0,
                "Flying": 0.0, "Bug": 0.5, "Rock": 2.0, "Steel": 2.0},
    "Flying":  {"Electric": 0.5, "Grass": 2.0, "Fighting": 2.0, "Bug": 2.0,
                "Rock": 0.5, "Steel": 0.5},
    "Psychic": {"Fighting": 2.0, "Poison": 2.0, "Psychic": 0.5, "Dark": 0.0,
                "Steel": 0.5},
    "Bug":     {"Fire": 0.5, "Grass": 2.0, "Fighting": 0.5, "Poison": 0.5,
                "Flying": 0.5, "Psychic": 2.0, "Ghost": 0.5, "Dark": 2.0,
                "Steel": 0.5},
    "Rock":    {"Fire": 2.0, "Ice": 2.0, "Fighting": 0.5, "Ground": 0.5,
                "Flying": 2.0, "Bug": 2.0, "Steel": 0.5},
    "Ghost":   {"Normal": 0.0, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5,
                "Steel": 0.5},
    "Dragon":  {"Dragon": 2.0, "Steel": 0.5},
    "Dark":    {"Fighting": 0.5, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5,
                "Steel": 0.5},
    "Steel":   {"Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2.0,
                "Rock": 2.0, "Steel": 0.5},
}


def type_effectiveness(move_type, defender_types):
    mult = 1.0
    chart = TYPE_CHART.get(move_type, {})
    for t in defender_types:
        mult *= chart.get(t, 1.0)
    return mult


# ============================================================================
# STATS
# ============================================================================

STAT_NAMES = ["hp", "atk", "def", "spa", "spd", "spe"]

# nature -> (boosted stat, lowered stat); neutral natures map a stat to itself
NATURES = {
    "hardy": (None, None), "docile": (None, None), "serious": (None, None),
    "bashful": (None, None), "quirky": (None, None),
    "lonely": ("atk", "def"), "brave": ("atk", "spe"), "adamant": ("atk", "spa"),
    "naughty": ("atk", "spd"),
    "bold": ("def", "atk"), "relaxed": ("def", "spe"), "impish": ("def", "spa"),
    "lax": ("def", "spd"),
    "timid": ("spe", "atk"), "hasty": ("spe", "def"), "jolly": ("spe", "spa"),
    "naive": ("spe", "spd"),
    "modest": ("spa", "atk"), "mild": ("spa", "def"), "quiet": ("spa", "spe"),
    "rash": ("spa", "spd"),
    "calm": ("spd", "atk"), "gentle": ("spd", "def"), "sassy": ("spd", "spe"),
    "careful": ("spd", "spa"),
}


def nature_multiplier(nature, stat):
    nature = (nature or "hardy").lower()
    boosted, lowered = NATURES.get(nature, (None, None))
    if stat == boosted and stat != lowered:
        return 1.1
    if stat == lowered and stat != boosted:
        return 0.9
    return 1.0


def calc_stat(base, iv, ev, level, stat, nature="hardy"):
    core = (2 * base + iv + ev // 4) * level // 100
    if stat == "hp":
        if base == 1:  # Shedinja-style, not used here but kept for safety
            return 1
        return core + level + 10
    val = core + 5
    return int(math.floor(val * nature_multiplier(nature, stat)))


def stage_multiplier(stage):
    stage = max(-6, min(6, stage))
    if stage >= 0:
        return (2 + stage) / 2
    return 2 / (2 - stage)


# ============================================================================
# MOVES
# ============================================================================

class Cat(Enum):
    PHYSICAL = auto()
    SPECIAL = auto()
    STATUS = auto()


class Target(Enum):
    SELF = auto()          # affects the user
    SINGLE = auto()        # user chooses one opposing Pokemon (or ally, rare)
    ALL_FOES = auto()      # hits both opposing Pokemon, not the user's ally
    ALL_OTHERS = auto()    # hits both opposing Pokemon AND the user's ally
    FIELD = auto()         # weather
    SIDE = auto()          # Light Screen / Reflect - user's side of the field


@dataclass
class Move:
    name: str
    type_: str
    category: Cat
    power: int
    accuracy: int          # 0 = never misses
    pp: int
    priority: int = 0
    target: Target = Target.SINGLE
    effect: str = None            # tag handled in battle engine
    effect_chance: int = 0        # percent chance effect triggers (0/100 = always for status moves that always apply on hit)
    self_boosts: tuple = ()       # e.g. (("spa", 1), ("spd", 1))
    target_boosts: tuple = ()     # e.g. (("spd", -1),)
    high_crit: bool = False
    charge: bool = False          # two-turn move that charges (Solar Beam)
    semi_invulnerable: bool = False  # charge turn also dodges incoming moves (Dive)
    heal_fraction: float = 0.0    # e.g. Recover = 0.5
    drain_fraction: float = 0.0   # e.g. Giga Drain = 0.5 of damage dealt
    recoil_fraction: float = 0.0  # e.g. Double-Edge = 1/3
    sound: bool = False

    def __post_init__(self):
        self.max_pp = self.pp


def M(*args, **kwargs):
    return Move(*args, **kwargs)


# A reasonably broad Gen-3-legal movedex. Add more entries the same way if
# you pick a move for your Pokemon that isn't listed here -- the setup
# wizard will also let you type in custom move stats on the fly.
MOVEDEX = {}


def _add(mv):
    MOVEDEX[mv.name] = mv


# --- Tate & Liza's actual moves -------------------------------------------
_add(M("Psychic", "Psychic", Cat.SPECIAL, 90, 100, 10,
       effect="lower_spd", effect_chance=10, target_boosts=(("spd", -1),)))
_add(M("Light Screen", "Psychic", Cat.STATUS, 0, 0, 30, target=Target.SIDE,
       effect="light_screen"))
_add(M("Hypnosis", "Psychic", Cat.STATUS, 0, 60, 20, target=Target.SINGLE,
       effect="sleep"))
_add(M("Calm Mind", "Psychic", Cat.STATUS, 0, 0, 20, target=Target.SELF,
       self_boosts=(("spa", 1), ("spd", 1))))
_add(M("Sunny Day", "Fire", Cat.STATUS, 0, 0, 5, target=Target.FIELD,
       effect="sun"))
_add(M("Hail", "Ice", Cat.STATUS, 0, 0, 10, target=Target.FIELD,
       effect="hail"))
_add(M("Solar Beam", "Grass", Cat.SPECIAL, 120, 100, 10, charge=True))
_add(M("Flamethrower", "Fire", Cat.SPECIAL, 95, 100, 15,
       effect="burn", effect_chance=10))

# --- Common player-side options (offense) ----------------------------------
_add(M("Surf", "Water", Cat.SPECIAL, 95, 100, 15, target=Target.ALL_FOES))
_add(M("Dive", "Water", Cat.SPECIAL, 60, 100, 10, charge=True, semi_invulnerable=True))
_add(M("Fly", "Flying", Cat.PHYSICAL, 70, 95, 15, charge=True, semi_invulnerable=True))
_add(M("Hydro Pump", "Water", Cat.SPECIAL, 120, 80, 5))
_add(M("Water Gun", "Water", Cat.SPECIAL, 40, 100, 25))
_add(M("Waterfall", "Water", Cat.SPECIAL, 80, 100, 15))
_add(M("Ice Beam", "Ice", Cat.SPECIAL, 95, 100, 10, effect="freeze", effect_chance=10))
_add(M("Blizzard", "Ice", Cat.SPECIAL, 120, 70, 5, target=Target.ALL_FOES,
       effect="freeze", effect_chance=10))
_add(M("Powder Snow", "Ice", Cat.SPECIAL, 40, 100, 25, target=Target.ALL_FOES,
       effect="freeze", effect_chance=10))
_add(M("Ice Punch", "Ice", Cat.SPECIAL, 75, 100, 15, effect="freeze", effect_chance=10))
_add(M("Earthquake", "Ground", Cat.PHYSICAL, 100, 100, 10, target=Target.ALL_OTHERS))
_add(M("Rock Slide", "Rock", Cat.PHYSICAL, 75, 90, 10, target=Target.ALL_FOES,
       effect="flinch", effect_chance=30))
_add(M("Rock Tomb", "Rock", Cat.PHYSICAL, 50, 80, 10, effect="lower_spe",
       effect_chance=100, target_boosts=(("spe", -1),)))
_add(M("Shadow Ball", "Ghost", Cat.PHYSICAL, 80, 100, 15, effect="lower_spd",
       effect_chance=20, target_boosts=(("spd", -1),)))
_add(M("Crunch", "Dark", Cat.SPECIAL, 80, 100, 15, effect="lower_def",
       effect_chance=20, target_boosts=(("def", -1),)))
_add(M("Bite", "Dark", Cat.SPECIAL, 60, 100, 25, effect="flinch", effect_chance=30))
_add(M("Faint Attack", "Dark", Cat.SPECIAL, 60, 0, 20))  # never misses
_add(M("Thunderbolt", "Electric", Cat.SPECIAL, 95, 100, 15, effect="paralyze",
       effect_chance=10))
_add(M("Thunder", "Electric", Cat.SPECIAL, 120, 70, 10, effect="paralyze",
       effect_chance=10))
_add(M("Thunder Punch", "Electric", Cat.SPECIAL, 75, 100, 15, effect="paralyze",
       effect_chance=10))
_add(M("Sludge Bomb", "Poison", Cat.PHYSICAL, 90, 100, 10, effect="poison",
       effect_chance=30))
_add(M("Giga Drain", "Grass", Cat.SPECIAL, 60, 100, 5, drain_fraction=0.5))
_add(M("Razor Leaf", "Grass", Cat.SPECIAL, 55, 95, 25, high_crit=True))
_add(M("Aerial Ace", "Flying", Cat.PHYSICAL, 60, 0, 20))
_add(M("Wing Attack", "Flying", Cat.PHYSICAL, 60, 100, 35))
_add(M("Cross Chop", "Fighting", Cat.PHYSICAL, 100, 80, 5, high_crit=True))
_add(M("Rock Smash", "Fighting", Cat.PHYSICAL, 20, 100, 15, effect_chance=50,
       target_boosts=(("def", -1),)))
_add(M("Brick Break", "Fighting", Cat.PHYSICAL, 75, 100, 15))
_add(M("Sky Uppercut", "Fighting", Cat.PHYSICAL, 85, 90, 15))
_add(M("Dragon Claw", "Dragon", Cat.SPECIAL, 80, 100, 15))
_add(M("Iron Tail", "Steel", Cat.PHYSICAL, 100, 75, 15, effect="lower_def",
       effect_chance=30, target_boosts=(("def", -1),)))
_add(M("Meteor Mash", "Steel", Cat.PHYSICAL, 100, 85, 10, effect="raise_atk_self",
       effect_chance=20, self_boosts=(("atk", 1),)))
_add(M("Slash", "Normal", Cat.PHYSICAL, 70, 100, 20, high_crit=True))
_add(M("Return", "Normal", Cat.PHYSICAL, 102, 100, 20))
_add(M("Strength", "Normal", Cat.PHYSICAL, 80, 100, 15))
_add(M("Body Slam", "Normal", Cat.PHYSICAL, 85, 100, 15, effect="paralyze",
       effect_chance=30))
_add(M("Double-Edge", "Normal", Cat.PHYSICAL, 120, 100, 15, recoil_fraction=1/3))
_add(M("Hyper Beam", "Normal", Cat.PHYSICAL, 150, 90, 5))  # recharge not modeled
_add(M("Facade", "Normal", Cat.PHYSICAL, 70, 100, 20))
_add(M("Explosion", "Normal", Cat.PHYSICAL, 250, 100, 5, target=Target.ALL_OTHERS))
_add(M("Self-Destruct", "Normal", Cat.PHYSICAL, 200, 100, 5, target=Target.ALL_OTHERS))
_add(M("Fire Blast", "Fire", Cat.SPECIAL, 120, 85, 5, effect="burn", effect_chance=10))
_add(M("Fire Punch", "Fire", Cat.SPECIAL, 75, 100, 15, effect="burn", effect_chance=10))
_add(M("Overheat", "Fire", Cat.SPECIAL, 140, 90, 5, self_boosts=(("spa", -2),)))

# --- Status / support -------------------------------------------------------
_add(M("Toxic", "Poison", Cat.STATUS, 0, 85, 10, effect="toxic"))
_add(M("Thunder Wave", "Electric", Cat.STATUS, 0, 100, 20, effect="paralyze"))
_add(M("Will-O-Wisp", "Fire", Cat.STATUS, 0, 75, 15, effect="burn"))
_add(M("Spore", "Grass", Cat.STATUS, 0, 100, 15, effect="sleep"))
_add(M("Sleep Powder", "Grass", Cat.STATUS, 0, 75, 15, effect="sleep"))
_add(M("Stun Spore", "Grass", Cat.STATUS, 0, 75, 30, effect="paralyze"))
_add(M("Confuse Ray", "Ghost", Cat.STATUS, 0, 100, 10, effect="confuse"))
_add(M("Swords Dance", "Normal", Cat.STATUS, 0, 0, 20, target=Target.SELF,
       self_boosts=(("atk", 2),)))
_add(M("Amnesia", "Psychic", Cat.STATUS, 0, 0, 20, target=Target.SELF,
       self_boosts=(("spd", 2),)))
_add(M("Iron Defense", "Steel", Cat.STATUS, 0, 0, 15, target=Target.SELF,
       self_boosts=(("def", 2),)))
_add(M("Agility", "Psychic", Cat.STATUS, 0, 0, 30, target=Target.SELF,
       self_boosts=(("spe", 2),)))
_add(M("Growth", "Normal", Cat.STATUS, 0, 0, 40, target=Target.SELF,
       self_boosts=(("atk", 1), ("spa", 1))))
_add(M("Dragon Dance", "Dragon", Cat.STATUS, 0, 0, 20, target=Target.SELF,
       self_boosts=(("atk", 1), ("spe", 1))))
_add(M("Reflect", "Psychic", Cat.STATUS, 0, 0, 20, target=Target.SIDE,
       effect="reflect"))
_add(M("Rain Dance", "Water", Cat.STATUS, 0, 0, 5, target=Target.FIELD,
       effect="rain"))
_add(M("Recover", "Normal", Cat.STATUS, 0, 0, 20, target=Target.SELF,
       heal_fraction=0.5))
_add(M("Rest", "Psychic", Cat.STATUS, 0, 0, 10, target=Target.SELF,
       effect="rest"))
_add(M("Protect", "Normal", Cat.STATUS, 0, 0, 10, priority=4, target=Target.SELF,
       effect="protect"))
_add(M("Substitute", "Normal", Cat.STATUS, 0, 0, 10, target=Target.SELF,
       effect="substitute"))
_add(M("Taunt", "Dark", Cat.STATUS, 0, 100, 20, effect="taunt"))
_add(M("Helping Hand", "Normal", Cat.STATUS, 0, 0, 20, target=Target.SELF))
_add(M("Roar", "Normal", Cat.STATUS, 0, 100, 20))  # switch-forcing not modeled (no bench)
_add(M("Endure", "Normal", Cat.STATUS, 0, 0, 10, priority=4, target=Target.SELF,
       effect="endure"))
_add(M("Charm", "Normal", Cat.STATUS, 0, 100, 20, effect_chance=100,
       target_boosts=(("atk", -2),)))
_add(M("Screech", "Normal", Cat.STATUS, 0, 85, 40, effect_chance=100,
       target_boosts=(("def", -2),)))
_add(M("Growl", "Normal", Cat.STATUS, 0, 100, 40, target=Target.ALL_FOES,
       target_boosts=(("atk", -1),)))
_add(M("Teleport", "Psychic", Cat.STATUS, 0, 0, 20, target=Target.SELF))
# Teleport has no in-battle effect against a trainer (you can't flee a
# trainer battle in Gen 3); apply_self_move's "nothing happened" fallback
# covers this correctly since no self_boosts/heal/effect are set.


def get_move(name):
    """Look up a move by (case-insensitive) name."""
    for k, v in MOVEDEX.items():
        if k.lower() == name.lower():
            return v
    return None


# ============================================================================
# POKEMON
# ============================================================================

STATUSES = ("none", "slp", "psn", "brn", "prz", "frz", "tox")


@dataclass
class MoveSlot:
    move: Move
    pp: int = None

    def __post_init__(self):
        if self.pp is None:
            self.pp = self.move.pp


@dataclass(eq=False)  # identity-based equality/hash so Pokemon can key a dict (actions map)
class Pokemon:
    name: str
    species_types: tuple
    level: int
    base_stats: dict          # {"hp":.., "atk":.., "def":.., "spa":.., "spd":.., "spe":..}
    moves: list                # list[MoveSlot]
    ivs: dict = field(default_factory=lambda: {s: 31 for s in STAT_NAMES})
    evs: dict = field(default_factory=lambda: {s: 0 for s in STAT_NAMES})
    nature: str = "hardy"
    ability: str = ""
    item: str = ""
    is_player: bool = True

    def __post_init__(self):
        self.stats = {s: calc_stat(self.base_stats[s], self.ivs[s], self.evs[s],
                                    self.level, s, self.nature) for s in STAT_NAMES}
        self.max_hp = self.stats["hp"]
        self.current_hp = self.max_hp
        self.status = "none"
        self.sleep_turns = 0
        self.toxic_counter = 0
        self.confused = False
        self.confuse_turns = 0
        self.stages = {s: 0 for s in ("atk", "def", "spa", "spd", "spe", "acc", "eva")}
        self.item_used_up = False
        self.protecting = False
        self.enduring = False
        self.flinched = False
        self.charging_move = None   # Move object mid-charge for two-turn moves
        self.charging_target = None
        self.substitute_hp = 0
        self.taunt_turns = 0
        self.first_turn = True
        self.fainted = False
        self.last_move_used = None   # Move object last actually attempted (for AI's SpDefUp check)
        # doubles-position label filled in by Battle
        self.side = None
        self.slot = None

    # -- convenience -------------------------------------------------------
    def hp_fraction(self):
        return self.current_hp / self.max_hp if self.max_hp else 0

    def is_fainted(self):
        return self.current_hp <= 0

    def effective_stat(self, stat, is_crit_attacker_side=None):
        base = self.stats[stat]
        stage = self.stages.get(stat, 0)
        mult = stage_multiplier(stage)
        val = int(math.floor(base * mult))
        return max(1, val)

    def label(self):
        tag = "Your" if self.is_player else "Foe"
        return f"{tag} {self.name}"

    def take_damage(self, amount):
        amount = max(0, amount)
        if self.substitute_hp > 0:
            absorbed = min(self.substitute_hp, amount)
            self.substitute_hp -= absorbed
            remaining = amount - absorbed
            if remaining > 0:
                self.current_hp = max(0, self.current_hp - remaining)
        else:
            self.current_hp = max(0, self.current_hp - amount)
        if self.current_hp <= 0:
            self.fainted = True

    def heal(self, amount):
        self.current_hp = min(self.max_hp, self.current_hp + int(amount))

    def reset_charge(self):
        self.charging_move = None


def make_pokemon_from_spec(spec):
    """spec: dict as produced by the setup wizard (or hand-authored)."""
    moves = []
    for mv in spec["moves"]:
        if isinstance(mv, Move):
            moves.append(MoveSlot(mv))
        else:
            found = get_move(mv)
            if found is None:
                raise ValueError(f"Unknown move: {mv}")
            moves.append(MoveSlot(found))
    p = Pokemon(
        name=spec["name"],
        species_types=tuple(spec["types"]),
        level=spec["level"],
        base_stats=spec["base_stats"],
        moves=moves,
        ivs=spec.get("ivs", {s: 31 for s in STAT_NAMES}),
        evs=spec.get("evs", {s: 0 for s in STAT_NAMES}),
        nature=spec.get("nature", "hardy"),
        ability=spec.get("ability", ""),
        item=spec.get("item", ""),
        is_player=spec.get("is_player", True),
    )
    if "current_hp" in spec and spec["current_hp"] is not None:
        p.current_hp = clamp(int(spec["current_hp"]), 0, p.max_hp)
        if p.current_hp <= 0:
            p.fainted = True
    if spec.get("status"):
        p.status = spec["status"]
    return p


# ============================================================================
# CANONICAL TATE & LIZA DATA (Ruby/Sapphire)
# ============================================================================
# All figures below were pulled directly from the pokeruby decompilation
# (src/data/trainer_parties.h -> gTrainerParty_TateAndLiza) rather than
# guessed:
#   - Party order is Lunatone (slot 0) then Solrock (slot 1), both Lv.42.
#   - Lunatone: Light Screen / Psychic / Hypnosis / Calm Mind.
#   - Solrock:  Sunny Day / Solar Beam / Psychic / Flamethrower.
#   - Base stats confirmed against src/data/pokemon/base_stats.h: both are
#     Rock/Psychic, Levitate; Solrock 70/95/85/55/65/70 (HP/Atk/Def/SpA/SpD/Spe),
#     Lunatone 70/55/65/95/85/70.
#   - The party entry's "iv" byte is 250. Gen 3 converts that with
#     fixedIV = iv * 31 / 255 (src/battle_main.c), giving fixedIV = 30 --
#     i.e. IV 30 in every stat for both Pokemon (not a perfect 31).
#   - Nature comes from the Pokemon's personality value, which for trainer
#     Pokemon is derived from a hash of the trainer's name plus each species'
#     name (src/battle_main.c: nameHash accumulates trainerName's bytes once
#     per party slot, plus that slot's species name, using the game's custom
#     charmap encoding; personality = 0x80 + (nameHash << 8); nature =
#     personality % 25). Hand-computing that hash for "TATE&LIZA" +
#     "LUNATONE"/"SOLROCK" via charmap.txt gives Lunatone = Bold (+Def/-Atk)
#     and Solrock = Serious (neutral).

TATE_LIZA_IV = 30

SOLROCK_BASE = {"hp": 70, "atk": 95, "def": 85, "spa": 55, "spd": 65, "spe": 70}
LUNATONE_BASE = {"hp": 70, "atk": 55, "def": 65, "spa": 95, "spd": 85, "spe": 70}


def make_solrock():
    return make_pokemon_from_spec({
        "name": "Solrock", "types": ("Rock", "Psychic"), "level": 42,
        "base_stats": SOLROCK_BASE,
        "moves": ["Sunny Day", "Solar Beam", "Psychic", "Flamethrower"],
        "ivs": {s: TATE_LIZA_IV for s in STAT_NAMES},
        "nature": "serious",
        "ability": "Levitate", "is_player": False,
    })


def make_lunatone():
    return make_pokemon_from_spec({
        "name": "Lunatone", "types": ("Rock", "Psychic"), "level": 42,
        "base_stats": LUNATONE_BASE,
        "moves": ["Light Screen", "Psychic", "Hypnosis", "Calm Mind"],
        "ivs": {s: TATE_LIZA_IV for s in STAT_NAMES},
        "nature": "bold",
        "ability": "Levitate", "is_player": False,
    })


# ============================================================================
# FIELD / BATTLE STATE
# ============================================================================

@dataclass
class SideConditions:
    light_screen: int = 0   # turns remaining
    reflect: int = 0


class Battle:
    def __init__(self, player_mons, foe_mons, log=print, replacement_callback=None):
        assert 2 <= len(player_mons) <= 4 and len(foe_mons) == 2
        # First two are the active battlers; any beyond that are reserves
        # that can be sent in when an active slot faints (up to a 4-mon
        # party, per the person's request -- Tate & Liza's own team is
        # fixed at 2 with no bench, matching their real in-game roster).
        self.player = player_mons[:2]
        self.player_reserves = list(player_mons[2:])
        self.foe = foe_mons
        for i, p in enumerate(self.player):
            p.side, p.slot = "player", i
        for i, p in enumerate(self.foe):
            p.side, p.slot = "foe", i
        self.weather = None       # "sun" / "rain" / None
        self.weather_turns = 0
        self.sides = {"player": SideConditions(), "foe": SideConditions()}
        self.turn_count = 0
        self.log = log
        self.foe_hyper_potions_left = 4   # Tate & Liza's 4 Hyper Potions
        self.time_estimate = 0.0          # cumulative estimated real seconds elapsed
                                           # (see the REAL-TIME COST MODEL section below)
        self.enemy_faints_witnessed_by_partner = 0  # for the Abra/Wingull level-up cost
        self.rng = random.Random()
        # Called as replacement_callback(battle, slot_index) whenever an
        # active player slot faints and a living reserve is available; must
        # return the chosen reserve Pokemon (or None to decline, e.g. if a
        # test harness doesn't supply one -- in which case the first alive
        # reserve is auto-sent-in so automated runs don't stall).
        self.replacement_callback = replacement_callback
        update_forecast_forms(self)

    # -- helpers -------------------------------------------------------
    def all_active(self):
        return [p for p in (self.player + self.foe)]

    def alive(self, mons):
        return [p for p in mons if p is not None and not p.is_fainted()]

    def opposing(self, mon):
        return self.foe if mon.side == "player" else self.player

    def own_side(self, mon):
        return self.player if mon.side == "player" else self.foe

    def ally(self, mon):
        team = self.own_side(mon)
        for m in team:
            if m is not mon:
                return m
        return None

    def side_conditions(self, mon):
        return self.sides[mon.side]

    def player_party_members(self):
        """Every Pokemon the person owns this battle: active slots plus
        reserves, in a stable order, regardless of fainted status."""
        return [p for p in self.player if p is not None] + self.player_reserves

    def player_alive_reserves(self):
        return [m for m in self.player_reserves if not m.is_fainted()]

    def player_targetable_party(self):
        """Non-fainted party members, for choosing an item's target."""
        return [m for m in self.player_party_members() if not m.is_fainted()]

    def switch_in(self, slot_index, new_mon):
        old = self.player[slot_index]
        self.player[slot_index] = new_mon
        new_mon.side, new_mon.slot = "player", slot_index
        if new_mon in self.player_reserves:
            self.player_reserves.remove(new_mon)
        self.log(f"You send out {new_mon.name}!")
        return old

    def resolve_player_faints(self):
        """Checks every active player slot for a fresh faint and, if a
        living reserve exists, sends one in immediately (per-slot, so this
        can be called right after any single action as well as at the end
        of a turn)."""
        for i, p in enumerate(self.player):
            if not p.is_fainted():
                continue
            reserves = self.player_alive_reserves()
            if not reserves:
                continue
            if self.replacement_callback is not None:
                choice = self.replacement_callback(self, i)
            else:
                choice = reserves[0]
            if choice is not None:
                self.switch_in(i, choice)

    def is_over(self):
        player_defeated = all(p.is_fainted() for p in self.player) \
            and not self.player_alive_reserves()
        foe_defeated = len(self.alive(self.foe)) == 0
        return player_defeated or foe_defeated

    def winner(self):
        player_defeated = all(p.is_fainted() for p in self.player) \
            and not self.player_alive_reserves()
        foe_defeated = len(self.alive(self.foe)) == 0
        if player_defeated and foe_defeated:
            return "draw"
        if player_defeated:
            return "foe"
        if foe_defeated:
            return "player"
        return None


# ============================================================================
# DAMAGE CALCULATION (Gen 3 formula)
# ============================================================================
# Rewritten against src/calculate_base_damage.c (CalculateBaseDamage) and
# src/battle_script_commands.c (atk05_damagecalc / atk06_typecalc /
# atk07_adjustnormaldamage) to get the modifier ORDER exactly right. This
# matters more than it sounds: the real order is
#   base -> burn -> screens -> multi-target halving -> weather -> (+2)
#        -> critical hit -> STAB -> type effectiveness -> random (85-100%)
# Notably, the flat "+2" happens LAST inside the base-damage step (after
# burn/screens/weather already modified the running total, not right after
# the power*Atk/Def division like most fan write-ups of the formula show),
# and the 85-100% random roll is applied dead last, after STAB and type
# effectiveness -- not before them.

HIGH_CRIT_STAGE_CHANCE = {0: 1/16, 1: 1/8, 2: 1/4, 3: 1/3, 4: 1/2}

# Type-boosting held items (src/calculate_base_damage.c's gHoldEffectToType
# table). Only Mystic Water was requested; the table is written so any
# other same-family item (Charcoal/Miracle Seed/Magnet/...) is a one-line
# addition. Applied to the attacker's raw Atk/SpA (whichever the move's
# category uses), *before* stat stages are applied.
TYPE_BOOST_ITEMS = {
    "mystic water": ("Water", 10),
}

# Ability-based move-POWER boosts (not stat boosts) at <=1/3 max HP, ported
# verbatim from CalculateBaseDamage:
#   if (type==WATER && ability==TORRENT && hp<=maxHP/3) power = power*150/100
# (and the Grass/Fire/Bug equivalents, same mechanism).
TYPE_BOOST_ABILITIES = {
    "torrent": "Water",
    "overgrow": "Grass",
    "blaze": "Fire",
    "swarm": "Bug",
}

# Gym badge stat boosts (a real Gen 3 mechanic, verified in
# src/calculate_base_damage.c's BADGE_BOOST macro and src/battle_main.c's
# speed calc): each badge gives a flat +10% to exactly one stat, applied
# only to the PLAYER's Pokemon, only in trainer battles. The mapping is
#   Badge 1 (Stone/Roxanne)   -> Attack
#   Badge 3 (Dynamo/Wattson)  -> Speed
#   Badge 5 (Balance/Norman)  -> Defense
#   Badge 7 (Mind/Tate&Liza)  -> Special Attack AND Special Defense
# Badge 7 is deliberately NOT modeled: it's the badge Tate & Liza *give
# you* for winning this very fight, so it's impossible to already have it
# walking in, and there is no other badge that boosts the special stats in
# Gen 3. PLAYER_BADGES defaults to the 6 badges you'd realistically hold
# by the time you reach Mossdeep (everything through Feather Badge).
PLAYER_BADGES = {1, 2, 3, 4, 5, 6}


def badge_attack_boost():
    return 1 in PLAYER_BADGES


def badge_defense_boost():
    return 5 in PLAYER_BADGES


def badge_speed_boost():
    return 3 in PLAYER_BADGES


def crit_chance(move, attacker):
    stage = 0
    if move.high_crit:
        stage += 1
    # (Focus Energy would add a stage here in an accurate model; Gen 3
    #  famously has a bug where Focus Energy/Dire Hit *lower* crit rate
    #  instead of raising it. Not relevant to Tate & Liza's moveset, so
    #  it isn't wired in -- flagged here for transparency.)
    stage = min(stage, 4)
    return HIGH_CRIT_STAGE_CHANCE[stage]


def effective_move_power(attacker, move):
    power = move.power
    boosted_type = TYPE_BOOST_ABILITIES.get((attacker.ability or "").lower())
    if boosted_type == move.type_ and attacker.hp_fraction() <= 1/3:
        power = power * 150 // 100
    return power


def raw_attacking_stat(attacker, stat, move):
    """Attacker's un-stage-modified Atk/SpA, with type-boosting held items
    (Mystic Water etc.) and the Attack badge boost applied -- these all
    apply before stat stages do. The badge boost only affects Attack, per
    BADGE_BOOST(1, attack, ...); Special Attack never gets a badge boost
    here (see PLAYER_BADGES note above)."""
    val = attacker.stats[stat]
    item_info = TYPE_BOOST_ITEMS.get((attacker.item or "").lower())
    if item_info:
        boosted_type, pct = item_info
        if boosted_type == move.type_:
            val = val * (100 + pct) // 100
    if stat == "atk" and attacker.is_player and badge_attack_boost():
        val = val * 110 // 100
    return val


def raw_defending_stat(defender, stat):
    """Defender's un-stage-modified Def, with the Defense badge boost
    applied (BADGE_BOOST(5, defense, ...) -- Special Defense is never
    badge-boosted, see PLAYER_BADGES note above)."""
    val = defender.stats[stat]
    if stat == "def" and defender.is_player and badge_defense_boost():
        val = val * 110 // 100
    return val


def _screen_reduce(battle, dmg, defender, move, is_crit, defending_side_alive_count):
    """Reflect/Light Screen: skipped entirely on a crit. Halves damage, or
    only reduces it to 2/3 if both Pokemon on the defending side are still
    alive (real Gen 3/4 doubles quirk) -- src/calculate_base_damage.c."""
    if is_crit:
        return dmg
    side = battle.side_conditions(defender)
    relevant = (move.category == Cat.SPECIAL and side.light_screen > 0) or \
               (move.category == Cat.PHYSICAL and side.reflect > 0)
    if not relevant:
        return dmg
    return 2 * (dmg // 3) if defending_side_alive_count == 2 else dmg // 2


def _multi_target_reduce(dmg, move, defending_side_alive_count):
    """Gen 3's real (and surprisingly inconsistent) doubles spread rule:
    moves with target == TARGET_BOTH_ENEMIES (Surf, Rock Slide, Blizzard,
    Powder Snow -- our Target.ALL_FOES) are halved when both opposing
    Pokemon were alive when the move started, and NOT halved if only one
    was a valid target to begin with. Moves with target ==
    TARGET_ALL_EXCEPT_USER / FOES_AND_ALLY (Earthquake, Explosion -- our
    Target.ALL_OTHERS) are NEVER halved, full stop; this asymmetry is real
    (it's why "double Earthquake" was a notorious RSE doubles strategy).

    IMPORTANT: defending_side_alive_count must be a snapshot taken BEFORE
    any of this move's hits were resolved, not recomputed after each hit.
    Real Gen 3 does not mark a fainted Pokemon "absent" (removed from the
    field) until the separate openpartyscreen/replacement step runs,
    which only happens after the WHOLE move -- including every target a
    spread move hits -- has finished (src/battle_script_commands.c's
    ATK49_NEXT_TARGET loops back and re-runs the hit script for the
    second target *before* moveend ever reaches the fainted-Pokemon
    replacement handling). So if Surf's first hit faints Solrock, the
    second hit on Lunatone still sees "2 alive" and is still halved --
    only a Pokemon that was ALREADY fainted before this move began (e.g.
    from an earlier action this same turn) drops the count to 1."""
    if move.target != Target.ALL_FOES:
        return dmg
    if defending_side_alive_count == 2:
        return dmg // 2
    return dmg


def _weather_adjust(battle, dmg, move):
    if battle.weather in ("rain", "sandstorm", "hail") and move.name == "Solar Beam":
        dmg = dmg // 2
    if battle.weather == "rain":
        if move.type_ == "Fire":
            dmg = dmg // 2
        elif move.type_ == "Water":
            dmg = (15 * dmg) // 10
    elif battle.weather == "sun":
        if move.type_ == "Fire":
            dmg = (15 * dmg) // 10
        elif move.type_ == "Water":
            dmg = dmg // 2
    return dmg


def _base_damage(battle, attacker, defender, move, is_crit, defending_side_alive_count=None):
    """Everything through the real CalculateBaseDamage: base formula,
    burn, screens, multi-target halving, weather, and the final +2.
    Does NOT include crit/STAB/type/random -- those are applied outside,
    exactly matching the real call order.

    defending_side_alive_count should be a snapshot from before the
    current move's hits started resolving (see _multi_target_reduce for
    why); if not supplied it's computed fresh against the defender's
    current side, which is fine for isolated estimates (AI scoring, "what
    would this move do") that aren't part of an in-progress multi-hit
    resolution."""
    if defending_side_alive_count is None:
        defending_side_alive_count = len(battle.alive(battle.own_side(defender)))

    if move.category == Cat.PHYSICAL:
        atk_stat, def_stat = "atk", "def"
    else:
        atk_stat, def_stat = "spa", "spd"

    if is_crit:
        atk_stage = max(0, attacker.stages[atk_stat])
        def_stage = min(0, defender.stages[def_stat])
    else:
        atk_stage = attacker.stages[atk_stat]
        def_stage = defender.stages[def_stat]

    raw_atk = raw_attacking_stat(attacker, atk_stat, move)
    atk = max(1, int(math.floor(raw_atk * stage_multiplier(atk_stage))))
    raw_def = raw_defending_stat(defender, def_stat)
    defn = max(1, int(math.floor(raw_def * stage_multiplier(def_stage))))

    power = effective_move_power(attacker, move)
    level_factor = (2 * attacker.level) // 5 + 2
    dmg = (level_factor * power * atk // defn) // 50

    if attacker.status == "brn" and move.category == Cat.PHYSICAL and (attacker.ability or "").lower() != "guts":
        dmg = dmg // 2
    if dmg == 0:
        dmg = 1

    dmg = _screen_reduce(battle, dmg, defender, move, is_crit, defending_side_alive_count)
    dmg = _multi_target_reduce(dmg, move, defending_side_alive_count)
    dmg = _weather_adjust(battle, dmg, move)

    return dmg + 2


def _finish_damage(battle, attacker, defender, move, is_crit, base_plus_2, rand_pct):
    """Applies crit multiplier, STAB, type effectiveness (sequentially per
    defending type, matching ModulateDmgByType), then the random roll --
    in that order, matching atk05/atk06/atk07."""
    dmg = base_plus_2 * (2 if is_crit else 1)

    if move.type_ in attacker.species_types:
        dmg = (dmg * 15) // 10

    for def_type in defender.species_types:
        mult10 = round(TYPE_CHART.get(move.type_, {}).get(def_type, 1.0) * 10)
        dmg = (dmg * mult10) // 10
        if dmg == 0 and mult10 != 0:
            dmg = 1
        if mult10 == 0:
            return 0

    if dmg != 0:
        dmg = (dmg * rand_pct) // 100
        if dmg == 0:
            dmg = 1
    return max(0, dmg)


def calc_damage(battle, attacker, defender, move, is_crit, rng=random, log=None,
                 defending_side_alive_count=None):
    """Returns integer damage for an actual hit (fresh 85-100% random roll).
    Pass defending_side_alive_count when resolving one hit of a multi-target
    move -- see _multi_target_reduce's docstring for why it must be a
    snapshot taken before any of that move's hits were resolved."""
    if move.category == Cat.STATUS or move.power == 0:
        return 0
    if type_effectiveness(move.type_, defender.species_types) == 0:
        return 0
    base_plus_2 = _base_damage(battle, attacker, defender, move, is_crit, defending_side_alive_count)
    rand_pct = rng.randint(85, 100)
    dmg = _finish_damage(battle, attacker, defender, move, is_crit, base_plus_2, rand_pct)
    return max(1, dmg)


def estimate_avg_damage(battle, attacker, defender, move):
    """Deterministic 'expected' damage (92.5% average roll, no crit) used
    for general-purpose estimates elsewhere in the sim (not by the ported
    AI scorer itself, which uses _ai_calc_dmg with its own simulated roll
    per move slot, matching the real AI_CalcDmg)."""
    if move.category == Cat.STATUS or move.power == 0:
        return 0
    if type_effectiveness(move.type_, defender.species_types) == 0:
        return 0
    base_plus_2 = _base_damage(battle, attacker, defender, move, is_crit=False)
    dmg = _finish_damage(battle, attacker, defender, move, False, base_plus_2, rand_pct=92)
    return max(1, dmg)


# ============================================================================
# TRAINER AI -- ported directly from the pokeruby decompilation
# ============================================================================
# This section is a line-by-line port of the *actual* ASM-level AI scripts
# for Tate & Liza's exact aiFlags (0x7 = AI_CheckBadMove | AI_CheckViability
# | AI_TryToFaint -- confirmed in src/data/trainers_en.h / trainer_parties.h;
# NOT AI_SetupFirstTurn), read out of:
#   - data/battle_ai_scripts.s        (AI_CheckBadMove, AI_CheckViability,
#                                       AI_TryToFaint, and the AI_CV_* /
#                                       AI_CBM_* sub-scripts for our 7 moves)
#   - src/battle_ai_script_commands.c (opcode semantics: if_random_less_than
#                                       is `Random() % 256 < N`; if_can_faint
#                                       / is_most_powerful_move both use
#                                       AI_CalcDmg with NO crit and a
#                                       per-move-slot simulated random roll
#                                       of 100-(Random()%16) = 85-100%)
#   - src/battle_ai_switch_items.c    (ShouldUseItem: heal iff HP < 25% of
#                                       max, checked *before* any move
#                                       scoring, with no "skip if I can KO"
#                                       exception in the real code)
#
# The single most important, easy-to-miss real behavior this reproduces:
# in a double battle the AI does NOT evaluate each move against whichever
# target it hits best. `sub_810745C` (BattleAI_SetupAIData) rolls
# `Random() & 2` ONCE per Pokemon's turn to settle on a single opposing
# battler (re-rolling to the other one only if that slot is empty), and
# EVERY move that Pokemon considers this turn is scored against that same
# fixed target. This is a genuine, well-known quirk of Gen 3 doubles AI,
# not a simplification on our part -- it's really that "dumb".
#
# Anything not covered by an explicit AI_CBM_/AI_CV_ branch for one of these
# 7 moves (Psychic, Light Screen, Hypnosis, Calm Mind, Sunny Day, Solar
# Beam, Flamethrower) is intentionally left at its base score of 100,
# exactly like the real scripts do (e.g. Psychic and Flamethrower have no
# AI_CheckViability branch at all in Gen 3 -- their only non-default scoring
# comes from AI_TryToFaint's raw damage estimate).

PHYSICAL_CATEGORY_TYPES = {  # Gen 3's type-based physical/special split
    "Normal", "Fighting", "Poison", "Ground", "Flying", "Rock", "Bug",
    "Ghost", "Steel",
}


def ai_pick_turn_target(battle, mon):
    """Reproduces `Random() & 2` target selection: a single opposing
    Pokemon is chosen once per AI turn and reused for every move."""
    foes = battle.alive(battle.opposing(mon))
    if not foes:
        return None
    if len(foes) == 1:
        return foes[0]
    return battle.rng.choice(foes)


def ai_ability_immunity_penalty(move, target):
    """The generic ability-immunity preamble that runs at the top of the
    real AI_CheckBadMove for every damaging move (Volt Absorb / Water
    Absorb / Flash Fire / Wonder Guard / Levitate)."""
    ability = (target.ability or "").lower()
    if ability == "volt absorb" and move.type_ == "Electric":
        return -12
    if ability == "water absorb" and move.type_ == "Water":
        return -12
    if ability == "flash fire" and move.type_ == "Fire":
        return -12
    if ability == "wonder guard" and type_effectiveness(move.type_, target.species_types) <= 1:
        return -10
    if ability == "levitate" and move.type_ == "Ground":
        return -10
    return 0


def ai_check_bad_move(battle, user, move, target):
    """AI_CheckBadMove, restricted to the branches reachable by our 7 moves."""
    score = 0
    if move.category != Cat.STATUS and move.power > 1:
        eff = type_effectiveness(move.type_, target.species_types)
        if eff == 0:
            return -10  # real code returns/ends immediately here
        score += ai_ability_immunity_penalty(move, target)
        return score

    if move.name == "Hypnosis":
        ability = (target.ability or "").lower()
        if ability in ("insomnia", "vital spirit"):
            return -10
        if target.status != "none":
            return -10
    elif move.name == "Calm Mind":
        if user.stages["spa"] == 6:
            return -10
        elif user.stages["spd"] == 6:
            return -8
    elif move.name == "Light Screen":
        if battle.side_conditions(user).light_screen > 0:
            return -8
    elif move.name == "Sunny Day":
        if battle.weather == "sun":
            return -8
    return score


def ai_check_viability(battle, user, move, target, rng):
    """AI_CheckViability, restricted to the branches our 7 moves actually
    reach: AI_CV_SpDefUp (Calm Mind), AI_CV_LightScreen, AI_CV_SunnyDay,
    AI_CV_ChargeUpMove (Solar Beam). Psychic/Flamethrower/Hypnosis have no
    branch here in the real game and score 0 from this script."""
    r256 = lambda pct: rng.random() * 256 < pct  # Random() % 256 < N

    if move.name == "Calm Mind":
        score = 0
        if user.stages["spd"] < 3:
            if user.hp_fraction() == 1.0 and not r256(128):
                score += 2
            # else: falls through with no change (SpDefUp3)
        else:
            if not r256(100):
                score -= 1

        # AI_CV_SpDefUp3 / 4 / 5
        went_to_4 = False
        if user.hp_fraction() < 0.70:
            went_to_4 = True
        elif not r256(200):
            went_to_4 = True  # the ~22% fall-through
        if went_to_4:
            if user.hp_fraction() < 0.40:
                score -= 2
            else:
                last = target.last_move_used
                if last is not None and last.power > 0:
                    if last.type_ in PHYSICAL_CATEGORY_TYPES:
                        score -= 2
                    # else: SpDefUp5 -- both branches are a no-op in the
                    # real script (a fall-through bug), so no change either way
        return score

    if move.name == "Light Screen":
        if user.hp_fraction() < 0.50:
            return -2
        special_types = {"Fire", "Water", "Grass", "Electric", "Psychic",
                          "Ice", "Dragon", "Dark"}
        if user.species_types[0] in special_types or target.species_types[0] in special_types:
            return 0
        return 0 if r256(50) else -2

    if move.name == "Sunny Day":
        if user.hp_fraction() < 0.40:
            return -1
        if battle.weather in ("rain", "hail", "sandstorm"):
            return 1
        return 0

    if move.name == "Solar Beam":
        eff = type_effectiveness(move.type_, target.species_types)
        if eff <= 0.5:
            return -2
        if any(ms.move.name == "Protect" for ms in target.moves):
            return -2
        if user.hp_fraction() <= 0.38:
            return -1
        return 0

    return 0


# Moves excluded from the AI's "is this my most powerful move" comparison
# entirely (src/battle_ai_script_commands.c's sDiscouragedPowerfulMoveEffects
# -- verified directly, not guessed). Of the ones representable in this
# sim's movedex, Solar Beam is the one that actually matters: none of
# Solrock/Lunatone's other moves are in this list. A discouraged move can
# still get AI_TryToFaint's +4 KO bonus, but it never earns the -1
# "weaker than my other options" penalty, AND IT DOESN'T COUNT AGAINST
# other moves being considered "the most powerful" either. Net effect:
# Solar Beam scores a flat, unremarkable 100 against a target it can't
# one-shot (tying with Solrock's other options rather than beating them on
# raw power), and only clearly wins the score comparison when it can
# secure a KO outright -- e.g. against a target quadruply weak to Grass.
AI_DISCOURAGED_POWER_MOVES = {"Explosion", "Self-Destruct", "Solar Beam", "Hyper Beam", "Overheat"}


def ai_try_to_faint(battle, user, move, target, sim_rolls, move_index):
    """AI_TryToFaint: +4 for a guaranteed KO (the real Explosion exception
    doesn't apply to any of our 7 moves), -1 for a non-KO move that isn't
    the strongest option available, 0 otherwise. Status moves (power<2)
    never get the -1 (they aren't in the power comparison at all), and
    neither do AI_DISCOURAGED_POWER_MOVES (see note above).

    EMPIRICAL ADJUSTMENT (flagged separately from the rest of this file's
    source-verified logic): if a non-charging move in the same moveset can
    ALSO secure the KO this turn, a charging move like Solar Beam does not
    get the +4 bonus here, so the non-charging option wins outright rather
    than tying with it. This specific interaction -- a charge move being
    passed over in favor of an equally-lethal immediate attack -- was
    reported from extensive hands-on play (hundreds of reps of this exact
    fight) and reproduced here on that basis. Despite an extensive pass
    through data/battle_ai_scripts.s and src/battle_ai_script_commands.c
    (including cross-checking against an independently documented,
    structurally-parallel Gen 4 AI reference), no byte-level mechanism for
    this specific tie-break was found, so unlike the rest of this AI port
    it should be read as a reproduction of observed behavior rather than a
    confirmed instruction-for-instruction match."""
    if move.category == Cat.STATUS or move.power <= 1:
        return 0

    # if_can_faint is NOT subject to the discouraged-moves exclusion --
    # that exclusion only affects the separate is_most_powerful_move
    # comparison below -- so this uses the real damage estimate regardless.
    dmg_this = _ai_calc_dmg(battle, user, target, move, sim_rolls[move_index])
    if dmg_this >= target.current_hp:
        if move.name in AI_DISCOURAGED_POWER_MOVES:
            for i, ms in enumerate(user.moves):
                other = ms.move
                if other is move or other.name in AI_DISCOURAGED_POWER_MOVES:
                    continue
                if other.category == Cat.STATUS or other.power <= 1:
                    continue
                if _ai_calc_dmg(battle, user, target, other, sim_rolls[i]) >= target.current_hp:
                    return 0  # a non-charging move can also secure the KO
        return 4  # can_faint branch (Quick Attack bonus N/A -- not in our movesets)

    if move.name in AI_DISCOURAGED_POWER_MOVES:
        return 0  # excluded from the power comparison -- so no penalty either

    def move_dmg(mv, roll):
        if mv is None or mv.category == Cat.STATUS or mv.power <= 1:
            return 0
        if mv.name in AI_DISCOURAGED_POWER_MOVES:
            return 0  # doesn't count toward making another move "not the strongest"
        return _ai_calc_dmg(battle, user, target, mv, roll)

    all_dmgs = [move_dmg(ms.move, sim_rolls[i]) for i, ms in enumerate(user.moves)]
    if dmg_this < max(all_dmgs):
        return -1
    return 0


def _ai_calc_dmg(battle, attacker, defender, move, rand_pct):
    """Equivalent to the real AI_CalcDmg: no crit, a specific (not
    fresh-random) 85-100 damage roll, otherwise the same corrected-order
    formula as a real hit (see _base_damage/_finish_damage above)."""
    if move.category == Cat.STATUS or move.power == 0:
        return 0
    if type_effectiveness(move.type_, defender.species_types) == 0:
        return 0
    base_plus_2 = _base_damage(battle, attacker, defender, move, is_crit=False)
    return max(1, _finish_damage(battle, attacker, defender, move, False, base_plus_2, rand_pct))


def ai_should_heal(battle, mon):
    """ShouldUseItem: real threshold is HP < 25% of max, checked before any
    move scoring happens, with no "unless I could KO this turn" carve-out."""
    if battle.foe_hyper_potions_left <= 0:
        return False
    return mon.current_hp < mon.max_hp // 4


def ai_choose_move(battle, mon):
    usable = [(i, ms) for i, ms in enumerate(mon.moves) if ms.pp > 0]
    if not usable:
        return None, None, None

    target = ai_pick_turn_target(battle, mon)
    sim_rolls = [100 - battle.rng.randint(0, 15) for _ in mon.moves]  # 85-100

    candidates = []
    for i, ms in usable:
        mv = ms.move
        score = 100

        if target is not None:
            # Real Gen 3 clamps the score to a minimum of 0 after EACH
            # individual script's contribution (not just once at the end),
            # per src/battle_ai_script_commands.c's score accumulation --
            # so a step that would push the score deeply negative can't be
            # "refunded" by a later positive step.
            score = max(0, score + ai_check_bad_move(battle, mon, mv, target))
            score = max(0, score + ai_check_viability(battle, mon, mv, target, battle.rng))
            score = max(0, score + ai_try_to_faint(battle, mon, mv, target, sim_rolls, i))
        elif mv.category != Cat.STATUS:
            continue  # no living target to hit with a damaging move

        chosen_target = target if mv.target == Target.SINGLE else (
            mon if mv.target == Target.SELF else None)
        candidates.append((score, mv, chosen_target, ms))

    if not candidates:
        return None, None, None
    top_score = max(c[0] for c in candidates)
    best_choices = [c for c in candidates if c[0] == top_score]
    chosen = battle.rng.choice(best_choices)
    return chosen[1], chosen[2], chosen[3]


def ai_take_turn(battle, mon):
    if mon.charging_move is not None:
        ms = next((m for m in mon.moves if m.move is mon.charging_move), None)
        return {"kind": "move", "mon": mon, "move": mon.charging_move,
                "target": mon.charging_target, "moveslot": ms}
    if ai_should_heal(battle, mon):
        battle.foe_hyper_potions_left -= 1
        return {"kind": "item", "mon": mon, "item": "Hyper Potion"}
    move, target, moveslot = ai_choose_move(battle, mon)
    if move is None:
        return {"kind": "struggle", "mon": mon}
    return {"kind": "move", "mon": mon, "move": move, "target": target,
            "moveslot": moveslot}


# ============================================================================
# TURN EXECUTION ENGINE
# ============================================================================

def clamp(v, a, b):
    return max(a, min(b, v))


def get_speed(mon):
    spe = mon.effective_stat("spe")
    if mon.is_player and badge_speed_boost():
        spe = spe * 110 // 100  # Dynamo Badge: +10% Speed, player only
    if mon.status == "prz":
        spe = spe // 4  # Gen 3: paralysis cuts Speed to 1/4
    return spe


def change_stat(battle, mon, stat, delta):
    old = mon.stages[stat]
    new = clamp(old + delta, -6, 6)
    mon.stages[stat] = new
    name = mon.label()
    if new == old:
        battle.log(f"  {name}'s {stat.upper()} won't go any {'higher' if delta > 0 else 'lower'}!")
    else:
        direction = "rose" if delta > 0 else "fell"
        qualifier = "sharply " if abs(delta) == 2 else ("drastically " if abs(delta) >= 3 else "")
        battle.log(f"  {name}'s {stat.upper()} {qualifier}{direction}!")


def inflict_status(battle, mon, status):
    if mon.status != "none":
        return
    name = mon.label()
    if status == "burn" and "Fire" in mon.species_types:
        battle.log(f"  It doesn't affect {name}...")
        return
    if status == "freeze" and ("Ice" in mon.species_types or battle.weather == "sun"):
        battle.log(f"  It doesn't affect {name}...")
        return
    if status == "paralyze" and "Electric" in mon.species_types:
        battle.log(f"  It doesn't affect {name}...")
        return
    if status in ("poison", "toxic") and ("Poison" in mon.species_types or "Steel" in mon.species_types):
        battle.log(f"  It doesn't affect {name}...")
        return
    if status == "sleep":
        mon.status = "slp"
        mon.sleep_turns = battle.rng.randint(1, 4)
        battle.log(f"  {name} fell asleep!")
    elif status == "paralyze":
        mon.status = "prz"
        battle.log(f"  {name} is paralyzed! It may be unable to move!")
    elif status == "burn":
        mon.status = "brn"
        battle.log(f"  {name} was burned!")
    elif status == "poison":
        mon.status = "psn"
        battle.log(f"  {name} was poisoned!")
    elif status == "toxic":
        mon.status = "tox"
        mon.toxic_counter = 1
        battle.log(f"  {name} was badly poisoned!")
    elif status == "freeze":
        mon.status = "frz"
        battle.log(f"  {name} was frozen solid!")


def inflict_confusion(battle, mon):
    if mon.confused:
        return
    mon.confused = True
    mon.confuse_turns = battle.rng.randint(1, 4)
    battle.log(f"  {mon.label()} became confused!")


def confusion_self_damage(mon):
    level_factor = (2 * mon.level) // 5 + 2
    atk = mon.effective_stat("atk")
    defn = mon.effective_stat("def")
    base = (level_factor * 40 * atk // defn) // 50 + 2
    return max(1, int(base))


def apply_secondary_effect(battle, attacker, target, move):
    if move.effect in ("burn", "paralyze", "freeze", "poison", "toxic", "sleep"):
        inflict_status(battle, target, move.effect)
    elif move.effect == "confuse":
        inflict_confusion(battle, target)
    elif move.effect == "flinch":
        target.flinched = True
    for stat, delta in move.target_boosts:
        before = target.stages[stat]
        change_stat(battle, target, stat, delta)
        if stat == "spd" and target.stages[stat] < before:
            # Psychic's secondary effect specifically (the only stat-drop
            # this fight's movepool can trigger) -- an extra animation
            # beat on top of the hit's own cost, already charged in
            # resolve_hit_on_target. Guarded on the stage actually moving
            # (not just being selected) so a drop at -6 that "won't go any
            # lower" -- no visible stat-change animation -- isn't charged.
            _add_time(battle, TIME_COST_SPD_DROP)
    for stat, delta in move.self_boosts:
        change_stat(battle, attacker, stat, delta)


def apply_status_move_effect(battle, attacker, target, move):
    if move.effect in ("sleep", "paralyze", "burn", "poison", "toxic", "freeze"):
        inflict_status(battle, target, move.effect)
    elif move.effect == "confuse":
        inflict_confusion(battle, target)
    for stat, delta in move.target_boosts:
        change_stat(battle, target, stat, delta)
    if not move.effect and not move.target_boosts:
        battle.log("  But nothing happened...")


def apply_self_move(battle, attacker, move):
    did_something = False
    if move.self_boosts:
        for stat, delta in move.self_boosts:
            change_stat(battle, attacker, stat, delta)
        did_something = True
    if move.heal_fraction:
        amt = int(attacker.max_hp * move.heal_fraction)
        attacker.heal(amt)
        battle.log(f"  {attacker.label()} restored {amt} HP!")
        did_something = True
    if move.effect == "rest":
        attacker.status = "none"
        attacker.current_hp = attacker.max_hp
        attacker.status = "slp"
        attacker.sleep_turns = 2
        battle.log(f"  {attacker.label()} went to sleep and became healthy!")
        did_something = True
    if move.effect == "protect":
        attacker.protecting = True
        battle.log(f"  {attacker.label()} protected itself!")
        did_something = True
    if move.effect == "substitute":
        cost = attacker.max_hp // 4
        if attacker.current_hp > cost and attacker.substitute_hp == 0:
            attacker.current_hp -= cost
            attacker.substitute_hp = cost
            battle.log(f"  {attacker.label()} put up a substitute!")
        else:
            battle.log("  But it failed!")
        did_something = True
    if move.effect == "endure":
        attacker.enduring = True
        battle.log(f"  {attacker.label()} braced itself!")
        did_something = True
    if not did_something:
        battle.log("  But nothing happened...")


def apply_side_effect(battle, attacker, move):
    side = battle.side_conditions(attacker)
    if move.effect == "light_screen":
        if side.light_screen > 0:
            battle.log("  But it failed!")
        else:
            side.light_screen = 5
            battle.log(f"  {attacker.label()}'s team is protected by Light Screen!")
    elif move.effect == "reflect":
        if side.reflect > 0:
            battle.log("  But it failed!")
        else:
            side.reflect = 5
            battle.log(f"  {attacker.label()}'s team is protected by Reflect!")


def apply_field_effect(battle, attacker, move):
    """Weather-setting moves (Sunny Day/Rain Dance/Hail): fails (no turn
    counter refresh) if the weather AT THE MOMENT THIS MOVE ACTUALLY
    RESOLVES already matches -- checked here rather than at selection
    time, since this function only runs when do_move() reaches this
    mon's turn in speed order, by which point any earlier-resolving move
    this same turn (e.g. Solrock's Sunny Day going first) has already
    updated battle.weather. This is deliberate: selecting Rain Dance
    while it's already raining is legal (not masked out) and can still
    succeed, e.g. if Solrock's Sunny Day resolves first that same turn
    and overwrites rain with sun -- Rain Dance then sees "sun", not
    "rain", and correctly overwrites it back. It only fails when the
    weather it's trying to set is still in effect at resolution time,
    matching the real game (fixed 2026-09-21; this used to unconditionally
    reset weather_turns to 5 regardless, letting weather be refreshed
    indefinitely by repeated use)."""
    effect_to_weather = {"sun": "sun", "rain": "rain", "hail": "hail"}
    target_weather = effect_to_weather.get(move.effect)
    if target_weather is None:
        return
    if battle.weather == target_weather:
        battle.log("  But it failed!")
        _add_time(battle, TIME_COST_WEATHER_MOVE_FAILS)
        return
    battle.weather = target_weather
    battle.weather_turns = 5
    battle.log({"sun": "  The sunlight got intense!",
                "rain": "  It started to rain!",
                "hail": "  It started to hail!"}[target_weather])
    if attacker.name == "Castform":
        # Bundled number: the move's own animation plus Castform reacting
        # to its own weather. A non-Castform attacker's passive-transform
        # cost (if Castform is on the field to react) is charged below,
        # after update_forecast_forms actually runs the transform.
        _add_time(battle, TIME_COST_CASTFORM_WEATHER_MOVE)
    else:
        _add_time(battle, TIME_COST_NONCASTFORM_WEATHER_MOVE)
    castform = next((m for m in battle.player + battle.foe
                      if m.name == "Castform" and not m.is_fainted()), None)
    types_before = castform.species_types if castform else None
    update_forecast_forms(battle)
    if attacker.name != "Castform" and castform is not None \
            and castform.species_types != types_before:
        # Castform reacting to a weather change it didn't cause itself
        # (e.g. Solrock's Sunny Day overwriting Castform's own rain) --
        # NOT charged when Castform caused the change (already bundled
        # into TIME_COST_CASTFORM_WEATHER_MOVE above), and not charged for
        # the battle-start transform or a weather-clearing transform,
        # since this hook only runs from inside a weather move's own
        # resolution.
        _add_time(battle, TIME_COST_CASTFORM_PASSIVE_TRANSFORM)


def update_forecast_forms(battle):
    """Forecast (Castform's ability): its type becomes Fire in sun, Water
    in rain, Ice in hail, or Normal otherwise -- ported directly from
    src/battle_util.c's Castform form-change check, which literally
    overwrites the Pokemon's type1/type2 fields the same way."""
    weather_type = {"sun": "Fire", "rain": "Water", "hail": "Ice"}.get(battle.weather, "Normal")
    for mon in battle.player + battle.foe:
        if mon.is_fainted() or (mon.ability or "").lower() != "forecast":
            continue
        new_types = (weather_type,)
        if mon.species_types != new_types:
            mon.species_types = new_types
            battle.log(f"  {mon.label()} transformed with the weather!")


# ============================================================================
# REAL-TIME COST MODEL (2026-09-21)
# ============================================================================
# Empirical, stopwatch-measured animation-length estimates from
# tnl_timings.txt (this project's owner), used so the RL reward can target
# minimizing real elapsed speedrun time on a win instead of a turn-count
# proxy (see tl_env.py's REWARD_CONFIG / compute_reward -- the time bonus
# there only applies on a win; a loss is scored the same regardless of how
# long it took, per an explicit instruction, not an oversight).
#
# Every value below traces to a specific line in tnl_timings.txt EXCEPT
# those commented "ASSUMED" or "DERIVED" -- tnl_timings.txt does not cover
# every move in this fight (notably: none of Solrock/Lunatone's own move
# animations -- Psychic, Solar Beam, Sunny Day, Flamethrower, Calm Mind,
# Light Screen, Hypnosis -- were measured directly, nor Rock Smash, nor a
# *failed* weather move). Where a number is ASSUMED, it's a proxy from the
# closest structurally-similar measured event; where DERIVED, it's
# arithmetic on two measured numbers. Both are flagged inline and called
# out again in CLAUDE.md -- treat them as placeholders to replace with real
# measurements, not as verified data, the same way this file already flags
# Solar Beam AI targeting as empirically-motivated rather than
# source-confirmed.
TIME_COST_FLAT = {
    # Swampert (all measured)
    "Strength": 4.1,
    "Rock Smash": 4.1,   # ASSUMED: not in tnl_timings.txt; same shape as Strength
                          # (single-target physical hit, no secondary flourish)
    # Wingull (measured)
    "Water Gun": 3.4,
    "Growl": 7.6,
    # Abra (measured)
    "Teleport": 2.4,
    # Solrock/Lunatone -- ALL ASSUMED, not in tnl_timings.txt at all
    "Psychic": 4.8,        # ASSUMED: proxy = Surf (1 target) -- single-target special hit
    "Flamethrower": 4.8,   # ASSUMED: proxy = Surf (1 target) -- single-target special hit
    "Solar Beam": 4.8,     # ASSUMED: release-turn hit only (charge turn priced separately
                            # via TIME_COST_SOLAR_BEAM_CHARGE); proxy = Surf (1 target)
    "Calm Mind": 3.0,      # ASSUMED: generic self-buff fallback, no measured analog
    "Light Screen": 3.0,   # ASSUMED: generic side-buff fallback, no measured analog
    "Hypnosis": 3.0,       # ASSUMED: generic single-target status fallback
}
TIME_COST_DIVE_FLY_CHARGE = 2.75     # measured (Dive first turn); applied to Fly too, per
                                      # tnl_timings.txt's "Fly: Same as Dive"
TIME_COST_DIVE_FLY_RELEASE = 4.5     # measured (Dive second turn); Fly likewise
TIME_COST_SOLAR_BEAM_CHARGE = 2.75   # ASSUMED: proxy = Dive/Fly charge turn (structurally
                                      # the same "charging" beat); no release-turn analog
                                      # needed since Solar Beam's damage uses the normal
                                      # single-target attack cost below
TIME_COST_SURF = {2: 8.7, 1: 4.8}            # measured, keyed by live-target count this turn
TIME_COST_POWDER_SNOW = {2: 3.7, 1: 2.1}     # measured, keyed by live-target count this turn
TIME_COST_X_ITEM = 3.7                # measured (flat regardless of which X-item)
TIME_COST_FULL_HEAL = 3.9             # measured
TIME_COST_ENEMY_HYPER_POTION = 6.5    # measured (flat regardless of amount restored --
                                       # matches tnl_timings.txt giving one flat number,
                                       # unlike the player potions' amount-scaled cost)
TIME_COST_CASTFORM_WEATHER_MOVE = 8.3        # measured ("including transform animation")
TIME_COST_NONCASTFORM_WEATHER_MOVE = 8.3 - 3.5   # DERIVED = 4.8: the bare weather-move
                                       # animation with Castform's own transform time
                                       # subtracted back out, for when a NON-Castform
                                       # attacker (only Solrock's Sunny Day, in this fight)
                                       # sets the weather
TIME_COST_WEATHER_MOVE_FAILS = 2.0    # ASSUMED: not in tnl_timings.txt (no measurement for
                                       # a weather move used while that weather is already
                                       # up, i.e. the 2026-09-21 "But it failed!" case);
                                       # short text-only fallback
TIME_COST_CASTFORM_PASSIVE_TRANSFORM = 3.5   # measured ("on its own", i.e. weather changed
                                       # by something other than Castform's own move)
TIME_COST_WEATHER_TURN_TEXT = 2.0     # measured ("Rain text") -- read as the generic
                                       # per-turn weather-continues reminder, not
                                       # rain-specific; applied once per turn any weather
                                       # (sun/rain/hail) is already active going into that
                                       # turn, on top of whatever else happens that turn
TIME_COST_HAIL_HIT = 2.8              # measured, per Pokemon actually damaged by hail
                                       # that turn (non-fainted, non-Ice-type)
TIME_COST_CRIT = 2.0                  # measured, additional to the base hit's own cost
TIME_COST_SPD_DROP = 3.1              # measured -- Psychic's 10% Special Defense secondary
                                       # effect specifically (the only stat-drop this fight's
                                       # movepool can trigger); NOT applied to Calm
                                       # Mind's self-boost or an X-item's boost, which are
                                       # already priced into that move/item's own flat cost
TIME_COST_FAINT_WITH_REPLACEMENT = 6.9   # measured
TIME_COST_FAINT_NO_REPLACEMENT = 1.8     # measured (also covers every foe faint, since
                                       # Solrock/Lunatone never have a reserve to send in)
TIME_COST_LEVEL_UP_FIRST = 8.25       # measured -- first enemy faint witnessed by an
                                       # active Abra or Wingull this battle
TIME_COST_LEVEL_UP_SECOND = 4.1       # measured -- second such faint (there is no third:
                                       # only two foes exist)

# Player healing items (Potion/Super Potion/Hyper Potion): tnl_timings.txt
# gives 4 measured (HP restored, seconds) points -- (20, 5), (63, 5.2),
# (90, 5.5), (110, 6) -- explicitly "roughly linear ... for any amount in
# between". Least-squares line through all 4 (not just the endpoints, so
# every measurement pulls its weight): seconds = 4.6886 + 0.010409 * HP.
# Extrapolated the same way outside the 20-110 HP range Jacob actually
# measured (e.g. a small top-off heal, or a near-full Hyper Potion up to
# Swampert's 133 max) -- flagged here since that's reading past the
# measured interval, not interpolating within it.
_HEAL_TIME_SLOPE = 0.010408533567933148
_HEAL_TIME_INTERCEPT = 4.6885962500687315


def heal_time_cost(hp_restored):
    if hp_restored <= 0:
        return 0.0
    return _HEAL_TIME_INTERCEPT + _HEAL_TIME_SLOPE * hp_restored


def _add_time(battle, seconds):
    battle.time_estimate += seconds


def _charge_faint_time(battle, mon):
    """Call exactly once per Pokemon the instant it's confirmed fainted
    (there are three call sites: a damaging hit, end-of-turn hail, and
    end-of-turn status damage -- burn/poison/toxic never actually trigger
    in this fight's movepool, but are wired up for correctness anyway).
    Charges the faint animation itself, plus -- for a foe faint witnessed
    by an active Abra or Wingull -- the resulting level-up animation."""
    if mon.side == "player":
        has_reserve = bool(battle.player_alive_reserves())
        _add_time(battle, TIME_COST_FAINT_WITH_REPLACEMENT if has_reserve
                   else TIME_COST_FAINT_NO_REPLACEMENT)
    else:
        _add_time(battle, TIME_COST_FAINT_NO_REPLACEMENT)  # foes are never replaced
        for p in battle.player:
            if not p.is_fainted() and p.name in ("Abra", "Wingull"):
                battle.enemy_faints_witnessed_by_partner += 1
                if battle.enemy_faints_witnessed_by_partner == 1:
                    _add_time(battle, TIME_COST_LEVEL_UP_FIRST)
                elif battle.enemy_faints_witnessed_by_partner == 2:
                    _add_time(battle, TIME_COST_LEVEL_UP_SECOND)
                break  # only one partner can ever be active at a time


def move_usage_time_cost(move, num_live_targets=None):
    """Base cost for actually USING a move (the '<mon> used <move>!' beat
    and whatever immediately follows it, e.g. a weather flash or a stat
    line) -- NOT the per-hit extras (crit, SpD drop, faint), which are
    charged separately where they occur. Dive/Fly and weather moves are
    intentionally excluded here -- they have their own charge/release and
    success/fail hooks with different logic, called directly from
    do_move/apply_field_effect instead of through this function."""
    if move.name == "Surf":
        return TIME_COST_SURF[2 if num_live_targets >= 2 else 1]
    if move.name == "Powder Snow":
        return TIME_COST_POWDER_SNOW[2 if num_live_targets >= 2 else 1]
    return TIME_COST_FLAT.get(move.name, 3.0)  # ASSUMED fallback for anything
                                                # not in tnl_timings.txt at all


def resolve_hit_on_target(battle, attacker, target, move, defending_side_alive_count=None):
    if target.protecting and move.target != Target.SELF:
        battle.log(f"  {target.label()} protected itself!")
        return

    if (target.charging_move is not None and target.charging_move.semi_invulnerable
            and move.target != Target.SELF):
        battle.log(f"  {attacker.label()}'s attack missed {target.label()}"
                   f" -- it's out of reach!")
        return

    if move.accuracy:
        combined_stage = clamp(attacker.stages["acc"] - target.stages["eva"], -6, 6)
        acc = move.accuracy * stage_multiplier(combined_stage)
        if battle.rng.random() * 100 > acc:
            battle.log(f"  {attacker.label()}'s attack missed {target.label()}!")
            return

    if move.category == Cat.STATUS:
        apply_status_move_effect(battle, attacker, target, move)
        return

    is_crit = battle.rng.random() < crit_chance(move, attacker)
    dmg = calc_damage(battle, attacker, target, move, is_crit, rng=battle.rng,
                       defending_side_alive_count=defending_side_alive_count)
    target.take_damage(dmg)
    eff = type_effectiveness(move.type_, target.species_types)
    battle.log(f"  -> {target.label()} took {dmg} damage "
               f"({max(0, target.current_hp)}/{target.max_hp} HP left).")
    if is_crit:
        battle.log("  A critical hit!")
        _add_time(battle, TIME_COST_CRIT)
    if eff > 1:
        battle.log("  It's super effective!")
    elif 0 < eff < 1:
        battle.log("  It's not very effective...")

    if move.recoil_fraction and not attacker.is_fainted():
        recoil = max(1, int(dmg * move.recoil_fraction))
        attacker.take_damage(recoil)
        battle.log(f"  {attacker.label()} is hit by recoil! ({recoil} dmg)")
    if move.drain_fraction:
        healed = max(1, int(dmg * move.drain_fraction))
        attacker.heal(healed)
        battle.log(f"  {attacker.label()} had its HP restored by {healed}.")

    if target.is_fainted():
        battle.log(f"  {target.label()} fainted!")
        _charge_faint_time(battle, target)
        return

    if move.effect_chance and battle.rng.random() * 100 < move.effect_chance:
        apply_secondary_effect(battle, attacker, target, move)


def resolve_single_target(battle, attacker, original_target):
    """Gen 3 targets a BATTLE POSITION (a field slot), not a specific
    Pokemon identity -- source-verified in src/battle_main.c's
    HandleAction_UseMove, which re-resolves the stored target at the
    moment the move actually executes:
        gBattlerTarget = <the position chosen when the move was selected>
        if that position is "absent" (gAbsentBattlerFlags):
            redirect to the FLANK (the other active position on that
            same side) instead of just failing
    Concretely, that means two distinct real behaviors, both different
    from "the move just fails if its original target already fainted":
      1. If a REPLACEMENT has already been sent into the original
         target's position (because it fainted earlier this same turn),
         the move simply hits whoever is there now -- targeting was
         always about the position, so this isn't even a redirect from
         the game's point of view.
      2. If that position is genuinely empty (fainted, no replacement
         available), the move redirects to the other active Pokemon on
         that same side rather than failing outright.
    Only if EVERY Pokemon on that side is gone does the move truly have
    no target (which in practice means the battle is already over)."""
    if original_target is None:
        return None
    side_list = battle.player if original_target.side == "player" else battle.foe
    slot = original_target.slot
    current_occupant = side_list[slot]
    if not current_occupant.is_fainted():
        return current_occupant
    other_occupant = side_list[1 - slot]
    if not other_occupant.is_fainted():
        return other_occupant
    return None


def do_move(battle, attacker, move, chosen_target, moveslot):
    if attacker.is_fainted():
        return

    if attacker.flinched:
        battle.log(f"{attacker.label()} flinched and couldn't move!")
        attacker.flinched = False
        return

    if attacker.status == "frz":
        if battle.rng.random() < 0.20:
            attacker.status = "none"
            battle.log(f"{attacker.label()} thawed out!")
        else:
            battle.log(f"{attacker.label()} is frozen solid!")
            return

    if attacker.status == "slp":
        if attacker.sleep_turns <= 0:
            attacker.status = "none"
            battle.log(f"{attacker.label()} woke up!")
        else:
            attacker.sleep_turns -= 1
            battle.log(f"{attacker.label()} is fast asleep.")
            return

    if attacker.status == "prz" and battle.rng.random() < 0.25:
        battle.log(f"{attacker.label()} is fully paralyzed!")
        return

    if attacker.confused:
        attacker.confuse_turns -= 1
        if attacker.confuse_turns < 0:
            attacker.confused = False
            battle.log(f"{attacker.label()} snapped out of confusion!")
        else:
            battle.log(f"{attacker.label()} is confused!")
            if battle.rng.random() < 1 / 3:
                dmg = confusion_self_damage(attacker)
                attacker.take_damage(dmg)
                battle.log(f"  It hurt itself in its confusion! ({dmg} dmg)")
                return

    attacker.last_move_used = move   # tracked for the real AI's "target's last move" check

    is_release_turn = attacker.charging_move is move
    if move.charge and not is_release_turn:
        skip_charge = (move.name == "Solar Beam" and battle.weather == "sun")
        if not skip_charge:
            moveslot.pp -= 1
            battle.log(f"{attacker.label()} used {move.name}!")
            attacker.charging_move = move
            attacker.charging_target = chosen_target
            if move.name == "Dive":
                battle.log(f"  {attacker.label()} hid underwater!")
                _add_time(battle, TIME_COST_DIVE_FLY_CHARGE)
            elif move.name == "Fly":
                battle.log(f"  {attacker.label()} is glowing with energy!")
                _add_time(battle, TIME_COST_DIVE_FLY_CHARGE)
            else:
                battle.log(f"  {attacker.label()} is glowing with energy!")
                _add_time(battle, TIME_COST_SOLAR_BEAM_CHARGE)
            return
    if is_release_turn:
        attacker.charging_move = None
        chosen_target = attacker.charging_target
        attacker.charging_target = None
        if move.name in ("Dive", "Fly"):
            _add_time(battle, TIME_COST_DIVE_FLY_RELEASE)
    else:
        moveslot.pp -= 1
        battle.log(f"{attacker.label()} used {move.name}!")

    if move.target == Target.SIDE:
        apply_side_effect(battle, attacker, move)
        _add_time(battle, move_usage_time_cost(move))
        return
    if move.target == Target.FIELD:
        apply_field_effect(battle, attacker, move)   # its own time cost, handled internally
        return
    if move.target == Target.SELF:
        apply_self_move(battle, attacker, move)
        _add_time(battle, move_usage_time_cost(move))
        return

    if move.target == Target.SINGLE:
        resolved = resolve_single_target(battle, attacker, chosen_target)
        targets = [resolved] if resolved is not None else []
        if not targets:
            battle.log("  But it failed!")
            return
    elif move.target == Target.ALL_FOES:
        targets = battle.alive(battle.opposing(attacker))
    elif move.target == Target.ALL_OTHERS:
        targets = battle.alive(battle.opposing(attacker))
        ally = battle.ally(attacker)
        if ally and not ally.is_fainted():
            targets = targets + [ally]
    else:
        targets = []

    if not targets:
        battle.log("  But there was no target...")
        return

    if move.name not in ("Dive", "Fly"):
        # Dive/Fly's cost is entirely covered by the dedicated charge/
        # release hooks above -- not a per-target attack cost like this.
        _add_time(battle, move_usage_time_cost(move, len(targets)))

    # Snapshot each side's alive count BEFORE any of this move's hits are
    # resolved. Gen 3 does not mark a fainted Pokemon "absent" until the
    # separate openpartyscreen/replacement step, which only runs after the
    # WHOLE move (every target a spread move hits) has finished -- so if
    # Surf's first hit faints Solrock, the second hit on Lunatone still
    # sees "2 alive" for multi-target-halving and screen purposes. See
    # _multi_target_reduce's docstring for the full source citation.
    side_snapshot = {"player": len(battle.alive(battle.player)),
                      "foe": len(battle.alive(battle.foe))}

    for t in targets:
        if t.is_fainted():
            continue
        resolve_hit_on_target(battle, attacker, t, move, side_snapshot[t.side])


def _has_unresolved_replacement(battle):
    """True iff some player slot is fainted, a living reserve exists to
    fill it, but the replacement_callback declined (returned None) --
    i.e. a real decision is still pending and hasn't been answered yet.
    Used by _resolve_turn_gen to know when it must pause rather than
    continue on to later, not-yet-executed actions."""
    return any(p.is_fainted() for p in battle.player) and bool(battle.player_alive_reserves())


def _resolve_turn_gen(battle, actions):
    """Generator form of turn resolution (2026-09-23): yields (pausing
    execution) immediately after ANY point where a player replacement
    decision is still unresolved, BEFORE processing any later, not-yet-
    executed action for that turn -- see resolve_turn's docstring for why
    this matters. A caller whose replacement_callback answers
    synchronously (returns a real Pokemon immediately, as
    evaluate.py's watch_one_battle and collect_corrections.py's callbacks
    do) will never actually observe a pause here, since
    battle.resolve_player_faints() will already have filled the slot via
    switch_in by the time _has_unresolved_replacement is checked --
    behavior for every such caller is therefore identical to the old
    non-generator implementation. TateLizaEnv (tl_env.py) is the only
    caller whose callback can't answer synchronously (the answer has to
    come from an external agent's env.step() call), so it's the only
    caller that needs to actually hold onto this generator across
    multiple calls and resume it once the pending decision is filled in."""
    if battle.weather:
        # The "weather continues" reminder text, shown once per turn any
        # weather was ALREADY active going into that turn -- checked here,
        # before this turn's own actions run, so the turn weather is first
        # set doesn't also get charged this (that's priced into the
        # weather move's own cost instead, in apply_field_effect).
        _add_time(battle, TIME_COST_WEATHER_TURN_TEXT)
    for mon in battle.player + battle.foe:
        mon.protecting = False
    order = []
    for mon, act in actions.items():
        if mon.is_fainted() or act is None:
            continue
        if act["kind"] == "item":
            pr = 6
        elif act["kind"] == "move":
            pr = act["move"].priority
        else:
            pr = 0
        order.append((pr, get_speed(mon), battle.rng.random(), mon, act))
    order.sort(key=lambda x: (-x[0], -x[1], x[2]))

    for _, _, _, mon, act in order:
        if mon.is_fainted():
            continue
        if battle.is_over():
            break
        if act["kind"] == "item":
            do_item(battle, mon, act)
        elif act["kind"] == "struggle":
            battle.log(f"{mon.label()} has no moves left to use!")
        elif act["kind"] == "move":
            do_move(battle, mon, act["move"], act["target"], act["moveslot"])
        battle.resolve_player_faints()
        if _has_unresolved_replacement(battle):
            yield

    # This turn happened -- actions were taken and resolved -- regardless
    # of whether the battle also ended partway through it, so the count
    # increments unconditionally here (2026-09-22 fix: it used to live
    # inside end_of_turn() below, which is skipped once the battle is
    # over, silently undercounting the final turn of every battle that
    # ended during the main action phase rather than surviving to its own
    # end-of-turn phase -- e.g. a battle-ending KO. That's exactly the
    # majority case, since a win/loss is usually decided by a hit landing,
    # not by end-of-turn residual damage. Confirmed via evaluate.py's
    # --watch output showing "--- Turn 5 ---" for the turn the win
    # actually happened on, then "RESULT: WIN after 4 turns").
    battle.turn_count += 1

    if not battle.is_over():
        end_of_turn(battle)
        if not battle.is_over():
            battle.resolve_player_faints()
            if _has_unresolved_replacement(battle):
                yield


def resolve_turn(battle, actions):
    """Synchronous entry point -- drains _resolve_turn_gen to completion
    in one call. Identical behavior to before 2026-09-23's generator
    refactor for every caller with a synchronous replacement_callback
    (which is every caller except tl_env.py's TateLizaEnv -- see
    _resolve_turn_gen's docstring)."""
    for _ in _resolve_turn_gen(battle, actions):
        pass


PLAYER_ITEMS = {
    "potion": {"heal": 20},
    "super potion": {"heal": 50},
    "hyper potion": {"heal": 200},
    "full heal": {"cure": True},
    "x attack": {"stage": ("atk", 1)},
    "x special": {"stage": ("spa", 1)},  # Gen 3's real name -- boosts Special Attack.
    "x speed": {"stage": ("spe", 1)},
}
# This is the exact, deliberately-trimmed item set requested for this
# simulator/training project: Potion, Super Potion, Hyper Potion, Full
# Heal, X Attack, X Special, X Speed -- all usable an unlimited number of
# times (no bag/inventory tracking). Note Gen 3 has no "X Sp. Def" item at
# all (verified against include/constants/item_effects.h: only X_ATTACK,
# X_SPEED, X_DEFEND, X_SPATK, and X_ACCURACY exist as stat-boost items),
# and the correct Gen 3 name for the Special Attack booster is "X
# Special", not "X Sp. Atk" -- an earlier version of this file had both of
# those wrong.


def item_targets_any_party_member(item_name):
    """Healing/status items (Potion, Full Heal, ...) can be used on any of
    your Pokemon, active or benched, regardless of which Pokemon is 'using'
    it. X-items only ever affect the Pokemon using them (a stat stage isn't
    a thing a benched Pokemon has)."""
    info = PLAYER_ITEMS.get(item_name.lower(), {})
    return bool("heal" in info or info.get("cure"))


def do_item(battle, mon, act):
    item_name = act["item"]
    if item_name == "Hyper Potion" and not mon.is_player:
        healed = min(200, mon.max_hp - mon.current_hp)
        mon.heal(200)
        battle.log(f"Tate & Liza used a Hyper Potion on {mon.name}! Restored {healed} HP.")
        _add_time(battle, TIME_COST_ENEMY_HYPER_POTION)
        return
    key = item_name.lower()
    info = PLAYER_ITEMS.get(key)
    if not info:
        battle.log(f"(Unknown item '{item_name}' - no effect.)")
        return
    target = act.get("item_target", mon)
    if target.is_fainted():
        battle.log(f"{target.name} has already fainted -- the item has no effect.")
        return
    if "heal" in info:
        healed = min(info["heal"], target.max_hp - target.current_hp)
        target.heal(info["heal"])
        battle.log(f"You used a {item_name} on {target.name}! Restored {healed} HP.")
        _add_time(battle, heal_time_cost(healed))
    if info.get("cure"):
        if target.status != "none":
            battle.log(f"{target.name}'s {target.status.upper()} status was cured!")
        target.status = "none"
        target.confused = False
        _add_time(battle, TIME_COST_FULL_HEAL)
    if "stage" in info:
        stat, delta = info["stage"]
        change_stat(battle, target, stat, delta)
        _add_time(battle, TIME_COST_X_ITEM)


def end_of_turn(battle):
    if battle.weather:
        battle.weather_turns -= 1
        if battle.weather_turns <= 0:
            which = {"sun": "sunlight faded", "rain": "rain stopped",
                     "hail": "hail stopped"}.get(battle.weather, "weather cleared")
            battle.log(f"The {which}.")
            battle.weather = None
            update_forecast_forms(battle)

    for side_name, side in battle.sides.items():
        if side.light_screen > 0:
            side.light_screen -= 1
            if side.light_screen == 0:
                battle.log(f"{side_name.capitalize()} side's Light Screen wore off!")
        if side.reflect > 0:
            side.reflect -= 1
            if side.reflect == 0:
                battle.log(f"{side_name.capitalize()} side's Reflect wore off!")

    if battle.weather == "hail":
        for mon in battle.player + battle.foe:
            if mon.is_fainted() or "Ice" in mon.species_types:
                continue
            dmg = max(1, mon.max_hp // 16)
            mon.take_damage(dmg)
            battle.log(f"{mon.label()} is pelted by hail! ({dmg} dmg)")
            _add_time(battle, TIME_COST_HAIL_HIT)
            if mon.is_fainted():
                battle.log(f"{mon.label()} fainted!")
                _charge_faint_time(battle, mon)

    for mon in battle.player + battle.foe:
        if mon.is_fainted():
            continue
        if mon.status == "brn":
            dmg = max(1, mon.max_hp // 16)
            mon.take_damage(dmg)
            battle.log(f"{mon.label()} is hurt by its burn! ({dmg} dmg)")
        elif mon.status == "psn":
            dmg = max(1, mon.max_hp // 16)
            mon.take_damage(dmg)
            battle.log(f"{mon.label()} is hurt by poison! ({dmg} dmg)")
        elif mon.status == "tox":
            dmg = max(1, mon.max_hp * mon.toxic_counter // 16)
            mon.take_damage(dmg)
            battle.log(f"{mon.label()} is badly hurt by poison! ({dmg} dmg)")
            mon.toxic_counter += 1
        if mon.is_fainted():
            battle.log(f"{mon.label()} fainted!")
            _charge_faint_time(battle, mon)
            continue
        if mon.item and mon.item.lower() == "leftovers":
            heal = max(1, mon.max_hp // 16)
            mon.heal(heal)
            battle.log(f"{mon.label()} restored a little HP using its Leftovers!")

    for mon in battle.player + battle.foe:
        mon.first_turn = False


# ============================================================================
# CLI: SETUP WIZARD
# ============================================================================

def ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else default


def ask_int(prompt, default):
    v = ask(prompt, str(default))
    try:
        return int(v)
    except ValueError:
        print("  (not a number, using default)")
        return default


def ask_yesno(prompt, default=True):
    d = "Y/n" if default else "y/N"
    v = ask(f"{prompt} ({d})", "")
    if not v:
        return default
    return v.lower().startswith("y")


def choose_from_list(prompt, options, allow_custom=False):
    print(prompt)
    for i, o in enumerate(options, 1):
        print(f"  {i}) {o}")
    while True:
        v = input("> ").strip()
        if v.isdigit() and 1 <= int(v) <= len(options):
            return options[int(v) - 1]
        if allow_custom and v:
            return v
        print("  Please enter a valid number.")


def wizard_custom_move():
    print("  -- defining a custom move --")
    name = ask("  Move name")
    type_ = ask("  Type (e.g. Water)", "Normal")
    cat_str = choose_from_list("  Category:", ["Physical", "Special", "Status"])
    cat = {"Physical": Cat.PHYSICAL, "Special": Cat.SPECIAL, "Status": Cat.STATUS}[cat_str]
    power = ask_int("  Base power (0 for status)", 0)
    accuracy = ask_int("  Accuracy 1-100 (0 = never misses)", 100)
    pp = ask_int("  PP", 10)
    mv = Move(name=name, type_=type_, category=cat, power=power, accuracy=accuracy, pp=pp)
    MOVEDEX[name] = mv
    return mv


def wizard_pick_move():
    name = ask("  Move name (or 'list' to browse, 'custom' to define one)")
    if name and name.lower() == "list":
        names = sorted(MOVEDEX.keys())
        for i in range(0, len(names), 6):
            print("   " + ", ".join(names[i:i + 6]))
        return wizard_pick_move()
    if name and name.lower() == "custom":
        return wizard_custom_move()
    mv = get_move(name) if name else None
    if mv is None:
        print(f"  '{name}' isn't in the movedex.")
        if ask_yesno("  Define it as a custom move?", True):
            return wizard_custom_move()
        return wizard_pick_move()
    return mv


def wizard_make_pokemon(slot_label):
    print(f"\n=== Setting up your {slot_label} Pokemon ===")
    name = ask("Nickname/species", "Pikachu")
    t1 = ask("Type 1 (e.g. Water)", "Normal")
    t2 = ask("Type 2 (blank if single-typed)", "")
    types = (t1,) if not t2 else (t1, t2)
    level = ask_int("Level", 50)
    print("Base stats (use Bulbapedia/Serebii values for your species):")
    base_stats = {}
    for s in STAT_NAMES:
        base_stats[s] = ask_int(f"  Base {s.upper()}", 70)

    use_default_ivs = ask_yesno("Use max IVs (31 in every stat)?", True)
    ivs = {s: 31 for s in STAT_NAMES}
    if not use_default_ivs:
        for s in STAT_NAMES:
            ivs[s] = ask_int(f"  IV {s.upper()} (0-31)", 31)

    use_evs = ask_yesno("Enter custom EVs? (No = all 0)", False)
    evs = {s: 0 for s in STAT_NAMES}
    if use_evs:
        for s in STAT_NAMES:
            evs[s] = ask_int(f"  EV {s.upper()} (0-252)", 0)

    nature = ask("Nature (e.g. Adamant, Modest, Hardy for neutral)", "Hardy").lower()
    if nature not in NATURES:
        print("  (unrecognized nature, treating as neutral)")
        nature = "hardy"

    ability = ask("Ability (Levitate/Torrent/Overgrow/Blaze/Swarm are mechanically "
                  "modeled here; others are informational only)", "")
    item = ask("Held item (Leftovers and Mystic Water are mechanically modeled; "
               "others are informational only)", "")

    moves = []
    print("Choose up to 4 moves.")
    for i in range(4):
        if i >= 2 and not ask_yesno(f"Add a {i+1}th move?", True):
            break
        mv = wizard_pick_move()
        moves.append(mv)

    spec = {
        "name": name, "types": types, "level": level, "base_stats": base_stats,
        "moves": moves, "ivs": ivs, "evs": evs, "nature": nature,
        "ability": ability, "item": item, "is_player": True,
    }
    p = make_pokemon_from_spec(spec)

    if ask_yesno(f"Start at full HP ({p.max_hp}/{p.max_hp})?", True):
        pass
    else:
        hp = ask_int(f"Starting HP (0-{p.max_hp})", p.max_hp)
        p.current_hp = clamp(hp, 0, p.max_hp)
        if p.current_hp <= 0:
            p.fainted = True
            print(f"  {p.name} starts the battle fainted.")
        status = ask("Starting status? (none/brn/psn/tox/prz/slp/frz)", "none").lower()
        if status in ("brn", "psn", "tox", "prz", "slp", "frz"):
            p.status = status
            if status == "slp":
                p.sleep_turns = random.randint(1, 4)
            if status == "tox":
                p.toxic_counter = 1
    return p


def quickstart_pokemon(slot_label, defaults):
    """Non-interactive convenience path used by --demo mode."""
    return make_pokemon_from_spec(defaults)


# ============================================================================
# CLI: BATTLE LOOP
# ============================================================================

def print_state(battle):
    print("\n" + "=" * 60)
    print(f"Turn {battle.turn_count + 1}"
          + (f"   Weather: {battle.weather} ({battle.weather_turns} left)" if battle.weather else ""))
    print("-" * 60)
    for p in battle.player:
        status = f" [{p.status.upper()}]" if p.status != "none" else ""
        print(f"  YOU: {p.name:12s} {p.current_hp:4d}/{p.max_hp:<4d} HP{status}"
              f"{'  (fainted)' if p.is_fainted() else ''}")
    if battle.player_reserves:
        bench = ", ".join(
            f"{m.name} ({m.current_hp}/{m.max_hp}{', fainted' if m.is_fainted() else ''})"
            for m in battle.player_reserves)
        print(f"       Bench: {bench}")
    ls_p = battle.sides['player'].light_screen
    rf_p = battle.sides['player'].reflect
    if ls_p or rf_p:
        print(f"       (Your side: {'Light Screen ' + str(ls_p) if ls_p else ''}"
              f"{'Reflect ' + str(rf_p) if rf_p else ''})")
    print("-" * 60)
    for f in battle.foe:
        status = f" [{f.status.upper()}]" if f.status != "none" else ""
        print(f"  FOE: {f.name:12s} {f.current_hp:4d}/{f.max_hp:<4d} HP{status}"
              f"{'  (fainted)' if f.is_fainted() else ''}")
    ls_f = battle.sides['foe'].light_screen
    rf_f = battle.sides['foe'].reflect
    if ls_f or rf_f:
        print(f"       (Foe side: {'Light Screen ' + str(ls_f) if ls_f else ''}"
              f"{'Reflect ' + str(rf_f) if rf_f else ''})   "
              f"Hyper Potions left: {battle.foe_hyper_potions_left}")
    print("=" * 60)


def get_player_action(battle, mon):
    if mon.charging_move is not None:
        ms = next(m for m in mon.moves if m.move is mon.charging_move)
        print(f"{mon.name} is locked into finishing {mon.charging_move.name}!")
        return {"kind": "move", "mon": mon, "move": mon.charging_move,
                "target": mon.charging_target, "moveslot": ms}

    print(f"\nWhat will {mon.name} do?")
    print("  1) Use a move")
    print("  2) Use an item")
    choice = ask("Choice", "1")
    if choice == "2":
        item_names = list(PLAYER_ITEMS.keys())
        for i, it in enumerate(item_names, 1):
            print(f"   {i}) {it.title()}")
        idx = ask_int("Pick an item", 1) - 1
        idx = clamp(idx, 0, len(item_names) - 1)
        item_name = item_names[idx].title()

        if item_targets_any_party_member(item_name):
            party = battle.player_targetable_party()
            if len(party) == 1:
                item_target = party[0]
            else:
                print("  Use it on which Pokemon?")
                for i, m in enumerate(party, 1):
                    st = f" [{m.status.upper()}]" if m.status != "none" else ""
                    print(f"    {i}) {m.name} ({m.current_hp}/{m.max_hp} HP){st}")
                tidx = ask_int("  Target", 1) - 1
                tidx = clamp(tidx, 0, len(party) - 1)
                item_target = party[tidx]
        else:
            item_target = mon  # X-items only ever affect the Pokemon using them

        return {"kind": "item", "mon": mon, "item": item_name, "item_target": item_target}

    usable = [(i, ms) for i, ms in enumerate(mon.moves) if ms.pp > 0]
    if not usable:
        return {"kind": "struggle", "mon": mon}
    print(f"{mon.name}'s moves:")
    for i, ms in enumerate(mon.moves):
        marker = " (no PP)" if ms.pp <= 0 else ""
        print(f"   {i+1}) {ms.move.name} (PP {ms.pp}/{ms.move.max_pp}, "
              f"{ms.move.type_}, {ms.move.category.name}){marker}")
    while True:
        sel = ask_int("Pick a move", usable[0][0] + 1) - 1
        match = next((ms for i, ms in usable if i == sel), None)
        if match:
            break
        print("  That move has no PP or isn't valid, pick another.")
    move = match.move

    target = None
    if move.category != Cat.STATUS and move.target == Target.SINGLE or \
       (move.category == Cat.STATUS and move.target == Target.SINGLE):
        foes = battle.alive(battle.opposing(mon))
        if len(foes) == 0:
            target = None
        elif len(foes) == 1:
            target = foes[0]
        else:
            print("  Choose a target:")
            for i, f in enumerate(foes, 1):
                print(f"    {i}) {f.name} ({f.current_hp}/{f.max_hp} HP)")
            idx = ask_int("  Target", 1) - 1
            idx = clamp(idx, 0, len(foes) - 1)
            target = foes[idx]

    return {"kind": "move", "mon": mon, "move": move, "target": target, "moveslot": match}


def cli_choose_replacement(battle, slot_index):
    fainted_mon = battle.player[slot_index]
    reserves = battle.player_alive_reserves()
    if not reserves:
        return None
    print(f"\n{fainted_mon.name} fainted! Choose a replacement:")
    for i, m in enumerate(reserves, 1):
        st = f" [{m.status.upper()}]" if m.status != "none" else ""
        print(f"  {i}) {m.name} ({m.current_hp}/{m.max_hp} HP){st}")
    idx = ask_int("Send out", 1) - 1
    idx = clamp(idx, 0, len(reserves) - 1)
    return reserves[idx]


def run_battle(player_mons, foe_mons):
    battle = Battle(player_mons, foe_mons, replacement_callback=cli_choose_replacement)
    bench_note = f" (plus {len(battle.player_reserves)} in reserve)" if battle.player_reserves else ""
    print("\nA double battle is about to start against TATE & LIZA!")
    print(f"{battle.player[0].name} and {battle.player[1].name}{bench_note} vs. "
          f"{foe_mons[0].name} and {foe_mons[1].name}!\n")

    while not battle.is_over():
        print_state(battle)
        actions = {}
        for mon in battle.player:
            if mon.is_fainted():
                continue
            actions[mon] = get_player_action(battle, mon)
        for mon in battle.foe:
            if mon.is_fainted():
                continue
            actions[mon] = ai_take_turn(battle, mon)
        resolve_turn(battle, actions)

    print_state(battle)
    w = battle.winner()
    if w == "player":
        print("\nYou defeated Tate & Liza! You received the MIND BADGE and TM04 (Calm Mind)!")
    elif w == "foe":
        print("\nYour Pokemon have all fainted... Tate & Liza win this time.")
    else:
        print("\nBoth sides were wiped out simultaneously -- a draw!")


# ============================================================================
# MAIN
# ============================================================================

DEMO_PLAYER_SPECS = [
    {
        "name": "Swampert", "types": ("Water", "Ground"), "level": 45,
        "base_stats": {"hp": 100, "atk": 110, "def": 90, "spa": 85, "spd": 90, "spe": 60},
        "moves": ["Surf", "Earthquake", "Ice Beam", "Protect"],
        "ivs": {s: 31 for s in STAT_NAMES}, "evs": {s: 0 for s in STAT_NAMES},
        "nature": "adamant", "ability": "Torrent", "item": "Leftovers", "is_player": True,
    },
    {
        "name": "Gengar", "types": ("Ghost", "Poison"), "level": 45,
        "base_stats": {"hp": 60, "atk": 65, "def": 60, "spa": 130, "spd": 75, "spe": 110},
        "moves": ["Shadow Ball", "Thunderbolt", "Ice Punch", "Giga Drain"],
        "ivs": {s: 31 for s in STAT_NAMES}, "evs": {s: 0 for s in STAT_NAMES},
        "nature": "modest", "ability": "Levitate", "item": "", "is_player": True,
    },
]

# A specific 4-Pokemon party, requested with exact final stats (post
# IV/EV/nature) rather than raw IVs/EVs. The IV/EV combinations below were
# solved by brute-force search against this simulator's own stat formula so
# each Pokemon's HP/Atk/Def/SpA/SpD/Spe reproduces the requested numbers
# exactly (preferring EV 0 wherever a solution existed at EV 0 -- Swampert's
# Atk/SpA/Spe needed a small EV top-up since no IV 0-31 at EV 0 lands on
# those exact integers, a real artifact of the stat formula's rounding).
# Abilities and held items the person didn't specify default to each
# species' real (only) Gen 3 ability; items are blank where "None" was
# requested.
PRESET_TEAM_SPECS = [
    {
        "name": "Swampert", "types": ("Water", "Ground"), "level": 38,
        "base_stats": {"hp": 100, "atk": 110, "def": 90, "spa": 85, "spd": 90, "spe": 60},
        "moves": ["Strength", "Dive", "Rock Smash", "Surf"],
        "ivs": {"hp": 26, "atk": 31, "def": 14, "spa": 31, "spd": 31, "spe": 31},
        "evs": {"hp": 0, "atk": 8, "def": 0, "spa": 8, "spd": 0, "spe": 20},
        "nature": "naughty", "ability": "Torrent", "item": "Mystic Water", "is_player": True,
    },
    {
        "name": "Wingull", "types": ("Water", "Flying"), "level": 5,
        "base_stats": {"hp": 40, "atk": 30, "def": 30, "spa": 55, "spd": 30, "spe": 85},
        "moves": ["Growl", "Water Gun", "Fly"],
        "ivs": {"hp": 19, "atk": 31, "def": 19, "spa": 29, "spd": 31, "spe": 29},
        "evs": {s: 0 for s in STAT_NAMES},
        "nature": "modest", "ability": "Keen Eye", "item": "", "is_player": True,
    },
    {
        "name": "Abra", "types": ("Psychic",), "level": 8,
        "base_stats": {"hp": 25, "atk": 20, "def": 15, "spa": 105, "spd": 55, "spe": 90},
        "moves": ["Teleport"],
        "ivs": {"hp": 12, "atk": 9, "def": 31, "spa": 14, "spd": 14, "spe": 19},
        "evs": {s: 0 for s in STAT_NAMES},
        "nature": "careful", "ability": "Synchronize", "item": "", "is_player": True,
    },
    {
        "name": "Castform", "types": ("Normal",), "level": 25,
        "base_stats": {"hp": 70, "atk": 70, "def": 70, "spa": 70, "spd": 70, "spe": 70},
        "moves": ["Powder Snow", "Rain Dance", "Sunny Day", "Hail"],
        "ivs": {"hp": 27, "atk": 27, "def": 11, "spa": 23, "spd": 31, "spe": 3},
        "evs": {s: 0 for s in STAT_NAMES},
        "nature": "calm", "ability": "Forecast", "item": "", "is_player": True,
    },
]


def make_preset_team(swampert_hp=None):
    """Builds the Swampert/Wingull/Abra/Castform preset party. Pass
    swampert_hp (an int, 1 <= swampert_hp <= 133) to start Swampert below
    full HP; None (the default) starts it at full HP as usual. Handy for
    setting up specific test scenarios (including for external tooling
    that drives this simulator programmatically, e.g. AI-training code)."""
    specs = [dict(spec) for spec in PRESET_TEAM_SPECS]
    if swampert_hp is not None:
        specs[0]["current_hp"] = swampert_hp
    return [make_pokemon_from_spec(s) for s in specs]


def main():
    print("=" * 60)
    print(" TATE & LIZA DOUBLE BATTLE SIMULATOR (Pokemon Ruby/Sapphire)")
    print("=" * 60)
    demo = "--demo" in sys.argv
    preset = "--preset" in sys.argv
    swampert_hp_arg = None
    for arg in sys.argv:
        if arg.startswith("--swampert-hp="):
            try:
                swampert_hp_arg = int(arg.split("=", 1)[1])
            except ValueError:
                pass
    if demo:
        print("\n[--demo mode: using two preset Pokemon, no prompts]\n")
        party = [make_pokemon_from_spec(DEMO_PLAYER_SPECS[0]),
                 make_pokemon_from_spec(DEMO_PLAYER_SPECS[1])]
    elif preset:
        print("\n[--preset mode: Swampert/Wingull/Abra/Castform, no prompts]\n")
        party = make_preset_team(swampert_hp_arg)
    else:
        print("\nHow would you like to set up your team?")
        choice = choose_from_list("", [
            "Build my own team (full setup wizard)",
            "Use the preset team (Swampert / Wingull / Abra / Castform)",
            "Quick demo (Swampert / Gengar, no prompts)",
        ])
        if choice.startswith("Use the preset"):
            swampert_max_hp = make_pokemon_from_spec(PRESET_TEAM_SPECS[0]).max_hp
            if ask_yesno(f"Start Swampert at full HP ({swampert_max_hp}/{swampert_max_hp})?", True):
                party = make_preset_team()
            else:
                hp = ask_int(f"Swampert's starting HP (1-{swampert_max_hp})", swampert_max_hp)
                hp = clamp(hp, 1, swampert_max_hp)
                party = make_preset_team(swampert_hp=hp)
        elif choice.startswith("Quick demo"):
            party = [make_pokemon_from_spec(DEMO_PLAYER_SPECS[0]),
                     make_pokemon_from_spec(DEMO_PLAYER_SPECS[1])]
        else:
            print("\nThis is a double battle, so at least 2 Pokemon fight at once, "
                  "but you can bring up to 4 -- the extra 2 sit in reserve and can be "
                  "sent in immediately if one of your active Pokemon faints.")
            party_size = ask_int("How many Pokemon in your party? (2-4)", 2)
            party_size = clamp(party_size, 2, 4)
            ordinals = ["first", "second", "third", "fourth"]
            party = [wizard_make_pokemon(ordinals[i]) for i in range(party_size)]

    solrock = make_solrock()
    lunatone = make_lunatone()
    run_battle(party, [solrock, lunatone])


if __name__ == "__main__":
    main()
