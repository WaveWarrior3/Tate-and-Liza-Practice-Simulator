"""
tl_env.py -- a Gym-like RL environment around tl_sim.py's Tate & Liza
double-battle simulator.

DESIGN OVERVIEW
-----------------
The underlying simulator resolves a whole TURN at once (every active
Pokemon's action is decided, then resolve_turn() executes them all in
speed order). RL algorithms want a stream of (observation, action, reward,
done) steps, so this environment exposes one step() call per INDIVIDUAL
DECISION rather than per turn:
  - a move (+ target, if needed) for one of your active Pokemon, or
  - which reserve to send in, when a Pokemon just fainted and you have a
    living reserve.
Internally it buffers decisions until a full turn's worth of player
actions are ready, then calls tl_sim.resolve_turn() (which also runs the
opponent's existing, source-verified AI) before handing back the next
observation. The same learned policy is used for whichever Pokemon is
acting -- one shared network plays your whole team, which is both the
natural reading of "train a network to play this fight" and much more
sample-efficient than training 4 separate specialists.

Reward is 0 on every non-terminal step; the full shaped reward (see
compute_reward below) is given on the step that ends the episode. This
matches the reward function as specified rather than inventing additional
per-turn shaping -- see the REWARD_SHAPING_NOTE at the bottom for how to
add dense shaping later if training turns out to need it.

OBSERVATION
-----------
A single flat float32 vector, fixed size regardless of which Pokemon is
acting or which are currently active/benched/fainted:
  - one block per canonical party member (Swampert, Wingull, Abra,
    Castform, in that fixed order) with: active flag, fainted flag, HP
    fraction, a 7-way status one-hot, 5 normalized stat stages, confused
    flag, charging flag, and up to 4 normalized PP fractions (21 floats
    each -- see mon_block)
  - one block per foe (Solrock, then Lunatone) using the same mon_block
    encoding (21 floats each)
  - a field block: weather one-hot (4), weather turns remaining, each
    side's Light Screen / Reflect turns remaining, the foe's remaining
    Hyper Potions, and the turn count (11 floats -- see field_block)
  - a 4-way one-hot for which canonical party member this decision is
    for, and a decision-type flag (0 = choose a move, 1 = choose a
    replacement) (5 floats)
Total size is computed once at import time as OBS_DIM (currently 142) --
always read env.obs_dim rather than hardcoding it, in case the encoding
changes.

ACTION SPACE
------------
A single discrete space of size ACTION_DIM = 12 for every decision:
  - indices 0-7: "use move slot i (0-3), target enemy slot t (0-1)" --
    index = i*2 + t. For moves that don't need a target choice (self,
    field, or hit-both-foes moves), t is ignored, so both of that slot's
    two indices do the same thing.
  - indices 8-11: "send out canonical party member i (0-3)" -- only used
    for replacement decisions.
A boolean mask of length 12 accompanies every observation; illegal
actions (0 PP, a move slot the Pokemon doesn't have, a reserve that's
fainted or already active, or the wrong half of the space for the
current decision type) are masked out. Always sample from the masked
action set -- passing a masked-out action raises an error.
"""

import os
import random
import sys
import numpy as np

# VENDORED COPY (webapp/vendor/tl_env.py) -- see webapp/README.md's
# "Standalone / vendoring" section. In the canonical tl_ai/tl_env.py this
# line reaches one directory up to the project root to find tl_sim.py;
# here tl_sim.py is a sibling in this same vendor/ directory instead, so
# the insert targets this file's own directory -- harmless either way
# since battle_engine.py already puts vendor/ on sys.path before
# importing this module, but kept correct/self-consistent for a reader.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tl_sim as sim

STATUS_LIST = ["none", "brn", "psn", "tox", "prz", "slp", "frz"]
STAT_KEYS = ("atk", "def", "spa", "spd", "spe")
WEATHER_LIST = [None, "sun", "rain", "hail"]
CANONICAL_NAMES = ["Swampert", "Wingull", "Abra", "Castform"]

# Action space layout. Indices 0-7 (moves) are unchanged from before item
# support was added; items and replacement slide up to make room.
#   0-7   : move slot i (0-3) * 2 + target t (0-1)  (t ignored for moves
#           that don't need a target choice)
#   8-10  : X-item i -- always self-target (the Pokemon using it), never
#           choosable for a different party member, matching how X-items
#           actually work
#   11-26 : healing/status item i (0-3) * 4 + target party index (0-3) --
#           these CAN go on any party member, active or benched
#   27-30 : send out canonical party member i (0-3), for replacement
#           decisions only
# Items are unlimited-use by design for this project (no bag/inventory
# tracking) -- the only masking for them is "is the target fainted".
X_ITEMS = ["x attack", "x special", "x speed"]
HEAL_ITEMS = ["potion", "super potion", "hyper potion", "full heal"]

MOVE_ACTIONS = 8
X_ITEM_START = MOVE_ACTIONS
X_SPEED_ACTION_IDX = X_ITEM_START + X_ITEMS.index("x speed")  # fixed: X-items are always self-target
X_SPECIAL_ACTION_IDX = X_ITEM_START + X_ITEMS.index("x special")
HEAL_ITEM_START = X_ITEM_START + len(X_ITEMS)
REPLACEMENT_START = HEAL_ITEM_START + len(HEAL_ITEMS) * 4
ACTION_DIM = REPLACEMENT_START + 4

# 100-point empirical-CDF samplers for reset_to_solar_beam_castform_
# scenario() (2026-09-25), built from a 541-sample characterization of
# real games hitting the exact "Solrock charging Solar Beam, Dive can't
# kill it" decision -- see that method's docstring and MODELS.md for the
# full investigation. np.percentile(values, q) for q in linspace(0.5,
# 99.5, 100), so a uniform self.rng.choice() over either list reproduces
# the real empirical distribution without needing a hand-fit parametric
# approximation.
SOLROCK_HP_FRAC_PERCENTILES = [
    0.5561, 0.561, 0.561, 0.561, 0.5691, 0.5691, 0.5691, 0.5691, 0.5772, 0.5772,
    0.5772, 0.5772, 0.5772, 0.5772, 0.5772, 0.5772, 0.5854, 0.5854, 0.5854, 0.5854,
    0.5854, 0.5854, 0.5854, 0.5935, 0.5935, 0.5935, 0.5935, 0.5935, 0.5935, 0.5935,
    0.5935, 0.6016, 0.6016, 0.6016, 0.6016, 0.6016, 0.6016, 0.6016, 0.6016, 0.6098,
    0.6098, 0.6098, 0.6098, 0.6098, 0.6098, 0.6098, 0.6098, 0.6179, 0.6179, 0.6179,
    0.6179, 0.626, 0.626, 0.626, 0.626, 0.626, 0.6268, 0.6341, 0.6341, 0.6423,
    0.648, 0.6504, 0.6504, 0.6504, 0.6504, 0.6504, 0.6585, 0.6585, 0.6585, 0.6585,
    0.6585, 0.6585, 0.6585, 0.6585, 0.6667, 0.6667, 0.6667, 0.6667, 0.6667, 0.6691,
    0.6748, 0.6748, 0.6748, 0.6748, 0.6772, 0.6829, 0.6829, 0.6829, 0.6829, 0.6829,
    0.6911, 0.6911, 0.6911, 0.6911, 0.6992, 0.6992, 0.6992, 0.6992, 0.6992, 0.6992,
]
LUNATONE_HP_FRAC_PERCENTILES = [
    0.122, 0.187, 0.2114, 0.2276, 0.2602, 0.7073, 0.7073, 0.7073, 0.7154, 0.7154,
    0.7154, 0.7162, 0.7236, 0.7236, 0.7236, 0.7236, 0.7236, 0.7236, 0.7317, 0.7317,
    0.7317, 0.7317, 0.7317, 0.7317, 0.7317, 0.7374, 0.7398, 0.7398, 0.7398, 0.7398,
    0.7398, 0.7398, 0.748, 0.748, 0.748, 0.7561, 0.7561, 0.7561, 0.7561, 0.7561,
    0.7561, 0.7561, 0.8618, 0.8927, 0.9024, 0.9024, 0.9187, 0.9187, 0.9187, 0.9187,
    0.9471, 0.9593, 0.9593, 0.9593, 0.9593, 0.9593, 0.9593, 0.9593, 0.9593, 0.9593,
    0.9593, 0.9593, 0.9593, 0.9593, 0.9593, 0.9593, 0.9593, 0.9593, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
]


def status_onehot(status):
    vec = [0.0] * len(STATUS_LIST)
    vec[STATUS_LIST.index(status) if status in STATUS_LIST else 0] = 1.0
    return vec


def visible_hp_fraction(mon):
    """What a human actually sees of an OPPONENT Pokemon's HP: a 48-pixel-
    wide bar, not the exact current_hp/max_hp fraction -- the player's own
    Pokemon's HP is shown as exact numeric text, but the foe's is not
    (source-confirmed: pokeruby's src/battle_interface.c,
    `GetScaledHPFraction(hp, maxhp, scale=48)` == `hp * 48 / maxhp`
    (integer division/floor), with a floor of 1 pixel whenever the
    Pokemon is alive -- `if (result == 0 && hp > 0) return 1` -- so a
    sliver of health is never visually indistinguishable from fainted;
    `GetHPBarLevel` builds the bar's green/yellow/red coloring on top of
    this same pixel count). Quantizing to these 49 discrete levels
    (0/48..48/48) is what actually reaches the player, not the raw
    fraction."""
    if mon.max_hp <= 0:
        return 0.0
    pixels = mon.current_hp * 48 // mon.max_hp
    if pixels == 0 and mon.current_hp > 0:
        pixels = 1
    return pixels / 48.0


def mon_block(mon, visible=False):
    """21 floats describing one Pokemon, active or not, alive or not.
    `visible=True` (used for the FOE's blocks -- see encode_observation)
    quantizes the HP-fraction feature to what a human actually sees on the
    opponent's HP bar (see visible_hp_fraction) instead of the true value;
    the player's own party always gets the exact fraction, matching the
    real game's exact-numeric-HP display for your own side."""
    vec = [
        1.0,  # placeholder for "is_active", overwritten by caller
        1.0 if mon.is_fainted() else 0.0,
        visible_hp_fraction(mon) if visible else mon.hp_fraction(),
    ]
    vec.extend(status_onehot(mon.status))
    for s in STAT_KEYS:
        vec.append(mon.stages[s] / 6.0)
    vec.append(1.0 if mon.confused else 0.0)
    vec.append(1.0 if mon.charging_move is not None else 0.0)
    pp = [ms.pp / ms.move.max_pp for ms in mon.moves]
    while len(pp) < 4:
        pp.append(0.0)
    vec.extend(pp[:4])
    return vec


def field_block(battle):
    vec = [1.0 if battle.weather == w else 0.0 for w in WEATHER_LIST]
    vec.append(battle.weather_turns / 5.0)
    vec.append(battle.sides["player"].light_screen / 5.0)
    vec.append(battle.sides["player"].reflect / 5.0)
    vec.append(battle.sides["foe"].light_screen / 5.0)
    vec.append(battle.sides["foe"].reflect / 5.0)
    vec.append(battle.foe_hyper_potions_left / 4.0)
    vec.append(min(battle.turn_count, 30) / 30.0)
    return vec


def encode_observation(battle, party, acting_idx, decision_type, foe_hp_visible=False):
    """party: the 4 canonical Pokemon objects (Swampert, Wingull, Abra,
    Castform), in that fixed order, regardless of who's currently active.
    acting_idx: canonical index (0-3) this decision concerns.
    decision_type: 0 = choose a move (or item), 1 = choose a replacement.
    Items are unlimited-use, so -- unlike moves' PP -- there's no "how
    many do I have left" feature to encode; each party member's existing
    fainted/HP/status fields already carry everything needed to decide
    who to heal, so no observation changes were needed to support them.
    foe_hp_visible (2026-09-26, default False): if True, the foe blocks'
    HP-fraction feature is quantized to visible_hp_fraction() -- what a
    human actually sees on the opponent's 48-pixel HP bar -- instead of
    the exact true fraction. Defaults to False so every EXISTING
    checkpoint (all trained on exact foe HP) keeps evaluating under the
    observation it was actually trained on; pass True only for a model
    deliberately trained/evaluated under the human-visible-HP variant
    (see best_model_healband.pt's visible-HP counterpart in MODELS.md)."""
    vec = []
    for mon in party:
        block = mon_block(mon)
        block[0] = 1.0 if mon in battle.player else 0.0
        vec.extend(block)
    for foe_mon in battle.foe:
        vec.extend(mon_block(foe_mon, visible=foe_hp_visible))
    vec.extend(field_block(battle))
    vec.extend(1.0 if i == acting_idx else 0.0 for i in range(4))
    vec.append(float(decision_type))
    return np.array(vec, dtype=np.float32)


OBS_DIM = len(encode_observation(
    sim.Battle(sim.make_preset_team(), [sim.make_solrock(), sim.make_lunatone()],
               log=lambda *a: None),
    sim.make_preset_team(), 0, 0))


def move_action_mask(mon, party):
    """Full mask for a 'move' decision: moves (by PP), X-items (always
    legal -- unlimited use, self-target only), and healing items. Matches
    the real game's item-select screen, which refuses a target that the
    item genuinely can't affect: HP-restoring items (Potion/Super
    Potion/Hyper Potion) require the target not already be at full HP,
    and Full Heal requires the target actually have a status condition --
    neither was checked before 2026-09-21 (only "not fainted"), so a
    Hyper Potion could be "used" on an already-full-HP ally for 0 HP
    restored, a real wasted-action pattern seen directly in --watch
    output."""
    mask = np.zeros(ACTION_DIM, dtype=bool)
    for slot_idx, ms in enumerate(mon.moves[:4]):
        if ms.pp > 0:
            mask[slot_idx * 2] = True
            mask[slot_idx * 2 + 1] = True
    for i in range(len(X_ITEMS)):
        mask[X_ITEM_START + i] = True
    for i, item_name in enumerate(HEAL_ITEMS):
        for t, target_mon in enumerate(party):
            if target_mon.is_fainted():
                continue
            if item_name == "full heal":
                if target_mon.status == "none":
                    continue
            else:
                if target_mon.hp_fraction() >= 1.0:
                    continue
            mask[HEAL_ITEM_START + i * 4 + t] = True
    return mask


def decode_move_action(battle, mon, party, action_idx):
    """Decodes any action in the 'move decision' range (0 to
    REPLACEMENT_START-1) into the kind of action dict tl_sim.resolve_turn
    expects: a move+target, or an item+item_target."""
    if action_idx < MOVE_ACTIONS:
        slot_idx, target_choice = divmod(action_idx, 2)
        ms = mon.moves[slot_idx]
        move = ms.move
        target = None
        if move.target == sim.Target.SINGLE:
            foes = battle.alive(battle.opposing(mon))
            if len(foes) == 1:
                target = foes[0]
            elif len(foes) >= 2:
                target = foes[target_choice]
        return {"kind": "move", "mon": mon, "move": move, "target": target, "moveslot": ms}

    if action_idx < HEAL_ITEM_START:
        item_name = X_ITEMS[action_idx - X_ITEM_START]
        return {"kind": "item", "mon": mon, "item": item_name.title(), "item_target": mon}

    offset = action_idx - HEAL_ITEM_START
    item_idx, target_idx = divmod(offset, 4)
    item_name = HEAL_ITEMS[item_idx]
    return {"kind": "item", "mon": mon, "item": item_name.title(), "item_target": party[target_idx]}


def replacement_action_mask(reserves, party):
    mask = np.zeros(ACTION_DIM, dtype=bool)
    for r in reserves:
        idx = party.index(r)
        mask[REPLACEMENT_START + idx] = True
    return mask


def decode_replacement_action(action_idx, party):
    return party[action_idx - REPLACEMENT_START]


# ============================================================================
# REWARD FUNCTION
# ============================================================================
# Priority order, exactly as specified:
#   1. Winning matters far more than anything else.
#   2. Losing but fainting one of Solrock/Lunatone beats losing without
#      fainting either.
#   3. Among losses, less remaining enemy HP (combined) is better.
#   4. Among wins, less cumulative estimated REAL TIME taken is better
#      (2026-09-21: replaces the old turn-count/Surf-count proxies now
#      that tl_sim.py's per-event time-cost model gives a direct signal
#      -- see battle.time_estimate. This is a speedrun tool: the goal was
#      always real elapsed time, and turns/Surf-count were only ever
#      crude stand-ins for it).
# Losses are NOT scored by time at all, by explicit instruction: a loss is
# a loss regardless of how long it took, so only faint_bonus/hp_damage_bonus
# apply there, exactly as before this change.
# The constants are deliberately layered so a WIN always beats every
# possible LOSS outcome, and so within each branch the secondary terms
# can only nudge the reward, never flip the win/loss ordering or (for
# wins) push the total below the guaranteed WIN_BASE.

REWARD_CONFIG = dict(
    win_base=1000.0,
    time_bonus_per_second_saved=8.0,   # wins: + this * max(0, time_cap_bonus - time_estimate)
                                        # (raised 8x from 1.0 on 2026-09-22 -- the first
                                        # time-reward run showed literally no measurable
                                        # effect on win-time; see CLAUDE.md/MODELS.md)
    time_cap_bonus=200.0,              # seconds (raised from 120 on 2026-09-22 -- the old
                                        # cap meant a large fraction of observed win times
                                        # (up to ~206s) got ZERO time-based differentiation
                                        # at all, clipped to the same 0 bonus regardless of
                                        # how much slower than 120s they were)
    faint_bonus=200.0,                 # losses: + this * (# of Solrock/Lunatone fainted)
    hp_damage_bonus=100.0,             # losses: + this * (1 - avg foe HP fraction remaining)
    time_bonus_scale=1.0,              # wins: multiplies the whole time-bonus term, 0-1.
                                        # Added 2026-09-22 for win-rate-gated training (see
                                        # ppo_train_torch.py's --time-gate-low/--time-gate-high):
                                        # an always-on strong time bonus was tried and made
                                        # win rate significantly worse without actually
                                        # improving win time (model_2026-09-22_stronger-time-
                                        # reward.pt) -- the theory here is that competing with
                                        # win_base before the policy can even win reliably is
                                        # what caused that regression, so this lets a
                                        # per-iteration scale fade the time bonus in only once
                                        # win rate is already healthy. Default 1.0 (full
                                        # strength, i.e. ungated) so any caller that doesn't
                                        # know about gating -- evaluate.py, older code --
                                        # measures/uses the full reward unchanged.
    shaping_time_coef=0.0,             # 0 = off (default). Potential-based dense per-step
                                        # shaping (2026-09-22): Phi(s) = -shaping_time_coef *
                                        # battle.time_estimate(s), added as
                                        # gamma*Phi(s') - Phi(s) to EVERY step's reward
                                        # (see step()'s "shaping" info-dict entry and
                                        # ppo_train_torch.py's _run_episodes, which is what
                                        # actually applies it -- WINS ONLY, see there for why).
                                        # Motivated by three failed terminal-only time-reward
                                        # attempts (weak/strong/gated, see CLAUDE.md) -- a
                                        # sparse once-per-episode bonus may simply be too hard
                                        # for PPO to assign credit from; this delivers the same
                                        # underlying signal every turn instead, which
                                        # Ng-et-al-style potential-based shaping guarantees
                                        # doesn't change the optimal policy (for a FIXED base
                                        # reward), only how easy it is to learn.
    shaping_gamma=0.99,                # must match ppo_train_torch.py's --gamma for the
                                        # potential-based telescoping identity to hold; not
                                        # auto-synced, since --gamma is never actually
                                        # overridden from its default in this project.
)


def compute_reward(stats, config=REWARD_CONFIG):
    if stats["won"]:
        r = config["win_base"]
        r += (config.get("time_bonus_scale", 1.0) * config["time_bonus_per_second_saved"]
              * max(0, config["time_cap_bonus"] - stats["time_estimate"]))
        return r
    r = config["faint_bonus"] * stats["foe_fainted_count"]
    r += config["hp_damage_bonus"] * (1.0 - stats["foe_hp_fraction_remaining"])
    return r


# ============================================================================
# ENVIRONMENT
# ============================================================================

class TateLizaEnv:
    def __init__(self, swampert_hp=None, turn_cap=50, reward_config=None, seed=None,
                 force_swampert_xspeed_turn1=False, force_swampert_xspeed_turn1_prob=None,
                 force_wingull_heal_item_turn1=None,
                 force_wingull_heal_item_turn1_hp_range=None,
                 foe_hp_visible=False):
        self.swampert_hp = swampert_hp
        self.turn_cap = turn_cap
        self.foe_hp_visible = foe_hp_visible
        self.reward_config = reward_config or REWARD_CONFIG
        self.rng = random.Random(seed)
        self.obs_dim = OBS_DIM
        self.action_dim = ACTION_DIM
        self.force_swampert_xspeed_turn1 = force_swampert_xspeed_turn1
        # Optional per-episode Bernoulli forcing probability (2026-09-23),
        # distinct from the always-on/off force_swampert_xspeed_turn1 above:
        # meant to be ANNEALED over training (see ppo_train_torch.py's
        # xspeed_forcing_prob_for_iteration) from 1.0 down to 0.0, so the
        # value function first learns X Speed's worth from a majority-
        # forced early phase, then a growing fraction of episodes hand the
        # opening decision back to the policy -- testing whether it keeps
        # choosing X Speed once shown its value, vs. needing to rediscover
        # it via unassisted exploration alone (which model_2026-09-21_
        # entropy-schedule.pt already showed doesn't work: 8x more usage
        # via entropy tuning alone still left it at ~0.18%, functionally
        # unused). None (default) disables this -- reset() then falls back
        # to the plain boolean flag above.
        self.force_swampert_xspeed_turn1_prob = force_swampert_xspeed_turn1_prob
        self.force_wingull_heal_item_turn1 = force_wingull_heal_item_turn1  # e.g. "Hyper Potion"
        # Optional (lo, hi) inclusive starting-Swampert-HP gate (2026-09-22):
        # only force the Wingull heal item when the episode's starting HP
        # falls in this range; outside it, Wingull is left completely free,
        # same as if force_wingull_heal_item_turn1 were unset. None (default)
        # means force unconditionally whenever the item is set, matching the
        # original analysis-tool behavior.
        self.force_wingull_heal_item_turn1_hp_range = force_wingull_heal_item_turn1_hp_range
        self._battle = None
        self._party = None
        self._pending = []       # queue of ("move", mon) / ("replace", slot_index) for this turn
        self._turn_actions = {}  # mon -> action dict, being built up for the current turn
        self._surf_count = 0
        self._item_counts = {name: 0 for name in ("Potion", "Super Potion", "Hyper Potion", "Full Heal")}
        self._done = False
        self._xspeed_forced_yet = False
        self._xspeed_forced_this_episode = force_swampert_xspeed_turn1
        self._policy_chose_xspeed_turn1 = False
        self._policy_turn1_action_idx = None  # generalized version, see step()
        self._wingull_heal_forced_yet = False
        self._last_decision = None  # (kind, mon_or_slot_index), see is_swampert_turn1_move_decision
        self._turn_gen = None  # active tl_sim._resolve_turn_gen(), or None between turns -- see
                                # _start_turn_resolution/_advance_to_next_decision

    # -- internal helpers -------------------------------------------------
    def _queue_replacement(self, battle, slot_index):
        """Registered as tl_sim's replacement_callback. We can't decide
        synchronously here -- the *agent* needs to choose -- so we just
        record that a decision is pending and decline the switch (return
        None). tl_sim.Battle.resolve_player_faints() leaves the fainted
        Pokemon in place when the callback declines, which is exactly
        what we want: the real switch_in happens later, from step(), once
        the agent's action for this decision arrives. We guard against
        resolve_turn() asking about the same still-vacant slot more than
        once within a single turn (it re-checks after every action and
        again after end-of-turn effects)."""
        key = ("replace", slot_index)
        if key not in self._pending:
            self._pending.append(key)
        return None

    def _autofill_charging_actions(self):
        """A Pokemon mid-charge on a two-turn move (Dive/Fly) is FORCED to
        continue that exact move -- it is not a free choice. This must be
        filled in directly rather than routed through the policy: if the
        agent were allowed to pick something else here, tl_sim.do_move()
        would just execute that other move normally and silently leave
        mon.charging_move pointing at the old move forever (nothing ever
        clears it, since only actually completing the release turn does).
        Concretely, that means a Pokemon could use Dive once, "switch" to
        a different move on the resulting free decision, and stay
        semi-invulnerable -- untargetable by every enemy attack -- for
        the rest of the battle. That's not a hypothetical: it happened in
        an earlier version of this file and produced a heavily inflated
        training win rate."""
        for mon in self._battle.player:
            if mon.is_fainted() or mon in self._turn_actions:
                continue
            if mon.charging_move is not None:
                ms = next((m for m in mon.moves if m.move is mon.charging_move), None)
                self._turn_actions[mon] = {"kind": "move", "mon": mon, "move": mon.charging_move,
                                            "target": mon.charging_target, "moveslot": ms}

    def _autofill_forced_xspeed(self):
        """Experimental ablation (2026-09-21), off by default: force
        Swampert's canonical-slot-0 Pokemon to use X Speed on its very
        first decision of the episode, exactly once, without ever asking
        the policy. Mirrors _autofill_charging_actions's approach
        deliberately -- an auto-filled action never goes through
        net.act(), so no logprob is recorded and it never enters the PPO
        buffer as something the policy "chose". This tests whether
        forcing the human's opening move helps, as a controlled
        environment-level ablation, without biasing the policy's own
        learned preferences the way a reward bonus for X Speed would.

        Whether THIS episode forces at all is decided once, in reset()
        (see force_swampert_xspeed_turn1_prob), and cached in
        self._xspeed_forced_this_episode -- not re-evaluated here, so a
        probabilistic schedule doesn't redraw mid-episode."""
        if not self._xspeed_forced_this_episode or self._xspeed_forced_yet:
            return
        swampert = self._party[0]  # canonical index 0 is always Swampert
        if swampert.is_fainted() or swampert in self._turn_actions or swampert not in self._battle.player:
            return
        self._turn_actions[swampert] = {"kind": "item", "mon": swampert,
                                         "item": "X Speed", "item_target": swampert}
        self._xspeed_forced_yet = True

    def _autofill_forced_wingull_heal(self):
        """Analysis tool (2026-09-21), off by default: force Wingull's
        canonical-slot-1 Pokemon to use a specific healing item on
        Swampert on its very first decision of the episode, exactly once.
        Same mechanism/rationale as _autofill_forced_xspeed -- lets a
        one-off comparison (e.g. Hyper Potion vs. Super Potion on turn 1)
        be evaluated without needing a dedicated training run per item.

        force_wingull_heal_item_turn1_hp_range, if set, additionally
        gates this on the episode's starting Swampert HP falling in that
        (lo, hi) inclusive band -- checked against swampert.current_hp at
        this point, which still equals the starting HP since nothing has
        resolved yet this turn. Outside the range, this gives up forcing
        for the rest of the episode (same as if the item were never set),
        so Wingull is trained/evaluated completely normally there."""
        if not self.force_wingull_heal_item_turn1 or self._wingull_heal_forced_yet:
            return
        wingull = self._party[1]  # canonical index 1 is always Wingull
        swampert = self._party[0]
        if wingull.is_fainted() or wingull in self._turn_actions or wingull not in self._battle.player:
            return
        if self.force_wingull_heal_item_turn1_hp_range is not None:
            lo, hi = self.force_wingull_heal_item_turn1_hp_range
            if not (lo <= swampert.current_hp <= hi):
                self._wingull_heal_forced_yet = True  # out of range: never force this episode
                return
        self._turn_actions[wingull] = {"kind": "item", "mon": wingull,
                                        "item": self.force_wingull_heal_item_turn1,
                                        "item_target": swampert}
        self._wingull_heal_forced_yet = True

    def _next_pending_from_active(self):
        for mon in self._battle.player:
            if not mon.is_fainted() and mon not in self._turn_actions:
                return ("move", mon)
        return None

    def _pending_replacement_obs_and_mask(self):
        while self._pending:
            kind, slot_index = self._pending[0]
            reserves = self._battle.player_alive_reserves()
            if not reserves:
                # Nothing left to send out (e.g. both slots fainted at
                # once with only one reserve available) -- nothing to
                # decide, so drop it and keep looking.
                self._pending.pop(0)
                continue
            fainted_mon_idx = self._last_fainted_idx.get(slot_index, 0)
            obs = encode_observation(self._battle, self._party, fainted_mon_idx, 1, foe_hp_visible=self.foe_hp_visible)
            mask = replacement_action_mask(reserves, self._party)
            return obs, mask, ("replace", slot_index)
        return None, None, None

    def _current_obs_and_mask(self):
        if self._turn_gen is not None:
            # Mid-resolution of a turn already in progress (2026-09-23 fix
            # -- see tl_sim.py's _resolve_turn_gen docstring for why this
            # branch exists at all): the ONLY thing that can legitimately
            # be pending here is a replacement forced by a faint that just
            # happened partway through this turn's action order. Every
            # other player action for this turn was already decided and
            # handed to the generator before it started running -- asking
            # _next_pending_from_active() here would be wrong, since
            # self._turn_actions was already cleared when the generator
            # was created, so every active mon would look "undecided" even
            # though its action is mid-execution.
            return self._pending_replacement_obs_and_mask()

        self._autofill_charging_actions()
        self._autofill_forced_xspeed()
        self._autofill_forced_wingull_heal()

        obs, mask, decision = self._pending_replacement_obs_and_mask()
        if decision is not None:
            return obs, mask, decision

        nxt = self._next_pending_from_active()
        if nxt is None:
            return None, None, None
        kind, mon = nxt
        idx = self._party.index(mon)
        obs = encode_observation(self._battle, self._party, idx, 0, foe_hp_visible=self.foe_hp_visible)
        mask = move_action_mask(mon, self._party)
        return obs, mask, ("move", mon)

    def _start_turn_resolution(self):
        """Collects the foe's moves (decided now, same as the player's --
        matching real mechanics: everyone chooses simultaneously, before
        speed order is known) and starts resolving the turn via tl_sim's
        resumable generator, advancing it up to its first pause (a
        pending replacement) or completion."""
        for mon in self._battle.foe:
            if not mon.is_fainted():
                self._turn_actions[mon] = sim.ai_take_turn(self._battle, mon)
        for mon, act in list(self._turn_actions.items()):
            if (act.get("kind") == "move" and act["move"].name == "Surf"
                    and mon.name == "Swampert"):
                self._surf_count += 1
            if mon.side == "player" and act.get("kind") == "item" and act.get("item") in self._item_counts:
                self._item_counts[act["item"]] += 1
        self._turn_gen = sim._resolve_turn_gen(self._battle, self._turn_actions)
        self._turn_actions = {}
        self._advance_turn_gen()

    def _advance_turn_gen(self):
        """Resumes the in-progress turn-resolution generator by one step
        (i.e. until its next pause or until it finishes). Only call this
        once any previously-pending replacement decisions have actually
        been filled in -- see _advance_to_next_decision."""
        try:
            next(self._turn_gen)
        except StopIteration:
            self._turn_gen = None

    def _advance_to_next_decision(self):
        """Resolves turns automatically for as long as there's nothing
        left for the agent to decide -- which happens whenever every
        currently-active player Pokemon is mid-charge (all auto-filled by
        _autofill_charging_actions with no real decision among them),
        most plausibly if both Swampert and Wingull happen to be
        charging Dive/Fly on the same turn. Returns either a real
        (obs, mask, decision) to hand back to the caller, or a terminal
        result tuple (None, None, "done") if the battle ends along the
        way. Bounded by an iteration cap so a future bug here fails
        loudly instead of hanging -- higher than before the 2026-09-23
        turn-pausing fix, since a single turn can now legitimately need
        several passes through this loop (e.g. two simultaneous faints
        from one spread-move hit, each needing its own replacement
        decision, interleaved with resuming the generator between them)."""
        for _ in range(20):
            if self._turn_gen is not None:
                # A turn is mid-resolution -- only a pending replacement
                # can be relevant; do NOT fall through to collecting a
                # fresh move decision for a turn that hasn't finished yet.
                obs, mask, decision = self._current_obs_and_mask()
                if decision is not None:
                    return obs, mask, decision
                self._advance_turn_gen()
                if self._battle.is_over() or self._battle.turn_count >= self.turn_cap:
                    self._turn_gen = None
                    return None, None, "done"
                continue

            obs, mask, decision = self._current_obs_and_mask()
            if decision is not None:
                return obs, mask, decision
            # Nothing pending and no active Pokemon needs a decision --
            # every active slot must already be auto-filled (charging) or
            # fainted-with-no-reserve. Start resolving the turn.
            self._start_turn_resolution()
            if self._battle.is_over() or self._battle.turn_count >= self.turn_cap:
                return None, None, "done"
        raise RuntimeError("_advance_to_next_decision looped without finding a decision or "
                            "ending the battle -- this indicates a new bug, not normal play")

    # -- public Gym-like API -----------------------------------------------
    def reset(self, swampert_hp=None, seed=None):
        hp = swampert_hp if swampert_hp is not None else self.swampert_hp
        if seed is not None:
            self.rng = random.Random(seed)
        self._party = sim.make_preset_team(swampert_hp=hp)
        self._last_fainted_idx = {}

        def replacement_callback(battle, slot_index):
            self._last_fainted_idx[slot_index] = self._party.index(battle.player[slot_index])
            return self._queue_replacement(battle, slot_index)

        self._battle = sim.Battle(self._party, [sim.make_solrock(), sim.make_lunatone()],
                                   log=lambda *a: None, replacement_callback=replacement_callback)
        self._battle.rng = self.rng
        self._pending = []
        self._turn_actions = {}
        self._turn_gen = None
        self._surf_count = 0
        self._item_counts = {name: 0 for name in ("Potion", "Super Potion", "Hyper Potion", "Full Heal")}
        self._done = False
        self._xspeed_forced_yet = False
        if self.force_swampert_xspeed_turn1_prob is not None:
            self._xspeed_forced_this_episode = self.rng.random() < self.force_swampert_xspeed_turn1_prob
        else:
            self._xspeed_forced_this_episode = self.force_swampert_xspeed_turn1
        self._policy_chose_xspeed_turn1 = False
        self._policy_turn1_action_idx = None
        self._wingull_heal_forced_yet = False
        # A fresh battle can never start mid-charge, so a real decision is
        # always available immediately -- but route through the shared
        # helper anyway for a single, uniform code path.
        obs, mask, decision = self._advance_to_next_decision()
        assert decision != "done", "battle ended before any decision was made -- shouldn't happen"
        self._last_decision = decision
        return obs, mask

    def reset_to_solar_beam_castform_scenario(self, seed=None, hp_range=(100, 133)):
        """Resets directly into "Solrock full-HP charging Solar Beam,
        Castform active, Dive-won't-kill" (2026-09-24) -- a targeted
        state-distribution curriculum addition, NOT forced-action
        training. The episode starts here, but every decision (including
        THIS turn's Swampert-Dive and Castform-Rain-Dance choices) is
        still made freely by the policy and receives normal, genuinely-
        attributed PPO gradient like any other episode -- only the
        STARTING STATE distribution is biased. This sidesteps the "no
        contrastive signal" failure mode that sank the forced-action
        X-Speed anneal attempts (see xspeed_forcing_prob_for_iteration's
        docstring): the policy's own entropy still naturally samples both
        Rain Dance and alternatives here, so GAE has real signal to learn
        from, unlike forcing a single action every time.

        Motivated by a directly-measured, real win-rate cost: this
        project's best model reliably Dives correctly here but picks Rain
        Dance ~64% of the time despite Solar Beam surviving Dive being
        able to set up a documented loss (Solrock's AI overwrites the
        rain with Sunny Day) -- see CLAUDE.md's mechanics notes. Free
        parameters are randomized each episode so the model learns a
        general rule instead of memorizing one exact state: Swampert's
        starting HP (uniform over `hp_range`, default the full documented
        40-133 span -- round 2+ fine-tunes can narrow/shift this to
        oversample a sub-range a prior round under-learned, e.g. near-full
        HP where healing stops being a plausible alternative to Rain
        Dance), whether Abra is alive in reserve, Castform's own current
        HP, and Lunatone's Calm Mind count.

        **Solrock's own current HP is randomized too (2026-09-25 fix),
        not fixed at full** -- rounds 1-2 always used exactly 123/123 HP,
        which was a real train/eval distribution mismatch: a large-sample
        from-scratch measurement (see MODELS.md) found Solrock is at
        ~55-68% HP (never full) every single time this scenario arises
        in actual play, because it's already taken damage earlier in the
        fight by the time Castform is out and it's charging Solar Beam.
        Rounds 1-2 accordingly fixed the mistake almost perfectly within
        their own (full-HP-Solrock) training distribution but left the
        real, precise-criterion mistake rate at ~100% (0.56-1.29% of
        games, but wrong essentially every time) -- the model had learned
        "avoid Rain Dance when Solrock is full HP," not the general rule.
        Fixed by computing Dive's actual max-roll damage against THIS
        episode's randomized Swampert (post `_battle` construction, so
        weather/side-condition context is real) and sampling Solrock's HP
        uniformly from the range that keeps "Dive genuinely can't kill it"
        true -- the same invariant the scenario is named for, now actually
        varied instead of pinned to one point on it.

        **Light Screen is now also set up on the foe's side (2026-09-25,
        same investigation)** -- a from-scratch instrumented check found
        it up in 111/111 (100%) of real occurrences of this scenario, not
        an incidental detail: Dive/Surf are Water-type, which is a
        SPECIAL move under this sim's Gen-3 type-based physical/special
        split, so Light Screen directly determines how much damage
        Solrock has actually taken by this point (almost always exactly
        one prior hit, a screened, already-halved 2-target Surf -- see
        MODELS.md for the full reconstructed battle logs). Turns
        remaining sampled to match the observed real distribution (2:
        ~25%, 3: ~74%, rare 1/4). Applied BEFORE the Dive-invariant
        computation below, so the sampled Solrock HP band correctly
        reflects Light-Screen-reduced Dive damage rather than the
        unscreened figure.

        **Castform is now always at full HP, and `hp_range` defaults to
        (100, 133) instead of (40, 133) (2026-09-25, the same day's THIRD
        and most consequential fix).** After rounds 3-6 all failed to
        close the real-game gap despite fixing Solrock's HP/Light Screen
        (round 3 helped partially; oversampling and longer training
        regressed it in rounds 4-5; attributed forcing toward the
        policy's own "best alternative" backfired in round 6, reinforcing
        Rain Dance instead of suppressing it, since the forced action had
        no guarantee of actually being good), a side-by-side comparison
        of scripted-reset episodes against real ones that hit this exact
        decision turned up the actual root cause: a 541-sample large-scale
        check found Castform is at EXACTLY full HP in 100.0% of real
        occurrences (it just switched in, hasn't taken damage yet), and
        Swampert's HP is heavily concentrated near-full (median/mode 133,
        5th percentile 114, full observed range 103-133) -- nothing like
        the old uniform 40-133 Swampert range or the 40-100%-of-max
        Castform range this method used to sample. Both were creating
        artificial "heal the low-HP ally instead of dealing with Rain
        Dance" escape hatches that essentially never exist in real games,
        diluting training with mostly-irrelevant states while
        underweighting the actual dominant real case (both attackers
        near/at full HP, where neither heal alternative is available and
        the model has to confront the real tactical question). This
        directly explains round 1's own finding ("reverts to Rain Dance
        specifically near full Swampert HP") -- that was never a rare
        edge case, it was the TRUE common case the whole time, just
        underrepresented in training. `hp_range` is still overridable for
        a future round that wants to shift it further.

        Fourth fix, same day (2026-09-25): Abra-fainted rate, Lunatone's
        HP, and Lunatone's Calm Mind rate were ALSO all wrong. A
        side-by-side comparison of round 7 (trained with the above two
        fixes) against real occurrences showed round 7's scripted-scenario
        behavior improving substantially (50pct to 28pct Rain-Dance-on-
        first-decision) while its REAL-game precise-criterion mistake
        rate barely moved -- meaning the training distribution still
        didn't match reality closely enough. A comprehensive 541-sample
        characterization (see MODELS.md) found: Abra is fainted in 91.7%
        of real occurrences (this method was using 50%), Lunatone's Calm
        Mind is up in only 8.7% (this method was using 50%), Lunatone's
        HP is highly variable with a real distribution nothing like
        "always full" (median 91.9%, full only 30.7% of the time, ranging
        down to 0%), and Solrock's own HP is actually capped around 70%
        in practice (real max observed ~70%), not the full 37-100%
        theoretical range the old "whatever keeps Dive from killing it"
        sampling allowed. Fixed via SOLROCK_HP_FRAC_PERCENTILES and
        LUNATONE_HP_FRAC_PERCENTILES (100-point empirical-CDF samplers
        built directly from the same 541-sample characterization, so no
        hand-fit parametric approximation is needed) plus corrected
        probabilities for Abra-fainted and Calm Mind.

        Fifth fix, same day (2026-09-25): Swampert's exact max HP
        (hp_range[1]) is now heavily oversampled, not just one point in a
        uniform range. Round 8 (with all four fixes above) showed a real,
        repeatable improvement in the 1-80 and 1-133 large-sample checks
        (90.0%->73.7% and 92.9%->88.4% real-game mistake rate) but the
        Full-HP evaluation range stayed at EXACTLY 100%, completely
        unmoved -- a genuine full-HP start means Swampert essentially
        never takes damage before this decision (nothing else attacks
        it), so it's ALWAYS still at exactly hp_range[1] by the time this
        scenario arises, yet the old uniform sampling only gave that
        single boundary value 1/(hp_range[1]-hp_range[0]+1) of the
        weight -- a classic function-approximation edge effect, on top of
        an already-known pattern (the OLD fine-tune arc's round 1/2 also
        found the mistake concentrated hardest right at the literal max
        HP). Weighted 35% of episodes to land exactly on hp_range[1]."""
        if seed is not None:
            self.rng = random.Random(seed)
        hp = hp_range[1] if self.rng.random() < 0.35 else self.rng.randint(*hp_range)
        self._party = sim.make_preset_team(swampert_hp=hp)
        swampert, wingull, abra, castform = self._party
        swampert.stages["spe"] = 1  # matches this model's own learned turn-1 X Speed opening
        wingull.current_hp = 0
        if self.rng.random() < 0.917:  # matches the observed real 91.7% rate
            abra.current_hp = 0
        # Castform is always full HP in real occurrences of this scenario
        # (it just switched in, hasn't taken damage) -- see the 2026-09-25
        # note above. No longer randomized down.

        solrock = sim.make_solrock()
        solrock.charging_move = next(ms.move for ms in solrock.moves if ms.move.name == "Solar Beam")
        solrock.charging_target = swampert
        lunatone = sim.make_lunatone()
        # Sixth fix ATTEMPTED and REVERTED, same day (2026-09-25): a
        # bucketed check of round 9 found mistake rate correlates with
        # Lunatone being at exact full HP too (56.7% vs 22.0%, and 78.3%
        # when BOTH Swampert and Lunatone are exactly full vs 33.0%
        # overall), so round 10 boosted Lunatone's exact-full-HP
        # probability to 0.40 on top of the percentile sampler's own
        # natural ~31% -- but this REGRESSED the 1-80 and 1-133 ranges
        # (27.8%->55.6%, 21.6%->48.3% real-game mistake rate) instead of
        # helping. Unlike Swampert's hp_range[1] fix (a genuine structural
        # guarantee -- a full-HP-START game is ALWAYS still at exactly
        # max HP by this decision, so undersampling that exact value was
        # a pure representation gap), Lunatone being at full HP is just a
        # common OUTCOME (~30% of the time), not a certainty -- there's no
        # structural reason to weight it above its own true empirical
        # rate, and doing so over-narrowed the training distribution the
        # same way round 4's HP-range oversampling did in the earlier
        # arc. Reverted to pure percentile-sampler weighting (no override).
        #
        # Seventh fix, targeted retry (2026-09-25, follow-up session): a
        # bucketed check of round 11's REAL Full-HP-range games (not the
        # pooled/scripted data round 10 used) found the same Lunatone-full
        # correlation, but specific to Swampert ALSO being at exact max
        # HP: mistake rate 61.1% (n=108) vs 12.5% (n=128) within Full-HP
        # games specifically, and Lunatone is at exact full HP in 45.8%
        # of real Full-HP-range occurrences (108/236) -- well above the
        # ~30% pooled-average rate the percentile sampler alone gives.
        # Unlike round 10's blanket, UNCONDITIONAL boost (which diluted
        # the 1-80/1-133 ranges by also raising Lunatone's full-HP rate
        # in episodes where Swampert ISN'T at max HP, where that
        # correlation doesn't hold), this boost is CONDITIONAL on
        # Swampert already being at exactly hp_range[1] this episode --
        # it shouldn't touch the other ranges' distribution at all.
        if hp == hp_range[1] and self.rng.random() < 0.202:
            lunatone.current_hp = lunatone.max_hp
        else:
            lunatone.current_hp = max(1, round(lunatone.max_hp * self.rng.choice(LUNATONE_HP_FRAC_PERCENTILES)))
        if self.rng.random() < 0.087:  # matches the observed real 8.7% rate, not 50%
            lunatone.stages["spa"] = 1
            lunatone.stages["spd"] = 1

        self._last_fainted_idx = {}

        def replacement_callback(battle, slot_index):
            self._last_fainted_idx[slot_index] = self._party.index(battle.player[slot_index])
            return self._queue_replacement(battle, slot_index)

        self._battle = sim.Battle([swampert, castform, wingull, abra], [solrock, lunatone],
                                   log=lambda *a: None, replacement_callback=replacement_callback)
        self._battle.rng = self.rng
        self._battle.turn_count = self.rng.choices([2, 3, 4], weights=[223, 315, 3])[0]

        # Light Screen on the foe's side -- see the docstring's 2026-09-25
        # note; matches the observed real-game distribution (up in 100%
        # of real occurrences, almost always 2-3 turns remaining).
        self._battle.sides["foe"].light_screen = self.rng.choices(
            [1, 2, 3, 4], weights=[3, 141, 421, 2])[0]

        # Solrock's current HP -- sampled from the real empirical
        # distribution (SOLROCK_HP_FRAC_PERCENTILES, capped ~70% in
        # practice, not the full theoretical range), then clamped up if
        # needed to preserve the "Dive genuinely can't kill it" invariant
        # the scenario is named for (should essentially never bind, since
        # the empirical range starts well above Dive's max damage).
        dive = next(ms.move for ms in swampert.moves if ms.move.name == "Dive")
        dive_base = sim._base_damage(self._battle, swampert, solrock, dive, False)
        dive_max_dmg = sim._finish_damage(self._battle, swampert, solrock, dive, False, dive_base, 100)
        sampled_hp = max(1, round(solrock.max_hp * self.rng.choice(SOLROCK_HP_FRAC_PERCENTILES)))
        solrock.current_hp = max(sampled_hp, dive_max_dmg + 1)
        self._pending = []
        self._turn_actions = {}
        self._turn_gen = None
        self._surf_count = 0
        self._item_counts = {name: 0 for name in ("Potion", "Super Potion", "Hyper Potion", "Full Heal")}
        self._done = False
        self._xspeed_forced_yet = True    # X Speed already "used" (stage already applied) -- never force it here
        self._xspeed_forced_this_episode = False
        self._policy_chose_xspeed_turn1 = False
        self._policy_turn1_action_idx = None
        self._wingull_heal_forced_yet = True  # no Wingull in play this scenario
        obs, mask, decision = self._advance_to_next_decision()
        assert decision != "done", "battle ended before any decision was made -- shouldn't happen"
        self._last_decision = decision
        return obs, mask

    def reset_to_surf_no_kill_dive_kill_scenario(self, seed=None, hp_range=(40, 133)):
        """Resets into "Surf (2-target) cannot kill Solrock even at its
        max possible roll, Dive always kills it even at its min possible
        roll, Solrock NOT charging Solar Beam" (2026-09-25) -- a second
        targeted state-distribution curriculum scenario, testing a
        DIFFERENT decision than reset_to_solar_beam_castform_scenario:
        Swampert's own move choice, isolated from the reactive-Dive-vs-
        Solar-Beam-charge pattern entirely (Solrock is never charging
        here).

        Motivated by an empirical finding, not an assumed one: both
        best_model_xspeed-learned.pt and a from-scratch baseline trained
        with none of the Dive+Rain-Dance fine-tuning STILL pick Surf
        essentially unconditionally here (Dive chosen under 0.05% of the
        time at Solrock HP 67-68, the cleanest boundary case, across both
        models and multiple Swampert HP points) -- even though
        Tate_and_Liza_Document.tex calls Dive the correct play in exactly
        this situation ("if your first Surf roll is very good, then you
        can choose to go for the Surf+Surf range" implies the converse:
        if Surf can't secure it, prefer Dive). Jacob is explicitly open
        to the documented strategy being wrong here, not just untested by
        the model -- this scenario plus attributed forcing (see
        ppo_train_torch.py's --dive-forcing-prob) is designed to find out
        which: if the model's FREE choice shifts toward Dive once given a
        fair, contrastive-signal-bearing chance to learn it, Dive was
        genuinely underexplored; if it reverts to Surf even with forcing
        washed out, Surf may actually be fine (or better) here despite
        the conventional wisdom.

        Solrock's HP band is computed dynamically each episode from the
        REAL damage functions (`tl_sim._base_damage`/`_finish_damage`),
        not a fixed threshold, since Torrent's 1.5x Water-move boost
        (Swampert HP <= 1/3 max) shifts both Surf's and Dive's damage
        ranges depending on this episode's randomized Swampert HP.
        Lunatone is always kept alive at a randomized HP -- the 2-target
        Surf halving is part of the tested comparison; a fainted Lunatone
        would let Surf single-target (double damage), changing the whole
        comparison. Light Screen is intentionally left off (unlike the
        other scenario) to keep this specific comparison clean and
        directly interpretable, a deliberate simplification -- a
        follow-up could extend this the way the other scenario's
        real-occurrence characterization did, if this initial version
        finds something worth refining further.

        **Partner identity is now randomized too (2026-09-25 fix, same
        day as the first version)** -- caught by Jacob after the first
        dive-forcing run: `sim.Battle`'s constructor treats the first two
        party entries as the active pair, and the original version of
        this method hardcoded `[swampert, castform, wingull, abra]`, so
        Castform was the active partner in EVERY episode; Wingull-active
        (not yet fainted) and Abra-active (Wingull fainted, Castform
        still untouched in reserve) were never trained on at all. Fixed:
        each episode picks uniformly among the three realistic partner
        configurations (Wingull still active; Abra active with Castform
        untouched in reserve; Castform active with Abra alive-in-reserve
        or already fainted, 50/50) -- a deliberate uniform simplification
        rather than a characterized real-occurrence distribution like the
        other scenario has, since the point here is COVERAGE of the full
        state space, not matching real frequencies. The active partner's
        own HP is also randomized (uniform 20-100% of its max), not
        pinned to full."""
        if seed is not None:
            self.rng = random.Random(seed)
        hp = self.rng.randint(*hp_range)
        self._party = sim.make_preset_team(swampert_hp=hp)
        swampert, wingull, abra, castform = self._party
        swampert.stages["spe"] = 1  # matches this model's own learned turn-1 X Speed opening

        partner_kind = self.rng.choice(["wingull", "abra", "castform"])
        if partner_kind == "wingull":
            wingull.current_hp = max(1, round(wingull.max_hp * self.rng.uniform(0.2, 1.0)))
            active_list = [swampert, wingull, abra, castform]
        elif partner_kind == "abra":
            wingull.current_hp = 0
            abra.current_hp = max(1, round(abra.max_hp * self.rng.uniform(0.2, 1.0)))
            active_list = [swampert, abra, wingull, castform]
        else:
            wingull.current_hp = 0
            if self.rng.random() < 0.5:
                abra.current_hp = 0
            castform.current_hp = max(1, round(castform.max_hp * self.rng.uniform(0.2, 1.0)))
            active_list = [swampert, castform, wingull, abra]

        solrock = sim.make_solrock()
        lunatone = sim.make_lunatone()
        lunatone.current_hp = max(1, round(lunatone.max_hp * self.rng.uniform(0.4, 1.0)))

        self._last_fainted_idx = {}

        def replacement_callback(battle, slot_index):
            self._last_fainted_idx[slot_index] = self._party.index(battle.player[slot_index])
            return self._queue_replacement(battle, slot_index)

        self._battle = sim.Battle(active_list, [solrock, lunatone],
                                   log=lambda *a: None, replacement_callback=replacement_callback)
        self._battle.rng = self.rng
        self._battle.turn_count = self.rng.choices([2, 3, 4], weights=[223, 315, 3])[0]

        # The precise invariant this scenario is named for: Surf's own
        # max-roll damage can't kill Solrock, but Dive's min-roll damage
        # always does. Computed post-Battle-construction so any
        # weather/side-condition context is real (none is set up here,
        # but this keeps the calculation correct if that ever changes).
        surf = next(ms.move for ms in swampert.moves if ms.move.name == "Surf")
        dive = next(ms.move for ms in swampert.moves if ms.move.name == "Dive")
        surf_base = sim._base_damage(self._battle, swampert, solrock, surf, False, defending_side_alive_count=2)
        surf_max = sim._finish_damage(self._battle, swampert, solrock, surf, False, surf_base, 100)
        dive_base = sim._base_damage(self._battle, swampert, solrock, dive, False)
        dive_min = sim._finish_damage(self._battle, swampert, solrock, dive, False, dive_base, 85)
        lo = surf_max + 1
        hi = min(dive_min, solrock.max_hp)
        if lo > hi:
            # Extremely rare edge case (e.g. an unusual Swampert HP making
            # the valid band empty) -- clamp to a single point rather than
            # crash; this scenario simply won't hold its own invariant
            # exactly for this one seed, which the regression test checks
            # for at scale, not per-seed.
            lo = hi = min(max(lo, 1), solrock.max_hp)
        solrock.current_hp = self.rng.randint(lo, hi)

        self._pending = []
        self._turn_actions = {}
        self._turn_gen = None
        self._surf_count = 0
        self._item_counts = {name: 0 for name in ("Potion", "Super Potion", "Hyper Potion", "Full Heal")}
        self._done = False
        self._xspeed_forced_yet = True
        self._xspeed_forced_this_episode = False
        self._policy_chose_xspeed_turn1 = False
        self._policy_turn1_action_idx = None
        self._wingull_heal_forced_yet = True
        obs, mask, decision = self._advance_to_next_decision()
        assert decision != "done", "battle ended before any decision was made -- shouldn't happen"
        self._last_decision = decision
        return obs, mask

    def reset_to_lunatone_cm1_heal_scenario(self, seed=None, hp_range=(79, 95)):
        """Resets into "Lunatone has already used Calm Mind once, Swampert
        is at a starting HP that lands inside the resulting BOOSTED
        Psychic pseudoroll band" (2026-09-26) -- a fourth state-
        distribution curriculum scenario, testing the heal-item decision
        (Hyper Potion / Super Potion / no heal) under a SECOND, previously
        undocumented danger threshold.

        Motivated directly by comparing best_model_xspeed-learned.pt
        against best_model_healband.pt across the full HP sweep: the two
        models' win rate and win-time diverge in a mirror-image pattern
        specifically around Swampert HP 86-92 (healband: higher win rate,
        slower; xspeed-learned: lower win rate, faster) -- the SAME
        signature the Hyper-vs-Super-Potion comparison shows at HP 45-59,
        just at a much higher starting HP where neither model's turn-1
        decision differs at all (both pick Water Gun ~99% of the time
        there). Traced paired episodes (same seed, both models, starting
        HP 90) confirmed the actual divergence is a MID-FIGHT healing
        decision, not the turn-1 one: in every traced game, Lunatone
        reaches Calm Mind x1 very early, before Swampert has taken
        meaningful damage -- and one Calm Mind shifts Lunatone's Psychic
        from the documented base-case pseudoroll band (54-64 vs. full-HP
        Swampert, source-confirmed, see CLAUDE.md) to a BOOSTED band,
        computed the same way: 79-94 (guaranteed KO at HP<=79, impossible
        at HP>=95). Swampert sitting at 86-92 -- its ORIGINAL starting HP,
        since little damage has been taken yet -- lands almost exactly
        inside this newly-dangerous band the moment Calm Mind lands.
        best_model_healband.pt's heal-band fine-tuning (targeted at the
        BASE-case threshold, starting HP 35-71) apparently generalized
        into a more general "heal when currently inside whatever Psychic-
        kill band Lunatone currently threatens" instinct that transfers
        to this boosted band too, even though the absolute HP numbers
        (86-92 vs. 35-71) don't overlap at all -- exactly the kind of
        incidental spillover this project's fine-tuning write-ups have
        repeatedly flagged as a real risk to watch for, not a designed
        fix. This scenario gives that specific decision genuine, direct
        training signal instead of relying on generalization from a
        different HP range.

        Partner identity, Solrock's alive/fainted status and HP, and the
        active partner's own HP are all randomized, the same generalized
        pattern reset_to_surf_no_kill_dive_kill_scenario uses, so the
        model doesn't just memorize one canonical state. Solrock is
        deliberately never charging Solar Beam here (kept simple, like
        that scenario's own "Light Screen intentionally left off"
        simplification) -- this scenario isolates the heal-item decision
        from the separate reactive-Dive pattern, which already has its
        own dedicated scenario. Light Screen is also left off, matching
        the real traces (Lunatone chose Calm Mind over Light Screen in
        every one)."""
        if seed is not None:
            self.rng = random.Random(seed)
        hp = self.rng.randint(*hp_range)
        self._party = sim.make_preset_team(swampert_hp=hp)
        swampert, wingull, abra, castform = self._party
        swampert.stages["spe"] = 1  # matches this model's own learned turn-1 X Speed opening

        partner_kind = self.rng.choice(["wingull", "abra", "castform"])
        if partner_kind == "wingull":
            wingull.current_hp = max(1, round(wingull.max_hp * self.rng.uniform(0.2, 1.0)))
            active_list = [swampert, wingull, abra, castform]
        elif partner_kind == "abra":
            wingull.current_hp = 0
            abra.current_hp = max(1, round(abra.max_hp * self.rng.uniform(0.2, 1.0)))
            active_list = [swampert, abra, wingull, castform]
        else:
            wingull.current_hp = 0
            if self.rng.random() < 0.5:
                abra.current_hp = 0
            castform.current_hp = max(1, round(castform.max_hp * self.rng.uniform(0.2, 1.0)))
            active_list = [swampert, castform, wingull, abra]

        solrock = sim.make_solrock()
        lunatone = sim.make_lunatone()
        lunatone.stages["spa"] = 1  # one Calm Mind already used -- the scenario's whole premise
        lunatone.current_hp = max(1, round(lunatone.max_hp * self.rng.uniform(0.4, 1.0)))
        if self.rng.random() < 0.3:
            solrock.current_hp = 0  # already fainted in some fraction of real occurrences
        else:
            solrock.current_hp = max(1, round(solrock.max_hp * self.rng.uniform(0.3, 1.0)))

        self._last_fainted_idx = {}

        def replacement_callback(battle, slot_index):
            self._last_fainted_idx[slot_index] = self._party.index(battle.player[slot_index])
            return self._queue_replacement(battle, slot_index)

        self._battle = sim.Battle(active_list, [solrock, lunatone],
                                   log=lambda *a: None, replacement_callback=replacement_callback)
        self._battle.rng = self.rng
        self._battle.turn_count = self.rng.choices([2, 3, 4], weights=[223, 315, 3])[0]

        self._pending = []
        self._turn_actions = {}
        self._turn_gen = None
        self._surf_count = 0
        self._item_counts = {name: 0 for name in ("Potion", "Super Potion", "Hyper Potion", "Full Heal")}
        self._done = False
        self._xspeed_forced_yet = True
        self._xspeed_forced_this_episode = False
        self._policy_chose_xspeed_turn1 = False
        self._policy_turn1_action_idx = None
        self._wingull_heal_forced_yet = True
        obs, mask, decision = self._advance_to_next_decision()
        assert decision != "done", "battle ended before any decision was made -- shouldn't happen"
        self._last_decision = decision
        return obs, mask

    def step(self, action):
        if self._done:
            raise RuntimeError("step() called after episode ended -- call reset() first")

        obs, mask, decision = self._advance_to_next_decision()
        if decision == "done":
            # Can only happen if a prior call left the battle mid-resolution
            # right at the terminal boundary; treat as a logic error since
            # step() should have already returned done=True at that point.
            raise RuntimeError("step() found the battle already over -- call reset() first")
        if not mask[action]:
            raise ValueError(f"Action {action} is not legal for this observation "
                              f"(mask={mask.tolist()})")

        kind, payload = decision
        if kind == "move":
            mon = payload
            act = decode_move_action(self._battle, mon, self._party, action)
            self._turn_actions[mon] = act
            # Tracks whether the POLICY (not an autofill) chose X Speed on
            # Swampert's own turn-1 decision (2026-09-23) -- the success
            # metric for the xspeed-forcing-anneal experiment. Only ever
            # set here, in the real decode path; the forced-autofill
            # version writes directly into _turn_actions and never reaches
            # this line, so a forced episode always reports False here,
            # even on iterations where forcing happened to apply.
            if mon is self._party[0] and self._battle.turn_count == 0:
                # Generalized alongside the X-Speed-specific flag above
                # (2026-09-23): records the RAW action index of whatever
                # Swampert's turn-1 decision actually was, so
                # ppo_train_torch.py can measure the free-choice rate for
                # ANY attributed-forced action (e.g. X Special), not just
                # X Speed, without needing a matching item-name check here.
                self._policy_turn1_action_idx = action
                if act.get("kind") == "item" and act.get("item") == "X Speed":
                    self._policy_chose_xspeed_turn1 = True
        else:
            slot_index = payload
            chosen = decode_replacement_action(action, self._party)
            self._battle.switch_in(slot_index, chosen)
            self._pending.pop(0)

        # More decisions left before the turn can resolve? (This also
        # transparently handles auto-resolving through any all-charging
        # turns via the same shared helper.) time_estimate only actually
        # changes as a side effect of THIS call (when it triggers
        # resolve_turn -- the earlier _advance_to_next_decision() above
        # never resolves anything new), so this is exactly where a
        # potential-based shaping delta for the transition belongs.
        time_before = self._battle.time_estimate
        next_obs, next_mask, next_decision = self._advance_to_next_decision()
        shaping = self._time_shaping_delta(time_before, self._battle.time_estimate)
        if next_decision != "done":
            self._last_decision = next_decision
            return next_obs, 0.0, False, {"shaping": shaping}, next_mask

        self._done = True
        stats = self._episode_stats()
        reward = compute_reward(stats, self.reward_config)
        zeros = np.zeros(self.obs_dim, dtype=np.float32)
        zero_mask = np.zeros(self.action_dim, dtype=bool)
        info = {"stats": stats, "shaping": shaping}
        return zeros, reward, True, info, zero_mask

    def _time_shaping_delta(self, time_before, time_after):
        """Potential-based dense shaping term for the transition
        time_before -> time_after: gamma*Phi(s') - Phi(s), where
        Phi(s) = -shaping_time_coef * time_estimate(s). Returns 0.0 (a
        true no-op, not just numerically negligible) when
        shaping_time_coef is 0, so this is exactly a no-op unless
        explicitly enabled. NOT applied to the returned `reward` here --
        callers (see ppo_train_torch.py's _run_episodes) are responsible
        for applying it to the training signal ONLY for episodes that end
        in a win, since Phi is a pure function of state and can't itself
        encode the eventual outcome; applying it to losses too would
        violate the explicit "a loss is a loss regardless of how long it
        took" requirement."""
        coef = self.reward_config.get("shaping_time_coef", 0.0)
        if not coef:
            return 0.0
        gamma = self.reward_config.get("shaping_gamma", 0.99)
        phi_before = -coef * time_before
        phi_after = -coef * time_after
        return gamma * phi_after - phi_before

    def is_swampert_turn1_move_decision(self):
        """True iff the decision whose obs/mask were most recently
        returned (by reset() or step()) is Swampert's own turn-1 move
        decision. Public, unlike _last_decision itself -- meant for
        ppo_train_torch.py's attributed-forcing rollout loop, which needs
        to know when to consider substituting a forced action while still
        routing it through the network's own act_forced() (so the actor
        gets a genuine, correctly-attributed policy-gradient update),
        rather than the gradient-blind autofill mechanism
        (force_swampert_xspeed_turn1(_prob)) above, which was shown
        (2026-09-23, model_2026-09-23_xspeed-forcing-anneal.pt) to never
        move the actor's own preference at all: since an autofilled action
        never goes through net.act(), the actor's logits for that decision
        never receive a single gradient step favoring X Speed, no matter
        how long or how often forcing is applied."""
        if self._last_decision is None or self._battle is None:
            return False
        kind, payload = self._last_decision
        return kind == "move" and payload is self._party[0] and self._battle.turn_count == 0

    def is_wingull_turn1_move_decision(self):
        """True iff the decision whose obs/mask were most recently
        returned is Wingull's own turn-1 move decision -- used by
        ppo_train_torch.py's heal-band attributed-forcing mechanism
        (2026-09-26, --heal-band-forcing-prob), which nudges the Super-
        Potion-vs-no-heal choice at the narrow HP band where the two are
        nearly tied (see MODELS.md's HP=61 investigation: at HP 61 the
        model's own decision was measured at ~46% Super Potion vs. ~51%
        no-heal, an almost-exact tie that happens to land on the wrong
        side of a real win-rate dip)."""
        if self._last_decision is None or self._battle is None:
            return False
        kind, payload = self._last_decision
        return kind == "move" and payload is self._party[1] and self._battle.turn_count == 0

    def is_castform_move_decision(self):
        """True iff the decision whose obs/mask were most recently
        returned is Castform's own move decision, at ANY point in the
        episode (2026-09-25) -- unlike is_swampert_turn1_move_decision,
        not restricted to turn 0. Used by ppo_train_torch.py's raindance-
        avoid attributed-forcing mechanism, which only actually forces the
        FIRST such decision within a scenario-reset episode (tracked by
        the caller), since that's the only point the "Dive genuinely
        can't kill, Solrock charging Solar Beam" invariant is guaranteed
        to hold by construction."""
        if self._last_decision is None or self._battle is None:
            return False
        kind, payload = self._last_decision
        return kind == "move" and payload is self._party[3]

    def castform_rain_dance_action_indices(self):
        """Action indices (within the move-decision range) that decode to
        Castform using Rain Dance right now -- both target-choice bits,
        since Rain Dance is a field-effect move where the target slot is
        irrelevant to its actual effect. Used by ppo_train_torch.py's
        raindance-avoid attributed-forcing mechanism to know which
        actions to exclude when forcing a policy-preferred alternative."""
        castform = self._party[3]
        indices = []
        for slot_idx, ms in enumerate(castform.moves[:4]):
            if ms.move.name == "Rain Dance":
                indices.extend([slot_idx * 2, slot_idx * 2 + 1])
        return indices

    def is_swampert_move_decision(self):
        """True iff the decision whose obs/mask were most recently
        returned is Swampert's own move decision, at ANY point in the
        episode -- unlike is_swampert_turn1_move_decision, not restricted
        to turn 0. Used by ppo_train_torch.py's dive-forcing mechanism
        (2026-09-25), which forces only the FIRST such decision within a
        surf-no-kill-dive-kill scenario-reset episode, since that's the
        only point the scenario's own invariant is guaranteed to hold by
        construction."""
        if self._last_decision is None or self._battle is None:
            return False
        kind, payload = self._last_decision
        return kind == "move" and payload is self._party[0]

    def swampert_dive_solrock_action_idx(self):
        """Action index for Swampert using Dive against Solrock right
        now -- used by ppo_train_torch.py's dive-forcing mechanism to
        know which action to force. Computed dynamically (not a fixed
        constant) from the actual move-slot and foe-ordering state, the
        same way castform_rain_dance_action_indices() does, rather than
        hardcoding an index that would silently break if either ever
        changed."""
        swampert = self._party[0]
        solrock = self._battle.foe[0]
        slot_idx = next(i for i, ms in enumerate(swampert.moves[:4]) if ms.move.name == "Dive")
        foes = self._battle.alive(self._battle.opposing(swampert))
        target_choice = foes.index(solrock) if solrock in foes else 0
        return slot_idx * 2 + target_choice

    def _episode_stats(self):
        won = self._battle.winner() == "player"
        foe_fainted = sum(1 for f in self._battle.foe if f.is_fainted())
        foe_hp_frac = sum(f.hp_fraction() for f in self._battle.foe) / len(self._battle.foe)
        return {
            "won": won,
            "turns": self._battle.turn_count,
            "surf_count": self._surf_count,
            "potion_count": self._item_counts["Potion"],
            "super_potion_count": self._item_counts["Super Potion"],
            "hyper_potion_count": self._item_counts["Hyper Potion"],
            "full_heal_count": self._item_counts["Full Heal"],
            "time_estimate": self._battle.time_estimate,
            "foe_fainted_count": foe_fainted,
            "foe_hp_fraction_remaining": foe_hp_frac,
            "timed_out": (not self._battle.is_over()) and self._battle.turn_count >= self.turn_cap,
            "policy_chose_xspeed_turn1": self._policy_chose_xspeed_turn1,
            "xspeed_was_forced_this_episode": self._xspeed_forced_this_episode,
            "policy_turn1_action_idx": self._policy_turn1_action_idx,
        }


# ============================================================================
# REWARD_SHAPING_NOTE
# ============================================================================
# If sparse terminal-only reward makes early training too slow, a common
# fix is dense shaping: e.g. add a small multiple of
#   (damage dealt to foes this turn - damage taken this turn) / total HP
# as an additional per-step reward inside step(), on top of (not instead
# of) the terminal reward above. That's intentionally NOT enabled by
# default here since it changes the objective from what was specified;
# add it in _start_turn_resolution()/step() if needed, and consider
# annealing its weight toward 0 over training so the agent ultimately
# optimizes the true objective.
