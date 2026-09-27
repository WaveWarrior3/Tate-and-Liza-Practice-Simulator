"""battle_engine.py -- the practice-tool's battle logic, kept separate from
app.py's Flask routes so the underlying engine can be reused by a future
richer frontend (the project's eventual image/GUI version) without
touching the web-serving code.

Wraps vendor/tl_env.py's TateLizaEnv (a vendored copy of tl_ai/tl_env.py --
see README.md; NOT tl_sim.py's Battle directly):
TateLizaEnv's step()-per-decision interface is exactly the abstraction a
web server needs, since a mid-turn replacement decision can't block
synchronously on an HTTP request the way advisor.py's CLI version blocks
on input() -- TateLizaEnv already solves this (see CLAUDE.md's 2026-09-23
note on the resumable-generator fix), because it was built for the
identical constraint an external RL loop has: decisions arrive one at a
time via separate calls, not synchronous blocking. Reusing it here means
this webapp automatically inherits that fix rather than needing its own.

A full human-readable turn log is recovered by monkeypatching
`env._battle.log` right after reset() -- TateLizaEnv hardcodes
`log=lambda *a: None` internally (it doesn't need one for RL), but
`Battle.log` is just a plain callable attribute (`self.log = log`),
so reassigning it after construction is safe and doesn't touch any
other behavior.

The model is always the project's human-realistic checkpoint,
best_model_visible-hp.pt, evaluated with `foe_hp_visible=True` -- the
observation it actually trained on. This also drives the practice
tool's own opponent-HP display: the human player never gets told the
foe's exact current_hp, only the same 48-pixel-bar-derived visual the
model itself receives (see tl_env.py's visible_hp_fraction /
pokeruby's GetScaledHPFraction), so practicing here can't cheat via
information a real player wouldn't have. The player's OWN party HP is
always shown exactly, matching the real game.
"""
import contextlib
import os
import sys
import random as _random

# Standalone: everything this module needs lives inside webapp/ itself --
# vendor/ holds snapshot copies of tl_sim.py/tl_env.py plus a trimmed
# inference-only policy loader, and models/ holds a copy of the checkpoint.
# See README.md's "Standalone / vendoring" section for why, and how to
# refresh vendor/ after the canonical project files change.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(THIS_DIR, "vendor")
if VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

import numpy as np
import tl_sim as sim
from tl_env import (TateLizaEnv, move_action_mask, replacement_action_mask,
                     decode_move_action, decode_replacement_action, visible_hp_fraction,
                     X_ITEMS)
from actor_critic import load_policy

# Category labels for grouping the option buttons in the UI, in display
# order. "replace" decisions (send-out choices) aren't move decisions at
# all, so they're never grouped -- see current_decision().
CATEGORY_ORDER = ["move", "heal_item", "x_item"]
CATEGORY_LABELS = {"move": "Moves", "heal_item": "Healing Items", "x_item": "X Items"}

MODEL_PATH = os.environ.get(
    "TNL_MODEL_PATH", os.path.join(THIS_DIR, "models", "best_model_visible-hp.pt"))
FOE_HP_VISIBLE = True  # always -- this is the whole point of this model/tool
TURN_CAP = 10  # matches this project's speedrun-oriented default since 2026-09-20

_probe = TateLizaEnv()
OBS_DIM, ACTION_DIM = _probe.obs_dim, _probe.action_dim
NET = load_policy(MODEL_PATH, OBS_DIM, ACTION_DIM)


def _describe_move_action(battle, mon, party, action_idx):
    act = decode_move_action(battle, mon, party, action_idx)
    if act["kind"] == "item":
        return f"{act['item']} on {act['item_target'].name}"
    target_str = f" -> {act['target'].name}" if act["target"] is not None else ""
    return f"{act['move'].name}{target_str}"


def _categorize_move_action(battle, mon, party, action_idx):
    """"move" (an attack/status move), "heal_item" (Potion/Super Potion/
    Hyper Potion/Full Heal), or "x_item" (X Attack/X Special/X Speed)."""
    act = decode_move_action(battle, mon, party, action_idx)
    if act["kind"] != "item":
        return "move"
    return "x_item" if act["item"].lower() in X_ITEMS else "heal_item"


def _describe_replacement_action(action_idx, party):
    mon = decode_replacement_action(action_idx, party)
    return f"Send out {mon.name} ({mon.current_hp}/{mon.max_hp} HP)"


def hp_bar(mon, bar_width=20):
    """A human-authentic HP readout for an OPPONENT Pokemon: a text bar
    sized from visible_hp_fraction() (the same 48-pixel-bar quantization
    the model itself sees), plus the same green/yellow/red tier real Gen 3
    HP bars use (GetHPBarLevel's own thresholds) -- never an exact number,
    since a real player never sees one either."""
    if mon.is_fainted():
        return "[[ FAINTED ]]"
    frac = visible_hp_fraction(mon)
    filled = max(1, round(frac * bar_width)) if mon.current_hp > 0 else 0
    bar = "#" * filled + "-" * (bar_width - filled)
    pixels48 = round(frac * 48)
    if mon.current_hp == mon.max_hp:
        tier = "GREEN"
    elif pixels48 >= 25:
        tier = "GREEN"
    elif pixels48 >= 10:
        tier = "YELLOW"
    else:
        tier = "RED"
    return f"[{bar}] ({tier})"


def exact_hp(mon):
    if mon.is_fainted():
        return "0/%d HP (fainted)" % mon.max_hp
    return f"{mon.current_hp}/{mon.max_hp} HP"


@contextlib.contextmanager
def _trace_turn_actions(session):
    """Temporarily wraps tl_sim's module-level do_move/do_item so that
    EVERY Pokemon's action (player's own included, not just the foe's)
    executes into its own batch on `session` -- see
    PracticeSession._begin_batch/_end_batch -- instead of going straight
    to the visible log. Batches land in `pending_batches` in true
    chronological (speed-resolved) order, since do_move/do_item are
    called synchronously in that same order by tl_sim's turn-resolution
    loop. PracticeSession then auto-reveals PLAYER batches immediately
    (see _autoreveal_player_batches) while gating FOE batches behind an
    explicit "Continue" click.

    This used to only trace FOE actions, on the theory that the player's
    own actions could just be shown "instantly" since the player already
    chose them. That was a real bug (2026-09-27): a single submit() call
    can resolve several actions at once (once every active player mon's
    decision for the turn is in), and instantly-shown player actions
    would jump straight into the visible log ahead of already-executed
    but not-yet-revealed foe actions that actually happened EARLIER in
    true speed order -- e.g. a slower Castform's move appearing before
    an already-resolved-but-still-queued Solrock/Lunatone action. Tracing
    every actor and only auto-revealing (not skipping the queue for)
    player batches fixes this: nothing can be shown out of true
    chronological order, since a player batch can only auto-reveal once
    every earlier-queued batch (foe or player) ahead of it is gone.

    This is a genuine (if temporary and narrowly-scoped) monkeypatch of
    tl_sim.py's module-level functions, not a change to tl_sim.py itself
    -- deliberately NOT implemented by adding new pause points to
    tl_sim._resolve_turn_gen, since that generator is shared, carefully-
    tested core code for the whole RL pipeline (see its own docstring),
    and this project's history has a specific precedent (the mid-turn
    replacement-decision leak) for how easy it is to introduce a subtle
    bug by changing what it pauses on. Scoping the patch to only run
    during this one submit() call, and restoring the originals in a
    `finally`, keeps it invisible to every other caller (training,
    evaluate.py, the test suite) and safe even if this call raises."""
    orig_do_move = sim.do_move
    orig_do_item = sim.do_item

    def traced_do_move(battle, attacker, move, chosen_target, moveslot):
        session._begin_batch(attacker.label(), attacker.is_player)
        result = orig_do_move(battle, attacker, move, chosen_target, moveslot)
        session._end_batch()
        return result

    def traced_do_item(battle, mon, act):
        session._begin_batch(mon.label(), mon.is_player)
        result = orig_do_item(battle, mon, act)
        session._end_batch()
        return result

    sim.do_move = traced_do_move
    sim.do_item = traced_do_item
    try:
        yield
    finally:
        sim.do_move = orig_do_move
        sim.do_item = orig_do_item


class PracticeSession:
    """One in-progress (or finished) practice battle. Held server-side,
    keyed by a per-browser-session id in app.py -- never serialized, since
    it holds live TateLizaEnv/Battle objects."""

    def __init__(self, swampert_hp=None, seed=None):
        self.env = TateLizaEnv(turn_cap=TURN_CAP, foe_hp_visible=FOE_HP_VISIBLE)
        self.obs, self.mask = self.env.reset(swampert_hp=swampert_hp, seed=seed)
        self.revealed_history = []
        # Actions not yet shown to the player, in true chronological
        # (speed-resolved) order: a queue of (actor_label_or_None,
        # is_player, [log lines], battle-state-snapshot) 4-tuples. PLAYER
        # batches auto-drain from the front as soon as they're reached
        # (see _autoreveal_player_batches); FOE batches (and the final
        # result batch, actor_label=None) wait for an explicit "Continue"
        # click -- see reveal_next(). See _trace_turn_actions for why
        # EVERY action goes through this queue, not just foe ones.
        self.pending_batches = []
        self._current_actor = None
        self._current_is_player = None
        self._current_lines = None
        self.last_revealed_state = None  # set by the very first _log() call below
        self.env._battle.log = self._log
        self.done = False
        self.result_stats = None
        self.decisions_total = 0
        self.decisions_agreed = 0
        self.show_advice = False
        self._log(f"=== New battle: Swampert starting HP "
                   f"{self.env._party[0].current_hp}/{self.env._party[0].max_hp} ===")

    def _log(self, msg):
        """The Battle's log callback. While ANY action is being traced
        (_current_lines is set -- see _trace_turn_actions; this covers
        both player and foe actions now), lines are buffered into that
        pending batch instead of shown immediately. Otherwise (anything
        outside a traced action, like the opening banner or end-of-turn
        weather/residual-damage messages, which happen after the last
        action's batch has already closed) the line is revealed right
        away, and last_revealed_state is refreshed to match -- so it
        always reflects the true state as of everything actually shown
        to the player so far, whether that arrived directly here or via
        a batch being revealed (instantly for a player batch via
        _autoreveal_player_batches, or on a Continue click for a foe
        batch -- see reveal_next())."""
        if self._current_lines is not None:
            self._current_lines.append(msg)
        else:
            self.revealed_history.append(msg)
            self.last_revealed_state = self.state_lines()

    def _begin_batch(self, actor_label, is_player):
        self._current_actor = actor_label
        self._current_is_player = is_player
        self._current_lines = []

    def _end_batch(self):
        snapshot = self.state_lines()
        self.pending_batches.append(
            (self._current_actor, self._current_is_player, self._current_lines, snapshot))
        self._current_actor = None
        self._current_is_player = None
        self._current_lines = None

    def _autoreveal_player_batches(self):
        """Drains any PLAYER batches sitting at the FRONT of the queue
        straight into the visible transcript, with no Continue click
        needed -- the player already knows what they chose. Stops at the
        first FOE batch (or the final result batch, actor_label=None),
        which still requires an explicit click. Since batches are queued
        in true chronological order, a player batch can only ever drain
        here once everything that actually happened before it has
        already been revealed -- this is what keeps the displayed order
        matching true turn/speed order (see _trace_turn_actions)."""
        while self.pending_batches and self.pending_batches[0][1]:
            actor_label, _is_player, lines, snapshot = self.pending_batches.pop(0)
            if actor_label:
                self.revealed_history.append(f"--- {actor_label}'s turn ---")
            self.revealed_history.extend(lines)
            self.last_revealed_state = snapshot

    def reveal_next(self):
        """Pops and reveals the single next queued FOE action (or the
        final result batch): its log lines go into the visible
        transcript, and last_revealed_state advances to match -- so the
        displayed HP/status moves forward by exactly one action per
        click, in step with the log. Any player batches immediately
        following it are then auto-drained too (they don't need their
        own click), which can only be correct because they're now at the
        front of a chronologically-ordered queue."""
        if not self.pending_batches:
            return
        actor_label, _is_player, lines, snapshot = self.pending_batches.pop(0)
        if actor_label:
            self.revealed_history.append(f"--- {actor_label}'s turn ---")
        self.revealed_history.extend(lines)
        self.last_revealed_state = snapshot
        self._autoreveal_player_batches()

    def next_actor_label(self):
        """Who the next pending Continue click will reveal, for the
        button's own label -- None for the final battle-result reveal."""
        if not self.pending_batches:
            return None
        return self.pending_batches[0][0]

    # -- current decision -------------------------------------------------
    def _describe_fn(self):
        kind, payload = self.env._last_decision
        party = self.env._party
        battle = self.env._battle
        if kind == "move":
            return lambda a: _describe_move_action(battle, payload, party, a)
        return lambda a: _describe_replacement_action(a, party)

    def _category_fn(self):
        """"move"/"heal_item"/"x_item" for a move decision (see
        _categorize_move_action), or "replace" for a send-out decision --
        the latter is never actually grouped in the UI (there's only ever
        one kind of choice there), but it's tagged for consistency."""
        kind, payload = self.env._last_decision
        party = self.env._party
        battle = self.env._battle
        if kind == "move":
            return lambda a: _categorize_move_action(battle, payload, party, a)
        return lambda a: "replace"

    def _deduped_actions(self):
        """(representative_action_idx, description, category,
        aggregated_probability) tuples, sorted by descending probability.
        Spread moves like Surf occupy two functionally-identical action
        indices (one per target-choice bit); those are merged into a
        single entry here -- the FIRST matching index becomes that
        entry's representative index. This is the ONE place that dedup
        happens: current_decision (what's offered), advice (what's shown
        as the model's ranking), and model_top_pick (what a submitted
        action is compared against) all call this, so the representative
        index used for "did you match the model's top pick" is always the
        exact same index the option list actually offers -- previously
        model_top_pick() took a raw argmax over the un-deduplicated probs,
        which could land on the OTHER (unexposed) duplicate index for the
        same real choice purely from floating-point noise between two
        identical actions, making a genuinely-correct match register as a
        mismatch."""
        describe = self._describe_fn()
        categorize = self._category_fn()
        probs = NET.action_probs(self.obs, self.mask)
        order = {}
        category = {}
        agg = {}
        for a in np.flatnonzero(self.mask):
            desc = describe(int(a))
            order.setdefault(desc, int(a))
            category.setdefault(desc, categorize(int(a)))
            agg[desc] = agg.get(desc, 0.0) + float(probs[a])
        entries = [(order[desc], desc, category[desc], agg[desc]) for desc in order]
        entries.sort(key=lambda e: -e[3])
        return entries

    def current_decision(self):
        """Returns (kind, label, options_by_category), where
        options_by_category is a list of (category_label, [(form_value,
        description), ...]) pairs, in CATEGORY_ORDER, each already sorted
        by the model's own descending confidence -- see _deduped_actions
        for the dedup rule. A "replace" decision has exactly one category
        ("Send out"), since there's only one kind of choice there."""
        kind, payload = self.env._last_decision
        battle = self.env._battle
        if kind == "move":
            mon = payload
            if mon.charging_move is not None:
                return "forced", f"{mon.name} is locked into finishing {mon.charging_move.name}!", []
            label = f"{mon.name}'s turn"
        else:
            slot_index = payload
            label = f"{battle.player[slot_index].name} fainted -- choose a replacement"
        entries = self._deduped_actions()
        if kind == "move":
            grouped = []
            for cat in CATEGORY_ORDER:
                opts = [(idx, desc) for idx, desc, c, _ in entries if c == cat]
                if opts:
                    grouped.append((CATEGORY_LABELS[cat], opts))
        else:
            grouped = [("Send out", [(idx, desc) for idx, desc, _, _ in entries])]
        return kind, label, grouped

    def advice(self):
        """Ranked (description, probability) pairs -- see _deduped_actions
        for the dedup rule."""
        return [(desc, p) for _, desc, _, p in self._deduped_actions()]

    def model_top_pick(self):
        """The representative action index for the model's single highest-
        probability CHOICE (after merging duplicate indices for the same
        real action) -- directly comparable to a submitted action_idx,
        since both come from the same _deduped_actions() representative
        indices."""
        return self._deduped_actions()[0][0]

    def submit(self, action_idx):
        """Advances the battle by exactly one decision. If the decision
        was a forced charging-move continuation, no action is submitted
        (TateLizaEnv autofills these before ever surfacing a real
        decision, so this path is only reached for player choices).

        A single call can resolve an entire turn (once every active
        player Pokemon's decision for it is in) -- every action that
        resolves as part of that (player's and foe's alike) goes into
        pending_batches in true speed order (see _trace_turn_actions);
        player batches then auto-drain immediately (no Continue needed),
        while foe batches wait for one, without ever letting a later
        player action jump ahead of an earlier, still-unrevealed foe
        one."""
        describe = self._describe_fn()
        chosen_desc = describe(action_idx)
        top_desc = describe(self.model_top_pick())
        self.decisions_total += 1
        if chosen_desc == top_desc:
            self.decisions_agreed += 1
        with _trace_turn_actions(self):
            self.obs, reward, done, info, self.mask = self.env.step(action_idx)
        self._autoreveal_player_batches()
        self.show_advice = False
        if done:
            self.done = True
            self.result_stats = info["stats"]
            w = self.env._battle.winner()
            result_str = "WIN" if w == "player" else "LOSS" if w == "foe" else "DRAW/TIMEOUT"
            result_line = f"=== RESULT: {result_str} after {self.env._battle.turn_count} turns ==="
            # Queued (not logged immediately) so the result only appears
            # once the player has clicked through every action that led
            # to it, same as any other pending batch. is_player=False so
            # _autoreveal_player_batches never skips its Continue gate.
            self.pending_batches.append((None, False, [result_line], self.state_lines()))

    def state_lines(self):
        battle = self.env._battle
        lines = []
        for p in battle.player:
            status = f" [{p.status.upper()}]" if p.status != "none" else ""
            lines.append(f"  YOU: {p.name:10s} {exact_hp(p)}{status}")
        for f in battle.foe:
            status = f" [{f.status.upper()}]" if f.status != "none" else ""
            lines.append(f"  FOE: {f.name:10s} {hp_bar(f)}{status}")
        if battle.weather:
            lines.append(f"  Weather: {battle.weather} ({battle.weather_turns} turns left)")
        lines.append(f"  Turn {battle.turn_count + 1}")
        return lines
