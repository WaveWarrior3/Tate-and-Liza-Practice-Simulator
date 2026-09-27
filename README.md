# Tate & Liza Practice Tool

A local webpage for practicing the Tate & Liza fight yourself, with the
ability to ask the trained model (`best_model_visible-hp.pt`) what it
would do at any decision point before you commit to your own choice.
This is a study tool, not an autoplay tool — the model never plays for
you, it only advises when asked.

## Setup

```bash
pip install -r requirements.txt
python3 app.py
```

Then open <http://127.0.0.1:5000/> in a browser.

I am not interested in maintaining/updating this tool, so anyone can steal/fork it and do whatever you want.

## What it does

- Start a battle with a chosen Swampert starting HP (or full HP) and an
  optional RNG seed.
- Make every decision yourself — moves, items, targets, replacements —
  exactly as in a real fight. The opponent AI plays normally.
- At any decision, click "Ask TNLbot What It Would Do" to see its full
  ranked probability distribution over every legal action here, then
  still pick anything you want.
- The opponent's HP is shown only as a bar (with the same green/yellow/red
  banding the real game uses), never an exact number — matching exactly
  what the model itself is trained to see. Your own party's HP is shown
  exactly, matching the real game's own-side display. This means you
  can't accidentally practice with more information than a real player
  has.
- A running, scrollable battle log shows exactly what happened each turn.
- At the end of a battle, you'll see how often your own choice matched
  the model's own top pick — a cheap signal for tracking how your
  intuition is developing across sessions.

## Notes / limitations

- **Single-user, local only.** Battles are held in server memory, keyed
  by a browser session cookie. There's no database and no
  multi-process/multi-worker story — restarting the server drops any
  in-progress battle. This is intentional for a personal practice tool,
  not a design gap to fix later.
- **The model is fixed** to `models/best_model_visible-hp.pt` (this
  directory's own copy — see "Standalone / vendoring" below; override via
  the `TNL_MODEL_PATH` environment variable to point at a different
  checkpoint file, e.g. a copy of `best_model_healband.pt` — note that
  any exact-HP-trained checkpoint should be evaluated with
  `foe_hp_visible=False`, which currently requires editing
  `battle_engine.py`'s `FOE_HP_VISIBLE` constant, not just the env var).
- Set `TNL_WEBAPP_SECRET` to a fixed value if you want session cookies
  (and therefore in-progress battles) to survive a server restart across
  runs on the same machine; otherwise a new random key is generated every
  time the server starts, which invalidates old sessions.

