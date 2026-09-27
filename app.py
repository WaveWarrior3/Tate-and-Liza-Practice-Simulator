"""app.py -- Flask frontend for the Tate & Liza practice tool.

Run with:
    python3 app.py
then open http://127.0.0.1:5000/ .
"""
import os
import uuid

from flask import Flask, session, request, redirect, url_for, render_template, abort

import battle_engine

app = Flask(__name__)
app.secret_key = os.environ.get("TNL_WEBAPP_SECRET", os.urandom(24))

# In-memory battle store: {battle_id: PracticeSession}. Deliberately not
# persisted anywhere -- see module docstring.
SESSIONS = {}


def _get_session():
    battle_id = session.get("battle_id")
    if battle_id is None or battle_id not in SESSIONS:
        return None
    return SESSIONS[battle_id]


@app.route("/")
def new_battle_form():
    return render_template("new_battle.html", model_path=battle_engine.MODEL_PATH)


@app.route("/new", methods=["POST"])
def new_battle():
    hp_raw = request.form.get("swampert_hp", "").strip()
    seed_raw = request.form.get("seed", "").strip()
    swampert_hp = int(hp_raw) if hp_raw else None
    seed = int(seed_raw) if seed_raw else None
    if swampert_hp is not None and not (1 <= swampert_hp <= 133):
        abort(400, "Swampert HP must be between 1 and 133.")

    battle_id = str(uuid.uuid4())
    SESSIONS[battle_id] = battle_engine.PracticeSession(swampert_hp=swampert_hp, seed=seed)
    session["battle_id"] = battle_id
    return redirect(url_for("battle_view"))


@app.route("/battle")
def battle_view():
    sess = _get_session()
    if sess is None:
        return redirect(url_for("new_battle_form"))

    # Any foe action (or the final result) still waiting to be revealed
    # takes priority over everything else -- the player must click
    # through each one before seeing the next real decision or the
    # result screen. See battle_engine.PracticeSession.reveal_next.
    if sess.pending_batches:
        return render_template(
            "battle.html", sess=sess, revealing=True, done=False,
            state_lines=sess.last_revealed_state, history=sess.revealed_history,
            next_actor=sess.next_actor_label(),
        )

    if sess.done:
        return render_template(
            "battle.html", sess=sess, revealing=False, done=True,
            state_lines=sess.last_revealed_state, history=sess.revealed_history,
        )

    kind, label, option_groups = sess.current_decision()
    advice = sess.advice() if sess.show_advice else None
    return render_template(
        "battle.html", sess=sess, revealing=False, done=False,
        state_lines=sess.last_revealed_state, history=sess.revealed_history,
        decision_kind=kind, decision_label=label, option_groups=option_groups,
        advice=advice,
    )


@app.route("/battle/advise", methods=["POST"])
def battle_advise():
    sess = _get_session()
    if sess is None:
        return redirect(url_for("new_battle_form"))
    sess.show_advice = True
    return redirect(url_for("battle_view"))


@app.route("/battle/act", methods=["POST"])
def battle_act():
    sess = _get_session()
    if sess is None:
        return redirect(url_for("new_battle_form"))
    action_raw = request.form.get("action")
    if action_raw is None or not action_raw.isdigit():
        abort(400, "No action selected.")
    sess.submit(int(action_raw))
    return redirect(url_for("battle_view"))


@app.route("/battle/continue", methods=["POST"])
def battle_continue():
    sess = _get_session()
    if sess is None:
        return redirect(url_for("new_battle_form"))
    sess.reveal_next()
    return redirect(url_for("battle_view"))


@app.route("/battle/reset", methods=["POST"])
def battle_reset():
    """Abandons the current battle (in progress or finished) and sends
    the player back to the new-battle form -- available at any point
    during play, not just after a battle ends."""
    battle_id = session.pop("battle_id", None)
    SESSIONS.pop(battle_id, None)
    return redirect(url_for("new_battle_form"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
