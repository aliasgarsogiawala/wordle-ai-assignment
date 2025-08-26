import os
import json
import time
import random
from datetime import datetime
from collections import defaultdict
import numpy as np
import urllib.request
import urllib.error

WORDS_URL = 'https://raw.githubusercontent.com/tabatkins/wordle-list/main/words'

def load_words() -> list[str]:
    # Try remote list first
    try:
        with urllib.request.urlopen(WORDS_URL, timeout=10) as resp:
            data = resp.read().decode('utf-8', errors='ignore')
        words = [w.strip().lower() for w in data.splitlines() if len(w.strip()) == 5 and w.strip().isalpha()]
        # dedupe, preserve order
        return list(dict.fromkeys(words))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        pass
    # Fallback to local file if present
    if os.path.exists('wordle.txt'):
        try:
            with open('wordle.txt', 'r', encoding='utf-8') as f:
                words = [w.strip().lower() for w in f if len(w.strip()) == 5 and w.strip().isalpha()]
                return list(dict.fromkeys(words))
        except Exception:
            pass
    # Minimal fallback
    return [
        'crane','slate','flint','pride','crown','glare','shine','grape','stone','brink','apple'
    ]

WORDS = load_words()

# Environment for Wordle
def compute_feedback(secret, guess):
    """Return feedback string using G (green), Y (yellow), X (gray)."""
    feedback = []
    secret_counts = {}
    for s, g in zip(secret, guess):
        if s == g:
            feedback.append('G')
        else:
            feedback.append(None)
            secret_counts[s] = secret_counts.get(s, 0) + 1
    for i, (s, g) in enumerate(zip(secret, guess)):
        if feedback[i] is None:
            if g in secret_counts and secret_counts[g] > 0:
                feedback[i] = 'Y'
                secret_counts[g] -= 1
            else:
                feedback[i] = 'X'
    return ''.join(feedback)

def matches(word, guess, feedback):
    """Return True if `word` would yield the same feedback for `guess`."""
    return compute_feedback(word, guess) == feedback

class QLearningAgent:
    def __init__(self, words, alpha=0.1, gamma=0.9, epsilon=0.3, persist_path: str = "q_values.json"):
        self.words = words
        self.q = defaultdict(float)  # Q-values per word
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.persist_path = persist_path
        # Load persisted Q-table if available
        self._load()

    def choose(self, candidates):
        if not candidates:
            candidates = self.words
        if np.random.rand() < self.epsilon:
            return random.choice(candidates)
        q_vals = np.array([self.q[w] for w in candidates])
        return candidates[int(np.argmax(q_vals))]

    def update(self, action, reward, next_candidates):
        max_q = max([self.q[w] for w in next_candidates], default=0.0)
        self.q[action] += self.alpha * (reward + self.gamma * max_q - self.q[action])

    def _load(self):
        try:
            if os.path.exists(self.persist_path):
                with open(self.persist_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Restore into defaultdict(float)
                self.q = defaultdict(float, {k: float(v) for k, v in data.get('q', {}).items()})
                # Optionally restore hyperparams
                self.alpha = float(data.get('alpha', self.alpha))
                self.gamma = float(data.get('gamma', self.gamma))
                self.epsilon = float(data.get('epsilon', self.epsilon))
        except Exception:
            # Ignore corrupt file
            pass

    def save(self):
        try:
            data = {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'q': dict(self.q),
            }
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception:
            pass


def train(agent, episodes=100):
    for _ in range(episodes):
        secret = random.choice(WORDS)
        candidates = WORDS.copy()

        # Always open with 'slate'
        guess = "slate"
        feedback = compute_feedback(secret, guess)
        reward = 10 if guess == secret else -1
        candidates = [w for w in candidates if matches(w, guess, feedback) and w != guess]
        agent.update(guess, reward, candidates)
        if guess == secret:
            continue

        for _ in range(4):
            guess = agent.choose(candidates)
            feedback = compute_feedback(secret, guess)
            reward = 10 if guess == secret else -1
            candidates = [w for w in candidates if matches(w, guess, feedback) and w != guess]
            agent.update(guess, reward, candidates)
            if guess == secret:
                break
    # Persist learned Q-values after training
    agent.save()

# Terminal color codes
COLORS = {
    'G': '\033[42m',  # Green background
    'Y': '\033[43m',  # Yellow background
    'X': '\033[47m'   # Grey background
}
RESET = '\033[0m'

def colorize(guess, feedback):
    return ''.join(f"{COLORS[f]}{c.upper()}{RESET}" for c, f in zip(guess, feedback))


def play(agent, delay_seconds: float = 2.0):
        secret = random.choice(WORDS)
        candidates = WORDS.copy()
        print("Secret word selected. Start guessing!\n")

        # Session log for frontend/reporting
        session = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'secret': secret,  # kept for report; not printed live
                'steps': [],
                'won': False,
        }

        # First guess is always 'slate'
        turn = 1
        guess = "slate"
        feedback = compute_feedback(secret, guess)
        print(f"Guess {turn}: {colorize(guess, feedback)}  (candidates: {len(candidates)})")
        reward = 10 if guess == secret else -1
        candidates = [w for w in candidates if matches(w, guess, feedback) and w != guess]
        agent.update(guess, reward, candidates)
        session['steps'].append({
                'turn': turn,
                'guess': guess,
                'feedback': feedback,
                'candidates_after': len(candidates),
                'reward': reward,
                'q_value': agent.q[guess],
        })
        if guess == secret:
                session['won'] = True
                print("\nAI won!")
                _persist_session(session)
                agent.save()
                _generate_html_report(session)
                return
        time.sleep(delay_seconds)

        for turn in range(2, 6 ):
                guess = agent.choose(candidates)
                feedback = compute_feedback(secret, guess)
                print(f"Guess {turn}: {colorize(guess, feedback)}  (candidates: {len(candidates)})")
                reward = 10 if guess == secret else -1
                candidates = [w for w in candidates if matches(w, guess, feedback) and w != guess]
                agent.update(guess, reward, candidates)
                session['steps'].append({
                        'turn': turn,
                        'guess': guess,
                        'feedback': feedback,
                        'candidates_after': len(candidates),
                        'reward': reward,
                        'q_value': agent.q[guess],
                })
                if guess == secret:
                        session['won'] = True
                        print("\nAI won!")
                        _persist_session(session)
                        agent.save()
                        _generate_html_report(session)
                        return
                if turn < 6:
                        time.sleep(delay_seconds)
        print(f"\nAI lost! The word was {secret}.")
        _persist_session(session)
        agent.save()
        _generate_html_report(session)


def _persist_session(session: dict):
        """Write session to last_run.json and append to logs/session-*.jsonl."""
        try:
                with open('last_run.json', 'w', encoding='utf-8') as f:
                        json.dump(session, f)
        except Exception:
                pass
        try:
                os.makedirs('logs', exist_ok=True)
                stamp = datetime.utcnow().strftime('%Y%m%d')
                path = os.path.join('logs', f'session-{stamp}.jsonl')
                with open(path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(session) + '\n')
        except Exception:
                pass


def _generate_html_report(session: dict):
        """Generate a standalone HTML file with inline data to present the run visually."""
        html = f"""
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Wordle RL — Last Run</title>
    <style>
        body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 24px; color: #222; }}
        h1 {{ margin-top: 0; }}
        .row {{ display: flex; gap: 8px; margin: 8px 0; }}
        .tile {{ width: 44px; height: 44px; display: grid; place-items: center; font-weight: 700; color: #fff; border-radius: 6px; }}
        .G {{ background: #2e7d32; }}
        .Y {{ background: #f9a825; }}
        .X {{ background: #9e9e9e; }}
        .meta {{ color: #555; font-size: 14px; }}
        .controls {{ margin: 16px 0; display: flex; gap: 8px; align-items: center; }}
        button {{ padding: 8px 12px; border: 1px solid #ccc; background: #fafafa; border-radius: 6px; cursor: pointer; }}
        button:hover {{ background: #f0f0f0; }}
        .hidden-secret {{ filter: blur(6px); }}
    </style>
    <script>const data = {json.dumps(session)};</script>
    <script>
        let idx = 0; let timer = null; let delay = 2000;
        function render() {{
            const c = document.getElementById('content');
            c.innerHTML = '';
            const h = document.createElement('div');
            h.className = 'meta';
            h.innerHTML = `Won: <b>${{data.won}}</b> &nbsp;|&nbsp; Steps: <b>${{data.steps.length}}</b> &nbsp;|&nbsp; Secret: <span id='secret' class='hidden-secret'>${{data.secret.toUpperCase()}}</span>`;
            c.appendChild(h);
            data.steps.slice(0, idx).forEach(step => {{
                const row = document.createElement('div'); row.className = 'row';
                const meta = document.createElement('div'); meta.className = 'meta';
            meta.textContent = `Guess ${{step.turn}} — candidates left: ${{step.candidates_after}} — reward: ${{step.reward}} — Q(${{step.guess}})=${{step.q_value.toFixed(3)}}`;
                row.appendChild(meta);
                const row2 = document.createElement('div'); row2.className = 'row';
                for (let i = 0; i < step.guess.length; i++) {{
                    const d = document.createElement('div');
                    d.className = 'tile ' + step.feedback[i];
                    d.textContent = step.guess[i].toUpperCase();
                    row2.appendChild(d);
                }}
                c.appendChild(row);
                c.appendChild(row2);
            }});
        }}
        function play() {{ if (timer) return; timer = setInterval(() => {{ if (idx < data.steps.length) {{ idx++; render(); }} else {{ stop(); reveal(); }} }}, delay); }}
        function pause() {{ if (!timer) return; clearInterval(timer); timer = null; }}
        function reset() {{ pause(); idx = 0; render(); hide(); }}
        function stepOnce() {{ if (idx < data.steps.length) {{ idx++; render(); }} else {{ reveal(); }} }}
        function faster() {{ delay = Math.max(300, delay - 300); if (timer) {{ pause(); play(); }} document.getElementById('spd').textContent = (delay/1000).toFixed(1)+'s'; }}
        function slower() {{ delay = Math.min(5000, delay + 300); if (timer) {{ pause(); play(); }} document.getElementById('spd').textContent = (delay/1000).toFixed(1)+'s'; }}
        function reveal() {{ document.getElementById('secret').classList.remove('hidden-secret'); }}
        function hide() {{ document.getElementById('secret').classList.add('hidden-secret'); }}
        window.onload = () => {{ render(); }}
    </script>
    </head>
    <body>
        <h1>Wordle RL — Last Run</h1>
        <div class="controls">
            <button onclick="play()">Play</button>
            <button onclick="pause()">Pause</button>
            <button onclick="stepOnce()">Step</button>
            <button onclick="reset()">Reset</button>
            <button onclick="slower()">-</button>
            <span>Speed: <span id="spd">2.0s</span></span>
            <button onclick="faster()">+</button>
            <button onclick="reveal()">Reveal Secret</button>
        </div>
        <div id="content"></div>
    </body>
</html>
"""
        try:
                with open('report_last_run.html', 'w', encoding='utf-8') as f:
                        f.write(html)
        except Exception:
                pass


def main():
    agent = QLearningAgent(WORDS)
    train(agent)
    play(agent, delay_seconds=2.0)

if __name__ == "__main__":
    main()