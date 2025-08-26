import random
from collections import defaultdict
import numpy as np
import pandas as pd

# Load words from wordle.txt using pandas
word_df = pd.read_csv('wordle.txt', header=None, names=['word'])
WORDS = word_df['word'].tolist()

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
    def __init__(self, words, alpha=0.1, gamma=0.9, epsilon=0.3):
        self.words = words
        self.q = defaultdict(float)  # Q-values per word
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

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

# Terminal color codes
COLORS = {
    'G': '\033[42m',  # Green background
    'Y': '\033[43m',  # Yellow background
    'X': '\033[47m'   # Grey background
}
RESET = '\033[0m'

def colorize(guess, feedback):
    return ''.join(f"{COLORS[f]}{c.upper()}{RESET}" for c, f in zip(guess, feedback))


def play(agent):
    secret = random.choice(WORDS)
    candidates = WORDS.copy()
    print("Secret word selected. Start guessing!\n")

    # First guess is always 'slate'
    guess = "slate"
    feedback = compute_feedback(secret, guess)
    print(f"Guess 1: {colorize(guess, feedback)}")
    reward = 10 if guess == secret else -1
    candidates = [w for w in candidates if matches(w, guess, feedback) and w != guess]
    agent.update(guess, reward, candidates)
    if guess == secret:
        print("\nAI won!")
        return

    for turn in range(2, 6):
        guess = agent.choose(candidates)
        feedback = compute_feedback(secret, guess)
        print(f"Guess {turn}: {colorize(guess, feedback)}")
        reward = 10 if guess == secret else -1
        candidates = [w for w in candidates if matches(w, guess, feedback) and w != guess]
        agent.update(guess, reward, candidates)
        if guess == secret:
            print("\nAI won!")
            return
    print(f"\nAI lost! The word was {secret}.")


def main():
    agent = QLearningAgent(WORDS)
    train(agent)
    play(agent)

if __name__ == "__main__":
    main()