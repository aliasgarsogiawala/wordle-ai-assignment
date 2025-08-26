# **Wordle RL Agent - README**

## **Overview**

This project implements a **Reinforcement Learning (RL) agent** that plays **Wordle**, a popular word-guessing game. The agent learns to guess the secret word by receiving feedback after each guess and updates its **Q-values** based on that feedback. It explores possible words, learns from past attempts, and improves its guesses over time.

## **How the Project Works**

### **1. Word List Loading (`load_words()`)**

* The agent begins by **fetching a list of 5-letter words** from a GitHub repository (`https://raw.githubusercontent.com/tabatkins/wordle-list/main/words`).
* If it cannot fetch the word list (e.g., offline), it falls back to a **local file (`wordle.txt`)** containing words or defaults to a small built-in word list.
* The list is filtered to ensure that only valid 5-letter words are included.

### **2. Wordle Feedback (`compute_feedback()`)**

* After each guess, the agent compares the guess with the **secret word** and computes feedback based on the Wordle rules:

  * **Green (`G`)**: Correct letter, correct position.
  * **Yellow (`Y`)**: Correct letter, wrong position.
  * **Gray (`X`)**: Incorrect letter.
* This feedback helps the agent narrow down possible candidate words for future guesses.

### **3. Matching Words (`matches()`)**

* The `matches()` function checks if a given word produces the same feedback when compared to the agent's guess.
* This function is essential for filtering out invalid candidate words that do not match the given feedback.

### **4. QLearningAgent Class**

The `QLearningAgent` class is the core of the RL agent. It utilizes **Q-learning** to choose guesses and learn from feedback.

* **Q-values**: These represent the value of each word in terms of how useful it is for guessing the secret word. The agent updates these values after each guess.

* **Exploration vs Exploitation**: The agent uses an **epsilon-greedy strategy** to balance exploration (trying new words) and exploitation (choosing high-value words based on previous experience).

* **Alpha (learning rate)**: Determines how much new information influences the Q-values.

* **Gamma (discount factor)**: How much future rewards are considered when updating Q-values.

* **Epsilon (exploration factor)**: Probability of choosing a random word rather than the word with the highest Q-value.

```python
class QLearningAgent:
    def __init__(self, words, alpha=0.1, gamma=0.9, epsilon=0.3, persist_path: str = "q_values.json"):
        self.words = words
        self.q = defaultdict(float)  
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.persist_path = persist_path
        self._load()
```

### **5. Training the Agent (`train()`)**

* The agent undergoes **training** via self-play before starting a real game.
* During training, the agent:

  * Randomly selects a secret word.
  * Makes a guess (starting with “slate”).
  * Updates its **Q-values** based on feedback (reward of +10 for a correct guess and -1 for an incorrect guess).
  * Filters out candidate words based on feedback to refine future guesses.

```python
def train(agent, episodes=100):
    for _ in range(episodes):
        secret = random.choice(WORDS)
        candidates = WORDS.copy()
        guess = "slate"
        feedback = compute_feedback(secret, guess)
        reward = 10 if guess == secret else -1
        candidates = [w for w in candidates if matches(w, guess, feedback) and w != guess]
        agent.update(guess, reward, candidates)
```

### **6. Playing the Game (`play()`)**

* After training, the agent plays a real game:

  * It guesses the secret word using its learned **Q-values**.
  * It prints the feedback in **colored Wordle format** (`G/Y/X`).
  * The agent continues guessing until it either guesses correctly or exhausts all 6 attempts.

```python
def play(agent, delay_seconds: float = 2.0):
    secret = random.choice(WORDS)
    candidates = WORDS.copy()
    print("Secret word selected. Start guessing!\n")
    session = {'timestamp': datetime.utcnow().isoformat() + 'Z', 'secret': secret, 'steps': [], 'won': False}
    ...
    time.sleep(delay_seconds)
```

### **7. Persistence and Reporting**

* After each game, the **Q-table** is saved to a **JSON file** (`q_values.json`) so that the agent’s learning can persist across runs.
* The session details (including guesses and feedback) are saved to **`last_run.json`** and a **log file** (`logs/session-*.jsonl`).
* A **HTML report** is generated for each game to visually replay the agent’s guesses, feedback, and the result.

```python
def _persist_session(session: dict):
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
```

### **8. Running the Program**

The `main()` function runs the agent:

1. **Training**: The agent is trained on a few episodes to learn how to guess words.
2. **Playing**: After training, the agent plays a real Wordle game.

```python
def main():
    agent = QLearningAgent(WORDS)
    train(agent)
    play(agent, delay_seconds=2.0)

if __name__ == "__main__":
    main()
```
