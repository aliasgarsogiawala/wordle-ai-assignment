from collections import Counter, defaultdict
import random

WORDS = [
    "raise","adieu","stare","stone","tears","rates","cigar","rebut","sissy",
    "humph","awake","blush","focal","evade","naval","serve","heath","dwarf",
    "model","karma","grade","pride","crane","slate","trace","alert","later"
]
ANSWERS_STACK = WORDS[:]  
ALLOWED_GUESSES = WORDS[:]  

SQ = {"G": "🟩", "Y": "🟨", "B": "⬛"}

def wordle_feedback(secret: str, guess: str) -> str:
    """Return 5 chars: G (green) / Y (yellow) / B (gray). Handles duplicates correctly."""
    fb = ["B"] * 5
    counts = Counter(secret)

    for i, (s, g) in enumerate(zip(secret, guess)):
        if g == s:
            fb[i] = "G"
            counts[g] -= 1

    for i, g in enumerate(guess):
        if fb[i] == "G":
            continue
        if counts[g] > 0:
            fb[i] = "Y"
            counts[g] -= 1

    return "".join(fb)

def filter_candidates(cands, guess, fb):
    """Keep only candidates that would yield this feedback if they were the secret."""
    return [w for w in cands if wordle_feedback(w, guess) == fb]

def score_words(cands):
    """
    Simple, clear scoring:
    - Count how often each letter appears at each position among remaining candidates.
    - Score = sum of positional frequencies for the word's letters.
    - Small bonus for unique letters (avoid repeats early).
    """
    pos_counts = [Counter() for _ in range(5)]
    for w in cands:
        for i, ch in enumerate(w):
            pos_counts[i][ch] += 1

    overall = Counter("".join(cands))

    def score(w):
        s = 0
        for i, ch in enumerate(w):
            s += pos_counts[i][ch]
        s += sum(overall[ch] for ch in set(w)) * 0.05
        s -= (len(w) - len(set(w))) * 2
        return s

    return score

def pick_next_guess(cands, allowed, tried):
    """Pick the allowed word with the best frequency score (skip already tried)."""
    if len(cands) == 1:
        return cands[0]
    scorer = score_words(cands)
    best = None
    best_score = -1
    for w in allowed:
        if w in tried:
            continue
        s = scorer(w)
        if s > best_score:
            best, best_score = w, s
    return best or random.choice(cands)

def show_row(guess, fb):
    print(" ".join(list(guess.upper())), "  ", "".join(SQ[c] for c in fb))

def play_auto(secret=None, tries=10, verbose=True):
    """
    Auto-play like real Wordle:
    - The AI guesses, sees feedback, prints the board, and learns.
    - Up to `tries` attempts (default 10 per your request).
    """
    secret = secret or random.choice(ANSWERS_STACK)
    candidates = ANSWERS_STACK[:]    
    tried = []
    history = []

    opener = "raise" if "raise" in ALLOWED_GUESSES else ALLOWED_GUESSES[0]

    for step in range(1, tries + 1):
        guess = opener if step == 1 else pick_next_guess(candidates, ALLOWED_GUESSES, tried)
        tried.append(guess)

        fb = wordle_feedback(secret, guess)
        history.append((guess, fb))

        if verbose:
            print(f"\nStep {step}: Guess -> {guess.upper()}")
            show_row(guess, fb)
            candidates = filter_candidates(candidates, guess, fb)
            print(f"Learned: {fb.replace('G','G').replace('Y','Y').replace('B','B')} "
                  f"| Remaining candidates: {len(candidates)}")
        else:
            candidates = filter_candidates(candidates, guess, fb)

        if fb == "GGGGG":
            if verbose:
                print(f"\nSolved in {step} steps! Secret = {secret.upper()}")
            return True, history

    if verbose:
        print(f"\nOut of tries. Secret was: {secret.upper()}")
    return False, history

if __name__ == "__main__":
    solved, hist = play_auto(secret="plant", tries=10, verbose=True)
