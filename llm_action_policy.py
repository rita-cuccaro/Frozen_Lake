import gymnasium as gym
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------------
# CONFIGURAZIONE CLIENT LLM LOCALE
# ----------------------------------
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="meta-llama-3-8b-instruct"
)

# -----------------------------
# CREAZIONE ENVIRONMENT
# -----------------------------
def create_env(map_name="4x4"):
    return gym.make("FrozenLake-v1", map_name=map_name, is_slippery=True, success_rate=1.0/3.0, reward_schedule=(1, 0, 0))

# -----------------------------
# MAPPATURA AZIONI
# -----------------------------
ACTION_MAP = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3} # associa nomi delle azioni agli indici numerici
ACTIONS = list(ACTION_MAP.keys())

# -----------------------------
# UTILS
# -----------------------------
# per convertire lo stato numerico in coordinate (righe, colonne) sulla mappa
def state_to_position(state, ncols):
    return state // ncols, state % ncols

# per controllare i 4 vicini dello stato corrente
def get_adjacent_tiles(env, state):
    n = env.unwrapped.ncol
    r, c = state_to_position(state, n)
    directions = {"UP": (r-1, c), "DOWN": (r+1, c),
                  "LEFT": (r, c-1), "RIGHT": (r, c+1)}
    tiles = {}
    for a, (rr, cc) in directions.items():
        if 0 <= rr < n and 0 <= cc < n:
            tiles[a] = env.unwrapped.desc[rr, cc].decode("utf-8")
        else:
            tiles[a] = "WALL" # fuori mappa
    return tiles

# -----------------------------
# GRAFICI
# -----------------------------
# serve per valutare se e quanto l’agente migliora nel tempo
def plot_success_history(success_list, window=50, title="LLM Success Rate per Episodio"):
    if len(success_list) < window:
        ma = success_list
    else:
        ma = [np.mean(success_list[i:i+window]) for i in range(len(success_list)-window+1)]

    plt.figure(figsize=(10,5))
    plt.plot(success_list, alpha=0.3, label="Successo episodio")
    plt.plot(range(window-1, len(success_list)), ma, color='red', linewidth=2, label=f"Media mobile {window}")
    plt.xlabel("Episodio")
    plt.ylabel("Successo (0/1)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# serve per capire l'efficienza del comportamento dell'agente
def plot_steps_distribution(steps_list, title="Distribuzione Passi per Episodio"):
    plt.figure(figsize=(8,5))
    plt.hist(steps_list, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Numero di passi")
    plt.ylabel("Frequenza")
    plt.title(title)
    plt.grid(True)
    plt.show()

# per confronto finale
def plot_success_bar(success_rate, map_name="4x4"):
    plt.figure(figsize=(6,4))
    plt.bar(["LLM-action"], [success_rate], color='orange')
    plt.ylabel("Success rate")
    plt.title(f"LLM-action – Success rate finale ({map_name})")
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.show()

# -----------------------------
# COSTRUZIONE PROMPT LLM
# -----------------------------
def build_prompt(state, env):
    n = env.unwrapped.ncol
    r, c = state_to_position(state, n) # posizione corrente
    adjacent = get_adjacent_tiles(env, state) # tile adiacenti
    adjacent_info = "\n".join([f"{a}: {t}" for a, t in adjacent.items()])

    prompt = f"""
FrozenLake {n}x{n} (slippery)
Legend: S=start, F=frozen, H=hole, G=goal

Current position: ({r},{c})
Adjacent tiles:
{adjacent_info}

Choose ONE safe action to reach the goal.
- Avoid holes (H) and walls.
- Prefer frozen tiles (F) or the goal (G).
- Respond ONLY with LEFT, DOWN, RIGHT, or UP.
"""
    return prompt.strip()

# -----------------------------
# CACHE LLM
# -----------------------------
LLM_CACHE = {} # memorizza le azioni già decise per ogni stato così non si chiede ogni volta all'LLM

# per scegliere azione
def llm_choose_action(state, env, epsilon=0.05):
    # esplorazione casuale (epsilon-greedy)
    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)

    # se abbiamo già calcolato l'azione la riutilizziamo
    if state in LLM_CACHE:
        return LLM_CACHE[state]

    prompt = build_prompt(state, env)
    response = client.chat.completions.create(
        model="meta-llama-3-8b-instruct",
        messages=[{"role": "system", "content": "You are an expert at Frozen Lake."},
                  {"role": "user", "content": prompt}],
        temperature=0.0, # risposta deterministica
        max_tokens=5
    )

    action = response.choices[0].message.content.strip().upper() # pulisce la risposta LLM e la memorizza in cache
    if action not in ACTION_MAP:
        action = np.random.choice(ACTIONS) # sceglie casuale

    LLM_CACHE[state] = action
    return action

# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_llm(n_episodes=1000, map_name="4x4", epsilon=0.05):
    env = create_env(map_name)
    success_list = []
    steps_list = []

    with tqdm(total=n_episodes, desc=f"LLM Evaluation {map_name}") as pbar:
        for ep in range(n_episodes):
            state, _ = env.reset()
            done = False
            steps = 0
            success = 0

            while not done and steps < 100:
                action_text = llm_choose_action(state, env, epsilon) # llm decide azione
                action = ACTION_MAP[action_text]

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if reward == 1:
                    success = 1

                state = next_state
                steps += 1

            # aggiornamento statistiche per barra di progresso
            success_list.append(success)
            steps_list.append(steps)
            pbar.update(1)
            pbar.set_postfix({
                "success_rate": f"{np.mean(success_list)*100:.1f}%",
                "avg_steps": f"{np.mean(steps_list):.1f}"
            })

    success_rate = np.mean(success_list)
    print(f"\n LLM-only policy evaluation ({map_name}) ")
    print(f"Success rate: {np.mean(success_list)*100:.2f}%")
    print(f"Average steps: {np.mean(steps_list):.2f}") # episodi medi

    plot_success_history(success_list, window=50, title=f"{map_name} - Success Rate")
    plot_steps_distribution(steps_list, title=f"{map_name} - Distribuzione Passi")
    plot_success_bar(success_rate, map_name=map_name) 

    return success_list, steps_list, success_rate

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    success_4x4, steps_4x4, rate_4x4 = evaluate_llm(n_episodes=500, map_name="4x4", epsilon=0.05)
    success_8x8, steps_8x8, rate_8x8 = evaluate_llm(n_episodes=1000, map_name="8x8", epsilon=0.05)
