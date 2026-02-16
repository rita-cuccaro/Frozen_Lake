import gymnasium as gym
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------------
# CONFIGURAZIONE CLIENT LLM
# ----------------------------------
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="meta-llama-3-8b-instruct"
)

# ----------------------------------
# CREAZIONE ENVIRONMENT
# ----------------------------------
def create_env(map_name="4x4"):
    return gym.make("FrozenLake-v1",
                    map_name=map_name,
                    is_slippery=True,
                    success_rate=1.0/3.0,
                    reward_schedule=(1,0,0))

# ----------------------------------
# MAPPATURA AZIONI
# ----------------------------------
ACTION_MAP = {"LEFT":0, "DOWN":1, "RIGHT":2, "UP":3} # mappa nomi azioni in indici numerici
ACTIONS = list(ACTION_MAP.keys())

# ----------------------------------
# LLM REWARD CACHE
# ----------------------------------
LLM_CACHE = {} # memorizza il reward suggerito dall'LLM per coppia (stato, azione)

# ----------------------------------
# UTILS
# ----------------------------------
# per convertire stato numerico in coordinate sulla mappa
def state_to_pos(state, ncols):
    return state // ncols, state % ncols

def build_llm_prompt(state, env):
    n = env.unwrapped.ncol
    r,c = state_to_pos(state,n)
    prompt = f"""
FrozenLake {n}x{n} (slippery)
Legend: S=start, F=frozen, H=hole, G=goal

Current position: ({r},{c})

Choose ONE action that seems safe.
Respond ONLY with LEFT, DOWN, RIGHT, or UP.
"""
    return prompt.strip()

# funzione llm reward
def llm_reward(state, action_text, env):
    key = (state, action_text)
    # controlla se il reward llm è già in cache
    if key in LLM_CACHE:
        return LLM_CACHE[key]

    # chiamata llm
    prompt = build_llm_prompt(state, env)
    response = client.chat.completions.create(
        model="meta-llama-3-8b-instruct",
        messages=[{"role":"system","content":"You are an expert at Frozen Lake."},
                  {"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=5
    )

    # chiede all'llm di suggerire un azione 
    suggested_action = response.choices[0].message.content.strip().upper()
    reward = 1.0 if action_text == suggested_action else 0.0 # reward=1 se l'azione coincide con quella suggerita dall'llm, altrimenti 0
    LLM_CACHE[key] = reward
    return reward

# ----------------------------------
# GRAFICI
# ----------------------------------
# serve per valutare se e quanto l’agente migliora nel tempo
def plot_training_progress(success_history, window=50, title="Success Rate per Episodio"):
    ma = [np.mean(success_history[i:i+window]) for i in range(len(success_history)-window+1)]
    plt.figure(figsize=(10,5))
    plt.plot(success_history, alpha=0.3, label="Successo episodio")
    plt.plot(range(window-1, len(success_history)), ma, color='red', linewidth=2, label=f"Media mobile {window}")
    plt.xlabel("Episodio")
    plt.ylabel("Reward / Successo")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# serve per capire quali stati sono più sicuri o vantaggiosi
def plot_q_heatmap(Q, title="Q-table Heatmap"):
    max_q = np.max(Q, axis=1)  # max valore Q per stato
    n = int(np.sqrt(len(max_q)))
    if n*n == len(max_q):  # solo se mappa quadrata
        grid = max_q.reshape((n,n))
        plt.figure(figsize=(5,5))
        plt.imshow(grid, cmap="viridis", aspect="equal")
        plt.colorbar(label="Q massimo per stato")
        plt.title(title)
        plt.show()

# serve per capire l'efficienza del comportamento dell'agente
def plot_steps_distribution(steps_list, title="Distribuzione passi per episodio"):
    plt.figure(figsize=(8,5))
    plt.hist(steps_list, bins=30, edgecolor="black")
    plt.xlabel("Numero di passi")
    plt.ylabel("Frequenza")
    plt.title(title)
    plt.grid(True)
    plt.show()

# per confronto finale
def plot_success_bar(success_rate, agent_name="RL + LLM", map_name="4x4"):
    plt.figure(figsize=(6,4))
    plt.bar([agent_name], [success_rate])
    plt.ylabel("Success rate")
    plt.title(f"{agent_name} – Success rate finale ({map_name})")
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.show()

# ----------------------------------
# TRAINING RL CON REWARD LLM
# ----------------------------------
def train_rl_llm_fast(n_episodes, map_name="4x4",
                      alpha=0.4, gamma=0.95,
                      epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                      max_steps_per_episode=50):
    env = create_env(map_name)
    Q = np.zeros((env.observation_space.n, env.action_space.n)) # inizializza Q-table a 0
    epsilon = epsilon_start
    success = 0
    success_history = [] 

    with tqdm(total=n_episodes, desc=f"Training RL + LLM reward ({map_name})") as pbar:
        for ep in range(n_episodes):
            state,_ = env.reset()
            done = False
            step_count = 0
            ep_success = 0

            while not done and step_count < max_steps_per_episode:
                # epsilon-greedy
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q[state])

                next_state, env_reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # reward extra dall'LLM
                action_text = list(ACTION_MAP.keys())[action]
                r_llm = llm_reward(state, action_text, env)
                total_reward = env_reward + 0.3*r_llm  # LLM reward più leggero. Ponderato con 0.3 cosi non domina il reward originale

                # aggiornamento Q-learning
                Q[state, action] += alpha*(total_reward + gamma*np.max(Q[next_state]) - Q[state, action])
                state = next_state
                step_count += 1

                if env_reward == 1:
                    ep_success = 1  # goal raggiunto

            success += ep_success
            success_history.append(ep_success)
            epsilon = max(epsilon_min, epsilon*epsilon_decay)
            pbar.update(1)
            pbar.set_postfix({
                "success_rate": f"{success/(ep+1)*100:.1f}%",
                "epsilon": f"{epsilon:.2f}"
            })
    return Q, success_history

# ----------------------------------
# EVALUATION
# ----------------------------------
def evaluate(Q, n_episodes=500, map_name="4x4", max_steps=50):
    env = create_env(map_name)
    success = 0
    steps_list = []

    for _ in range(n_episodes):
        state,_ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        success += reward
        steps_list.append(steps)

    success_rate = success / n_episodes

    print(f"\n RL + LLM reward evaluation ({map_name}) ")
    print(f"Success rate: {success/n_episodes*100:.2f}%")
    print(f"Average steps: {np.mean(steps_list):.2f}")

    return success_rate, steps_list

# ----------------------------------
# MAIN
# ----------------------------------
if __name__ == "__main__":

    Q4, hist4 = train_rl_llm_fast(n_episodes=3000, map_name="4x4")
    rate4, steps4 = evaluate(Q4, n_episodes=500, map_name="4x4")

    plot_training_progress(hist4, title="4x4 - RL + LLM Success Rate")
    plot_q_heatmap(Q4, title="4x4 - Q-table Heatmap")
    plot_steps_distribution(steps4, title="4x4 - Distribuzione passi")
    plot_success_bar(rate4, agent_name="RL + LLM reward", map_name="4x4")

    Q8, hist8 = train_rl_llm_fast(n_episodes=5000, map_name="8x8", max_steps_per_episode=100)
    rate8, steps8 = evaluate(Q8, n_episodes=500, map_name="8x8", max_steps=100)

    plot_training_progress(hist8, title="8x8 - RL + LLM Success Rate")
    plot_q_heatmap(Q8, title="8x8 * Q-table Heatmap")
    plot_steps_distribution(steps8, title="8x8 - Distribuzione passi")
    plot_success_bar(rate8, agent_name="RL + LLM reward", map_name="8x8")

