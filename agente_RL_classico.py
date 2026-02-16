import gymnasium as gym
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# CREAZIONE ENVIRONMENT
# -----------------------------
def create_env(map_name="4x4", is_slippery=True):
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
    return env, env.observation_space.n, env.action_space.n

# -----------------------------
# TRAINING Q-LEARNING / SARSA
# -----------------------------
def train_agent(env, n_states, n_actions, episodes, method="q_learning",
                alpha=0.3, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995):

    Q = np.zeros((n_states, n_actions)) # creazione Q-table inizializzata a zero
    reward_history = []
    success_history=[]

    # scelta dell'azione usando epsilon-greedy
    def choose_action(state, epsilon):
        return env.action_space.sample() if random.random() < epsilon else np.argmax(Q[state])

    # ciclo sugli episodi
    for ep in tqdm(range(episodes), desc=f"Training {method}"):
        state, _ = env.reset()
        state = int(state)
        done = False
        total_reward = 0

        if method == "sarsa":
            action = choose_action(state, epsilon)

        while not done:
            if method == "q_learning":
                action = choose_action(state, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)
            done = terminated or truncated

            # aggiornamento Q-table
            if method == "q_learning":
                # off-policy: usa il max valore futuro tra tutte le azioni dello stato successivo
                td_target = reward + gamma * np.max(Q[next_state])
                Q[state, action] += alpha * (td_target - Q[state, action])
            else:  # SARSA
                # on-policy: usa l'azione effettivamente scelta per lo stato successivo
                next_action = choose_action(next_state, epsilon)
                td_target = reward + gamma * Q[next_state, next_action]
                Q[state, action] += alpha * (td_target - Q[state, action])
                action = next_action

            state = next_state
            total_reward += reward

        reward_history.append(total_reward)
        success_history.append(1 if total_reward == 1 else 0)
        epsilon = max(epsilon_min, epsilon * epsilon_decay) # riduzione epsilon per diminuire esplorazione nel tempo

    return Q, reward_history, success_history

# -----------------------------
# TEST POLICY
# -----------------------------
def test_agent(env, Q, runs=100):
    successes = 0
    steps_list = []

    for _ in range(runs):
        state, _ = env.reset()
        state = int(state)
        done = False
        steps = 0

        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        if reward == 1:
            successes += 1
            steps_list.append(steps)

    return successes / runs, steps_list

# -----------------------------
# GRAFICI 
# -----------------------------
# serve per valutare se e quando l'agente migliora le prestazioni nel tempo
def plot_success_rate(success_history, window=100, title="Success rate"):
    ma = [np.mean(success_history[i:i+window])
          for i in range(len(success_history)-window+1)]
    plt.figure(figsize=(10,5))
    plt.plot(success_history, alpha=0.3, label="Successo episodio")
    plt.plot(range(window-1, len(success_history)), ma,
             linewidth=2, label=f"Media mobile {window}")
    plt.xlabel("Episodio")
    plt.ylabel("Successo (0/1)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# serve per capire l'efficienza del comportamento dell'agente
def plot_steps_distribution(steps_list, title="Distribuzione passi"):
    plt.figure(figsize=(7,5))
    plt.hist(steps_list, bins=30)
    plt.xlabel("Numero di passi")
    plt.ylabel("Frequenza")
    plt.title(title)
    plt.grid(True)
    plt.show()

# capire quali stati sono più sicuri o vantaggiosi
def plot_q_heatmap(Q, title="Q-table Heatmap"):
    max_q = np.max(Q, axis=1)
    n = int(np.sqrt(len(max_q)))
    if n*n == len(max_q):
        grid = max_q.reshape((n,n))
        plt.figure(figsize=(5,5))
        plt.imshow(grid, cmap="viridis", aspect="equal")
        plt.colorbar(label="Q massimo per stato")
        plt.title(title)
        plt.show()

# serve per confrontare le prestazioni finali degli agenti
def plot_final_success(results, map_name):
    filtered = [r for r in results if r["map"] == map_name]
    labels = [r["method"] for r in filtered]
    values = [r["success"] for r in filtered]

    plt.figure(figsize=(6,4))
    plt.bar(labels, values)
    plt.ylabel("Success rate")
    plt.title(f"Confronto finale – mappa {map_name}")
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.show()

# ---------------------------------
# ESECUZIONE ESPERIMENTI
# ---------------------------------
def run_experiment(cfg, results):
    print(f"\n--- {cfg['map']} | {cfg['method']} ---") # quale esperimento si sta eseguendo

    env, n_states, n_actions = create_env(cfg["map"])

    Q, rewards, success_hist = train_agent(
        env,
        n_states,
        n_actions,
        episodes=cfg["episodes"],
        method=cfg["method"],
        epsilon_decay=cfg["epsilon_decay"]
    )

    runs = 2000 if cfg["map"] == "8x8" else 100 # quante volte testare la policy
    success_rate, steps = test_agent(env, Q, runs=runs)

    results.append({
        "map": cfg["map"],
        "method": cfg["method"],
        "success": success_rate
    })

    print(f"Success rate: {success_rate:.2f}")

    plot_success_rate(success_hist, title=f"{cfg['map']} - {cfg['method']} - Success rate")
    plot_steps_distribution(steps, title=f"{cfg['map']} - {cfg['method']} - Passi")
    plot_q_heatmap(Q, title=f"{cfg['map']} - {cfg['method']} - Q-table")


# ---------------------------------
# MAIN
# ---------------------------------
experiments = [
    {"map":"4x4","method":"q_learning","episodes":8000,"epsilon_decay":0.9995},
    {"map":"4x4","method":"sarsa", "episodes":8000,"epsilon_decay":0.9995},
    {"map":"8x8","method":"q_learning","episodes":50000,"epsilon_decay":0.99995},
    {"map":"8x8","method":"sarsa", "episodes":50000,"epsilon_decay":0.99995}
]

results = []

for cfg in experiments:
    run_experiment(cfg, results)

plot_final_success(results, "4x4")
plot_final_success(results, "8x8")
