import gymnasium as gym
import ale_py
import numpy as np
import math
import random
import time
from tqdm import tqdm

np.bool8 = np.bool_


class MCTSNode:
    def __init__(self, action=None, parent=None):
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0


def uct_value(node, total_visits, c=1.41):
    if node.visits == 0:
        return float('inf')
    return (node.value / node.visits) + c * math.sqrt(math.log(
        total_visits) / node.visits)  # c est le parametre de l'exploration, c proche de 0 --> exploitation, c grand --> exploration, c = 1.41 --> standard


def run_mcts(env, iterations=60):
    root_state = env.unwrapped.clone_state(include_rng=True)
    root = MCTSNode()

    actions = [1, 2, 3, 4, 5] # 1=Fire, 2=Up, 3=Right, 4=Left, 5=Down
    # (les autres commandes sont une combinaison de plusieurs commandes et 0=NoOperation)

    for _ in tqdm(range(iterations)):
        terminated, truncated, reward = False, False, 0
        node = root

        # selection : On descend dans l'arbre connu via UCT
        while node.children:
            node = max(node.children, key=lambda n: uct_value(n, node.visits))
            _, reward, terminated, truncated, _ = env.step(node.action)
            if terminated or truncated:
                break

        reward = 0

        # expansion : On ajoute des noeuds si la partie n'est pas finie
        while not (terminated or truncated):
            # On ajoute toutes les actions possibles de l'émulateur
            for action in actions:
                node.children.append(MCTSNode(action=action, parent=node))

            # On en choisit une au hasard pour commencer la simulation
            if node.children:
                node = random.choice(node.children)
                _, reward, terminated, truncated, _ = env.step(node.action)

        # backpropagation : On remonte les scores en inversant les signes
        temp_node = node
        while temp_node is not None:
            temp_node.visits += 1
            temp_node.value += reward
            temp_node = temp_node.parent

        # On restaure l'état pour que la partie réelle puisse continuer
        env.unwrapped.restore_state(root_state)


    #if not root.children:
    return root


# Utilise render_mode=None pour enlever l'interface et gagner de la vitesse de calcul, "human" pour voir (mais très lent)
env = gym.make("ALE/TicTacToe3D-v5", render_mode=None)
obs, _ = env.reset()

start_time = time.time()
root = run_mcts(env, iterations=300)

elapsed = time.time() - start_time
print(f"Résultats :\tRoot value: {root.value} | Root visits: {root.visits} | Temps: {elapsed:.2f}s")

env.close()