import gym
import numpy as np
import math
import random
import time

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
    return (node.value / node.visits) + c * math.sqrt(math.log(total_visits) / node.visits) #c est le parametre de l'exploration, c proche de 0 --> exploitation, c grand --> exploration, c = 1.41 --> standard

def run_mcts(env, iterations=60):
    # Sauvegarde de l'état actuel
    root_state = env.unwrapped.clone_state(include_rng=True)
    root = MCTSNode()
    
    # Espace d'action complet de l'émulateur
    num_actions = env.action_space.n
    
    for _ in range(iterations):
        # On repart de l'état racine pour chaque simulation
        env.unwrapped.restore_state(root_state)
        node = root
        terminated, truncated, reward = False, False, 0
        
        # Premiere etape --> la selection : On descend dans l'arbre connu via UCT
        while node.children:
            node = max(node.children, key=lambda n: uct_value(n, node.visits))
            _, reward, terminated, truncated, _ = env.step(node.action)
            if terminated or truncated:
                break
            
        # Deuxieme etape --> l'expansion : On ajoute des noeuds si la partie n'est pas finie
        if not (terminated or truncated):
            # On ajoute toutes les actions possibles de l'émulateur
            for a in range(num_actions):
                node.children.append(MCTSNode(action=a, parent=node))
            
            # On en choisit une au hasard pour commencer la simulation
            if node.children:
                node = random.choice(node.children)
                _, reward, terminated, truncated, _ = env.step(node.action)

        # Troisieme etape --> la simulation (Rollout) : On joue au hasard jusqu'à la fin ou limite de temps
        current_res = reward
        if not (terminated or truncated):
            # On simuleavec une profondeur de 120 pour espérer toucher une fin de partie (victoire/défaite)
            for _ in range(120): 
                _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                if terminated or truncated:
                    current_res = reward
                    break
        else:
            current_res = reward
        
        # Quatrieme etape --> backpropagation : On remonte les scores en inversant les signes
        res = current_res
        temp_node = node
        while temp_node is not None:
            temp_node.visits += 1
            temp_node.value += res
            # Inversion Minimax : une victoire pour l'enfant est une défaite pour le parent
            res = -res 
            temp_node = temp_node.parent

    # On restaure l'état pour que la partie réelle puisse continuer
    env.unwrapped.restore_state(root_state)
    
    if not root.children:
        return env.action_space.sample()
    
    # On choisit l'action la plus visitée (plus stable que la valeur brute)
    best_move_node = max(root.children, key=lambda n: n.visits)
    
    return best_move_node.action


# Utilise render_mode=None pour enlever l'interface et gagner de la vitesse de calcul, "human" pour voir (mais très lent)
env = gym.make("ALE/TicTacToe3D-v5", render_mode=None)
obs, _ = env.reset()
done = False
step_count = 0

while not done:
    step_count += 1
    start_time = time.time()

    action = run_mcts(env, iterations=60) 
    
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    elapsed = time.time() - start_time
    print(f"Coup {step_count} | Action: {action:2} | Reward: {reward:+.1f} | Temps: {elapsed:.2f}s")

    # Sécurité pour éviter les parties infinies si l'IA stagne
    if step_count > 300:
        print("Limite de coups atteinte.")
        break

print(f"Resultat final : {reward}")

env.close()