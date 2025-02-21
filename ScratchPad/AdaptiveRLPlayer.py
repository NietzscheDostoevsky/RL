import numpy as np
import random

class AdaptivePlayer:
    def __init__(self):
        self.q_table = {}  # Q-learning table to store values of (state, action) pairs
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.2  # Exploration rate

    def choose_action(self, state, available_actions):
        if random.random() < self.epsilon:  # Explore
            return random.choice(available_actions)
        return max(available_actions, key=lambda a: self.q_table.get((state, a), 0))  # Exploit

    def update_q_value(self, state, action, reward, next_state, next_available_actions):
        max_next_q = max([self.q_table.get((next_state, a), 0) for a in next_available_actions], default=0)
        self.q_table[(state, action)] = self.q_table.get((state, action), 0) + self.learning_rate * (reward + self.discount_factor * max_next_q - self.q_table.get((state, action), 0))

# Example usage:
player = AdaptivePlayer()
state = "OpponentWeakInEndgame"
actions = ["AggressiveMove", "DefensiveMove"]
chosen_action = player.choose_action(state, actions)
print(f"Chosen Action: {chosen_action}")
