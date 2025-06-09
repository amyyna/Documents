#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 11:04:22 2025

@author: amina
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

def draw_markov_chain_graph_circular(P):
    num_states = P.shape[0]
    angles = np.linspace(0, 2 * np.pi, num_states, endpoint=False)
    states_x = np.cos(angles)
    states_y = np.sin(angles)
    
    fig, ax = plt.subplots(figsize=(8,8))
    
    for i in range(num_states):
        circle = plt.Circle((states_x[i], states_y[i]), 0.1, color='lightgray', ec='black', zorder=2)
        ax.add_patch(circle)
        ax.text(states_x[i], states_y[i], f'{i}', fontsize=12, ha='center', va='center', zorder=3)
    
    for i in range(num_states):
        for j in range(num_states):
            if P[i,j] > 0:
                if i == j:
                    arc = patches.Arc((states_x[i], states_y[i]+0.2), 0.3, 0.3, theta1=0, theta2=300, color='gray', lw=1.5)
                    ax.add_patch(arc)
                    ax.text(states_x[i], states_y[i] + 0.45, f'{P[i,j]:.2f}', fontsize=9, ha='center', color='blue')
                else:
                    start = np.array([states_x[i], states_y[i]])
                    end = np.array([states_x[j], states_y[j]])
                    direction = end - start
                    length = np.linalg.norm(direction)
                    direction /= length
                    start_adj = start + direction * 0.12
                    end_adj = end - direction * 0.12
                    
                    ax.annotate(
                        "",
                        xy=end_adj,
                        xytext=start_adj,
                        arrowprops=dict(arrowstyle="->", color='gray', lw=1.5),
                        zorder=1
                    )
                    mid_point = (start_adj + end_adj) / 2
                    ax.text(mid_point[0], mid_point[1], f'{P[i,j]:.2f}', fontsize=9, ha='center', color='blue')
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title('Markov Chain with Small Probabilities to Absorbing State (Circular Layout)')
    return fig, ax, states_x, states_y

def run_markov_chain_visual_circular(P, start_state, absorbing_states, pause=1):
    fig, ax, states_x, states_y = draw_markov_chain_graph_circular(P)
    current_state = start_state
    
    current_marker = plt.Circle((states_x[current_state], states_y[current_state]), 0.13, color='red', alpha=0.5, zorder=4)
    ax.add_patch(current_marker)
    
    fig.canvas.draw()
    plt.show(block=False)
    
    print(f"Starting at state {current_state}")
    time.sleep(pause)
    
    while current_state not in absorbing_states:
        next_state = np.random.choice(len(P), p=P[current_state])
        print(f"Moved to state {next_state}")
        
        current_marker.center = (states_x[next_state], states_y[next_state])
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        current_state = next_state
        time.sleep(pause)
    
    print(f"Reached absorbing state {current_state}. Chain stopped.")
    plt.show()

# Transition matrix with 6 states; state 5 absorbing
# Small probabilities (0.02-0.05) leading to absorbing state 5
"""
P = np.array([
    [0.30, 0.30, 0.25, 0.10, 0.03, 0.02],  # state 0
    [0.25, 0.30, 0.25, 0.10, 0.05, 0.05],  # state 1
    [0.15, 0.30, 0.35, 0.10, 0.05, 0.05],  # state 2
    [0.25, 0.10, 0.20, 0.38, 0.05, 0.02],  # state 3
    [0.20, 0.10, 0.10, 0.50, 0.08, 0.02],  # state 4 (not absorbing, small chance to 5)
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # state 5 absorbing
])
"""

P = np.array([
    [0.25, 0.28, 0.22, 0.12, 0.08, 0.05],  # state 0
    [0.22, 0.25, 0.25, 0.10, 0.10, 0.08],  # state 1
    [0.15, 0.25, 0.30, 0.10, 0.08, 0.12],  # state 2
    [0.20, 0.10, 0.15, 0.35, 0.10, 0.10],  # state 3
    [0.18, 0.10, 0.10, 0.40, 0.10, 0.12],  # state 4 (not absorbing, some chance to 5)
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # state 5 absorbing
])

start_state = 0
absorbing_states = {5}

run_markov_chain_visual_circular(P, start_state, absorbing_states, pause=1)
