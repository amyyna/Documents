#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 12:03:34 2025

@author: amina
"""
import numpy as np
import matplotlib.pyplot as plt

# Parameters
p = 0.47            # Probability of winning
q = 1 - p            # Probability of losing
initial_money = 50   # Starting amount
n_steps = 1000       # Maximum number of steps

# Simulate the gambler's ruin walk
money = [initial_money]
for _ in range(n_steps):
    if money[-1] == 0:
        break  # Gambler is ruined
    step = np.random.choice([1, -1], p=[p, q])
    money.append(money[-1] + step)

# Time steps
t = np.arange(len(money))

# Drift curve: expected position over time = initial + (p - q) * t
drift = initial_money + (p - q) * t

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, money, label="Gambler's Ruin Walk (p=0.47)", color='blue')
plt.plot(t, drift[:len(t)], label='Drift Curve (Expected Value)', linestyle='--', color='red')
plt.xlabel('Steps')
plt.ylabel('Money ($)')
plt.title("Gambler's Ruin with Drift (Starting at $50)")
plt.axhline(0, color='black', linewidth=1, linestyle=':')
plt.legend()
plt.grid(True)
plt.show()
