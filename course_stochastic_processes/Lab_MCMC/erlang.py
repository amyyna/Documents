#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 10:57:06 2025

@author: amina
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Parameters
lambda_ = 0.5  # rate parameter
k_values = [1, 2, 3, 5, 20]  # different orders (shape parameters)
x = np.linspace(0, 50, 1000)

# Plot Erlang distributions for different orders
plt.figure(figsize=(10, 6))
for k in k_values:
    pdf = gamma.pdf(x, a=k, scale=1/lambda_)  # Erlang is gamma with integer shape
    plt.plot(x, pdf, label=f'k = {k}')

plt.title('Erlang Distribution for Different Orders (k)')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend(title='Order (k)')
plt.grid(True)
plt.show()
