#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 10:51:42 2025

@author: amina
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom, expon

# Parameters
p = 0.2  # probability of success for geometric distribution
lambda_ = p  # rate parameter for exponential distribution

# Geometric distribution (discrete)
x_geom = np.arange(1, 20)
pmf_geom = geom.pmf(x_geom, p)

# Exponential distribution (continuous)
x_exp = np.linspace(0, 20, 500)
pdf_exp = expon.pdf(x_exp, scale=1/lambda_)

# Plotting
plt.figure(figsize=(10, 6))

# Plot geometric PMF
plt.stem(x_geom, pmf_geom, linefmt='C0-', markerfmt='C0o', basefmt=" ", label='Geometric PMF (discrete)')

# Plot exponential PDF
plt.plot(x_exp, pdf_exp, 'C1-', label='Exponential PDF (continuous)', linewidth=2)

plt.title('Geometric vs Exponential Distribution')
plt.xlabel('x')
plt.ylabel('Probability / Density')
plt.legend()
plt.grid(True)
plt.show()
