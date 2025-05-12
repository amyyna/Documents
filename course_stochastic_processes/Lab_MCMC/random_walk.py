import random
import numpy as np
import matplotlib.pyplot as plt

def function(T):
    X_n = 0
    list_of_X_n = [X_n]
    for _ in range(T):
        delta = random.choice([-1, 1]) 
        X_n += delta
        list_of_X_n.append(X_n)
    return list_of_X_n

# Run
T = 1000000
X_n = function(T)

x_pos = np.linspace(0, T, 400)  # x >= 0 for sqrt(x)
x_neg = np.linspace(0, T, 400)  # x <= 0 for sqrt(-x)

# Define y = sqrt(x) and y = sqrt(-x)
y_pos = np.sqrt(x_pos)
y_neg = -np.sqrt(x_neg)

# Plotting
#plt.plot(x_pos, y_pos, label="y = sqrt(x)", color='green')
#plt.plot(x_neg, y_neg, label="y = sqrt(-x)", color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y = sqrt(x) and y = sqrt(-x)')

# Plot the result
#plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(X_n)), X_n)
plt.title('1D process')
plt.xlabel('seconds')
plt.ylabel('X_n')
plt.grid(True)
plt.show()