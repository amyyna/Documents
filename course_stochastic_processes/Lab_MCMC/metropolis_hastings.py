import numpy as np
import matplotlib.pyplot as plt

# Target distribution: Normal(0, 1)
def target_distribution(x):
    # Normal distribution PDF
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

# Proposal distribution: Uniform(current_state - delta, current_state + delta)
def proposal_distribution(x, delta):
    return np.random.uniform(x - delta, x + delta)

# Metropolis-Hastings algorithm
def metropolis_hastings(target, proposal, initial_state, n_samples, burn_in, step_size):
    samples = []
    current_state = initial_state
    # Run the chain for the burn-in period
    for _ in range(burn_in):
        proposed_state = proposal(current_state, step_size)
        acceptance_ratio = min(1, target(proposed_state) / target(current_state))
        if np.random.rand() < acceptance_ratio:
            current_state = proposed_state

    # Collect samples after burn-in
    for _ in range(n_samples):
        proposed_state = proposal(current_state, step_size)
        acceptance_ratio = min(1, target(proposed_state) / target(current_state))
        if np.random.rand() < acceptance_ratio:
            current_state = proposed_state
        samples.append(current_state)

    return np.array(samples)

# Hyperparameters
initial_state = 0.0  # Starting point
burn_in = 1000
num_samples = 100000  # Number of samples
step_size = 1.0      # Step size for proposal distribution

# Run Metropolis-Hastings to generate samples
samples = metropolis_hastings(target_distribution, proposal_distribution, initial_state, num_samples, burn_in, step_size)

# Plotting the histogram of the samples and comparing with the target distribution
plt.hist(samples, bins=50, density=True, alpha=0.6, color='b')

# Plot the target distribution for comparison
x = np.linspace(-5, 5, 1000)
y = target_distribution(x)
plt.plot(x, y, 'r-', lw=2)

plt.title('Metropolis-Hastings Sampling from N(0, 1)')
plt.xlabel('x')
plt.ylabel('Density')
plt.show()
