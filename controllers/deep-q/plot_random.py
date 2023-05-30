import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('q_value_distributions_random.txt', names=['current', 'expected'])

# Scatter plot
plt.scatter(data['current'], data['expected'])
plt.xlabel('Current Q-values')
plt.ylabel('Expected Q-values')
plt.title('Distribution of Q-values')
plt.show()