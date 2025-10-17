# -*- coding: utf-8 -*-
"""
@author: parra
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

theta_vector = np.array([-25,1,2])

np.random.seed(10)
x1_inputs = np.random.uniform(-10, 10, size=100)
np.random.seed(100)
x2_inputs = np.random.uniform(-10, 10, size=100)

row_list = []
labels = []

for x1,x2 in zip(x1_inputs,x2_inputs):
    x_i_vector = np.array([1,x1,x2])
    row_list.append(x_i_vector)
    
    hypothesis_prediction = np.dot(theta_vector, x_i_vector)
    if hypothesis_prediction >= 25:
        labels.append(1)
    else:
        labels.append(0)

design_matrix = np.array(row_list)
data = pd.DataFrame(row_list,columns = ['intercept','x1','x2'])
data['label'] = labels
np.random.seed(123)
test_ids = np.random.choice(np.arange(0, data.shape[0]), size=int(.20*data.shape[0]), replace=False)
test_data = data[data.index.isin(test_ids)]
training_data = data[~data.index.isin(test_ids)]

test_data['type'] = 'test'
training_data['type'] = 'train'

full_data = pd.concat([test_data,training_data])

cols_to_keeps = ['x1','x2','label']

# Create the scatter plot
plt.scatter(x1_inputs, x2_inputs, c=labels, cmap='viridis', s=100, edgecolor='k')

# Add a color bar
plt.colorbar(label='Color intensity')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Logitstic Regression with linear Decision Boundary')

# Show the plot
plt.show()
