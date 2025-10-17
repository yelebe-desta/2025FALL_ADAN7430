# -*- coding: utf-8 -*-
"""
@author: parra
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

theta_vector = np.array([-25,0,0,1,1,1,1,1])
N = 1_200_000
np.random.seed(10)
x1_inputs = np.random.uniform(-10, 10, size=N)
np.random.seed(100)
x2_inputs = np.random.uniform(-10, 10, size=N)

row_list = []
labels = []

for x1,x2 in zip(x1_inputs,x2_inputs):
    x_i_vector = np.array([1,x1,x2,x1**2,x2**2,x1**3,x2**3,x1*x2])
    #x_i_vector = np.array([1,x1,x2,x1**2,x2**2,x1**3,x2**3,x1**4])
    row_list.append(x_i_vector)
    
    hypothesis_prediction = np.dot(theta_vector, x_i_vector)
    if hypothesis_prediction >= 25:
        labels.append(1)
    else:
        labels.append(0)

## add some noise to labels so that the two classes are not perfectly seperable

ctr = 1
noisy_labels = []
for x in labels:
    if np.random.random()>.92:
        if x == 0:
            noisy_labels.append(1)
        elif x ==1:
            noisy_labels.append(1)
        print(f"Swapped label on {ctr} records")
        ctr += 1
    else:
        noisy_labels.append(x)
        


design_matrix = np.array(row_list)
col_names = ['intercept','x1','x2','x1_squared','x2_squared','x1_cubed','x2_cubed','x1_times_x2']
data = pd.DataFrame(row_list,columns = col_names)
data['label'] = noisy_labels
np.random.seed(123)
test_portion = .10
test_ids = np.random.choice(np.arange(0, data.shape[0]), size=int(test_portion*data.shape[0]), replace=False)
test_data = data[data.index.isin(test_ids)]
training_data = data[~data.index.isin(test_ids)]

training_data = training_data.reset_index()
test_data = test_data.reset_index()

test_data['type'] = 'test'
training_data['type'] = 'train'

full_data = pd.concat([test_data,training_data])

cols_to_keeps = ['x1','x2','label']

# draw a sample to use for actual model selection and evaluation

np.random.seed(123)
sample_portion = 1/5_000
sample_ids = np.random.choice(np.arange(0, training_data.shape[0]), size=int(sample_portion*data.shape[0]), replace=False)
sample_data = training_data[training_data.index.isin(sample_ids)]
sample_data[cols_to_keeps].to_csv(os.path.join(output_path,'SAMPLE_DATA.csv'),
                                  index=False)


# Create the scatter plot
plot_size = 10_000
plt.scatter(x1_inputs[0:plot_size],
            x2_inputs[0:plot_size],
            c=noisy_labels[0:plot_size],
            cmap='viridis',
            s=100,
            edgecolor='k')

# Add a color bar
plt.colorbar(label='Color intensity')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Logitstic Regression Data: Non linear Decision Boundary')

# Show the plot
plt.show()
