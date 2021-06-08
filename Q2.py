# Q2_graded
# Do not change the above line.

# This cell is for your imports.

import numpy as np
import matplotlib.pyplot as plt
import random 
from time import time
from numpy import loadtxt
from google.colab import drive
drive.mount("/content/drive")

class Perceptron(object):

  def __init__(self, learning_rate, number_of_inputs, stop_iteration):
    self.learning_rate = learning_rate
    random.seed(time())
    self.weights = np.random.random(number_of_inputs + 1)
    #self.weights = np.zeros(number_of_inputs + 1)
    self.stop_iteration = stop_iteration
    self.error = []

  def activation_function(self, inputs):
    summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
    #summation = np.dot(inputs, self.weights)
    activation = 0
    if summation > 0:          
      activation = 1        
    else:          
      activation = 0
    return summation, activation


  def training(self, training_inputs, labels):
    for _ in range(self.stop_iteration):
      for inputs, label in zip(training_inputs, labels):
        summation, activation = self.activation_function(inputs)
        self.weights[1:] = self.weights[1:] + self.learning_rate * (label - summation) * inputs
        self.weights[0] = self.weights[0] + self.learning_rate * (label - summation)
        self.error.append(abs(label - summation))

inputs = []
labels = []
training_inputs = []
training_labels = []

learning_rate = 0.01
number_of_inputs = 2
stop_iteration = 100

with open('/content/drive/My Drive/Colab Notebooks/data.txt') as dataset:
    mydatas = [line.rstrip('\n') for line in dataset]
    for data in mydatas:
      s_data = data.split(',')
      inputs.append(np.array([float(s_data[0]), float(s_data[1])]))
      labels.append(float(s_data[2]))

for i in range(int(len(inputs))):
  training_inputs.append(inputs[i]/100)
  if labels[i] == 0.0:
    training_labels.append(labels[i] - 1)
  else:
    training_labels.append(labels[i])

#for i in range(int(len(inputs) * 3/4) , len(inputs)):
#  test_inputs.append(inputs[i]/100)
#  test_labels.append(labels[i])

perceptron = Perceptron(learning_rate, number_of_inputs, stop_iteration)
perceptron.training(training_inputs, training_labels)
print(perceptron.weights)

fig, ax = plt.subplots()
xmin, xmax = -3, 1
X = np.arange(xmin, xmax, 0.1)
for i in range(len(training_inputs)):
  if training_labels[i] == 1.0:
    ax.scatter(training_inputs[i][0], training_inputs[i][1], color="b")
  else:
    ax.scatter(training_inputs[i][0], training_inputs[i][1], color="r")

ax.set_xlim([xmin, xmax])
ax.set_ylim([-1, 4])
m = -perceptron.weights[1] / perceptron.weights[2]
c = -perceptron.weights[0] / perceptron.weights[2]
ax.plot(X, m * X + c )
#ax.plot(X, m * X + 1.2, label="decision boundary")
plt.plot()
plt.show()
#print(perceptron.error)
plt.plot(perceptron.error)


# Q2_graded
# Do not change the above line.

# This cell is for your codes.

# Q2_graded
# Do not change the above line.

# This cell is for your codes.

