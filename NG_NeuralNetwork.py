# Neural Gas
# Two-spiral problem (length is 5 * pi) with depth!
# Mini-batch mechanism included

# ----------------------------------------------------- Changes / Imports -----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------------------------------------------- Imports / Hyperparameters -----------------------------------------------------

noise = 0.75 # Standard deviation of a normal distribution of noise added to the dataset
batch_size = 128 # The number of samples drawn per iteration
length = 5 # The length of the spiral arm, multiplied by pi

'''
lambdaConstant - Meant to be adjusted so that higher values will slow the decrease
of the learning rate and radius of influence over time and vice versa.
initDims - The squared number of nodes within the neural gas network.
initRadius - The initial radius of influence for best matching units on other units, measured
             in their Euclidean distance.
initLearnRate - The initial learning rate of every node.
iterations - The number of times mini-batches are presented to the network.
'''

lambdaConstant = 1e+2
initDims = 25
initRadius = initDims / 2
initLearnRate = 0.1
iterations = 1000

# ----------------------------------------------------- Hyperparameters / Dataset -----------------------------------------------------

def generateTwoSpiralData(num_samples=int(batch_size), noise=noise, length=length):
    t = np.linspace(0, np.pi * length, num_samples) + np.random.normal(0, noise, num_samples)
    x1 = t * np.cos(t) + np.random.normal(0, noise, num_samples)
    y1 = t * np.sin(t) + np.random.normal(0, noise, num_samples)
    x2 = t * np.cos(t + np.pi) + np.random.normal(0, noise, num_samples)
    y2 = t * np.sin(t + np.pi) + np.random.normal(0, noise, num_samples)
    X = np.vstack((np.vstack((x1, y1, t)).T, np.vstack((x2, y2, -t)).T))
    random_indices = np.arange(0, int(X.shape[0]))
    np.random.shuffle(random_indices)
    X = X[random_indices[:num_samples], :]
    return X

# ----------------------------------------------------- Dataset / Classes -----------------------------------------------------

class NeuralGas:

    def __init__(self, initDims, sampleFromData=None):

        '''
        Initializing node values, wherein they will first occupy a cube in space spanning from
        the lowest to the highest observed values in the first mini-batch.
        '''

        self.width = initDims
        if sampleFromData is None:
            self.weights = np.random.uniform(-self.width / 2, self.width / 2, size=(initDims, initDims, 1, 3))
            self.initRadius = initRadius
        else:
            self.weights = np.random.uniform(low=sampleFromData.min(), high=sampleFromData.max(), size=(initDims, initDims, 1, 3))
            self.initRadius = (sampleFromData.max() - sampleFromData.min() / 2)

    def findBU(self, input):

        '''
        Locates the best-matching unit (referred to as the BU), qualified by being the closest node to the points
        received in the mini-batch.
        '''

        self.input = input[np.newaxis,np.newaxis,:,:]
        self.euclideanDistance = np.sqrt(np.sum((self.input - self.weights) ** 2, axis=3))
        bestUnitPerSample = np.argmin(self.euclideanDistance.reshape(int(self.width ** 2),self.input.shape[2]), axis=0)
        self.bestUnit = ((bestUnitPerSample / self.width).astype(int), (bestUnitPerSample % self.width).astype(int), (np.arange(0,self.input.shape[2])).astype(int))
        self.bestUnit = np.asarray(self.bestUnit[0:2]).T
        self.indices = np.arange(0,self.bestUnit.shape[0]).reshape((1,1,self.bestUnit.shape[0],1)) + np.zeros((self.width,self.width,self.bestUnit.shape[0],1))

    def adjustWeights(self, initRadius, lambdaConstant, timeStep, initLearningRate):

        '''
        Points around a radius (that decreases over time) are drawn towards their closest
        best-matching unit, although the strength of the BU's influence decreases over distance.
        The learning rate and affected radius decreases over time so that the structure can
        settle over time. This results in the orientation of each node resembling that of the
        input data's structure over time.
        '''

        learningRate = initLearningRate * np.exp(-timeStep / lambdaConstant)
        radius = initRadius * np.exp(-timeStep / lambdaConstant)
        bestUnits = np.squeeze(self.weights[self.bestUnit[:,0],self.bestUnit[:,1],:,:],axis=1)
        affectedRadius = np.sqrt(np.sum((self.weights - bestUnits) ** 2,axis=3))
        radiusBUMask = np.where(np.argmin(self.euclideanDistance,axis=2).astype(int)[:,:,np.newaxis,np.newaxis] + np.zeros((self.width,self.width,self.bestUnit.shape[0],1)) == self.indices, 1.0, 0.0)
        radiusBUMask *= np.where((affectedRadius <= radius), 1.0, 0.0)[:,:,:,np.newaxis]
        influence = np.exp(-(affectedRadius ** 2) / (2 * (radius ** 2)))[:,:,:,np.newaxis]
        deltaWeights = np.sum((influence * learningRate * (self.input - self.weights) * radiusBUMask),axis=2)[:,:,np.newaxis,:]
        radiusBUMaskTotal = np.sum(radiusBUMask,axis=2)[:,:,:,np.newaxis] + 1
            
        self.weights += (deltaWeights / radiusBUMaskTotal) * np.sign(radiusBUMaskTotal - 1)

# ----------------------------------------------------- Classes / Training -----------------------------------------------------

'''
Points from a 3d spiral dataset are given to the neural gas network to learn from,
with decreasing movement over time to encourage a settled orientation. Two graphs are
shown, one being a sample of input data and another being the untrained nodes of the
neural gas.
'''

X = generateTwoSpiralData()
fig = plt.figure(figsize=(6, 6))
plot = fig.add_subplot(projection='3d')
plot.scatter(list(X[:,0]), list(X[:,1]), list(X[:,2]))
fig.suptitle("Input data example:", fontsize=15)
plt.show()

selfOrganizingMap = NeuralGas(initDims = initDims, sampleFromData = X)
selfOrganizingMap.findBU(X)
selfOrganizingMap.adjustWeights(selfOrganizingMap.initRadius, lambdaConstant, 0, initLearnRate)

fig = plt.figure(figsize=(6, 6))
plot = fig.add_subplot(projection='3d')
plot.scatter(list(selfOrganizingMap.weights[:, :, :, 0]), list(selfOrganizingMap.weights[:, :, :, 1]),
             list(selfOrganizingMap.weights[:, :, :, 2]), c=selfOrganizingMap.weights[:, :, :, 2], cmap='rainbow_r', alpha=1.0)
fig.suptitle("Neural gas starting structure:", fontsize=15)
plt.show()

startTime = time.time()
for i in range(iterations):

    X = generateTwoSpiralData()

    selfOrganizingMap.findBU(X)
    selfOrganizingMap.adjustWeights(selfOrganizingMap.initRadius, lambdaConstant, (i + 1), initLearnRate)

# ----------------------------------------------------- Training / Results -----------------------------------------------------

print("Training completed in", str((time.time() - startTime) / 60), "minutes.")

'''
With the training completed, the new structure of nodes in the neural gas is
overlayed by a large sample of the input data, showing how well it learned its structure
in 3d space.
'''

fig = plt.figure(figsize=(6, 6))
plot = fig.add_subplot(projection='3d')

plot.scatter(list(selfOrganizingMap.weights[:, :, :, 0]), list(selfOrganizingMap.weights[:, :, :, 1]), 
             list(selfOrganizingMap.weights[:, :, :, 2]), c=selfOrganizingMap.weights[:, :, :, 2], cmap='rainbow_r', alpha=0.5)
X = generateTwoSpiralData(num_samples=256)
plot.scatter(list(X[:,0]), list(X[:,1]), list(X[:,2]), c=list(X[:,2]), cmap='binary', alpha=1)
fig.suptitle("True data overlaying learned structure:", fontsize=15)
plt.show()

fig = plt.figure(figsize=(6, 6))
plot = fig.add_subplot(projection='3d')
plot.scatter(list(selfOrganizingMap.weights[:, :, :, 0]), list(selfOrganizingMap.weights[:, :, :, 1]), 
             list(selfOrganizingMap.weights[:, :, :, 2]), c=selfOrganizingMap.weights[:, :, :, 2], cmap='rainbow_r', alpha=1.0)
fig.suptitle("Learned structure:", fontsize=15)
plt.show()