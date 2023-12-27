'''
One hidden layer with 256 activation neurons and one output neuron
Least squares error solution applied for training
4096 samples
Average best accuracy after training with degree of noise being 0.25: >98.5%
Two-spiral problem (length is 5 * pi)
'''
# ----------------------------------------------------- Changes / Imports -----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------------------------------------------- Imports / Hyperparameters -----------------------------------------------------

noise = 0.25 # Standard deviation of a normal distribution of noise added to the dataset
batch_size = 4096 # Total number of samples (validation and training combined)
test_proportion = 0.5 # Proportion of samples allocated out of the total for validating
length = 5 # The length of the spiral arm, multiplied by pi
center_n = 512 # Width of the RBF's hidden layer
random_seed = np.random.randint(1,100) # Edit this to be an integer for repeatable results
np.random.seed(random_seed)

# ----------------------------------------------------- Hyperparameters / Dataset -----------------------------------------------------

def generate_two_spiral_data(num_samples=int(batch_size / 2), noise=noise, length=length):
    t = np.linspace(0, np.pi * length, num_samples) + np.random.normal(0, noise, num_samples)
    x1 = t * np.cos(t) + np.random.normal(0, noise, num_samples)
    y1 = t * np.sin(t) + np.random.normal(0, noise, num_samples)
    x2 = t * np.cos(t + np.pi) + np.random.normal(0, noise, num_samples)
    y2 = t * np.sin(t + np.pi) + np.random.normal(0, noise, num_samples)
    X = np.vstack((np.vstack((x1, y1)).T, np.vstack((x2, y2)).T))
    y = np.hstack((np.zeros(num_samples), np.ones(num_samples)))
    np.random.seed(random_seed)
    np.random.shuffle(X[:,0])
    np.random.seed(random_seed)
    np.random.shuffle(X[:,1])
    np.random.seed(random_seed)
    np.random.shuffle(y)
    return X, y

# ----------------------------------------------------- Dataset / Classes -----------------------------------------------------

class RBF:

    def __init__(self, comparisoncenters=center_n):
        self.center_n = comparisoncenters
    
    def train(self, X, y, center=None):

        if center is not None:
            self.center_n = center

        '''
        numpy.random.choice:
        1. Takes in dimensions of array and spits out "indices" in that shape
        2. Takes in number of samples to draw (will increase accuracy the more are drawn, at the risk of overfitting)
        3. If "replace" is equal to True, it can draw multiple of the same sample
        Using this command to take random inputs from X and use as centers for radial comparison of other points,
        Used for initialization
        '''
        self.centers = X[np.random.choice(X.shape[0], self.center_n, replace=False)]

        '''
        Taking the variance of X batch-wise, then taking the mean of it
        This will be the width of RBF activation functions
        This is then turned into a vector the size of the number of centers, or the activation layer
        '''
        self.variance = np.var(X, axis=0)[:, np.newaxis].T * np.ones((self.center_n, 2))

        '''
        Finds the least squares solution of the RBF activation according to y
        Least squares solution command (np.linalg.lstsq(a, b)):
        Given Ax = B (wherein A is a matrix and B is a vector),
        the least squares solution would be the solutions for (A^T)x = (A^T)B.
        By forming the augmented matrix from such and row reducing,
        x_hat will be the least-squares solution.
        Here, A = self.RBF_Act(X) and B = y:
        x_hat will be the vector of linear transformations that,
        when multiplied with A, will yield the closest values to y.
        Only the first return (the [0] at the end) of this command is taken,
        since the rest of the returns are irrelevant here (such as the residuals and rank of matrix a).
        '''
        self.w = np.linalg.lstsq(self.RBF_Act(X), y, rcond=None)[0]
        
    def RBF_Act(self, X):

        '''
        RBF_ACT finds the distance between each sample point and each given center
        The distance calculation occurs as follows:
        (batch_size, placeholder dim, dimensions) - (center_num, dimensions) = (batch_size, center_num, dimensions)
        distance_x/(2*variance_x) = ((batch_size, center_num, dimension_x) / (2 * sigma_x ** 2))
        distance_y/(2*variance_y) = ((batch_size, center_num, dimension_y) / (2 * sigma_y ** 2))
        Exponentiate the negative of their sum to get the Gaussian function, or the RBF activation
        In other words:
        1 / ( (e^(distance_x/(2*variance_x))) * (e^(distance_y/(2*variance_y))) )
        Then stacks a column of ones onto results to be scaled by the least squares solution command,
        meaning that it will act as the bias for the output from being added with every other RBF activation.
        '''
        distance_xy = (((X[:, np.newaxis] - self.centers) ** 2) / (2 * self.variance)).sum(axis=2)
        return np.column_stack((np.exp(-(distance_xy)), np.ones(X.shape[0])))
    
    def validate(self, X):

        '''
        Puts validation portion of input through respective activation functions, add the bias to the result,
        and takes their dot product with respect to the weights to generate predictions.
        '''
        self.predictions = np.dot(self.RBF_Act(X), self.w)

rbf = RBF()

# ----------------------------------------------------- Classes / Training & Validation -----------------------------------------------------

'''
The network is trained and tested on the data.
'''

start_time = time.time()

X, y = generate_two_spiral_data()
proportion = int(test_proportion * batch_size)
X_test, X_train = X[:proportion,:], X[proportion:,:]
y_test, y_train = y[:proportion], y[proportion:]
rbf.train(X=X_train, y=y_train)
rbf.validate(X=X_test)

end_time = time.time()

# ----------------------------------------------------- Training & Validation / Results -----------------------------------------------------

'''
The accuracy, latency, and loss of the network is recorded.
The predictions are then prepared to be more interpretable for presentation.
'''

accuracy = np.mean((rbf.predictions > 0.5) == y_test) * 100
MSE = np.sum((y_test - rbf.predictions) ** 2) / len(y_test)
clipped_predictions = np.clip(rbf.predictions,0,1)
absolute_y = (clipped_predictions + 0.5).astype(int)
latency = end_time - start_time

print("\n" + str(accuracy) + "%", "accuracy and %s mean squared error loss"
      % MSE, "in %s seconds!" % latency)

confirmshowresult = input("\nShow results? (y/n)\n")
if confirmshowresult == "y" or confirmshowresult == "Y":

    '''
    Alongside the mean squared error and accuracy of each predicted classification,
    six graphs are shown:

    Flat predictions - Which class the network thinks each point belongs to, indicated by color.

    Raw predictions - Where the network actually placed each point when performing predicting their labels.
                      This one can be hard to read at times, as there are some points that go beyond "100% confidence,"
                      or their labels are predicted to be either over 1 or under 0. Since the employed method of training 
                      here is using the least squares solution, this means that the output must be a linear transformation
                      of the activation function of the previous layer. In other words, training the network is fitting
                      the hidden layer's outputs to be as closely matching with their respective labels, which can mean
                      sacrificing the alignment of a few points from the range of the binary label.

    Clipped predictions - Almost equivalent to the raw predictions. However, the output is made so that predictions
                          that are higher than 1 are set to 1, and those lower than 0 are set to 0. This allows for the
                          most informative overview of the model's confidence, as the bounded range allows greater visibility
                          of areas of uncertainty between two classifications.

    Clipped Decision Boundary - This takes the longest to show, as it essentially tests what the network classifies
                                each point as on a highly dense grid, which is made into a mesh to demonstrate the
                                boundaries at which classification decisions are made. The result has a clipped range
                                as well, so as to better show contours of uncertainty. It demonstrates how the network
                                is essentially classifying inputs based on a linearly transformed distance from 
                                its selected "centers," which is even more apparent if the "center_n" hyperparameter
                                is made to be a very small integer, such as 1.

    Actual values - The actual values of each point for comparison.

    Difficulty - To show what the model is seeing before classification.
    '''

    fig, plot = plt.subplots(2, 3, figsize=(11,6))
    fig.subplots_adjust(hspace=0.35, top=0.9, bottom=0.1)
    MSEtext = str(MSE)
    accuracy_text = str(accuracy)
    fig.suptitle("Accuracy: " + accuracy_text + "%, MSE Loss: " + MSEtext, fontsize=15)

    cs = plot[1,1].scatter(X_test[:, 0], X_test[:, 1], c=rbf.predictions, cmap='rainbow_r', alpha=1)
    fig.colorbar(cs)
    plot[1,1].set_title("Raw Predictions:")

    cs = plot[0,0].scatter(X_test[:, 0], X_test[:, 1], c=absolute_y, cmap='rainbow_r', alpha=1)
    fig.colorbar(cs)
    plot[0,0].set_title("Flat Predictions:")
    
    cs = plot[0,1].scatter(X_test[:, 0], X_test[:, 1], c=clipped_predictions, cmap='rainbow_r', alpha=1)
    fig.colorbar(cs)
    plot[0,1].set_title("Clipped Predictions:")
    
    cs = plot[1,0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='rainbow_r', alpha=1)
    fig.colorbar(cs)
    plot[1,0].set_title("Actual Values:")

    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    rbf.validate(np.c_[xx.ravel(), yy.ravel()])
    Z = np.clip(rbf.predictions, 0, 1)
    Z = Z.reshape(xx.shape)
    cs = plot[0,2].pcolor(xx, yy, Z, cmap='rainbow_r', vmin=Z.min(), vmax=Z.max(), alpha=1)
    fig.colorbar(cs)
    plot[0,2].set_title("Clipped Decision Boundary:")

    plot[1,2].set_title("Difficulty:")
    plot[1,2].scatter(X_test[:, 0], X_test[:, 1], alpha=1)

    plt.show()