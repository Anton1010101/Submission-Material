# Dependencies:
The system requirements for the software to be executed are as follows:

Python 3.11.3.

NumPy, version 1.24.3 or any subsequent release.

PyPlot (Matplotlib), version 3.7.1 or any subsequent release.

I used the Visual Studio Code environment to develop and test these programs, utilizing Python version v2023.22.1.

To ensure the availability of necessary packages, it is recommended to have 'pip' installed, enabling the installation of NumPy and Matplotlib via the following commands:

pip install numpy

pip install matplotlib
# Neural Gas Network:
The Python script "NG_NeuralNetwork.py" is a demonstration of a neural gas network, designed by Thomas Martinetz and Klaus Schulten in 1991, and inspired by Kohonen maps. Neural gas, a type of unsupervised learning neural network, excels in data clustering, topology preservation, and feature mapping. Their adaptability makes them useful for applications such as data compression, pattern recognition, and high-dimensional data visualization.

Neural gas networks aim to find the best underlying representation of data by minimizing the difference between input feature vectors and nodes within the network that are of the same shape. It works by finding the best-matching unit to an input in the neural gas, determined by a metric of distance that is commonly Euclidean, and pulling this node partway to the input. Neighbors to this node, also determined by some measure of distance, are dragged along as well, although with decreasing force if they are more distant. The amount that each node is pulled towards an input slowly decreases over time, so that stability can eventually be achieved. The end result is an assembly of nodes from the neural gas that are oriented in such a way that the input distribution of data is captured in a more interpretable pattern.

In my demonstration, an unmentioned step of this process is skipped. It is often the case that each node may hold connections with related nodes, allowing a graph-like representation of results. I did not incorporate this into my program for the fact that it would slow down the whole process and because the two-spiral dataset it uses is not nuanced enough to necessarily benefit from such.

Upon running "NG_NeuralNetwork.py," an interactable 3D graph of a sample of the input will be shown, and a graph of each node from the neural gas will follow upon closing the first. They will be randomly dispersed in space, as they have only just been initialized. After closing this window, the network will be shown input data to conform to for 1,000 iterations. After the training finishes, the training time will be printed in terms of minutes, and a graph of a fuller sample of input data (monochromatically-colored) will be shown overlaying the learned structure (rainbow-colored). Closing this window will result in a new graph appearing, demonstrating the learned structure by itself. The program will finish when this window is closed.
# Radial Basis Function Network:
The Python script "RBF_NeuralNetwork.py" is a demonstration of a radial basis function network, introduced by Broomhead and Lowe in 1988. These kinds of networks are commonly employed for function approximation, interpolation, and regression tasks, such as for time-series prediction, pattern-recognition, and system control jobs. They typically consist of two layers, the first layer possessing radial basis functions as their activations. An advantage they hold over other networks is their capacity for nonlinear processing despite their "shallow" architecture, which can allow them to be much faster than many networks that may have to possess more layers to capture all nonlinear aspects of input data.

The way RBF networks work is by using chosen "centers" for which to compare other input feature vectors. If a given center is associated with a desired output, then points closer to this center are also associated to this output. This gives the network the ability to train on relatively few samples of data with a high degree of accuracy. However, the disadvantage of RBF networks is that they have difficulty in modeling particularly linear data. To do so, they may require more centers and more training data to approximate strictly linear functions.

My demonstration of a radial basis function network is configured to interpolate a binary classification for the two-spiral problem, so the training process uses the least-squares solution of a single layer of weights. Upon running the "RBF_NeuralNetwork.py" program, the network quickly trains on noisy data, prints its accuracy and loss, and then asks if the user would like to see the results. By typing "y" and pressing enter, the network will take a longer time to construct six side-by-side graphs demonstrating what it learned and what the input data looks like. When the window is closed, the program will finish running. By typing "n" and pressing enter, the program will finish running without showing any graphs.
