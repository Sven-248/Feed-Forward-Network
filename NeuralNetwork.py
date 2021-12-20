import numpy as np
import matplotlib.pyplot as plt
import Perceptron

class NeuralNetwork():

    #initialize the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, iterations=100):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.n_iters = iterations
        self.Wih = np.random.random_sample((200,784))
        self.Who = np.random.random_sample((10,200))
        
    #sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #derivative of sigmoid function
    def sigmoid_derivative(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    # backpropagatoin
    def backpropagation(self, inputs, targets, output, weights):
        output_error = targets - output 
        output_delta = output_error * self.sigmoid_derivative(output)
        if weights == 'Wih':
            self.Wih = self.Wih + (np.dot(inputs, output_delta.T) * self.lr)
        
    #train the neural network
    def train(self, inputs, targets):
        for _ in range(self.n_iters):
            prediction = self.think(inputs)
            print(prediction.shape)
            E_out = targets - prediction
            E_hidden = np.dot(self.Who.T, E_out)
            
            delta_Who = np.dot(self.hidden_output, (E_out * self.sigmoid_derivative(prediction)).T)
            self.Who = self.Who + self.lr * delta_Who.T
            
            delta_Wih = np.dot(inputs, (E_hidden * self.sigmoid_derivative(self.hidden_output)).T)
            self.Wih = self.Wih + self.lr * delta_Wih.T             

    #one calculation step of the network
    def think(self, inputs):
        # inputlayer -> hiddenlayer
        self.hidden_output = self.sigmoid(np.dot(self.Wih, inputs))
        # hiddenlayer -> outputlayer
        output = self.sigmoid(np.dot(self.Who, self.hidden_output))
        return output
    
    def calc_output(self, inputs):
        return self.sigmoid(np.dot(inputs, self.Who))
    
    def test(self, inputs):
        return self.calc_output(inputs)
        

def plot(image_list):
    print("plotting image: ")
    #image_array = np.asfarray(raw_values[1:]).reshape((28,28))
    plt.imshow(np.array(image_list).reshape((28,28)),cmap='Greys', interpolation='None')
    plt.show(block = False)

def data_prep(input_list):
    targets = []
    inputs = []
    for imglist in input_list:
        imglist = [float(x) for x in np.array(imglist.split(','))]
        label = int(imglist[0])
        imglist = np.delete(imglist,0)
        input_img = [np.round(y/255, decimals=2) for y in imglist]
        # input_arr = np.array(input_img).reshape((28,28))
        target = [0.01 for n in range(10)]
        target[label] = 0.99
        targets.append(target)
        inputs.append(input_img)
    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, targets

if __name__ == "__main__":

    input_nodes = 784 #28*28 pixel
    hidden_nodes = 200 #voodoo magic number
    output_nodes = 10 #numbers from [0:9]

    learning_rate = 0.1 #feel free to play around with

    training_data_file = open("./Feed-Forward-Network/data/mnist_train_100.csv")
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    test_data_file = open("./Feed-Forward-Network/data/mnist_test_10.csv")
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)
    
    # get inputs as array of lists(784) and targets as lists(10,)
    inputs, targets = data_prep(training_data_list)
    
    # train network
    n.train(inputs.T, targets.T)
    
    test_inputs, test_targets = data_prep(test_data_list)
    output = n.test(test_inputs.T)
    
    print(test_targets)
    print(output.shape)