package com.github.baileydalton007.models;

import com.github.baileydalton007.exceptions.IncompatibleInputException;
import com.github.baileydalton007.exceptions.NetworkTooSmallException;

import com.github.baileydalton007.models.components.Layer;
import com.github.baileydalton007.models.components.Neuron;
import com.github.baileydalton007.models.components.WeightMatrix;

/**
 * A class for Dense Neural Networks where every neuron in a layer is connected
 * to every neuron in the previous layer.
 * 
 * @author Bailey Dalton
 */
public class DenseNeuralNetwork {
    // Array for storing the layer objects in the network.
    private Layer[] layerArray;

    // Array for storing the weight matrix objects connecting each layer to the
    // previous in the network.
    private WeightMatrix[] layerWeights;

    /**
     * Constructor for a dense neural network object.
     * 
     * @param layerArray An array of layers that will make up the network
     * @throws NetworkTooSmallException Will be thrown if a network is initialized
     *                                  with less than 2 layers
     */
    public DenseNeuralNetwork(Layer[] layerArray) throws NetworkTooSmallException {
        // Throw an exception if a network is created with less than two layers.
        if (layerArray.length < 2)
            throw new NetworkTooSmallException();

        this.layerArray = layerArray;

        // Weight matrix array will be the same size as the amount of layers minus one
        // as the input layer layer will not have a weight matrix.
        this.layerWeights = new WeightMatrix[layerArray.length - 1];

        // Iterates through each set of weights (between two layers) and initializes an
        // appropriately sized weight matrix.
        for (int i = 0; i < layerWeights.length; i++) {
            layerWeights[i] = new WeightMatrix(layerArray[i + 1].size(), layerArray[i].size());
        }
    }

    /**
     * Forward propagation algorithm calculating activations through the network.
     * 
     * @param input The input array to the network, should be the same size as the
     *              input layer
     * @return The output of the network propagated from the input
     * @throws IncompatibleInputException Thrown if the size of the input array is
     *                                    not the same as the input layer.
     */
    public double[] ForwardPropagation(double[] input) throws IncompatibleInputException {
        // Checks that the input array is the same size as the input layer.
        if (input.length != layerArray[0].size())
            throw new IncompatibleInputException(
                    "Input size does not match size of first layer. Make sure that the input layer has the same amount of neurons as the input array.");

        // Feeds the input to the first layer.
        layerArray[0].input(input);

        // For each layer in the network, for each neuron in that layer, calculate the
        // activation by taking the weighted sum with all the previous layer's neurons.
        for (int layerIndex = 1; layerIndex < layerArray.length; layerIndex++) {

            // Stores the current layer being propagated.
            Layer currLayer = layerArray[layerIndex];

            // Stores the previous layer for calculating the weighted sum for the current
            // layer.
            Layer previousLayer = layerArray[layerIndex - 1];

            for (int neuronIndex = 0; neuronIndex < currLayer.size(); neuronIndex++) {

                // Stores the weighted sum that will become the input for the current neuron.
                double weightedSum = 0.0;

                // Stores the current neuron to which the input is being calculated.
                Neuron currNeuron = currLayer.getNeuron(neuronIndex);

                // Iterates through neurons in the previous layer, multiplies each value by its
                // weight, then adds to the sum total.
                for (int i = 0; i < previousLayer.size(); i++) {

                    weightedSum += previousLayer.getNeuron(i).getActivation() // The previous neuron's activation.
                            * this.layerWeights[layerIndex - 1].getMatrix()[neuronIndex][i]; // The weight connecting
                                                                                             // the current neuron to
                                                                                             // the previous neuron.
                }

                currNeuron.setInput(weightedSum);
            }

        }

        return getOutput();
    }

    /**
     * Gets the output of the network, the activations of the last layer in the
     * network.
     * 
     * @return An array of values representing the activations of the output layer
     */
    private double[] getOutput() {
        return layerArray[layerArray.length - 1].getLayerActivations();
    }

    public void backPropagation(double[] input, double[] truth) {
        // Propagates the network forward with the given inputs to activate the
        // network's neurons.
        ForwardPropagation(input);

        // Propagates and stores a matrix storing the error for each neuron.
        // The first index for layer L is (L-1) since the input layer is not included.
        double[][] errorMatrix = propagateErrorMatrix(truth);
    }

    /**
     * Propagates the error for each neuron in the network working backwards.
     * 
     * The first index for layer L is (L-1) since the input layer is not included.
     * The second index is that of the neuron in the layer.
     * 
     * EX: The error value for the 2nd index neuron on layer 4 would be:
     * errorMatrix[3][2]
     * 
     * @param truth The expected value of the network.
     * @return 2D error matrix giving the error for each neuron in the network
     *         (excluding the input layer).
     */
    public double[][] propagateErrorMatrix(double[] truth) {
        // Creates a matrix to store each neuron's error.
        // Amount of errors - 1 since input layer is not included.
        double[][] neuronError = new double[layerArray.length - 1][];

        // Iterates backward through the netowrk's layers.
        // Does not include the input layer as the error is not needed.
        for (int layerIndex = this.layerArray.length - 1; layerIndex > 0; layerIndex--) {

            // Stores the current layer that will have its error calculated.
            Layer currLayer = layerArray[layerIndex];

            // Creates a row in the matrix with the amount of columns needed to store the
            // error for each neuron in the layer.
            // layerIndex - 1 since input layer is not included.
            neuronError[layerIndex - 1] = new double[currLayer.size()];

            // Stores the activations for the current layer.
            double[] activations = currLayer.getLayerActivations();

            // Checks if the current layer is the output layer to use different error
            // calculation.
            if (layerIndex == this.layerArray.length - 1) {

                // Iterates through neurons in current (output) layer to calculate error for
                // each.
                for (int neuronIndex = 0; neuronIndex < currLayer.size(); neuronIndex++) {
                    // ERROR = f'(ACTIVATION) * (TRUTH - ACTIVATION)
                    // where f'() is the derivative of the neuron's activation function.
                    // layerIndex - 1 since input layer is not included.
                    neuronError[layerIndex - 1][neuronIndex] = (currLayer.getActivationFunction()
                            .derivative(activations[neuronIndex])) * (truth[neuronIndex] - activations[neuronIndex]);
                }

            } else {
                // If this layer is not an output layer.

                // Iterates through neurons in current (hidden) layer to calculate error for
                // each.
                for (int neuronIndex = 0; neuronIndex < currLayer.size(); neuronIndex++) {
                    // Stores the sum of the errors and weights in the next layer.
                    double errSum = 0.0;

                    // Iterates through each neuron, adding the product of the error and the weight
                    // connecting it to the current neuron to the sum.
                    for (int k = 0; k < layerArray[layerIndex + 1].size(); k++) {
                        // layerIndex + 1 - 1 for the next layer's errors since input layer is not
                        // included (for both error and weight matrices).
                        errSum += neuronError[layerIndex][k] * layerWeights[layerIndex].getMatrix()[k][neuronIndex];
                    }

                    // ERROR = f'(ACTIVATION) * (ERROR_SUM)
                    // Where f'() is the derivative of the neuron's activation function.
                    // And ERROR_SUM is the sum of the products of all the errors and weights in the
                    // next layer.
                    //
                    // layerIndex - 1 since input layer is not included.
                    neuronError[layerIndex - 1][neuronIndex] = currLayer.getActivationFunction()
                            .derivative(activations[neuronIndex]) * errSum;
                }
            }
        }

        return neuronError;
    }
}
