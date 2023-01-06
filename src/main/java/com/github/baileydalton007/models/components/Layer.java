package com.github.baileydalton007.models.components;

import com.github.baileydalton007.activationfunctions.ActivationFunction;
import com.github.baileydalton007.activationfunctions.ReLUFunction;
import com.github.baileydalton007.activationfunctions.SigmoidFunction;
import com.github.baileydalton007.exceptions.IncompatibleInputException;
import com.github.baileydalton007.exceptions.LayerTooSmallException;
import com.github.baileydalton007.exceptions.UnknownActivationFunction;

/**
 * Class for layers that make up neural networks.
 * 
 * @author Bailey Dalton
 */
public class Layer {
    // Array storing the layer's neurons.
    private Neuron[] neurons;

    /**
     * Constructor for a layer instance.
     * 
     * @param numNeurons         The number of neurons that should make up the
     *                           layer, must be 1 or more
     * @param activationFunction The activation function that each neuron in the
     *                           layer should use. For Rectified Linear Unit use
     *                           "relu", for sigmoid use "sigmoid"
     */
    public Layer(int numNeurons, String activationFunctionString) {

        // Throws an exception if a layer is made with less than one neuron.
        if (numNeurons < 1)
            throw new LayerTooSmallException();

        // Creates an array to store the neurons in the layer.
        neurons = new Neuron[numNeurons];

        // Sets the activation function based on the input string.
        ActivationFunction activationFunction;
        if (activationFunctionString.toLowerCase() == "relu")
            activationFunction = new ReLUFunction();

        else if (activationFunctionString.toLowerCase() == "sigmoid")
            activationFunction = new SigmoidFunction();

        else
            // If the function is unknown, throw an exception.
            throw new UnknownActivationFunction(activationFunctionString
                    + " is an unknown activation function string, try one listed in the documentation.");

        // Creates and stores neurons in the array.
        for (int i = 0; i < numNeurons; i++) {
            neurons[i] = new Neuron(activationFunction);
        }
    }

    /**
     * Returns an array of the activations of the neurons in the layer.
     * 
     * @return Array of activations of the neurons in the layer.
     */
    public double[] getLayerActivations() {
        // Creates an array to store activations for the neurons in the layer.
        double[] activations = new double[neurons.length];

        // Iterates through neurons to calculate and store the activation for each.
        for (int i = 0; i < neurons.length; i++) {
            activations[i] = getNeuron(i).getActivation();
        }

        return activations;
    }

    /**
     * Getter for the amount of neurons in a layer.
     * 
     * @return The amount of neurons in a layer
     */
    public int size() {
        return neurons.length;
    }

    /**
     * Inputs an array of inputs to the layer of neurons.
     * 
     * @param input Array of inputs
     * @throws IncompatibleInputException Thrown if the size of input does not match
     *                                    the size of the layer.
     */
    public void input(double[] input) throws IncompatibleInputException {
        // Checks that the input is the same size as the layer.
        if (input.length != this.size())
            throw new IncompatibleInputException(
                    "Input size of (" + input.length + ") does not match size of layer (" + this.size() + ").");

        // Iterates through neurons and sets the input for each neuron.
        for (int i = 0; i < this.size(); i++) {
            getNeuron(i).setInput(input[i]);
        }
    }

    /**
     * Returns the neuron in a layer at a given index.
     * 
     * @param index The index of the neuron that should be accessed
     * @return The neuron in the layer at the passed in index
     */
    public Neuron getNeuron(int index) {
        return neurons[index];
    }
}
