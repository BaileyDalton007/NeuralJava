package com.github.baileydalton007.models.components;

import com.github.baileydalton007.activationfunctions.ActivationFunction;
import com.github.baileydalton007.activationfunctions.ReLUFunction;
import com.github.baileydalton007.activationfunctions.SigmoidFunction;
import com.github.baileydalton007.activationfunctions.SoftmaxFunction;
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

    // Stores the layer's activation function.
    private ActivationFunction activationFunction;

    // Storing the layer's current activations optimize run time.
    // Instead of iterating through each neuron every time the layer's activations
    // are needed, they will be stored and only updated when the activations change.
    private double[] currActivations;

    /**
     * Constructor for a layer instance.
     * 
     * @param numNeurons               The number of neurons that should make up the
     *                                 layer, must be 1 or more
     * @param activationFunctionString The activation function that each neuron in
     *                                 the
     *                                 layer should use. For Rectified Linear Unit
     *                                 use
     *                                 "relu", for sigmoid use "sigmoid"
     */
    public Layer(int numNeurons, String activationFunctionString) {

        // Throws an exception if a layer is made with less than one neuron.
        if (numNeurons < 1)
            throw new LayerTooSmallException();

        // Creates an array to store the neurons in the layer.
        neurons = new Neuron[numNeurons];

        // Creates an array to store the activations of a layer between propagation
        // cycles.
        currActivations = new double[numNeurons];

        // Sets the activation function based on the input string.
        if (activationFunctionString.toLowerCase().equals("relu"))
            activationFunction = new ReLUFunction();

        else if (activationFunctionString.toLowerCase().equals("sigmoid"))
            activationFunction = new SigmoidFunction();

        else if (activationFunctionString.toLowerCase().equals("softmax"))
            activationFunction = new SoftmaxFunction();

        else
            // If the function is unknown, throw an exception.
            throw new UnknownActivationFunction(activationFunctionString
                    + " is an unknown activation function string, try one listed in the documentation.");

        // Creates and stores neurons in the array.
        for (int i = 0; i < numNeurons; i++) {
            neurons[i] = new Neuron();
        }
    }

    /**
     * Returns a string describing the layer.
     * Number of neurons is the number of neurons in the layer, while activation is
     * the activation function on the layer.
     * 
     * @return String representation of the layer.
     */
    @Override
    public String toString() {
        String output = String.format("| Number of Neurons: %4d | Activation: %10s |", size(),
                activationFunction.toString());

        return output;
    }

    /**
     * Returns an array of the activations of the neurons in the layer.
     * 
     * @return Array of activations of the neurons in the layer.
     */
    public double[] getLayerActivations() {
        return this.currActivations;
    }

    /**
     * Updates the layer's activations to be stored for optimization purposes.
     */
    public void updateLayerActivations() {
        this.currActivations = activationFunction.apply(getLayerInputs());
    };

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
     * Getter for this layer's neurons inputs. Used to calculate the activations for
     * the layer.
     * 
     * @return Array of inputs to each neuron in the layer
     */
    public double[] getLayerInputs() {
        // Stores the input to each neuron in the layer.
        double[] inputs = new double[this.size()];

        // Iterates through each neuron in the layer and stores its input.
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = getNeuron(i).getInput();
        }

        return inputs;
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

    /**
     * A getter for a layer's activation function.
     * 
     * @return This layer's activation function object
     */
    public ActivationFunction getActivationFunction() {
        return this.activationFunction;
    }

}
