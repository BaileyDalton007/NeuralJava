package com.github.baileydalton007.models.components;

import com.github.baileydalton007.activationfunctions.ActivationFunction;

/**
 * Class object for a deep learning neuron.
 * 
 * @author Bailey Dalton
 */
public class Neuron {
    // The input to the neuron prior to applying the activation function.
    private double input = 0.0;

    // The activation of this neuron, the input to the neuron passed through its
    // activation function.
    private double activation = 0.0;

    // The non-linear activation function applied to the input to this neuron to
    // determine the activation of the neuron.
    private ActivationFunction activationFunction;

    /**
     * The constructor for a neuron instance.
     * 
     * @param activationFunction The non-linear activation function that should be
     *                           used to determine the activation of this neuron
     */
    protected Neuron(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    /**
     * Gives an input to the neuron.
     * 
     * @param input the value that should be input to the neuron
     */
    public void setInput(double input) {
        this.input = input;
    }

    /**
     * Calulates the activation of the neuron from the input currently being stored.
     * 
     * @return the activation of the neuron
     */
    public double getActivation() {
        // Applies the activation function to the input to calculate the activation.
        activation = activationFunction.apply(input);

        return activation;
    }
}
