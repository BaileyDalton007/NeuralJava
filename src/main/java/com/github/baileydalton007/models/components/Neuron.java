package com.github.baileydalton007.models.components;

/**
 * Class object for a deep learning neuron.
 * 
 * @author Bailey Dalton
 */
public class Neuron {
    // The input to the neuron prior to applying the activation function.
    private double input = 0.0;

    /**
     * The constructor for a neuron instance.
     */
    protected Neuron() {
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
     * Getter for the input a neuron is recieving prior to the application of the
     * activation function.
     * 
     * This is commonly denoted as the z term.
     * 
     * @return The weighted sum being passed into this neuron
     */
    public double getInput() {
        return this.input;
    }
}
