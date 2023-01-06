package com.github.baileydalton007.activationfunctions;

/**
 * Rectified Linear Unit (ReLU) activation function for deep learning neurons.
 * 
 * @author Bailey Dalton
 */
public class ReLUFunction extends ActivationFunction {

    /**
     * Method to apply the ReLU function to an input.
     * 
     * @param x Input to the ReLU function
     * @return The output of the ReLU function with x as an input.
     */
    @Override
    public Double apply(Double x) {
        if (x > 0.0)
            return x;
        return 0.0;
    }
}