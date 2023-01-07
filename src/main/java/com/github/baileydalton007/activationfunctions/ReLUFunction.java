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

    /**
     * Method to apply the derivative of the ReLU function to an input.
     * Used for back propagation error calculations.
     * 
     * relu'(0) is undefined, I have decided to follow the convention of other
     * deeplearning libraries like TensorFlow and have assigned it to 0.
     * 
     * @param x Input to the ReLU derivative function
     * @return The output of the derivative ReLU function with x as an input.
     */
    @Override
    public Double derivative(Double x) {
        if (x <= 0.0)
            return 0.0;
        else
            return 1.0;
    }
}