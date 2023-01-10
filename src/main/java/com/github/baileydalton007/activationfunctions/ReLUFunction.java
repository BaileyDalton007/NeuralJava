package com.github.baileydalton007.activationfunctions;

/**
 * Rectified Linear Unit (ReLU) activation function for deep learning neurons.
 * 
 * @author Bailey Dalton
 */
public class ReLUFunction extends ActivationFunction {

    /**
     * Method to apply the ReLU function to a layer of inputs.
     * 
     * @param x Input array to the ReLU function
     * @return The output array of the ReLU function with outputs mapping to the
     *         input array.
     */
    @Override
    public double[] apply(double[] x) {
        // Create an array to store the outputs.
        double[] output = new double[x.length];

        // Iterate through each input, and storing the calculated value in the output
        // array.
        for (int i = 0; i < x.length; i++) {
            if (x[i] > 0.0)
                output[i] = x[i];
            else
                output[i] = 0.0;
        }

        return output;
    }

    /**
     * Method to apply the derivative of the ReLU function to an input array.
     * Used for back propagation error calculations.
     * 
     * relu'(0) is undefined, I have decided to follow the convention of other
     * deeplearning libraries like TensorFlow and have assigned it to 0.
     * 
     * @param x Input array to the ReLU derivative function
     * @return The output array of the derivative ReLU function that maps to the
     *         input array.
     */
    @Override
    public double[] derivative(double[] x) {
        // Create an array to store the outputs.
        double[] output = new double[x.length];

        // For each input, calculate the derivative and store it in the output.
        for (int i = 0; i < x.length; i++) {
            if (x[i] <= 0.0)
                output[i] = 0.0;
            else
                output[i] = 1.0;
        }

        return output;

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
    public double derivative(double x) {
        if (x <= 0.0)
            return 0.0;
        return 1.0;
    }

    /**
     * Returns "relu", the name of the activation function.
     * 
     * @return String representation of the activation function
     */
    @Override
    public String toString() {
        return "relu";
    }
}