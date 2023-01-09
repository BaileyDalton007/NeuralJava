package com.github.baileydalton007.activationfunctions;

import java.util.function.Function;

/**
 * Abstract class encompassing all activation functions.
 * 
 * @author Bailey Dalton
 */
public abstract class ActivationFunction implements Function<double[], double[]> {

    /**
     * Method stub implementing Function's apply method for arrays of doubles.
     * 
     * @param x The input array to the activation function.
     * @return The output array of the activation function.
     */
    @Override
    public abstract double[] apply(double[] x);

    /**
     * Method stub implementing an activation function's derivative function for
     * input arrays. Should be used when propagating error in weights.
     * Used for back propagation error calculations.
     * 
     * @param x The input array to the derivative function.
     * @return The output array of the derivative function.
     */
    public abstract double[] derivative(double[] x);

    /**
     * Method stub implementing an activation function's derivative function.
     * Used for back propagation error calculations. Should be used when propagating
     * error in biases.
     * 
     * @param x The input to the derivative function.
     * @return The output of the derivative function.
     */
    public abstract double derivative(double x);
}
