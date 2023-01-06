package com.github.baileydalton007.activationfunctions;

import java.util.function.Function;

/**
 * Abstract class encompassing all activation functions.
 * 
 * @author Bailey Dalton
 */
public abstract class ActivationFunction implements Function<Double, Double> {

    /**
     * Method stub implementing Function's apply method for doubles.
     * 
     * @param x The input to the activation function.
     * @return The output of the activation function.
     */
    @Override
    public abstract Double apply(Double x);
}
