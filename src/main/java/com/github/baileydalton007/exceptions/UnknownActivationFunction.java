package com.github.baileydalton007.exceptions;

/**
 * Exception for when the inputted string for an activation function does not
 * match any currently known.
 * 
 * @author Bailey Dalton
 */
public class UnknownActivationFunction extends RuntimeException {
    /**
     * Constructor for a UnknownActivationFunction.
     */
    public UnknownActivationFunction(String message) {
        super(message);
    }
}