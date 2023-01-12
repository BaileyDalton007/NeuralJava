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
     * 
     * @param The message that will be output to the user
     */
    public UnknownActivationFunction(String message) {
        super(message);
    }
}