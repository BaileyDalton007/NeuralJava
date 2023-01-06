package com.github.baileydalton007.exceptions;

/**
 * Exception for when an input is not compatible with the network it is passed
 * to.
 * 
 * @author Bailey Dalton
 */
public class IncompatibleInputException extends RuntimeException {
    /**
     * Constructor for a IncompatibleInputException.
     */
    public IncompatibleInputException(String message) {
        super(message);
    }
}
