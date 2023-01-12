package com.github.baileydalton007.exceptions;

/**
 * Exception for when an input is not compatible with the network or method to
 * which it is passed.
 * 
 * @author Bailey Dalton
 */
public class IncompatibleInputException extends RuntimeException {
    /**
     * Constructor for a IncompatibleInputException.
     * 
     * @param message The message that will be output to the user.
     */
    public IncompatibleInputException(String message) {
        super(message);
    }
}
