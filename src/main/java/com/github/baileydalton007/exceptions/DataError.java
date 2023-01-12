package com.github.baileydalton007.exceptions;

/**
 * Exception for when an error arises
 * which it is passed.
 * 
 * @author Bailey Dalton
 */
public class DataError extends RuntimeException {
    /**
     * Constructor for a DataError.
     * 
     * @param The message that will be output to the user.
     */
    public DataError(String message) {
        super(message);
    }
}
