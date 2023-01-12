package com.github.baileydalton007.exceptions;

/**
 * Exception for there is an error when computing a tensor operation.
 * 
 * @author Bailey Dalton
 */
public class TensorOperationException extends RuntimeException {
    /**
     * Constructor for a TensorOperationException.
     * 
     * @param message The message that will be output to the user.
     */
    public TensorOperationException(String message) {
        super(message);
    }
}