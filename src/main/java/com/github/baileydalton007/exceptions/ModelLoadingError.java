package com.github.baileydalton007.exceptions;

/**
 * Exception for when a model is loaded from a JSON file, but fails.
 * 
 * @author Bailey Dalton
 */
public class ModelLoadingError extends RuntimeException {
    /**
     * Constructor for a ModelLoadingError.
     * 
     * @param message The message to output when the error is thrown.
     */
    public ModelLoadingError(String message) {
        super(message);
    }
}
