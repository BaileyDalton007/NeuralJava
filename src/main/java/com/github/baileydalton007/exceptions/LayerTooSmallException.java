package com.github.baileydalton007.exceptions;

/**
 * Exception for when a Layer is initialized with too few neurons
 * 
 * @author Bailey Dalton
 */
public class LayerTooSmallException extends RuntimeException {
    /**
     * Constructor for a LayerTooSmallException.
     */
    public LayerTooSmallException() {
        super("Layer is too small to be created. The minimum amount of neurons a layer can have is 1.");
    }
}