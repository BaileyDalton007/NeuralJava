package com.github.baileydalton007.exceptions;

/**
 * Exception for when a WeightMatrix is initialized with 0 as a dimension.
 * 
 * @author Bailey Dalton
 */
public class MatrixTooSmallException extends RuntimeException {
    /**
     * Constructor for a MatrixTooSmallException.
     */
    public MatrixTooSmallException() {
        super("WeightMatrix cannot have a dimension of 0");
    }
}
