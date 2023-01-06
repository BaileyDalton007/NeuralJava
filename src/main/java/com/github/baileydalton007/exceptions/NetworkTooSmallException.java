package com.github.baileydalton007.exceptions;

/**
 * Exception for when a Network is initialized with too few layers.
 * 
 * @author Bailey Dalton
 */
public class NetworkTooSmallException extends Exception {
    /**
     * Constructor for a NetworkTooSmallException.
     */
    public NetworkTooSmallException() {
        super("Network is too small to be created. The minimum amount of layers a network can have is 2, try adding more.");
    }
}
