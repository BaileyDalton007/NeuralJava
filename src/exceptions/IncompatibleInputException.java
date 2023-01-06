package exceptions;

/**
 * Exception for when an input is not compatible with the network it is passed
 * to.
 * 
 * @author Bailey Dalton
 */
public class IncompatibleInputException extends Exception {
    /**
     * Constructor for a IncompatibleInputException.
     */
    public IncompatibleInputException(String message) {
        super(message);
    }
}
