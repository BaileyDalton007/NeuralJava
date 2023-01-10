package com.github.baileydalton007.utils;

/**
 * Utility class used to wrap text in a box for terminal output.
 * 
 * @author Bailey Dalton
 */
public class TextBox {
    // Stores the text that will be surrounded in the box.
    private String[] text;

    // The width in characters of the box.
    private int width;

    /**
     * Constructor for a text box.
     * 
     * @param textToSurround The text that will be formatted to be put in the box.
     * @param minWidth       The minimum width of the box.
     */
    public TextBox(String[] textToSurround, int minWidth) {
        // Saves the input text.
        text = textToSurround;

        // Starts the width at the minimum width.
        width = minWidth;

        // Iterates through the input text and sees if the length is greater than the
        // minWidth, and if so saves it.
        for (String str : textToSurround) {
            if (str.length() > width)
                width = str.length();
        }

        // Iterates through the input text, padding them to the width of the box.
        for (int i = 0; i < text.length; i++) {
            text[i] = "| " + text[i].concat(fillChar(' ', width - text[i].length() - 4) + " | \n");
        }
    }

    /**
     * Getter for the padded text that will be put into the box.
     * 
     * @param i The Line of text to return, maps to the input text
     * @return Padded string
     */
    public String getText(int i) {
        return text[i];
    }

    /**
     * Returns a dash line of the box's width.
     * 
     * @return String containing line of the boxes width
     */
    public String getTopBox() {
        return fillChar('_', width) + "\n";
    }

    /**
     * Returns a dash line with edges of the box's width.
     * 
     * @return String containing line with edges of the box's width
     */
    public String getBottom() {
        return "|" + fillChar('_', width - 2) + "|" + "\n";
    }

    /**
     * Returns a dash line with edges of the boxes width that also has edges at each
     * of the specified indices.
     * 
     * @param dividerIndices indices to place edges
     * @return String containing line with edges of the box's width that also has
     *         edges at each of the input indices
     */
    public String getDividerBox(int... dividerIndices) {
        // String builder to store the output.
        StringBuilder sb = new StringBuilder();

        // Stores the last edge added to the string.
        int lastEdge = 0;

        for (int i = 0; i < width; i++) {
            // Iterates through the divider indices to see if the current index should have
            // an edge.
            for (int index : dividerIndices) {
                // If the index should be an edge, fill area in between with dashes and add an
                // edge.
                if (index == i) {
                    sb.append(fillChar('_', i - (1 + lastEdge)) + "|");

                    // Updates the index of the last edge added to the string.
                    lastEdge = i;
                }
            }

        }

        // Fills the rest of the width with dashes.
        sb.append(fillChar('_', width - sb.length() - 2));

        // Wraps the output with edges.
        return "|" + sb.toString() + "|";
    }

    /**
     * Returns a string of length n containing character c.
     * 
     * @param c The character that should be repeated in the string
     * @param n The length of the string
     * @return string of length n containing character c
     */
    private String fillChar(char c, int n) {
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < n; i++) {
            sb.append(c);
        }

        return sb.toString();
    }

}
