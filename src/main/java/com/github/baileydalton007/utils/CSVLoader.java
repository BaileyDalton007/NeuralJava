package com.github.baileydalton007.utils;

import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import com.github.baileydalton007.exceptions.DataError;
import com.opencsv.CSVReader;

/**
 * Utility class for loading a csv file into training data.
 * 
 * @author Bailey Dalton
 */
public class CSVLoader {
    /**
     * Loads an input CSV file into double arrays to be used with models. Will
     * return an array of matrices where the features are the first dimension while
     * the labels are the second.
     * 
     * EX:
     * double[][][] data = loadCSVToDataSet()
     * double[][] trainX = data[0]
     * double[][] trainY = data[1]
     * 
     * @param fileName     The name of the CSV file to parse.
     * @param targets      The number of targets (labels for training data) starting
     *                     at the beginning of each CSV row. The target values must
     *                     be in the first columns.
     * @param skipFirstRow CSVs often have column labels in the first row, this will
     *                     skip it.
     * @return An array containing two matrices, one being feature data and the
     *         other the labels (X and Y).
     */
    public static double[][][] loadCSVToDataset(String fileName, int targets, boolean skipFirstRow) {
        // Stores the CSV reader.
        CSVReader reader;

        // Creates two arrays for both the features and the labels.
        double[][] dataX;
        double[][] dataY;

        // Adjusts the file's name with the file extension.
        String file;
        if (fileName.contains(".csv"))
            file = fileName;
        else
            file = fileName + ".csv";

        try {
            // Opens the CSV file.
            reader = new CSVReader(new FileReader(file));

            // Stores the next line in the CSV.
            String[] nextLine;

            // Stores the total number of training examples (rows) in the CSV.
            int numRows;

            // Skips the first row if needed.
            if (skipFirstRow) {
                reader.readNext();
                numRows = countRows(file) - 1;
            } else {
                numRows = countRows(file);
            }

            // Makes a row in the matrices for each training example (row in the CSV).
            dataX = new double[numRows][];
            dataY = new double[numRows][];

            // Iterates through each row of the CSV file to parse.
            for (int lineIndex = 0; (nextLine = reader.readNext()) != null; lineIndex++) {
                // Fills the rows of the matrices.
                dataY[lineIndex] = new double[targets];
                dataX[lineIndex] = new double[nextLine.length - targets];

                // Iterates through the columns in the current row.
                for (int i = 0; i < nextLine.length; i++) {

                    // Appends the current column to either the features or label arrays (X or Y)
                    // depending on the amount of target columns passed in.
                    if (i < targets) {
                        dataY[lineIndex][i] = Double.parseDouble(nextLine[i]);
                    } else {
                        dataX[lineIndex][i - targets] = Double.parseDouble(nextLine[i]);
                    }
                }
            }

            // Returns a 3D tensor with the features (X) in the 0th index and the labels (Y)
            // in the 1st.
            return new double[][][] { dataX, dataY };

        } catch (

        Exception e) {
            throw new DataError("Error loading the dataset. Error Message: " + e.getMessage());
        }
    }

    /**
     * Simple method to count the number of rows in a CSV file.
     * 
     * @param fileName The name of the CSV file to count the rows of
     * @return The number of rows in the file
     * @throws IOException If the file cannot be found
     */
    private static int countRows(String fileName) throws IOException {
        int lines = (int) Files.lines(Paths.get(fileName)).count();

        return lines;
    }
}