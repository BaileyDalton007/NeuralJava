package com.github.baileydalton007.utils;

import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import com.opencsv.CSVReader;

public class CSVLoader {

    public static double[][][] loadCSVToDataset(String fileName, int targets, boolean skipFirstRow) {
        CSVReader reader;

        double[][] dataX;
        double[][] dataY;

        try {
            reader = new CSVReader(new FileReader(fileName));
            String[] nextLine;
            int numRows;

            if (skipFirstRow) {
                reader.readNext();
                numRows = countRows(fileName) - 1;
            } else {
                numRows = countRows(fileName);
            }

            dataX = new double[numRows][];
            dataY = new double[numRows][];

            // Iterates through each row of the CSV file to parse.
            for (int lineIndex = 0; (nextLine = reader.readNext()) != null; lineIndex++) {
                dataY[lineIndex] = new double[targets];
                dataX[lineIndex] = new double[nextLine.length - targets];

                for (int i = 0; i < nextLine.length; i++) {
                    if (i < targets) {
                        dataY[lineIndex][i] = Double.parseDouble(nextLine[i]);
                    } else {
                        dataX[lineIndex][i - targets] = Double.parseDouble(nextLine[i]);
                    }
                }
            }

            return new double[][][] { dataX, dataY };

        } catch (

        Exception e) {
            e.printStackTrace();
        }

        return null;
    }
    // reads one line at a time
    // while ((nextLine = reader.readNext()) != null) {
    // String[] targetData = Arrays.copyOfRange(nextLine, targetIndices[0],
    // targetIndices[1] + 1);
    // String[] featureBeforeIndexData = Arrays.copyOfRange(nextLine, 0,
    // targetIndices[0]);
    // String[] featureAfterIndexData = Arrays.copyOfRange(nextLine,
    // targetIndices[1] + 1,
    // nextLine.length + 1);

    // String[] featureData = Stream
    // .concat(Arrays.stream(featureBeforeIndexData),
    // Arrays.stream(featureAfterIndexData))
    // .toArray(String[]::new);

    // }

    private static int countRows(String fileName) throws IOException {
        int lines = (int) Files.lines(Paths.get(fileName)).count();

        return lines;
    }
}