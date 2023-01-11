package com.github.baileydalton007.utils;

import org.json.JSONArray;
import org.json.JSONObject;
import java.io.FileWriter;
import java.io.IOException;

import com.github.baileydalton007.models.components.BiasUnit;
import com.github.baileydalton007.models.components.Layer;
import com.github.baileydalton007.models.components.WeightMatrix;

/**
 * Class for handling JSON reading and writing for saving model configurations.
 * 
 * @author Bailey Dalton
 */
public class JSONHandler {
    /**
     * Takes the configuration of a model and writes them to a JSON file to saved to
     * disk and loaded later.
     * 
     * @param layerArray  The array of layers from the model to save
     * @param weightArray The array of weight matrices from the model to save
     * @param biasArray   The array of biases from the model to save
     * @param fileName    The name of the output JSON file
     */
    public static void saveModelToJSONFile(Layer[] layerArray, WeightMatrix[] weightArray, BiasUnit[] biasArray,
            String fileName) {
        // Creates an array to store the model layer JSON objects.
        JSONArray output = new JSONArray();

        // Interates through the layers in the model and adds them to the output.
        for (int i = 0; i < layerArray.length; i++) {
            // Stores the current layer.
            Layer layer = layerArray[i];

            // Adds the layer's size and activation to the JSON object.
            JSONObject jLayer = new JSONObject()
                    .put("size", layer.size())
                    .put("activation", layer.getActivationFunction().toString());

            // If the layer is not the input layer, add weights and biases to the JSON
            // object.
            if (i > 0) {
                // If the layer is not the
                jLayer
                        .put("weights", weightArray[i - 1].getMatrix())
                        .put("bias", biasArray[i - 1].getValue());
            }

            // Add the layer's JSON object to the output.
            output.put(jLayer);
        }

        // Write the output JSON array to a file with the name fileName.json.
        try {
            FileWriter file;

            // Will be the same fileName independent if the user puts .json or not.
            if (fileName.contains(".json"))
                file = new FileWriter(fileName);
            else
                file = new FileWriter(fileName + ".json");

            file.write(output.toString());
            file.close();

        } catch (IOException e) {
            e.printStackTrace();

        }
    }
}
