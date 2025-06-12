package com.example.seglo.ml

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

class MLModelManager(private val context: Context) {
    private var interpreter: Interpreter? = null
    private var labelMap: Map<String, Int> = emptyMap()
    private var reverseLabelMap: Map<Int, String> = emptyMap()

    init {
        loadModel()
        loadLabelMap()
    }

    private fun loadModel() {
        try {
            val modelBuffer = loadModelFile("sign_language_model.tflite")
            interpreter = Interpreter(modelBuffer)
            Log.d("MLModel", "Model loaded successfully")
        } catch (e: Exception) {
            Log.e("MLModel", "Error loading model", e)
        }
    }

    private fun loadLabelMap() {
        try {
            val jsonString = context.assets.open("label_map.json")
                .bufferedReader().use { it.readText() }
            val type = object : TypeToken<Map<String, Int>>() {}.type
            labelMap = Gson().fromJson(jsonString, type)
            reverseLabelMap = labelMap.entries.associate { it.value to it.key }
            Log.d("MLModel", "Label map loaded: $labelMap")
        } catch (e: Exception) {
            Log.e("MLModel", "Error loading label map", e)
        }
    }

    private fun loadModelFile(fileName: String): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd(fileName)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun predict(sequence: Array<FloatArray>): Pair<String, Float> {
        return try {
            interpreter?.let { interpreter ->
                // Prepare input: [1, 30, feature_size]
                val input = Array(1) { sequence }
                val output = Array(1) { FloatArray(labelMap.size) }

                // Run inference
                interpreter.run(input, output)

                // Get prediction
                val probabilities = output[0]
                val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
                val confidence = probabilities[maxIndex]
                val label = reverseLabelMap[maxIndex] ?: "Unknown"

                Log.d("MLModel", "Prediction: $label, Confidence: $confidence")
                Pair(label, confidence)
            } ?: Pair("Model not loaded", 0f)
        } catch (e: Exception) {
            Log.e("MLModel", "Error during prediction", e)
            Pair("Error", 0f)
        }
    }

    fun close() {
        interpreter?.close()
    }
}