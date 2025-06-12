package com.example.seglo.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class SignLanguageProcessor(context: Context) {
    private val mlModelManager = MLModelManager(context)
    private val poseDetectionManager = PoseDetectionManager()
    private val keypointExtractor = KeypointExtractor()

    suspend fun processFrame(bitmap: Bitmap): ProcessingResult = withContext(Dispatchers.IO) {
        try {
            // Step 1: Detect pose
            val pose = poseDetectionManager.detectPose(bitmap)
            if (pose == null) {
                return@withContext ProcessingResult.NoPoseDetected
            }

            // Step 2: Extract keypoints
            val keypoints = keypointExtractor.extractKeypoints(pose)

            // Step 3: Add to sequence and check if ready for prediction
            val sequence = keypointExtractor.addToSequence(keypoints)
            if (sequence == null) {
                return@withContext ProcessingResult.SequenceBuilding(keypointExtractor.sequenceLength, keypointExtractor.keypointSequence.size)
            }

            // Step 4: Make prediction
            val (label, confidence) = mlModelManager.predict(sequence)

            return@withContext ProcessingResult.Prediction(label, confidence)

        } catch (e: Exception) {
            Log.e("SignLanguageProcessor", "Error processing frame", e)
            return@withContext ProcessingResult.Error(e.message ?: "Unknown error")
        }
    }

    fun resetSequence() {
        keypointExtractor.resetSequence()
    }

    fun close() {
        mlModelManager.close()
        poseDetectionManager.close()
    }
}

sealed class ProcessingResult {
    object NoPoseDetected : ProcessingResult()
    data class SequenceBuilding(val required: Int, val current: Int) : ProcessingResult()
    data class Prediction(val label: String, val confidence: Float) : ProcessingResult()
    data class Error(val message: String) : ProcessingResult()
}