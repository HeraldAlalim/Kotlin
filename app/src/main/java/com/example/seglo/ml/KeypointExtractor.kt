package com.example.seglo.ml

import android.util.Log
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseLandmark

class KeypointExtractor {
    val sequenceLength = 30
    val keypointSequence = mutableListOf<FloatArray>()

    companion object {
        // Using a subset of pose landmarks for simplicity
        // You may need to adjust this based on your model's input requirements
        private val POSE_LANDMARKS = listOf(
            PoseLandmark.NOSE,
            PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EYE_OUTER,
            PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EYE_OUTER,
            PoseLandmark.LEFT_EAR, PoseLandmark.RIGHT_EAR,
            PoseLandmark.LEFT_MOUTH, PoseLandmark.RIGHT_MOUTH,
            PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.LEFT_ELBOW, PoseLandmark.RIGHT_ELBOW,
            PoseLandmark.LEFT_WRIST, PoseLandmark.RIGHT_WRIST,
            PoseLandmark.LEFT_PINKY, PoseLandmark.RIGHT_PINKY,
            PoseLandmark.LEFT_INDEX, PoseLandmark.RIGHT_INDEX,
            PoseLandmark.LEFT_THUMB, PoseLandmark.RIGHT_THUMB,
            PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP,
            PoseLandmark.LEFT_KNEE, PoseLandmark.RIGHT_KNEE,
            PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE,
            PoseLandmark.LEFT_HEEL, PoseLandmark.RIGHT_HEEL,
            PoseLandmark.LEFT_FOOT_INDEX, PoseLandmark.RIGHT_FOOT_INDEX
        )
    }

    fun extractKeypoints(pose: Pose): FloatArray {
        val keypoints = mutableListOf<Float>()

        for (landmarkType in POSE_LANDMARKS) {
            val landmark = pose.getPoseLandmark(landmarkType)
            if (landmark != null) {
                keypoints.add(landmark.position.x)
                keypoints.add(landmark.position.y)
                // Note: ML Kit doesn't provide Z coordinate like MediaPipe
                keypoints.add(0f) // Placeholder for Z
            } else {
                // Add zeros for missing landmarks
                keypoints.add(0f)
                keypoints.add(0f)
                keypoints.add(0f)
            }
        }

        Log.d("KeypointExtractor", "Extracted ${keypoints.size} keypoint values")
        return keypoints.toFloatArray()
    }

    fun addToSequence(keypoints: FloatArray): Array<FloatArray>? {
        keypointSequence.add(keypoints)

        // Maintain sequence length
        if (keypointSequence.size > sequenceLength) {
            keypointSequence.removeAt(0)
        }

        return if (keypointSequence.size == sequenceLength) {
            Log.d("KeypointExtractor", "Sequence ready for prediction")
            keypointSequence.toTypedArray()
        } else {
            Log.d("KeypointExtractor", "Sequence progress: ${keypointSequence.size}/$sequenceLength")
            null
        }
    }

    fun resetSequence() {
        keypointSequence.clear()
    }
}