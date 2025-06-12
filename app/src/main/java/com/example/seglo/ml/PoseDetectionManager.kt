package com.example.seglo.ml

import android.graphics.Bitmap
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.accurate.AccuratePoseDetectorOptions
import kotlinx.coroutines.tasks.await

class PoseDetectionManager {
    private val poseDetector: PoseDetector

    init {
        val options = AccuratePoseDetectorOptions.Builder()
            .setDetectorMode(AccuratePoseDetectorOptions.STREAM_MODE)
            .build()
        poseDetector = PoseDetection.getClient(options)
    }

    suspend fun detectPose(bitmap: Bitmap): Pose? {
        return try {
            val image = InputImage.fromBitmap(bitmap, 0)
            poseDetector.process(image).await()
        } catch (e: Exception) {
            Log.e("PoseDetection", "Error detecting pose", e)
            null
        }
    }

    fun close() {
        poseDetector.close()
    }
}