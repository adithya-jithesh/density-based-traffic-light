# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the density-based traffic light control system."""

import argparse
import sys
import time

import cv2
import RPi.GPIO as GPIO
from tflite_support.task import core, processor, vision


# ---------------------------------------------------------------------------
# GPIO Pin Definitions (BCM numbering)
# ---------------------------------------------------------------------------
NORTH_RED_PIN    = 17
NORTH_YELLOW_PIN = 18
NORTH_GREEN_PIN  = 27

EAST_RED_PIN     = 22
EAST_YELLOW_PIN  = 23
EAST_GREEN_PIN   = 24

WEST_RED_PIN     = 5
WEST_YELLOW_PIN  = 6
WEST_GREEN_PIN   = 13

SOUTH_RED_PIN    = 19
SOUTH_YELLOW_PIN = 20
SOUTH_GREEN_PIN  = 21

ALL_PINS = [
    NORTH_RED_PIN, NORTH_YELLOW_PIN, NORTH_GREEN_PIN,
    EAST_RED_PIN,  EAST_YELLOW_PIN,  EAST_GREEN_PIN,
    WEST_RED_PIN,  WEST_YELLOW_PIN,  WEST_GREEN_PIN,
    SOUTH_RED_PIN, SOUTH_YELLOW_PIN, SOUTH_GREEN_PIN,
]

# ---------------------------------------------------------------------------
# GPIO Setup
# ---------------------------------------------------------------------------
GPIO.setmode(GPIO.BCM)
for pin in ALL_PINS:
    GPIO.setup(pin, GPIO.OUT)


# ---------------------------------------------------------------------------
# Traffic Density Logic
# ---------------------------------------------------------------------------
def get_traffic_density(num_objects: int) -> str:
    """Classify traffic density based on detected vehicle count."""
    if num_objects == 1:
        return "Low Traffic"
    elif num_objects == 2:
        return "Medium Traffic"
    elif num_objects >= 3:
        return "High Traffic"
    else:
        return "Unknown"


def set_north_signal(density: str) -> None:
    """Control North direction LEDs based on traffic density."""
    if density == "High Traffic":
        # Green — extended time for heavy traffic
        GPIO.output(NORTH_RED_PIN,    GPIO.LOW)
        GPIO.output(NORTH_YELLOW_PIN, GPIO.LOW)
        GPIO.output(NORTH_GREEN_PIN,  GPIO.HIGH)
        time.sleep(5)
    elif density == "Medium Traffic":
        # Yellow — default timing
        GPIO.output(NORTH_RED_PIN,    GPIO.LOW)
        GPIO.output(NORTH_YELLOW_PIN, GPIO.HIGH)
        GPIO.output(NORTH_GREEN_PIN,  GPIO.LOW)
        time.sleep(3)
    elif density == "Low Traffic":
        # Red — shorter green time (stay red longer)
        GPIO.output(NORTH_RED_PIN,    GPIO.HIGH)
        GPIO.output(NORTH_YELLOW_PIN, GPIO.LOW)
        GPIO.output(NORTH_GREEN_PIN,  GPIO.LOW)
        time.sleep(2)
    else:
        # Unknown — all off
        GPIO.output(NORTH_RED_PIN,    GPIO.LOW)
        GPIO.output(NORTH_YELLOW_PIN, GPIO.LOW)
        GPIO.output(NORTH_GREEN_PIN,  GPIO.LOW)
        time.sleep(2)


# ---------------------------------------------------------------------------
# Main Detection Loop
# ---------------------------------------------------------------------------
def run(model: str, camera_id: int, width: int, height: int,
        num_threads: int, enable_edgetpu: bool) -> None:
    """Continuously run inference on camera frames and control traffic lights.

    Args:
        model:          Path to the TFLite object detection model.
        camera_id:      OpenCV camera index.
        width:          Frame capture width in pixels.
        height:         Frame capture height in pixels.
        num_threads:    Number of CPU threads for the model.
        enable_edgetpu: Whether to use an Edge TPU accelerator.
    """
    counter, fps = 0, 0.0
    start_time = time.time()

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization settings
    row_size       = 20
    left_margin    = 24
    text_color     = (0, 0, 255)  # Red
    font_size      = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Load TFLite object detection model
    base_options      = core.BaseOptions(file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.3)
    options           = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
    detector          = vision.ObjectDetector.create_from_options(options)

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                sys.exit("ERROR: Unable to read from webcam. Please verify your webcam settings.")

            counter += 1

            # Run inference
            rgb_image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = vision.TensorImage.create_from_array(rgb_image)
            detection_result     = detector.detect(input_tensor)
            num_objects_detected = len(detection_result.detections)
            traffic_density      = get_traffic_density(num_objects_detected)

            # Control traffic lights based on density
            set_north_signal(traffic_density)

            # Overlay traffic density text
            cv2.putText(image, f"Traffic Density: {traffic_density}",
                        (left_margin, row_size),
                        cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)

            # Calculate and overlay FPS
            if counter % fps_avg_frame_count == 0:
                end_time   = time.time()
                fps        = fps_avg_frame_count / (end_time - start_time)
                start_time = time.time()

            cv2.putText(image, f"FPS = {fps:.1f}",
                        (left_margin, row_size * 2),
                        cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)

            cv2.imshow("Density-Based Traffic Light System", image)

            # Press ESC to quit
            if cv2.waitKey(1) == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model",        help="Path to TFLite object detection model.", default="best.tflite")
    parser.add_argument("--cameraId",     help="Camera index for OpenCV.",               default=0,   type=int)
    parser.add_argument("--frameWidth",   help="Frame capture width in pixels.",         default=640, type=int)
    parser.add_argument("--frameHeight",  help="Frame capture height in pixels.",        default=480, type=int)
    parser.add_argument("--numThreads",   help="Number of CPU threads for the model.",   default=4,   type=int)
    parser.add_argument("--enableEdgeTPU",help="Run model on Edge TPU.",                 action="store_true", default=False)

    args = parser.parse_args()
    run(args.model, args.cameraId, args.frameWidth, args.frameHeight,
        args.numThreads, args.enableEdgeTPU)


if __name__ == "__main__":
    main()
