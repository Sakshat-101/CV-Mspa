# =========================
# Import Required Libraries
# =========================
import cv2
from fer.fer import FER
import matplotlib.pyplot as plt
import numpy as np
import os

# Initialize FER Emotion Detector
detector = FER(mtcnn=True)

# =========================
# Helper Function
# =========================
def draw_emotion_boxes(image, results):
    """
    Draws green rectangles and emotion labels on detected faces.
    """
    img_copy = image.copy()
    for face in results:
        (x, y, w, h) = face["box"]
        emotion, score = max(face["emotions"].items(), key=lambda x: x[1])
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img_copy, f"{emotion} ({score:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img_copy

# =========================
# Detect Emotion from Image
# =========================
def detect_emotion_in_image(image_path):
    """
    Detects emotion in a given image file and displays results.
    """
    img = plt.imread(image_path)
    results = detector.detect_emotions(img)
    if results:
        print("Detected emotions:")
        for r in results:
            print(r["emotions"])
        img_boxed = draw_emotion_boxes(img, results)
        plt.imshow(cv2.cvtColor(img_boxed, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        output_path = "output_" + os.path.basename(image_path)
        cv2.imwrite(output_path, cv2.cvtColor(img_boxed, cv2.COLOR_RGB2BGR))
        print(f"Saved output image as {output_path}")
    else:
        print("No face detected in the image.")

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    print("=== Facial Expression Detection ===")
    image_path = input("Enter image path (e.g., sample_images/test1.jpg): ")
    detect_emotion_in_image(image_path)
