# Live-Car-Number-Plates-Detection
This repository contains a live license plate recognition algorithm that utilizes the Haar cascade classifier model "haarcascade_russian_plate_number.xml" for license plate detection. The algorithm employs advanced techniques such as segmentation and contour detection to accurately extract license plate information from a live video stream or webcam feed.

The Haar cascade classifier model, specifically trained for Russian license plate detection, is included in the repository. This model has been trained using positive and negative samples to identify the specific patterns and features present in Russian license plates.

To use this algorithm, please follow the steps below:

Clone or download the repository to your local machine.

Ensure that you have the required dependencies installed. The algorithm relies on OpenCV, NumPy, and other common libraries for image processing and computer vision tasks. You can find the complete list of dependencies and their versions in the requirements.txt file.

Connect a webcam to your computer or prepare a live video stream source.

Run the live_license_plate_recognition.py script provided in the repository. This script initializes the webcam or video stream and performs real-time license plate recognition.

During the live video feed, the algorithm applies the Haar cascade classifier model, haarcascade_russian_plate_number.xml, to detect license plates. It then utilizes segmentation techniques to isolate the license plate region from the rest of the image. Finally, contour detection is employed to extract the alphanumeric characters from the license plate.

The recognized license plate numbers are displayed on the video feed or saved to a log file, depending on the configuration set in the script.

Please note that this algorithm is specifically trained and optimized for Russian license plates. For accurate recognition of license plates from other regions or countries, additional training or fine-tuning may be required.

The empty 'plates' folder allows for easy organization and retrieval of the recognized license plate images. As soon as the algorithm detects a license plate, it will save a snapshot of the plate as a separate image file in this folder. This enables further analysis, archiving, or post-processing of the recognized license plate

We encourage you to explore and modify the algorithm to suit your specific needs. If you encounter any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request on the GitHub repository.

Happy live license plate recognition!
