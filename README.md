# Assignment-1-WANG-JiaWei
1.Introduction
The objective of this project was to develop a robust text detection and extraction system using Python, OpenCV, and Tesseract OCR. The system was designed to preprocess images, extract text, and perform various post-processing tasks to enhance text extraction accuracy. The code provided demonstrates various preprocessing techniques such as grayscale conversion, noise removal, thresholding, morphological operations, edge detection, skew correction, template matching, and visualization of detected text. This report evaluates the performance of the text detection system based on the provided code.

2.Methodology
2.1 Image Loading and Preprocessing:
Image is loaded using OpenCV.
Various preprocessing techniques are applied to enhance the image quality for better OCR results.
2.2 Text Extraction:
Tesseract OCR is used to extract text from the preprocessed images.
Various configurations of Tesseract OCR are tested to optimize text detection.
2.3 Text Detection Visualization:
Bounding boxes are drawn around detected text to visualize the accuracy of text detection.
Confidence scores are used to filter out low-confidence detections.
2.4 Date Detection:
Regular expressions are used to detect date patterns within the extracted text.
Bounding boxes are drawn around detected dates.

3.Preprocessing Techniques
3.1Grayscale Conversion:
Conversion of the image to grayscale to reduce complexity and enhance contrast.
cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
3.2Noise Removal:
Median blur is applied to remove noise.
cv2.medianBlur(image, 5)
3.3Thresholding:
Otsu's thresholding is applied to binarize the image.
cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
3.4Morphological Operations:
Dilation, erosion, and opening are applied to remove small noise and enhance text regions.
cv2.dilate(image, kernel, iterations=1)
cv2.erode(image, kernel, iterations=1)
cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
3.5Edge Detection:
Canny edge detection is applied to highlight edges in the image.
cv2.Canny(image, 100, 200)
3.6Skew Correction:
Skew correction is performed to align the text horizontally.
cv2.getRotationMatrix2D(center, angle, 1.0)
cv2.warpAffine(image,M,(w,h),flags=cv2.INTER_CUBIC, 	borderMode=cv2.BORDER_REPLICATE)

4.Evaluation of Text Detection System
4.1Accuracy of Text Extraction:
The accuracy of text extraction was evaluated by comparing the extracted text with the original text in the images.
Various preprocessing techniques were tested to determine their impact on OCR accuracy.
In general, thresholding and grayscale conversion significantly improved text extraction accuracy.
4.2 Confidence Scores and Filtering:
The use of confidence scores to filter out low-confidence detections was effective in reducing false positives.
A confidence threshold of 60 was used to filter out low-confidence text detections.
4.3 Detection of Dates:
Regular expressions were used to detect date patterns in the extracted text.
The system successfully detected dates and drew bounding boxes around them.
4.4 Visualization of Detected Text:
Bounding boxes were drawn around detected text to visualize the accuracy of text detection.
This visualization helped identify areas where the OCR system struggled, such as small text or text with complex backgrounds.

5.Results
5.1 Preprocessing Impact:
Grayscale conversion and thresholding significantly improved OCR accuracy.
Noise removal and morphological operations further enhanced text extraction by reducing noise and enhancing text regions.
5.2 Text Extraction:
The system successfully extracted text from images with varying degrees of accuracy depending on the preprocessing technique applied.
Thresholding and grayscale conversion were the most effective preprocessing techniques.

5.3 Confidence Filtering:
Using confidence scores to filter out low-confidence detections reduced false positives and improved overall accuracy.
A threshold of 60 was found to be effective in balancing accuracy and recall.
5.4 Date Detection:
The system successfully detected dates using regular expressions.
Bounding boxes were accurately drawn around detected dates.
5.5 Visualization:
Visualization of detected text using bounding boxes provided a clear understanding of the system's performance.
It highlighted areas where the OCR system performed well and areas where it struggled.
6.Conclusion
The text detection system developed using OpenCV and Tesseract OCR demonstrates robust performance in extracting text from images. The preprocessing techniques applied significantly enhance the accuracy of text extraction. Thresholding and grayscale conversion were found to be particularly effective in improving OCR accuracy.
The use of confidence scores to filter out low-confidence detections and regular expressions for date detection further improved the system's performance. Visualization of detected text using bounding boxes provided valuable insights into the system's strengths and weaknesses.
Overall, the text detection system is effective in extracting text from images, and the preprocessing techniques applied play a crucial role in enhancing OCR accuracy. Future work could focus on further optimizing preprocessing techniques and exploring advanced methods for handling complex backgrounds and small text regions.
