# speedsigndetector

wip machine vision project detect and classify finnish speedsings from images or videos. Currently includes 5 different detection methods with varying levels of success.

# Dependencies

See requirements.txt and additionally install pytorch (torch & torchvision)

# Usage

Start with python3 main.py `--algo <algorithm>` `--classifier <optional classifier>`

Where the algorithm can be one of the following:

* `color`: Color based detection of a speed signs red outer rim and using pytesseract to read the sign
* `shape`: Using opencv inbuilt HoughCircles to detect circles with the same pytesseract for sign content detection
* `sift`: Using the SIFT algorithm for both locating and classifying traffic sings, works reasonably well, but is not realtime
* `brisk`: similar approach to SIFT, but using BRISK and a brute force matcher. This can find sings, but is bad at classifying them.
* `haar`: Haar Cascade based detection. A larger classifier XML file is currently being trained

Classifier parameter is an optional parameter for `color` & `shape` based detectors. Using `--classifier cnn` will replace the pytesseract based sign reader with a cnn classifier.
