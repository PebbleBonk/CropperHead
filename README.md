# CropperHead

A simple convolutional neural network trained to automatically crop photos.

## Motivation
Cropping photos is quite an subjective art form. I wanted to see how a neural network would replicate my style of cropping photos.

## Technologies
Training data:
- extracted from the Exif metadata with [ExifExtractor](https://github.com/PebbleBonk/ExifAnnotator).

Network:
- Keras & TensorFlow
- TensorBoard

Development:
- Jupyter Notebook

## Results:
Questionable yet deterministic crops on images. No noticeable effect on data augmentation or custom loss functions. However, an entertaining project and a working end result, which can be tested with the related webapp, [CropperHeadUI](https://github.com/PebbleBonk/CropperHeadUI) at https://cropper-head.herokuapp.com (might take a minute for the Heroku dyno to wake up)

Also, in the end, what would even be considered as a working crop? More explicit criterion should be worked in...
