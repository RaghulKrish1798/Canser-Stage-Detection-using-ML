# CSE573-Final-Project
Repository for the CSE573 course project for the Spring 2023 semester

Project Outline:

1) Retrieved WSI (Whole-Slide Image) of Cancer cells.
2) Performed patch extraction on each patient's cancer cell image.
3) Passed the payches of each image through a pre-trrained VGG-16 feature extractor and retrieved 1000 features for each patch.
4) Saved all the extraced features into a csv file.
5) Applied K-means++ clustering on the extracted features into 8 clusters to identify patterns.
6) Passed these clusters into a customized Neural network and trained the neural network by applying an attention mechanism to classify if the corresponding patient has Stage 2 or Stage 3 cancer.
7) Evaluated the ML model with unkown data and obtained an accuracy of 97%.
