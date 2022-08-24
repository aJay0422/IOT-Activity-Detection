# Project Description
This project is about understanding and classifying human activities in videos, 
specifically in an IOT scenario.  
Currently, we are working on videos which contain human interactions with a refrigerator. The dataset we are
 using contains 5 categories.  
|  Category  |  n_samples  |
|------------|-------------|
|no interaction|  190  |
|open close fridge| 187|
|put back item |194|
|screen interaction| 193  |
|take out item |187|


# Feature Extraction
Given the raw video, we first use [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to extract human pose features from the video. The 3D feature is generated from the 2D features, which are extracted by [detectron2](https://github.com/facebookresearch/detectron2). These two kind of features both have their own advantanges in our project. So we are trying use them separately and find out how they are related to our tasks. Currently, our model input only contains pose features. But there has been signs that adding image features to our model is beneficial.

# Video Downsample
Since each video has differnent number of frames, we need to fix their lengths to fit a machine learning model. Our method is __interpolating__ the feature trajectory along the temporal dimension. All videos after interpolation contain 100 frames.

# Models
The model that achieves the highest accuracy is a transformer encoder model.  We are using this architecture to deal with the sequence model and trying to use the attention mechanism to identify informative frames.
| Model | Test Accuracy  |
|-------|----------------|
|Transformer base| 78.87% |
|Transformer large| 83.10% |
|Transformer huge|  85.21%  |