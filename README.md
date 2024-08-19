# [Object Detection for Visually Impaired]{style="color:#237094;"}

![Python Version](https://img.shields.io/badge/python-3.12.4%2B-blue) ![Platform](https://img.shields.io/badge/platform-%20jupyter%20%7C%20python%20script-lightgreen) ![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)

## Table of Contents

<p>[1.]{style="color:#fb8500;"} <a href="#about">About</a><br> [2.]{style="color:#fb8500;"} <a href="#example">Example</a><br> [3.]{style="color:#fb8500;"} <a href="#workflow">Workflow</a><br> [4.]{style="color:#fb8500;"} <a href="#structure">Project structure</a><br> [6.]{style="color:#fb8500;"} <a href="#models"> Models</a><br> [7.]{style="color:#fb8500;"} <a href="#results"> Results</a><br> [8.]{style="color:#fb8500;"} <a href="#conclusions"> Conclusions</a><br> [9.]{style="color:#fb8500;"} <a href="#references"> References</a><br> [10.]{style="color:#fb8500;"} <a href="#contributors"> Contributors</a><br></p>

## [1. About]{#about style="color:#00bbd6;"}

The aim of the project is to create an object detection model which will help the visually impaired to navigate. A YOLOv8 model is used to locate identify the objects accurately. The model has undergone hyperparameter tuning to select suitable hyperparameters for the problem. The dataset used is the PASCAl VOC dataset, it has 20 classes which is a combination of indoor and outdoor objects.

## 2. Dataset

The Pascal VOC dataset used has the following 20 classes:

-   **Person***:* person

-   **Animal***:* bird, cat, cow, dog, horse, sheep

-   **Vehicle***:* aeroplane, bicycle, boat, bus, car, motorbike, train

-   **Indoor***:* bottle, chair, dining table, potted plant, sofa, tv/monitor

A total of 5008 images are used for training the model. The frequency distribution of the classes in the images can be seen below:

\######################## Add image

It can be seen that there is a class imabalanced. Due to this, it is best to use mAP (mean average precision) metric to asses the performance of the model.

## 3. Methodology

The dataset used has 5008 images. This has been split to 70% train, \@20% valid and 10% test data. The annotations were originally in XML format, this was converted to .txt format using Roboflow.

### Model Tuning

Parameters such as learning rate, image size, momentum and weight decay were tuned.

```{python}
learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
img_size = trial.suggest_categorical('img_size', [320, 416, 512, 640])
momentum = trial.suggest_uniform('momentum', 0.85, 0.99)
weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
```

Hyperparameter tuning was done using the **optuna** library. The parameters were set in a range and during the tuning, the parameters are selected randomly from the range and train the model with the selected parameters. This was done for 10 trials. This method was mainly followed due to hardware constraints and to get an idea of what range hyperparameters will be suitable for the problem. The set of parameters which maxmimises the mAP was selected to train the final model.

### Model Training

The model is then trained for 100 epochs with the selected hyperparameters using the optimizer 'AdamW'. Some layers were also frozen as a pre-trained model and to not lose whatever information was learned.

### Model Evaluation

The confusion matrix, performance metrics like mAP, precision and recall were looked at. The confusion matrix can be seen below:

![](images_readme/confusion_matrix_normalized.png)

The “train” class is the most accurate predication, 90% of the instances are classified accurately. The “aeroplane” class is a close second. Among 20 classes, only 4 classes (boat, bottle, chair, potted plant) have the correct prediction class proportion to be less than 0.65. This could be due to the complexity and the nature of the objects that the recall for these are relatively lower.

The performance metrics can be seen below:

![](images_readme/metrics.png)

The model performs well with mAP (mean average precision) of 0.746 for all classes. Due to class imbalance, it is best to evaluate the model using mAP as it takes into account of Precision and Recall metrics. Each class has mAP of at least 0.6 except for ‘potted plant’ class which is around 0.48. As mentioned above, this is probably because of the lower number of instances and how potted plants can come different varieties of sizes and shapes.

A visualisation of the above metrics are also shown below:

![](images_readme/mAP_plot.png)

![](images_readme/PR_curve.png)

### Model Testing

The trained model was also then used to detect and predict the objects in the test data. An image was created which combines multiple predictions and compares it with the original annotations. The image is shown below:

![](images_readme/2.png)

It can be seen that the detections are very accurate. The model can be further improved upon by letting it be tuned further with a bigger hyperparameter space and using a more complex model YOLOv8 model such as the YOLOv8m or YOLOv8l.
