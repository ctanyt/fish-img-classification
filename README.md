# What the Fish? - Supervised deep learning for Image Classification


## Executive Summary 
This project is centered around using computer vision to identify various species of marine life that can be found in Singapore waters. Through image classification, we hope that the general public will be more aware of endangered marine life and release them back if they were accidentally hooked. 

**Methodology:**
After pre-processing the images, and building an image pipeline, we ran 3 pre-trained models: ResNet50V2, VGG16 and EfficientNetB0. Feature extraction was first done, followed by fine tuning. Out of the 3, the EfficientNetB0 performed the best. However, we wanted to see if we were able to increase accuracy scores even further. We then aggregated the 3 fine-tuned models, built an ensemble model and used soft voting to predict the classes. Eventually, we managed to obtain a test accuracy score of 99.6%. 

Model performance aside, we also wanted to understand what was going on under the hood when our images were passed through the convolutional neural network. Visual exploration was done on the filters, and how features were extracted from the first to last convolutional blocks. Lastly, we used a grad-CAM to visualize which areas of the image were important in class predictions. This also gave us an insight as to why images were being misclassified, and how to further train the model to get higher accuracy scores in future.

**Future improvements:** 
At this initial stage of model building, the model is only trained on 8 species. We hope to be able to expand this further in future and build a mobile application. It would also be a good idea to incorporate size measurement into the computer vision algorithm, to alert people if they have caught a juvenile (babies!) species and release them as well.  


## Background
There is an increased demand for leisure fishing in Singapore. In tandem with this increased demand, there is an increasing number of news reports of people reeling in endangered creatures like [shovelnose rays](https://mothership.sg/2022/01/jives-fishing-shovelnose-ray/) and [honeycomb rays](https://mothership.sg/2020/07/giant-stingray-bedok-jetty/).  

There have been [reports](https://www.straitstimes.com/singapore/environment/video-clip-of-endangered-eagle-ray-caught-at-east-coast-park-goes-viral) of other endangered animals being caught and killed as well . These endangered creatures are threatened as they are very slow to reproduce, and only have young once a year. Furthermore, shovelnose rays are caught and killed as it is a delicacy. 

Singapore is rich in biodiversity, and it can be difficult to differentiate between species. However it is of utmost importance that catch and release is practiced, especially when it comes to the creatures that are more vulnerable.


## Problem Statement 
There is a lack of education about the marine biodiversity in Singapore - people are unable to differentiate between species that are vulnerable. 

People who fish recreationally also might not practice catch and release, which puts vulnerable species at risk.


## Contents

* [Creation of image directories](#link1)
* [Image pre-processing](#link2)
* [EDA](#link3)
* [Model preparation](#link4)
* [Modelling](#link5)
    * [Baseline CNN](#link5a)
    * [ResNet50V2](#link5b)
    * [VGG16](#link5c)
    * [EfficientNetB0](#link5d)
    * [EfficientNetB0](#link5e)
* [Understanding the convolutional NN](#link6)
* [Conclusion](#link7)
---

## Data Sets
The 8 classes chosen for this project are:

* Blue spotted ribbontail ray (bluespotray)
* Honeycomb grouper
* Honeycomb ray
* Hybrid grouper
* Queenfish
* Red sea bream
* Seabass
* Shovelnose ray (shovelnose)

Aside from sea breams, seabasses and queenfishes, which are very common fish, shovelnose rays and honeycomb rays were chosen as they are endangered, and we want the algorithm to be able to ID these species to encourage release back into the wild.

Hybrid groupers are also special, as they are an invasive species. These fish were originally bred (yes, it is a cross between 2 grouper species) in Johor, but have somehow found their way to Singapore waters. They have voracious appetites, and compete with native fish for food. There are also risks of them breeding with other species outside of captivity which may cause more strains of hybrid fish. As such, if caught, they should not be released but brought home for consumption.

The 2 classes honeycomb grouper and blue spot ribbontail ray were chosen to introduce some complexity into our model. They look similar to the hybrid grouper and honeycomb ray respectively, and we wanted to see if our model is able to differentiate between the species.


## Folder Structure
```
capstone/
├── datasets/
│   ├── rawimage/
│   │   ├── bluespotray
│   │   ├── hybridgrouper
│   │   ├── shovelnose
│   │   ├── honeycomb_grouper
│   │   ├── honeycomb_ray
│   │   ├── redseabream
│   │   ├── seabass
│   │   └── queenfish
│   ├── root/
│   │   ├── bluespotray
│   │   ├── hybridgrouper
│   │   ├── shovelnose
│   │   ├── honeycomb_grouper
│   │   ├── honeycomb_ray
│   │   ├── redseabream
│   │   ├── seabass
│   │   └── queenfish
│   ├── train/
│   │   ├── bluespotray
│   │   ├── hybridgrouper
│   │   ├── shovelnose
│   │   ├── honeycomb_grouper
│   │   ├── honeycomb_ray
│   │   ├── redseabream
│   │   ├── seabass
│   │   └── queenfish
│   └── test/
│       ├── bluespotray
│       ├── hybridgrouper
│       ├── shovelnose
│       ├── honeycomb_grouper
│       ├── honeycomb_ray
│       ├── redseabream
│       ├── seabass
│       └── queenfish
│
├── plots
│
└── model
```

--- 

## Model Summaries

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| Baseline CNN | 0.308|0.120|0.11|0.11|  
| ResNet50V2|0.991|0.970|0.97|0.97  | 
| ResNet50V2 (Fine-Tuned)|0.995|0.980|0.98|0.98  |  
| VGG16|0.990|0.967|0.96|0.97 | 
| VGG16 (Fine-Tuned)|0.996|0.990|0.99|0.99|
| EfficientNet|0.999|1.000|1.00|1.00|
| EfficientNet|0.999|1.000|1.00|1.00 |
| Ensemble NN|0.996|0.990|0.99|0.99|

--- 

## Model Analysis

4 different models were run in total:
* Baseline CNN (without transfer learning)
* ResNet50V2
* VGG16
* EfficientNetB0

During the feature extraction phase, each pre-trained model was regularized using early stopping and dropout in the last 2 layers. Keeping all batch normalization layers frozen so as not to undo the weights previously learnt, the models were then fine tuned with the last 2 layers unfrozen.

The last model we tried was an ensemble of all the 3 fine-tuned pre-trained models we have previously run.

The strategy was to do a soft voting of equal weights between the 3 models. We added up the confidence scores of each image prediction before passing it into the argmax function to return the class names.

However, this did not improve our results. Accuracy dropped by 0.1% from our best fine tuned pre-trained model (EfficientNetB0 - 99.7%) to 99.6% in the ensemble model.

We postulate that it is due to the misclassification of certain types of images that were common throughout the 3 pre-trained models. 

--- 

## Recommendations

A total of 1 baseline CNN and 3 pre-trained models were run. Feature extraction and fine tuning was done for all 3 pre-trained models.

Of the 3 pre-trained models, the EfficientNetB0 performed the best, with an accuracy score of 99.7%. An ensemble neural network was also explored, in hopes that accuracy could improve.

However, this was not the case, and the ensemble model's accuracy was 99.6%. On further investigation of the CNN black box and utilizing the grad-CAM, we found that there were 2 images commonly misclassfied across all 3 pre-trained models. These images contained subjects that were 'camouflaged' and did not have a 3D profile (flat stingrays). For future work, we could pass in more of such images into the model for training.

The production ensemble model stands at 99.6% accuracy, which would definitely help in identifying marine life for sustainable fishing. To err on the side of caution, we should always practice catch and release.

---

## Future Improvements

There are many improvements that can be made in order for this algorithm to be scalable.

* Adding in more classes to expand what the model can identify
* To explore colour correction for photos that were taken underwater, as these usually lack the colour red
* Work on a mobile application that people can use on the go
* Explore how to identify length of catch as this would help differentiate juveniles from adults