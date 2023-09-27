![cloud](https://github.com/pgrondein/cloud_big_data_model/assets/113172845/952afcbb-cb30-4041-9676-fc8ae073331b)

# Deploy a model in the cloud

## Problem Definition

The goal here is to study the feasibility of developing a mobile app that would allow users to take a photo of a fruit and obtain information about that fruit. The development of the app comes with with the necessity to build a first version of a Big Data architecture.

The steps are :

- Feature extraction from images
- Model development
- Big Data architecture building

## Data

Data are available [here](https://www.kaggle.com/datasets/moltean/fruits).

Data are divided by categories, 120 type of fruits and vegetables. With 492 images per folder, between 1 and 4 folder per category, and approximately 5 Ko per image, there’s more than 300 Mo of data to store.

![figue](https://github.com/pgrondein/cloud_big_data_model/assets/113172845/06ca4b1a-8560-4907-b5d9-069673ef2fb8)
![prune](https://github.com/pgrondein/cloud_big_data_model/assets/113172845/c2bb6d89-2388-45d3-a2fc-7f6d054391cf)
![banane](https://github.com/pgrondein/cloud_big_data_model/assets/113172845/73a6c7c0-c7d6-4363-8da8-14ee07554513)

Multiple storage options were available. I selected AWS S3, inexpensive solution with attractive transfer times, and a good free offer (5 GB for 12 months).

## Environment

![Untitled](https://github.com/pgrondein/cloud_big_data_model/assets/113172845/263f2f92-7e7a-4cd0-9d6a-8f8896d8cc4c)

I’m using Apache Spark, a unified, ultra-fast analytics engine for large-scale data processing. Like Hadoop before, it allows you to efficiently execute calculations that require rapid and iterative access to a large quantity of data, large-scale analyzes using Cluster machines.

As I’m coding with Python, I set up a Pyspark environment :

- Java 8
- Hadoop 3.3
- Spark 3.3.1

## Feature Extraction

In order to develop a Image Classification Model, we first need to be able to extract features from these images, that is to say tp take an image and return the key points of this image in the form of variables/vectors, the “digital fingerprint” of the image, invariant regardless of transformations.

For this task, I used VGG16. VGG16 is a pre-trained convolutional neural network. It takes an image as input and automatically returns the features of this image, by automatic extraction and prioritization of said features. There are several possible use of VGG16 (Classifier, Standalone Feature Extractor…), and several transfer learning strategies (total fine-tuning, partial fine-tuning…).

![Untitled (1)](https://github.com/pgrondein/cloud_big_data_model/assets/113172845/34885065-f7ad-46ba-bc96-a8a6efaf9017)

My version of VGG16 is a entirely pre-trained Feature Extractor, where I only remove the last classifying layers without retraining the model. 

## Feature Reduction with PCA

I ended up with more than 1000 variables, so a reduction was necessary.

![Untitled (2)](https://github.com/pgrondein/cloud_big_data_model/assets/113172845/17432dce-5792-4ead-9e76-e095bc8e5401)

The first two components make it possible to explain 45% of the variance. By also using components 3 and 4, we reach almost 60%.

Finally, I obtained 4 features after PCA.

![Untitled (3)](https://github.com/pgrondein/cloud_big_data_model/assets/113172845/9fbb07df-b617-416b-b2b3-3b102d2ca13f)

## Big Data Architecture

### Long Distant Access

API for Hadoop/Apache Spark to read and write Amazon S3 objects directly. 

Several options : 

- Boto3: allows python scripts (only) to interact with AWS resources
- S3: allows scripts to interact with AWS resources

I chose S3a, faster and more polyvalent.

### Output

Apache Parquet is a data storage format offering many advantages for projects of significant scale. 

With Apache Parquet, data is presented in columns. The benefits are :

- Reading attribute values faster
- Reading a subset of attributes also faster
- Complex data structures represented more efficiently

### Calculation Cluster

In order to access images stored in S3, and apply all steps, I chose EC2 (Elactic Compute Cloud), a AWS service for providing servers

For this project, the options are :

- **AMI** (Amazon Machine Image) : Amazon Linux
- **Instance type** : t2.medium avec 4G RAM
- **Storage** : 8 G


