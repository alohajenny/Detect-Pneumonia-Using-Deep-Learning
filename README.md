# Detecting Pneumonia using Deep Learning

---

## Description
Pneumonia is a severe lung infection that inflames and fills the air sacs in the lungs with fluid or puss. It's a common illness that affects many people each year in the US and around the world. To diagonose this disease, Radiologists looks for the presence and the severity of infiltrate on patient's chest X-Ray image. However, studies have shown that even experienced radiologists often have a hard time correctly identify whether something is an infiltrate on the X-ray. This causes delays in diagnosis, which increases the disease severity and associated mortality rate. 

Can AI help fill in the gap for humans, spotting Pneumonia the way that human's eye aren't able to do? 

## Objective
Build a deep learning model that detects Pneumonia on patient's chest X-Ray images. 

## Methodology
### 1. Data Collection
Data is obtained from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia), which are chest X-Ray images of pediatric patients under age of 5 from a Medical Center in Guangzhou, China. It contains a total of 5,856 images in different sizes. 

### 2. Data Exploring
As the validation dataset arranged by Kaggle is too small, I've re-split the entire dataset into this ratio: 70% train, 19% validation, and 11% test. The training dataset is distributed as 79% of X-rays has pneumonia and 21% doesn't. 

### 3. Image Preprocessing
To ensure efficient computation, **pixels were normalized** to values from 0 to 1. The **images were also augmented** to prevent model from overfitting. Lastly, since a chest X-ray shows the parts of the chest in different shades of black and white based on the amount of radiation the tissue can absorb, it makes sense to decrease the complexity of the model by **converting the images to grayscale** (one color channel). 

### 4. Data Modeling
#### Model Architecture
The basic structure of a CNN model consists of CNN layer, followed by a pooling layer and an optional dropout, then fully connected layers with activations and output layer. I've tried different combinations of activations, dropout, and batch normalization, and below is the high-level architecture of my final and best performing convolutional neural network (CNN) model.  

![CNN Model](cnn.png)

Since this is a binary classification problem, we used ReLU activation to output non-negative values and fed into a sigmoid softmax function that predicts an output probability, which can be converted to class values based on our threshold.

#### Optimize for AUC score
For the case of Pneumonia, we will aim to have high recall as any delayed diagnosis means that someone may get really sick and potentially lose their life. We do that by first having our model optimize for AUC score, focusing on having a good measure of separability for our binary classification. Then, we can crank the threshold up and down (from 0.5) if the business feels more comfortable with a higher or lower FP or FN. 

## Results

### Fit 
Through each epoch during the learning process, validation loss and training loss approach each other, which means our model doesn’t seem to have much overfitting or underfitting. Moreover, training and validation AUC score also converges to a point where they’re almost equal in the end. 

### Metrics
- The model has an AUC score of 97.98%, indicating that our CNN model can separate 97.98% of the images in the test set. 
- The model achieves 98.72% for recall, with 8.49% of False Positives and 0.80% of False Negatives, which is impressive and exactly what we’re aiming for. 

## Workflow
Follow the jupyter notebook in the order below:
- 00 Download dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- 01 EDA.ipynb
- 02 CNN Model.ipynb