To OPEN **Covid-19 Detection Tool** click here: 

[![Streamlit APP](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/rosariomoscato/covid19-detection-tool/covid19_detection_app.py)



# Covid-19 Detection Tool
### Simple Tool for Covid-19 Detection from Chest X-Ray


## Disclaimer
**This Tool is just a DEMO about Artificial Neural Networks so there is no clinical value in its diagnosis and the author is not a Doctor!**
**Please don't take the diagnosis outcome seriously and NEVER consider it valid!!!**

## Info
This Tool gets inspiration from the following works:
* [Detecting COVID-19 in X-ray images with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/)
* [Fighting Corona Virus with Artificial Intelligence & Deep Learning](https://www.youtube.com/watch?v=_bDHOwASVS4)
* [Deep Learning per la Diagnosi del COVID-19](https://www.youtube.com/watch?v=dpa8TFg1H_U&t=114s)

We used 206 Posterior-Anterior (PA) X-Ray [images](https://github.com/ieee8023/covid-chestxray-dataset/blob/master/metadata.csv) of patients infected by Covid-19 and 206 Posterior-Anterior X-Ray [images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) of healthy people to train a **Convolutional Neural Network** (made by about 5 million trainable parameters) in order to make a classification of pictures referring to infected and not-infected people.
Since dataset was quite small, some data augmentation techniques have been applied (rotation and brightness range). The result was quite good since we got 94.5% accuracy on the training set and 89.3% accuracy on the test set. Afterwards the model was tested using a new dataset of patients infected by pneumonia and in this case the performance was very good, only 2 cases in 206 were wrongly recognized. Last test was performed with 8 SARS X-Ray PA files, all these images have been classified as Covid-19.
Unfortunately in our test we got 5 cases of 'False Negative', patients classified as healthy that actually are infected by Covid-19. It's very easy to understand that these cases can be a huge issue.

We are aware the model is suffering of some limitations:
* small dataset (a bigger dataset for sure will help in improving performance)
* images coming only from the PA position
* a fine tuning activity is strongly suggested

Anybody has interest in this project can drop me an email ([rosario.moscato@outlook.com](mailto:rosario.moscato@outlook.com)) and I'll be very happy to reply and help.
