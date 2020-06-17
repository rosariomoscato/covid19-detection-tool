# Core Pkgs
import streamlit as st
import os
import time

# Viz Pkgs
import cv2
from PIL import Image,ImageEnhance
import numpy as np 

# AI Pkgs
import tensorflow as tf


def main():
	"""Simple Tool for Covid-19 Detection from Chest X-Ray"""
	html_templ = """
	<div style="background-color:blue;padding:10px;">
	<h1 style="color:yellow">Covid-19 Detection Tool</h1>
	</div>
	"""

	st.markdown(html_templ,unsafe_allow_html=True)
	st.write("A simple proposal for Covid-19 Diagnosis powered by Deep Learning")
	st.sidebar.image("covid19.jpeg",width=300)

	image_file = st.sidebar.file_uploader("Upload a PA X-Ray Image (jpg, png or jpeg)",type=['jpg','png','jpeg'])

	if image_file is not None:
		our_image = Image.open(image_file)
		
		if st.sidebar.button("Image Preview"):
			st.sidebar.image(our_image,width=300)

		activities = ["Image Enhancement","Diagnosis", "Disclaimer and Info"]
		choice = st.sidebar.selectbox("Select Activty",activities)

		if choice == 'Image Enhancement':
			st.subheader("Image Enhancement")

			enhance_type = st.sidebar.radio("Enhance Type",["Original","Contrast","Brightness"])
			
			if enhance_type == 'Contrast':
				c_rate = st.slider("Contrast",0.5,5.0)
				enhancer = ImageEnhance.Contrast(our_image)
				img_output = enhancer.enhance(c_rate)
				st.image(img_output,width=600,use_column_width=True)
				
			elif enhance_type == 'Brightness':
				c_rate = st.slider("Brightness",0.5,5.0)
				enhancer = ImageEnhance.Brightness(our_image)
				img_output = enhancer.enhance(c_rate)
				st.image(img_output,width=600,use_column_width=True)

			else:
				st.text("Original Image")
				st.image(our_image,width=600,use_column_width=True)
			
		elif choice == 'Diagnosis':

			if st.sidebar.button("Diagnosis"):

				# Convertiamo l'immagine in B&W
				new_img = np.array(our_image.convert('RGB')) #our image is binary we have to convert it in array
				new_img = cv2.cvtColor(new_img,1) # 0 is original, 1 is grayscale
				gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
				st.text("Chest X-Ray")
				st.image(gray,width=480,use_column_width=True)

				# Preprocessing della Radiografia
				IMG_SIZE = (200,200)
				img = cv2.equalizeHist(gray)
				img = cv2.resize(img,IMG_SIZE)
				img = img/255. #normalizzazione

				# Reshape della radiografia nel formato che piace a Tensorflow
				X_Ray = img.reshape(1,200,200, 1)
				
				# Importiamo il Modello di ConvNet pre-addestrato
				model = tf.keras.models.load_model("./models/Covid19_CNN_Classifier.h5")
				
				# Diagnosi (Previsione=Classificazione Binaria)
				diagnosis = model.predict_classes(X_Ray)
				diagnosis_proba = model.predict(X_Ray)
				probability_cov = diagnosis_proba*100
				probability_no_cov = (1-diagnosis_proba)*100

				my_bar = st.sidebar.progress(0)

				for percent_complete in range(100):
				 	time.sleep(0.05)
				 	my_bar.progress(percent_complete + 1)

				# with st.spinner('Diagnosis on going...'):
				# 	time.sleep(3)

				# Mappatura Diagnosi: No-Covid=0, Contagiato_da_Covid=1
				if diagnosis == 0:
					st.sidebar.success("DIAGNOSIS: NO COVID-19 (Probability: %.2f%%)" % (probability_no_cov))
				else:
					st.sidebar.error("DIAGNOSIS: COVID-19 (Probability: %.2f%%)" % (probability_cov))

				st.warning("This App is just a DEMO about Artificial Neural Networks so there is no clinical value in its diagnosis and the author is not a Doctor!")
		
		else:# choice == 'About':
			st.subheader("Disclaimer")
			st.write("**This Tool is just a DEMO about Artificial Neural Networks so there is no clinical value in its diagnosis and the author is not a Doctor!**")
			st.write("**Please don't take the diagnosis outcome seriously and NEVER consider it valid!!!**")
			st.subheader("Info")
			st.write("This Tool gets inspiration from the following works:")
			st.write("- [Detecting COVID-19 in X-ray images with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/)") 
			st.write("- [Fighting Corona Virus with Artificial Intelligence & Deep Learning](https://www.youtube.com/watch?v=_bDHOwASVS4)") 
			st.write("- [Deep Learning per la Diagnosi del COVID-19](https://www.youtube.com/watch?v=dpa8TFg1H_U&t=114s)")
			st.write("We used 206 Posterior-Anterior (PA) X-Ray [images](https://github.com/ieee8023/covid-chestxray-dataset/blob/master/metadata.csv) of patients infected by Covid-19 and 206 Posterior-Anterior X-Ray [images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) of healthy people to train a Convolutional Neural Network (made by about 5 million trainable parameters) in order to make a classification of pictures referring to infected and not-infected people.")
			st.write("Since dataset was quite small, some data augmentation techniques have been applied (rotation and brightness range). The result was quite good since we got 94.5% accuracy on the training set and 89.3% accuracy on the test set. Afterwards the model was tested using a new dataset of patients infected by pneumonia and in this case the performance was very good, only 2 cases in 206 were wrongly recognized. Last test was performed with 8 SARS X-Ray PA files, all these images have been classified as Covid-19.")
			st.write("Unfortunately in our test we got 5 cases of 'False Negative', patients classified as healthy that actually are infected by Covid-19. It's very easy to understand that these cases can be a huge issue.")
			st.write("The model is suffering of some limitations:")
			st.write("- small dataset (a bigger dataset for sure will help in improving performance)")
			st.write("- images coming only from the PA position")
			st.write("- a fine tuning activity is strongly suggested")
			st.write("")
			st.write("Anybody has interest in this project can drop me an email and I'll be very happy to reply and help.")

	if st.sidebar.button("About the Author"):
			st.sidebar.subheader("Covid-19 Detection Tool")
			st.sidebar.markdown("by [Ing. Rosario Moscato](https://www.linkedin.com/in/rosariomoscato)")
			st.sidebar.markdown("[rosario.moscato@outlook.com](mailto:rosario.moscato@outlook.com)")
			st.sidebar.text("All Rights Reserved (2020)")


if __name__ == '__main__':
		main()	