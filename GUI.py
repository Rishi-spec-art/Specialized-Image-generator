# Importing all the required libraries
import cv2
import numpy as np
from keras.models import load_model
from keras import backend as K
K.set_image_data_format('channels_first')
from tensorflow.python.keras.backend import eager_learning_phase_scope

import streamlit as st

#Loading the CSS file 
with open('test_file.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

num_params = 10
input_w = 124
input_h = 124

cur_face = np.zeros((3, input_h, input_w), dtype=np.uint8)

enc_model = load_model("Encoder.h5")
enc_fname = "Encoder.h5"

#Loading the model
enc_model = load_model(enc_fname)
enc = K.function([enc_model.get_layer('encoder').input],
				 [enc_model.layers[-1].output])

#Loading all the stats
means = np.load(file = "means.npy")
stds  = np.load('stds.npy')
evals = np.sqrt(np.load('evals.npy'))
evecs = np.load('evecs.npy')

sort_inds = np.argsort(-evals)
evals = evals[sort_inds]
evecs = evecs[:,sort_inds]

#Starting point of App creation
def main():
	st.title("Human Face Generator")
	st.markdown("You can here *generate images* that may be used by you for different purposes, such as using in your _model_ training without any need for privacy considerations, etc.")
	
	col1, col2, col3= st.columns([1,1,3])
	
	with col1:
	    num1 = st.slider("Face-color feature 1:", min_value = -3.0, max_value = 3.0, value = 0.0, step = 0.5)
	    num2 = st.slider("Gender feature 2:", min_value = -3.0, max_value = 3.0, value = 0.0, step = 0.5)
	    num3 = st.slider("Face-size feature 3:", min_value = -3.0, max_value = 3.0, value = 0.0, step = 0.5)
	    num5 = st.slider("Beard feature 5:", min_value = -3.0, max_value = 3.0, value = 0.0, step = 0.5)
	with col2:
	    num6 = st.slider("Beard feature 6:", min_value = -3.0, max_value = 3.0, value = 0.0, step = 0.5)
	    num7 = st.slider("Face-size feature 7:", min_value = -3.0, max_value = 3.0, value = 0.0, step = 0.5)
	    num8 = st.slider("Gender feature 8:", min_value = -3.0, max_value = 3.0, value = 0.0, step = 0.5)
	    num9 = st.slider("Face-cut feature 9:", min_value = -3.0, max_value = 3.0, value = 0.0, step = 0.5)
	
	num_list = np.array([num1, num2, num3, 0., num5, num6, num7, num8, num9, 0.]).astype('float32')
	
	x = means + np.dot(evecs, (num_list * evals).T).T
	x = means + stds * num_list
	x = np.expand_dims(x, axis = 0)
	with eager_learning_phase_scope(value = 0):
	    y = enc([x])[0][0]
	cur_face = (y * 255.0).astype(np.uint8)
	cur_face = cv2.rotate(cur_face.T, rotateCode = 0)
	
	with col3:
	    st.image(cur_face, width=350)
	    down = st.button("Download", help = "Click to download image")
	    if down:
	        cv2.imwrite("File_downloaded.png", cur_face)

if __name__ == '__main__':
	main()
