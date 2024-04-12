import streamlit as st
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import numpy as np
from streamlit_webrtc import webrtc_streamer
from excerciseTrack import LateralRaisesAnalyzer, HammerCurlsAnalyzer, SquatsAnalyzer, PushUpsAnalyzer, BicepCurlsAnalyzer

def load_image(file):
    img = image.load_img(file, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def predict_pose(image):
    model = load_model('Yoga_Pose_Classification_Model.h5')
    predictions = model.predict(image)
    return predictions


def print_menu():
    st.sidebar.title("Exercise Analysis")
    choice = st.sidebar.radio("Choose an option", ["Lateral Raises", "Hammer Curls", "Squats", "Push-ups", "Biceps Curls"])
    if choice == "Lateral Raises":
        st.subheader("You selected Lateral Raises, Click Start to begin analysis:")
        webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=LateralRaisesAnalyzer)

    elif choice == "Hammer Curls":
        st.subheader("You selected Hammer Curls. Click Start to begin analysis:")
        webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=HammerCurlsAnalyzer)

    elif choice == "Squats":
        st.subheader("You selected Squats.Click Start to begin analysis:")
        webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=SquatsAnalyzer)

    elif choice == "Push-ups":
        st.subheader("You selected Push-ups. Click Start to begin analysis:")
        webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=PushUpsAnalyzer)

    elif choice == "Biceps Curls":
        st.subheader("You selected Biceps Curls. Click Start to begin analysis:")
        webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=BicepCurlsAnalyzer)


navigation_menu=["Yoga Pose Detection","Excercise Tracker"]

def main():
    choice = st.sidebar.selectbox("Menu",navigation_menu)
    if choice == "Yoga Pose Detection":
        st.title("Yoga Pose Dectection")
        upload_image = st.file_uploader("Choose a file",type=["jpg", "jpeg", "png"])
        if upload_image is not None:
            st.image(upload_image,use_column_width=True)
            img =  load_image(upload_image)
            prediction = predict_pose(img)
            st.write("Prediction:")
            classes = ["DOWNDOG","GODDESS","PLANK","TREE","WARRIOR"]
            index = np.argmax(prediction)
            st.write(classes[index])
    elif choice=="Excercise Tracker":
        st.title("Excercise Tracker")
        print_menu()
            
            


if __name__ == '__main__':
    main()