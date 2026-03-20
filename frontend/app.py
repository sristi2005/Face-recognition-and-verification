import streamlit as st
import requests
from PIL import Image
import io

# FastAPI backend URL
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Modern Face Recognition", page_icon="👤", layout="wide")

st.title("👤 Modern Face Recognition & Verification System")
st.markdown("This system is powered by **FastAPI**, **DeepFace (ArcFace/Facenet)**, and **FAISS** for fast, scalable vector search.")

menu = ["Register Face", "Verify Face (1:1)", "Recognize Face (1:N)"]
choice = st.sidebar.selectbox("Select Action", menu)

st.sidebar.markdown("---")
st.sidebar.info("Upload an image or use your webcam to interact with the Face AI Engine.")

if choice == "Register Face":
    st.header("Register New User")
    user_id = st.text_input("Enter User ID", placeholder="e.g., user_123")
    
    img_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    img_camera = st.camera_input("Or Take a Picture")
    
    img_to_use = img_camera if img_camera else img_file
    
    if img_to_use and user_id:
        st.image(img_to_use, caption="Selected Image", width=300)
        if st.button("Register User", type="primary"):
            with st.spinner("Extracting face embedding and saving to FAISS..."):
                try:
                    files = {"file": (img_to_use.name, img_to_use.getvalue(), "image/jpeg")}
                    data = {"user_id": user_id}
                    response = requests.post(f"{API_URL}/register", data=data, files=files)
                    
                    if response.status_code == 200:
                        st.success(response.json()["message"])
                    else:
                        st.error(f"Error: {response.json()['detail']}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

elif choice == "Verify Face (1:1)":
    st.header("Verify Face Identity")
    user_id = st.text_input("Enter User ID to Verify Against", placeholder="e.g., user_123")
    
    img_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    img_camera = st.camera_input("Or Take a Picture")
    
    img_to_use = img_camera if img_camera else img_file
    
    if img_to_use and user_id:
        st.image(img_to_use, caption="Selected Image", width=300)
        if st.button("Verify Identity", type="primary"):
            with st.spinner("Checking face against registered embedding..."):
                try:
                    files = {"file": (img_to_use.name, img_to_use.getvalue(), "image/jpeg")}
                    data = {"user_id": user_id}
                    response = requests.post(f"{API_URL}/verify", data=data, files=files)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        if res_data["verified"]:
                            st.success(f"✅ Identity Verified! Distance: {res_data['distance']:.2f}")
                        else:
                            st.error(f"❌ Verification Failed. {res_data.get('message', '')}")
                    else:
                        st.error(f"Error: {response.json()['detail']}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

elif choice == "Recognize Face (1:N)":
    st.header("Recognize Face (Identify)")
    st.markdown("Identify the person in the image from the FAISS database.")
    
    img_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    img_camera = st.camera_input("Or Take a Picture")
    
    img_to_use = img_camera if img_camera else img_file
    
    if img_to_use:
        st.image(img_to_use, caption="Selected Image", width=300)
        if st.button("Identify Face", type="primary"):
            with st.spinner("Searching FAISS vector database..."):
                try:
                    files = {"file": (img_to_use.name, img_to_use.getvalue(), "image/jpeg")}
                    response = requests.post(f"{API_URL}/recognize", files=files)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        if res_data["user_id"]:
                            st.success(f"✅ Match Found! User ID: **{res_data['user_id']}** (Distance: {res_data['distance']:.2f})")
                        else:
                            st.warning("No matching face found in the database.")
                    else:
                        st.error(f"Error: {response.json()['detail']}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
