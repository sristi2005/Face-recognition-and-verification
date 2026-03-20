# Modern Face Recognition & Verification System

This project is a state-of-the-art Face Recognition system utilizing modern Machine Learning libraries and deployment architectures, suitable for real-world AI/ML engineering roles.

## 🚀 Tech Stack
- **Face Embedding & AI**: `DeepFace` (ArcFace / Facenet) and `RetinaFace` for robust detection. These are industry standards for face embedding.
- **Vector Database**: `FAISS` (Facebook AI Similarity Search) for scalable, lightning-fast 1:N identity resolution.
- **Backend Service**: `FastAPI` providing high-performance, asynchronous REST APIs.
- **Frontend UI**: `Streamlit` for a clean, interactive user experience with Webcam/File upload support.

## 📂 Project Structure
- `core/face_engine.py`: Handles model loading and embedding retrieval logic.
- `core/vector_store.py`: Wraps the FAISS database for persisting embeddings and mapping User IDs.
- `backend/main.py`: The FastAPI application defining `/register`, `/verify`, and `/recognize` endpoints.
- `frontend/app.py`: The Streamlit web interface.
- `requirements.txt`: Project dependencies.
- `run.py`: Runner script to start both Frontend & Backend concurrently.
- `data/`: Auto-generated folder holding the FAISS index and mappings.

## ⚙️ Setup Instructions

1. **Create Virtual Environment (Optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   You can run both the backend and frontend simultaneously using the provided script:
   ```bash
   python run.py
   ```

   Alternatively, run them separately:
   - **Backend**: `uvicorn backend.main:app --reload`
   - **Frontend**: `streamlit run frontend/app.py`

## 🎯 Features
1. **Register User**: Capture an image of a new user and add their 512-dimensional face embedding to the FAISS database.
2. **Verify Face (1:1)**: Take an image and a User ID, and assert whether the image belongs to the claimed User ID based on L2 Distance thresholding.
3. **Recognize Face (1:N)**: Retrieve the most likely identity for a given face across all registered users in milliseconds.
