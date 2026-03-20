import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import shutil
import os
import cv2
from core.face_engine import FaceEngine
from core.vector_store import VectorStore

app = FastAPI(title="Face Recognition API", description="Modern Face Recognition built with FastAPI, DeepFace and FAISS")

# Initialize AI Components
engine = FaceEngine() # default is Facenet512
vector_store = VectorStore(embedding_size=512)

class VerifyRequest(BaseModel):
    user_id: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Face Recognition API is running"}

@app.post("/register")
async def register_user(user_id: str = Form(...), file: UploadFile = File(...)):
    """
    Registers a new user by extracting their face embedding and adding it to FAISS.
    """
    temp_path = f"temp_{file.filename}"
    try:
        # Save uploaded file temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Extract embedding
        embedding = engine.extract_embedding(temp_path)
        
        # Add to vector store
        vector_store.add_user(user_id, embedding)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {"status": "success", "message": f"User {user_id} registered successfully."}
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/verify")
async def verify_user(user_id: str = Form(...), file: UploadFile = File(...)):
    """
    Verify if the uploaded face matches the registered user_id (1:1 Verification).
    Warning: Since FAISS is primarily for 1:N, for 1:1 we can just do a search and ensure 
    the top result is the right user AND within threshold. Or we could save per-user embeddings 
    in a DB. Here we use Vector search and check ID.
    """
    temp_path = f"temp_{file.filename}"
    try:
        # Save file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Extract embedding
        embedding = engine.extract_embedding(temp_path)
        
        # Search FAISS
        # Threshold for Facenet512 L2 distance is typically around 200-250 (empirical). 
        # DeepFace verification handles thresholding better, but for FAISS search we'll do:
        results = vector_store.search(embedding, k=5, threshold=300.0)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # Check if the requested user is in the valid top matches
        for match_id, dist in results:
            if match_id == user_id:
                return {"status": "success", "verified": True, "distance": dist, "message": "User verified successfully."}
                
        return {"status": "success", "verified": False, "message": "Face does not match the registered user."}
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recognize")
async def recognize_user(file: UploadFile = File(...)):
    """
    Identify the uploaded face among all registered users (1:N Recognition).
    """
    temp_path = f"temp_{file.filename}"
    try:
        # Save file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Extract embedding
        embedding = engine.extract_embedding(temp_path)
        
        # Search FAISS (get top 1)
        results = vector_store.search(embedding, k=1, threshold=300.0)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if results:
            match_id, dist = results[0]
            return {"status": "success", "user_id": match_id, "distance": dist}
        else:
            return {"status": "success", "user_id": None, "message": "No matching user found."}
            
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=400, detail=str(e))
