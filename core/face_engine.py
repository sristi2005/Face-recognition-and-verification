import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import cv2
import numpy as np
from deepface import DeepFace
from typing import List, Dict, Union

class FaceEngine:
    def __init__(self, model_name="Facenet512", detector_backend="retinaface"):
        # We use Facenet512 or ArcFace as they are state of the art
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.embedding_size = 512

    def extract_embedding(self, image_input: Union[np.ndarray, str]) -> List[float]:
        """
        Extract face embedding from an image. 
        image_input can be a numpy array or a path to an image.
        """
        try:
            # DeepFace.represent returns a list of dictionaries if multiple faces are found
            representations = DeepFace.represent(
                img_path=image_input,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True
            )
            
            if representations and len(representations) > 0:
                # Return the embedding of the primary detected face
                return representations[0]["embedding"]
            else:
                raise ValueError("No face detected in the image.")
                
        except Exception as e:
            raise ValueError(f"Error extracting embedding: {str(e)}")
            
    def verify_faces(self, img1: Union[np.ndarray, str], img2: Union[np.ndarray, str]) -> Dict:
        """
        Perform 1:1 verification directly using DeepFace.
        """
        try:
            result = DeepFace.verify(
                img1_path=img1,
                img2_path=img2,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True
            )
            return {
                "verified": result["verified"],
                "distance": result["distance"],
                "threshold": result["threshold"],
                "model": result["model"]
            }
        except Exception as e:
            raise ValueError(f"Error verifying faces: {str(e)}")
