import json
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import joblib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModel
import torch
import nltk
from nltk.tokenize import sent_tokenize
import logging
from flask_cors import CORS
import numpy as np
import PyPDF2
import pytesseract
from PIL import Image
import io
import joblib
import pandas as pd
from datetime import datetime
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image
import cv2

nltk.download('punkt')

app = Flask(__name__)

# Global variables to store models
medical_chatbot = None
biobert_model = None
biobert_tokenizer = None
heart_disease_model = None
heart_disease_features = None
disease_predictor = None
vitals_monitor = None
adr_model = None
w2v_model = None
drug_adr_map = None
models_loaded = False
model_tumor = None

# ------------------------------
# ✅ Medical Chatbot Class
# ------------------------------
class MedicalChatbot:
    def __init__(self):
        """Initialize the medical chatbot using a fine-tuned medical model."""
        model_name = "nikhil928/google-flan-t5-large-770-finetuned-medical-data"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)

    def get_response(self, user_input: str, max_length: int = 512) -> str:
        """Generate a detailed response from the chatbot model."""
        response = self.pipe(
            user_input, 
            max_length=max_length, 
            min_length=50,  
            temperature=1.0,  
            top_k=50,  
            top_p=0.95,  
            do_sample=True,  
            repetition_penalty=1.5,  
            num_return_sequences=1,  
            length_penalty=2.0  
        )
        return response[0]['generated_text']

# ------------------------------
# ✅ Medical Report Analysis Functions
# ------------------------------
def load_biobert_model():
    """Load BioBERT model for medical report analysis."""
    global biobert_model, biobert_tokenizer
    model_name = "dmis-lab/biobert-v1.1"
    biobert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    biobert_model = AutoModel.from_pretrained(model_name)

def get_sentence_embeddings(sentences):
    """Get sentence embeddings using BioBERT."""
    inputs = biobert_tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = biobert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_text_from_pdf(file_object):
    """Extract text from PDF file."""
    pdf_reader = PyPDF2.PdfReader(file_object)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_image(file_object):
    """Extract text from an image using OCR."""
    image = Image.open(file_object)
    text = pytesseract.image_to_string(image)
    return text

def analyze_text(text):
    """Analyze medical text using BioBERT embeddings."""
    sentences = sent_tokenize(text)
    embeddings = get_sentence_embeddings(sentences)

    key_phrases = [
        "The patient is diagnosed with",
        "The recommended treatment is",
        "Risk factors include",
        "The prognosis is",
        "Follow-up care includes"
    ]
    key_embeddings = get_sentence_embeddings(key_phrases)

    analysis_results = []
    for phrase, key_emb in zip(key_phrases, key_embeddings):
        similarities = [cosine_similarity(key_emb, sent_emb) for sent_emb in embeddings]
        most_similar_idx = np.argmax(similarities)
        analysis_results.append(f"{phrase}: {sentences[most_similar_idx]}")

    return "\n\n".join(analysis_results)

def analyze_reports(file_objects):
    """Analyze uploaded PDF or image medical reports."""
    combined_text = ""
    for file_object in file_objects:
        file_object.seek(0)  
        if file_object.name.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_object)
        elif file_object.name.lower().endswith(('.jpg', '.jpeg', '.png')):
            text = extract_text_from_image(file_object)
        else:
            return f"Unsupported file type: {file_object.name}. Please upload PDF or image files."
        combined_text += text + "\n\n"

    if not combined_text.strip():
        return "No text could be extracted from the uploaded files. Please ensure the files contain readable text."

    return analyze_text(combined_text)

# ------------------------------
# ✅ Heart Disease Prediction Functions
# ------------------------------
def load_heart_disease_model():
    """Load the heart disease prediction model and features."""
    global heart_disease_model, heart_disease_features
    heart_disease_model = joblib.load("heart_disease_model.pkl")
    heart_disease_features = joblib.load("heart_disease_features.pkl")

def predict_heart_disease(patient_data):
    """Predict heart disease risk for a patient."""
    try:
        # Convert input data to DataFrame
        patient_df = pd.DataFrame(patient_data)
        
        # Ensure feature order matches the model
        patient_df = patient_df[heart_disease_features]
        
        # Make prediction
        prediction = heart_disease_model.predict(patient_df)
        
        return bool(prediction[0] == 1)
    except Exception as e:
        raise ValueError(f"Error making prediction: {str(e)}")
    
# ------------------------------
# ✅ Disease Predictor Class
# ------------------------------
class ImprovedDiseasePredictor:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        
    def _parse_single_row(self, row_data):
        """Parse a single row of symptom data in string format"""
        try:
            symptoms_dict = {}
            symptoms_part = row_data
            
            # Split the string by "},{" to separate individual symptom entries
            entries = symptoms_part.strip('[]').split('},{')
            
            current_symptom = None
            for entry in entries:
                # Clean up the entry
                entry = entry.replace('{', '').replace('}', '').replace('"', '')
                if 'symptoms' not in entry:
                    continue
                    
                # Split into key-value pairs
                parts = entry.split(',')
                for part in parts:
                    if ':' not in part:
                        continue
                    key, value = part.split(':')
                    if key.strip() == 'symptoms':
                        value = value.strip()
                        # Check if the value is numeric (score) or text (symptom)
                        try:
                            score = float(value)
                            if current_symptom:  # If we have a pending symptom, add it with this score
                                symptoms_dict[current_symptom] = score
                                current_symptom = None
                        except ValueError:
                            current_symptom = value  # Store the symptom name for the next iteration
                            
            return symptoms_dict
        except Exception as e:
            print(f"Error parsing single row: {e}")
            return {}

    def _parse_symptoms_json(self, symptoms_list):
        """Parse symptoms from JSON list format"""
        try:
            symptoms_dict = {}
            current_symptom = None
            
            for item in symptoms_list:
                # Get the symptom value from the dict
                if not isinstance(item, dict) or 'symptoms' not in item:
                    continue
                    
                value = item['symptoms']
                
                # Try to convert to float for score
                try:
                    score = float(value)
                    if current_symptom:  # If we have a pending symptom, add it with this score
                        symptoms_dict[current_symptom] = score
                        current_symptom = None
                except (ValueError, TypeError):
                    current_symptom = str(value)  # Store the symptom name for the next iteration
                    
            return symptoms_dict
        except Exception as e:
            print(f"Error parsing symptoms JSON: {e}")
            return {}

    def _create_feature_vector(self, symptoms_dict):
        """Create a feature vector from symptoms dictionary"""
        if not symptoms_dict:
            return None
            
        # Convert symptoms to a space-separated string for TF-IDF
        symptoms_text = ' '.join(symptoms_dict.keys())
        
        # Get TF-IDF features
        features_sparse = self.vectorizer.transform([symptoms_text])
        features = features_sparse.toarray()
        
        # Add severity scores
        severity_features = np.zeros(len(self.vectorizer.get_feature_names_out()))
        for symptom, score in symptoms_dict.items():
            if symptom in self.vectorizer.vocabulary_:
                idx = self.vectorizer.vocabulary_[symptom]
                severity_features[idx] = score/100  # Normalize score to 0-1
                
        # Combine TF-IDF features with severity scores
        combined_features = np.concatenate([features, severity_features.reshape(1, -1)], axis=1)
        return combined_features

    def predict_disease(self, symptoms_data, top_n=3):
        """Predict top N most likely diseases"""
        try:
            # Handle both string and list input formats
            if isinstance(symptoms_data, list):
                symptoms_dict = self._parse_symptoms_json(symptoms_data)
            else:
                symptoms_dict = self._parse_single_row(symptoms_data)
                
            if not symptoms_dict:
                print("No symptoms could be parsed from the input")
                return []
            
            print(f"Parsed symptoms: {symptoms_dict}")  # Debug print
            
            # Create feature vector
            X_test = self._create_feature_vector(symptoms_dict)
            if X_test is None:
                print("Could not create feature vector")
                return []
            
            # Get probability predictions
            probas = self.model.predict_proba(X_test)
            
            # Get top N predictions
            top_n_idx = np.argsort(probas[0])[-top_n:][::-1]
            predictions = []
            
            for idx in top_n_idx:
                disease = self.model.classes_[idx]
                probability = probas[0][idx] * 100
                predictions.append({
                    'disease': disease,
                    'probability': round(probability, 2),
                    'key_symptoms': sorted(symptoms_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                })
            
            return predictions
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return []

# ------------------------------
# ✅ Disease Predictor Functions
# ------------------------------
def load_disease_predictor():
    """Load the disease prediction model and vectorizer from saved files"""
    try:
        predictor = ImprovedDiseasePredictor()
        predictor.model = joblib.load('disease_model.pkl')
        predictor.vectorizer = joblib.load('disease_vectorizer.pkl')
        print("Disease prediction model and vectorizer loaded successfully!")
        return predictor
    except Exception as e:
        print(f"Error loading disease prediction model: {e}")
        raise

#---------------------------------
#✅ Vitals Monitoring
#---------------------------------
# class VitalsMonitor:
#     def __init__(self, patient_id, window_size=20):
#         self.patient_id = patient_id
#         self.window_size = window_size
#         self.model = None  # Will be loaded later
#         self.scaler = None  # Will be loaded later
#         self.vital_features = ['heart_rate', 'blood_pressure_systolic', 
#                                'blood_pressure_diastolic', 'spo2', 'temperature']
        
#         # Initialize sliding windows for each vital sign
#         self.history = {feature: deque(maxlen=window_size) for feature in self.vital_features}
        
#     def add_to_history(self, vitals_data):
#         """Add new vitals data to sliding window"""
#         for feature in self.vital_features:
#             self.history[feature].append(vitals_data[feature])
            
#     def get_window_statistics(self):
#         """Calculate statistical features from the window"""
#         stats = {}
#         for feature in self.vital_features:
#             window_data = np.array(self.history[feature])
#             if len(window_data) == self.window_size:
#                 stats[f'{feature}_mean'] = np.mean(window_data)
#                 stats[f'{feature}_std'] = np.std(window_data)
#                 stats[f'{feature}_min'] = np.min(window_data)
#                 stats[f'{feature}_max'] = np.max(window_data)
#                 stats[f'{feature}_trend'] = np.polyfit(range(len(window_data)), window_data, 1)[0]
#         return stats
    
#     def preprocess_data(self, data):
#         """Preprocess vitals data with window statistics"""
#         # Handle missing values
#         processed_data = data[self.vital_features].ffill().bfill()
        
#         # Add window statistics if available
#         if all(len(self.history[feature]) == self.window_size for feature in self.vital_features):
#             stats = self.get_window_statistics()
#             for stat_name, value in stats.items():
#                 processed_data[stat_name] = value
                
#         # Convert DataFrame to NumPy array before scaling
#         processed_data = processed_data.to_numpy().reshape(1, -1)
        
#         # Ensure scaler is fitted before transforming
#         return self.scaler.transform(processed_data)
    
#     def train(self, data_path):
#         """Train the vitals monitoring model from a CSV file"""
#         raw_data = pd.read_csv(data_path)
        
#         # Initialize history with the first window_size records
#         for _, row in raw_data.head(self.window_size).iterrows():
#             self.add_to_history(row.to_dict())
        
#         # Prepare training data with sliding windows
#         X = []
#         for i in range(self.window_size, len(raw_data)):
#             window_data = raw_data.iloc[i:i+1].copy()
#             self.add_to_history(raw_data.iloc[i].to_dict())
#             stats = self.get_window_statistics()
#             for stat_name, value in stats.items():
#                 window_data[stat_name] = value
#             X.append(window_data[self.vital_features + list(stats.keys())].ffill().bfill())  
        
#         X = np.vstack(X)
        
#         # Fit and save scaler on the **final feature set**
#         self.scaler = StandardScaler()
#         self.scaler.fit(X)
#         joblib.dump(self.scaler, f'scaler_{self.patient_id}.pkl')

#         # Train the anomaly detection model
#         self.model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
#         self.model.fit(self.scaler.transform(X))
#         joblib.dump(self.model, f'vitals_monitor_{self.patient_id}.pkl')

#         print("Model trained successfully!")
    
#     def load_model(self):
#         """Load trained model and scaler"""
#         try:
#             self.model = joblib.load(f'vitals_monitor_{self.patient_id}.pkl')
#             self.scaler = joblib.load(f'scaler_{self.patient_id}.pkl')
#             print("Model and scaler loaded successfully!")
#         except FileNotFoundError:
#             print("Error: Model or scaler file not found. Train the model first.")
    
#     def convert_to_serializable(self, data):
#         """Convert non-serializable data types to serializable types."""
#         if isinstance(data, dict):
#             return {key: self.convert_to_serializable(value) for key, value in data.items()}
#         elif isinstance(data, list):
#             return [self.convert_to_serializable(item) for item in data]
#         elif isinstance(data, (np.int64, np.float64)):
#             return float(data) if isinstance(data, np.float64) else int(data)
#         return data

#     def detect_anomalies(self, vitals_data):
#         """Detect anomalies using current vitals and window history."""
#         if self.model is None or self.scaler is None:
#             self.load_model()

#         self.add_to_history(vitals_data)

#         if all(len(self.history[feature]) == self.window_size for feature in self.vital_features):
#             vitals_df = pd.DataFrame([vitals_data])
#             stats = self.get_window_statistics()
#             for stat_name, value in stats.items():
#                 vitals_df[stat_name] = value

#             scaled_data = self.preprocess_data(vitals_df)

#             predictions = self.model.predict(scaled_data)
#             scores = self.model.score_samples(scaled_data)
#             severity = np.exp(scores)

#             result = {
#                 'is_anomaly': bool(predictions[0] == -1),
#                 'severity_score': float(severity[0]),
#                 'timestamp': datetime.now().isoformat(),
#                 'patient_id': self.patient_id,
#                 'window_stats': stats
#             }

#             return self.convert_to_serializable(result)
#         else:
#             result = {
#                 'is_anomaly': False,
#                 'message': 'Insufficient history for prediction',
#                 'current_window_size': min(len(next(iter(self.history.values()))), self.window_size),
#                 'required_window_size': self.window_size
#             }

#             return self.convert_to_serializable(result)

class VitalsMonitor:
    def __init__(self):
        # Baseline normal ranges for vital signs with wider variation
        self.normal_baselines = {
            'heart_rate': {
                'values': [70, 75, 72, 78, 73, 76, 71, 74, 79, 72, 
                          80, 73, 75, 77, 74, 72, 76, 73, 71, 75],
                'threshold_std': 3.0  # Increased threshold for heart rate
            },
            'blood_pressure_systolic': {
                'values': [118, 115, 122, 119, 116, 117, 120, 118, 122, 114,
                          121, 118, 115, 119, 121, 118, 120, 116, 121, 117],
                'threshold_std': 3.0  # Increased threshold for BP
            },
            'blood_pressure_diastolic': {
                'values': [80, 78, 82, 79, 81, 77, 85, 78, 82, 79,
                          81, 83, 80, 79, 84, 78, 80, 82, 81, 77],
                'threshold_std': 3.0  # Increased threshold for BP
            },
            'spo2': {
                'values': [98, 97, 98, 99, 98, 97, 98, 99, 98, 97,
                          98, 99, 98, 97, 98, 99, 98, 97, 98, 99],
                'threshold_std': 3.0  # Increased threshold for SpO2
            },
            'temperature': {
                'values': [98.0, 98.2, 98.6, 98.3, 98.5, 98.1, 98.4, 98.2, 98.6, 98.3,
                          98.5, 98.2, 98.4, 98.3, 98.5, 98.2, 98.4, 98.3, 98.5, 98.1],
                'threshold_std': 3.0  # Increased threshold for temperature
            }
        }
        
    def update_and_check(self, vitals_data):
        """
        Update the baseline with new data and check for anomalies
        """
        anomalies = []
        timestamp = datetime.now().isoformat()
        
        for vital, value in vitals_data.items():
            if vital in self.normal_baselines:
                baseline = self.normal_baselines[vital]
                
                # Calculate current mean and std
                current_mean = np.mean(baseline['values'])
                current_std = np.std(baseline['values'])
                
                # Check if the new value is anomalous
                z_score = abs((value - current_mean) / current_std)
                is_anomaly = z_score > baseline['threshold_std']
                
                if is_anomaly:
                    anomalies.append({
                        'vital_sign': vital,
                        'value': value,
                        'mean': current_mean,
                        'std': current_std,
                        'z_score': z_score,
                        'threshold': baseline['threshold_std']
                    })
                
                # Update the baseline by removing oldest value and adding new one
                baseline['values'] = baseline['values'][1:] + [value]
        
        result = {
            'is_anomaly': len(anomalies) > 0,
            'timestamp': timestamp,
            'anomalies': anomalies,
            'current_readings': {
                vital: {
                    'value': vitals_data[vital],
                    'mean': np.mean(self.normal_baselines[vital]['values']),
                    'std': np.std(self.normal_baselines[vital]['values'])
                }
                for vital in self.normal_baselines.keys()
                if vital in vitals_data
            }
        }
        
        return result

    def get_baseline_stats(self):
        """
        Get current baseline statistics for all vital signs
        """
        stats = {}
        for vital, baseline in self.normal_baselines.items():
            stats[vital] = {
                'mean': np.mean(baseline['values']),
                'std': np.std(baseline['values']),
                'threshold_std': baseline['threshold_std']
            }
        return stats




##############################################################
        
def train_and_save_models():
    """Train and save both Word2Vec and Random Forest models if they don't exist."""
    global w2v_model, adr_model, drug_adr_map
    try:
        # Load and prepare data
        df = pd.read_csv('output.csv')
        df = df.dropna()
        
        # Create drug-ADR mapping
        drug_adr_map = df.groupby("drug")["effect"].apply(set).to_dict()
        
        # Prepare data for Word2Vec
        drug_sentences = [[drug] for drug in df["drug"].unique()]
        
        # Train Word2Vec model
        w2v_model = Word2Vec(sentences=drug_sentences, vector_size=100, window=5, min_count=1, workers=4)
        
        # Save Word2Vec model
        w2v_model.save('drug_w2v_model.bin')
        
        # Convert drugs to embeddings
        X = np.array([get_word_vector(drug) for drug in df["drug"]])
        y = np.ones(len(df))  # All cases are ADRs
        
        # Train Random Forest model
        adr_model = RandomForestClassifier(n_estimators=100, random_state=42)
        adr_model.fit(X, y)
        
        # Save Random Forest model
        joblib.dump(adr_model, 'adr_prediction_model.pkl')
        
        print("✅ Models trained and saved successfully!")
        return True
    except Exception as e:
        print(f"Error training models: {str(e)}")
        return False

def load_adr_models():
    """Load ADR prediction model and related components."""
    global adr_model, w2v_model, drug_adr_map
    try:
        # Check if models exist, if not train them
        if not os.path.exists('drug_w2v_model.bin') or not os.path.exists('adr_prediction_model.pkl'):
            print("Models not found. Training new models...")
            if not train_and_save_models():
                raise Exception("Failed to train models")
        
        # Load the models
        adr_model = joblib.load('adr_prediction_model.pkl')
        w2v_model = Word2Vec.load('drug_w2v_model.bin')
        
        # Load drug-ADR mapping from CSV
        df = pd.read_csv('output.csv')
        drug_adr_map = df.groupby("drug")["effect"].apply(set).to_dict()
        
        print("✅ ADR prediction models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading ADR models: {str(e)}")
        return False

def get_word_vector(word):
    """Get word vector from Word2Vec model."""
    if word in w2v_model.wv:
        return w2v_model.wv[word]
    else:
        return np.zeros(100)  # Return zero vector for unknown words
    
########################################################################
def dice_coefficients(y_true, y_pred, smooth=100):
    y_true_flatten = tf.keras.backend.flatten(y_true)
    y_pred_flatten = tf.keras.backend.flatten(y_pred)

    intersection = tf.keras.backend.sum(y_true_flatten * y_pred_flatten)
    union = tf.keras.backend.sum(y_true_flatten) + tf.keras.backend.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

def dice_coefficients_loss(y_true, y_pred, smooth=100):
    return -dice_coefficients(y_true, y_pred, smooth)

def iou(y_true, y_pred, smooth=100):
    intersection = tf.keras.backend.sum(y_true * y_pred)
    sum = tf.keras.backend.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou

# Define U-Net model architecture
def create_unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bridge
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    
    # Decoder
    up6 = concatenate([Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = concatenate([Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = concatenate([Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = concatenate([Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # Output
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    # Compile the model
    model.compile(optimizer='adam',
                 loss=dice_coefficients_loss,
                 metrics=[dice_coefficients, iou])
    
    return model

# Load weights from saved model
def load_weights_custom(model_path):
    model = create_unet_model()
    model.load_weights(model_path, by_name=True, skip_mismatch=True)
    return model

# Define image preprocessing function
def preprocess_image(img_path, target_size=(256, 256)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize the image
    img = img[np.newaxis, :, :, :]  # Add batch dimension
    return img


# ------------------------------
# ✅ Load Models Before Requests
# ------------------------------
# def load_models():
#     """Load all required models at startup."""
#     global medical_chatbot, disease_predictor, vitals_monitor, models_loaded
#     if not models_loaded:
#         print("🔄 Loading models...")
#         medical_chatbot = MedicalChatbot()
#         load_biobert_model()
#         load_heart_disease_model()
#         disease_predictor = load_disease_predictor()
        
#         models_loaded = True
#         print("✅ Models loaded successfully!")

def load_models():
    """Load all required models at startup."""
    global medical_chatbot, disease_predictor, vitals_monitor, models_loaded
    if not models_loaded:
        print("🔄 Loading models...")
        medical_chatbot = MedicalChatbot()
        load_biobert_model()
        load_heart_disease_model()
        disease_predictor = load_disease_predictor()
        load_adr_models()  # Add this line
        
        
        models_loaded = True
        print("✅ Models loaded successfully!")

model_tumor = load_weights_custom('unet.hdf5')


#####################################################################################
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    # Load the model and preprocessors
    logger.info("Loading model and preprocessors...")
    model = joblib.load('sleep_disorder_model.joblib')
    scaler = joblib.load('scaler.joblib')
    feature_names = joblib.load('feature_names.joblib')
    encoder_gender = joblib.load('encoder_Gender.joblib')
    encoder_occupation = joblib.load('encoder_Occupation.joblib')
    encoder_bmi = joblib.load('encoder_BMI Category.joblib')
    encoder_target = joblib.load('encoder_target.joblib')
    logger.info("All models and preprocessors loaded successfully")
    logger.info(f"Target classes: {encoder_target.classes_}")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise


@app.before_request
def before_request():
    load_models()

# ------------------------------
# ✅ Flask Routes
# ------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask_medical_chatbot', methods=['POST'])
def ask_medical_chatbot():
    """Chatbot API to handle medical-related questions."""
    data = request.json
    if 'question' not in data:
        return jsonify({'error': 'Missing required field: question'}), 400

    question = data['question']
    try:
        response = medical_chatbot.get_response(question, max_length=512)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_medical_reports', methods=['POST'])
def analyze_medical_reports():
    """API to analyze uploaded medical reports (PDFs or images)."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected for uploading'}), 400
    
    try:
        file_objects = []
        for file in files:
            file_object = io.BytesIO(file.read())
            file_object.name = file.filename
            file_objects.append(file_object)
        
        results = analyze_reports(file_objects)
        return jsonify({'analysis': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_heart_disease', methods=['POST'])
def heart_disease_prediction():
    """API endpoint for heart disease prediction."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    try:
        patient_data = request.json
        
        # Validate that all required features are present
        missing_features = set(heart_disease_features) - set(patient_data.keys())
        if missing_features:
            return jsonify({
                'error': f'Missing required features: {", ".join(missing_features)}'
            }), 400

        # Make prediction
        is_healthy = predict_heart_disease(patient_data)
        
        return jsonify({
            'prediction': 'Not at risk of heart disease' if is_healthy else 'At risk of heart disease',
            'risk_status': not is_healthy
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    """API endpoint for disease prediction based on symptoms."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    try:
        data = request.json
        symptoms_data = data.get('symptoms', [])
        top_n = data.get('top_n', 3)
        
        if not symptoms_data:
            return jsonify({'error': 'No symptoms provided'}), 400

        # Get predictions
        predictions = disease_predictor.predict_disease(symptoms_data, top_n=top_n)
        
        if not predictions:
            return jsonify({'error': 'Could not generate predictions from provided symptoms'}), 400

        return jsonify({
            'predictions': predictions
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Add error logging
        return jsonify({'error': str(e)}), 500
    
# @app.route('/monitor_vitals', methods=['POST'])
# def monitor_vitals():
#     """API endpoint for monitoring vital signs."""
#     if not request.is_json:
#         return jsonify({'error': 'Request must be JSON'}), 400

#     try:
#         vitals_data = request.json
#         required_features = ['heart_rate', 'blood_pressure_systolic', 
#                            'blood_pressure_diastolic', 'spo2', 'temperature']
        
#         # Validate input data
#         missing_features = set(required_features) - set(vitals_data.keys())
#         if missing_features:
#             return jsonify({
#                 'error': f'Missing required vital signs: {", ".join(missing_features)}'
#             }), 400

#         # Get predictions
#         monitor = VitalsMonitor(patient_id="patient_123", window_size=20)

#         # Step 1: Train model using existing vitals CSV file
#         monitor.train("vitals_data.csv")

#         # Step 2: Load trained model
#         monitor.load_model()
#         result = json.dumps(monitor.detect_anomalies(vitals_data))

#         #result = vitals_monitor.detect_anomalies(vitals_data)
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# Create a single instance of VitalsMonitor at the application level
vitals_monitor = VitalsMonitor()

@app.route('/monitor_vitals', methods=['POST'])
def monitor_vitals():
    """API endpoint for monitoring vital signs."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    try:
        vitals_data = request.json
        required_features = ['heart_rate', 'blood_pressure_systolic', 
                           'blood_pressure_diastolic', 'spo2', 'temperature']
        
        # Validate input data
        missing_features = set(required_features) - set(vitals_data.keys())
        if missing_features:
            return jsonify({
                'error': f'Missing required vital signs: {", ".join(missing_features)}'
            }), 400

        # Monitor vitals and get result
        result = vitals_monitor.update_and_check(vitals_data)
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_adr', methods=['POST'])
def predict_adr_route():
    """API endpoint for predicting adverse drug reactions from drug combinations."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    try:
        data = request.json
        drug_list = data.get('drugs', [])
        
        if not drug_list:
            return jsonify({'error': 'No drugs provided'}), 400
            
        if not isinstance(drug_list, list):
            return jsonify({'error': 'Drugs must be provided as a list'}), 400

        # Convert drugs to vectors
        input_vectors = np.array([get_word_vector(drug) for drug in drug_list])
        
        # Make prediction
        prediction = adr_model.predict(input_vectors)
        
        # Find ADRs related to the given drugs
        adrs = set()
        for drug in drug_list:
            if drug in drug_adr_map:
                adrs.update(drug_adr_map[drug])
        
        # Prepare response
        response = {
            'adr_detected': bool(any(prediction)),
            'drug_combination': drug_list,
            'potential_adverse_effects': list(adrs) if adrs else [],
            'recommendation': 'Monitor for potential adverse effects' if any(prediction) else 'No significant interactions predicted'
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Define Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_file = request.files['image']
    img_path = os.path.join('uploads', img_file.filename)
    
    # Save the uploaded image temporarily
    img_file.save(img_path)
    
    # Preprocess the image
    img = preprocess_image(img_path)
    
    # Predict the mask using the model
    pred_img = model_tumor.predict(img)
    
    # Threshold the predicted image to create a binary mask
    tumor_threshold = 0.5
    predicted_mask = (pred_img[0, :, :, 0] > tumor_threshold).astype(np.uint8)
    
    # Check if a tumor is present by summing the predicted mask
    tumor_present = np.sum(predicted_mask) > 0  # If any pixels are predicted as tumor
    
    # Return the result
    detection_result = "Tumor Present" if tumor_present else "No Tumor"
    return jsonify({"detection_result": detection_result}), 200


@app.route('/sleep_disorder', methods=['POST'])
def predict_sleep():
    try:
        data = request.get_json()
        logger.debug(f"Received data: {data}")
        
        # Extract blood pressure values
        systolic, diastolic = map(int, data['Blood Pressure'].split('/'))
        
        # Create feature dictionary with proper column names
        feature_dict = {
            'Gender': encoder_gender.transform([data['Gender']])[0],
            'Age': float(data['Age']),
            'Occupation': encoder_occupation.transform([data['Occupation']])[0],
            'Sleep Duration': float(data['Sleep Duration']),
            'Quality of Sleep': float(data['Quality of Sleep']),
            'Physical Activity Level': float(data['Physical Activity Level']),
            'Stress Level': float(data['Stress Level']),
            'BMI Category': encoder_bmi.transform([data['BMI Category']])[0],
            'Heart Rate': float(data['Heart Rate']),
            'Daily Steps': float(data['Daily Steps']),
            'Systolic': systolic,
            'Diastolic': diastolic
        } 
        
        # Create feature array in correct order
        feature_array = np.array([feature_dict[feature] for feature in feature_names]).reshape(1, -1)
        logger.debug(f"Feature array: {feature_array}")
        
        # Scale features
        scaled_features = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        logger.debug(f"Raw prediction: {prediction}")
        
        # Convert prediction to label
        predicted_label = encoder_target.inverse_transform(prediction)[0]
        logger.debug(f"Predicted label: {predicted_label}")
        
        return jsonify({
            'prediction': predicted_label,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        })


# ------------------------------
# ✅ Run Flask App
# ------------------------------
if __name__ == '__main__':
    app.run(debug=False, port=8000, host='0.0.0.0')