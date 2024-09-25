import os
import pickle
import pandas as pd
import numpy as np
import sys
import io
from sentence_transformers import SentenceTransformer
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class RunModel():
    def __init__(self, model, input_data):
        self.model = model
        self.input = input_data
        self.input_df = None
        self.colunms = ['품종', '색', '무게(Kg)', '나이', '성격', '건강', '성별', '중성화유무', '보유물건', '칩등록여부']
        # TODO: gpt embedding으로 변경
        self.embedding_llm = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS') 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load scalers and encoders
        with open('scaler/minmax_scaler_age_weight.pkl', 'rb') as f:
            self.age_weight_scaler = pickle.load(f)
        with open('scaler/breed_targetencoder.pkl', 'rb') as f:
            self.breed_targetencoder = pickle.load(f)
        with open('scaler/charater_kmeans.pkl', 'rb') as f:
            self.character_kmeans = pickle.load(f)
        with open('scaler/chip_labelencoder.pkl', 'rb') as f:
            self.chip_labelencoder = pickle.load(f)
        with open('scaler/color_kmeans.pkl', 'rb') as f:
            self.color_kmeans = pickle.load(f)
        with open('scaler/health_kmeans.pkl', 'rb') as f:
            self.health_kmeans = pickle.load(f)
        with open('scaler/neutering_labelencoder.pkl', 'rb') as f:
            self.neutering_labelencoder = pickle.load(f)
        with open('scaler/property_labelencoder.pkl', 'rb') as f:
            self.property_labelencoder = pickle.load(f)
        with open('scaler/sex_labelencoder.pkl', 'rb') as f:
            self.sex_labelencoder = pickle.load(f)
        with open('scaler/breed_labelencoder.pkl', 'rb') as f:
            self.breed_labelencoder = pickle.load(f)

    def preprocessing(self):
        # Process color using KMeans clustering
        color = str(self.input[1])
        print(f"[DEBUG] color: {color}")
        color_embeddings = self.embedding_llm.encode([color], device=self.device, convert_to_tensor=True)
        color_embeddings_np = color_embeddings.cpu().numpy().astype(np.float64)
        self.input[1] = self.color_kmeans.predict(color_embeddings_np)[0]
        
        # Scale age and weight using Min-Max Scaler
        age = float(self.input[3])
        weight = float(self.input[2])
        age_weight_scaled = self.age_weight_scaler.transform([[age, weight]])
        self.input[3], self.input[2] = age_weight_scaled[0][1], age_weight_scaled[0][0]
        
        # Process character using KMeans clustering
        character = str(self.input[4])
        character_embeddings = self.embedding_llm.encode([character], device=self.device, convert_to_tensor=True)
        character_embeddings_np = character_embeddings.cpu().numpy().astype(np.float64)
        self.input[4] = self.character_kmeans.predict(character_embeddings_np)[0]
        
        # Process health using KMeans clustering
        health = str(self.input[5])
        health_embeddings = self.embedding_llm.encode([health], device=self.device, convert_to_tensor=True)
        health_embeddings_np = health_embeddings.cpu().numpy().astype(np.float64)
        self.input[5] = self.health_kmeans.predict(health_embeddings_np)[0]
        
        # Label Encoding for categorical variables
        self.input[6] = self.sex_labelencoder.transform([self.input[6]])[0]
        self.input[7] = self.neutering_labelencoder.transform([self.input[7]])[0]
        self.input[8] = self.property_labelencoder.transform([self.input[8]])[0]
        self.input[9] = self.chip_labelencoder.transform([self.input[9]])[0]
        
        # Process breed encoding and target encoding
        breed = str(self.input[0])
        breed_labelencoded = self.breed_labelencoder.transform([breed])[0]
        
        # Update self.input with the correct breed encoding
        self.input[0] = breed_labelencoded

    def classification(self):
        self.preprocessing()

        # Convert input to DataFrame
        self.input_df = pd.DataFrame([self.input], columns=self.colunms)
        
        # Convert columns to float
        try:
            self.input_df = self.input_df.astype(float)
        except ValueError as e:
            print(self.input_df)
            return  # Stop execution if conversion fails
        
        # Model prediction
        result = self.model.predict_proba(self.input_df)

        # Print results
        if result[0][0] > result[0][1]:
            print('{:.2%}의 확률로 미입양'.format(result[0][0]))
        else:
            print('{:.2%}의 확률로 입양'.format(result[0][1])) 


input_line = sys.stdin.readlines()
input_list = list(map(lambda x:x.strip(), input_line))

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = script_dir + '\\model_state\\lgb_model.pkl'
dog_adopt_model = pickle.load(open(file_path, 'rb'))
run_model = RunModel(dog_adopt_model, input_list)
run_model.classification()
