import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os

# --- 1. Model Architecture 
class HandwritingRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(HandwritingRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- 2. Configuration & Loading ---
st.set_page_config(page_title="Handwriting RNN", page_icon="✍️")
st.title("📜 Handwriting Character Recognition")
st.markdown("Upload a handwritten character image (28x28) to see the RNN in action.")

@st.cache_resource
def load_resources():
    # Load Mapping
    char_map = {}
    map_path = 'emnist-balanced-mapping.txt'
    if os.path.exists(map_path):
        with open(map_path, 'r') as f:
            for line in f:
                idx, ascii_val = line.split()
                char_map[int(idx)] = chr(int(ascii_val))
    
    # Initialize and Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandwritingRNN(28, 128, 2, 47).to(device)
    
    model_path = "handwriting_rnn_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    return model, char_map, device

model, char_map, device = load_resources()

# --- 3. Image Preprocessing ---
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert('L') # Convert to Grayscale
    image = image.resize((28, 28))
    img_array = np.array(image).astype('float32') / 255.0
    
    # RNNs expect the image orientation to match training (Transpose if needed)
    img_tensor = torch.tensor(img_array.T, dtype=torch.float32).unsqueeze(0).to(device)
    return image, img_tensor

# --- 4. User Interface ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    display_img, input_tensor = preprocess_image(uploaded_file)
    
    with col1:
        st.image(display_img, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        with st.spinner('RNN is analyzing sequences...'):
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            char = char_map.get(predicted_idx.item(), "Unknown")
            
        st.metric(label="Predicted Character", value=char)
        st.metric(label="Confidence Score", value=f"{confidence.item()*100:.2f}%")

st.divider()
st.sidebar.info("Model: 2-Layer LSTM | Dataset: EMNIST Balanced")