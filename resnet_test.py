import os
import torch
import cv2
import numpy as np
from torchvision import transforms, models
from torch import nn
from PIL import Image


# Define the model path and load weights
model_save_dir = r'F:\face_recognition\model_weights_resnet' 
model_filename = r'F:\face_recognition\model_weights_resnet\hybrid_antispoofing_model_epoch_10.pth' 
model_path = os.path.join(model_save_dir, model_filename)

# ResNet as feature extractor
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Identity()

# Transformer Block for global feature capture
class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(input_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, input_dim)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        # Attention block
        attn_out, _ = self.attn(x, x, x)
        x = self.layer_norm1(attn_out + x) 
        
        # Feed-forward block
        ffn_out = self.ffn(x)
        x = self.layer_norm2(ffn_out + x) 
        return x

# Hybrid Model (CNN + Transformer)
class HybridModel(nn.Module):
    def __init__(self, resnet, transformer_block):
        super(HybridModel, self).__init__()
        self.resnet = resnet
        self.transformer_block = transformer_block
        self.fc1 = nn.Linear(2048, 512) 
        self.fc2 = nn.Linear(512, 1) 
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
       
        resnet_features = self.resnet(x)
        transformer_input = resnet_features.unsqueeze(0) 
        transformer_output = self.transformer_block(transformer_input)
        
        transformer_output = transformer_output.squeeze(0) 
        x = self.fc1(transformer_output)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer_block = TransformerBlock(input_dim=2048, num_heads=8, ff_dim=2048)
model = HybridModel(resnet, transformer_block).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((160, 160)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

categories = ['real', 'spoof']

while True:
    ret, frame = cap.read()  
    if not ret:
        print("Error: Failed to capture image.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Preprocess the detected face for model prediction
        face = frame[y:y+h, x:x+w]
        image = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) 
        image = Image.fromarray(image) 
        input_tensor = transform(image).unsqueeze(0).to(device) 

        # Get predictions from the model
        with torch.no_grad():
            outputs = model(input_tensor).squeeze()
            prediction = torch.sigmoid(outputs)
            threshold = 0.8 
            predicted_class = 1 if prediction > threshold else 0
            predicted_label = categories[int(predicted_class)]
        cv2.putText(frame, f"Prediction: {predicted_label} ({prediction.item():.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Real-time Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

