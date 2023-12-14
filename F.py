import cv2
import numpy as np
import urllib
import requests
from io import BytesIO
import torch
import torchvision.transforms as transforms
from tkinter import Tk, Label, Button, Entry, filedialog
from PIL import Image, ImageTk

# PyTorch transformations
transform = transforms.Compose([transforms.ToTensor()])

# Function to detect faces using PyTorch
def detect_faces_torch(frame):
    tensor_image = transform(frame).unsqueeze(0)
    
    # Use your face detection model here if available
    # Replace the following line with your face detection logic
    faces_tensor = torch.rand((1, 1, 100, 100))  
    
    faces = faces_tensor.numpy()
    if faces is not None:
        for face in faces:
            face_image = Image.fromarray(face[0] * 255).convert('RGB')
            if len(data) <= 150:
                data.append(face_image)
            else:
                cv2.putText(frame, 'complete', (250, 250),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    return frame

# Function to capture image from URL using requests
def capture_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)

# Function to update the Tkinter window with the captured frame
def update_frame():
    global frame, panel, data

    frame = capture_image(URL)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = detect_faces_torch(frame)
    
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    panel.config(image=photo)
    panel.image = photo

    root.after(100, update_frame)  # Update every 100 milliseconds

# Function to save images using PIL
def save_images():
    global data

    name = entry_name.get()
    folder_path = filedialog.askdirectory(title="Select Folder to Save Images")

    if name and folder_path:
        os.makedirs(os.path.join(folder_path, name), exist_ok=True)
        for i, face_image in enumerate(data):
            face_image.save(os.path.join(folder_path, name, f'{name}_{i}.jpg'))

# Tkinter GUI setup
root = Tk()
root.title("Face Capture GUI")

frame = capture_image(URL)
frame = detect_faces_torch(frame)
photo = ImageTk.PhotoImage(image=Image.fromarray(frame))

panel = Label(root, image=photo)
panel.pack(side="top", fill="both", expand="yes")

entry_name = Entry(root, width=20)
entry_name.pack(pady=10)

btn_save = Button(root, text="Save Images", command=save_images)
btn_save.pack(pady=10)

update_frame()  # Start the image update loop

root.mainloop()
