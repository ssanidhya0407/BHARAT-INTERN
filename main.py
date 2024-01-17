import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import models, transforms
import requests
from io import BytesIO

# Load pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])

    if file_path:
        try:
            # Load and preprocess the image
            img = Image.open(file_path)
            img_tensor = preprocess(img)
            img_tensor = img_tensor.unsqueeze(0)

            # Make a prediction using ResNet
            with torch.no_grad():
                outputs = model(img_tensor)

            predicted_class = torch.argmax(outputs[0]).item()

            # Display the result
            result_label.config(text=f"Prediction: {labels[predicted_class]}", foreground="#3498db")

            # Display the selected image
            img = img.resize((300, 300))
            img = ImageTk.PhotoImage(img)
            image_label.config(image=img)
            image_label.image = img

        except Exception as e:
            # Print the error to the console
            print(f"An error occurred: {e}")

            # Show an error message to the user
            messagebox.showerror("Error", "An error occurred. Please check the console for details.")

# Create the main window
root = tk.Tk()
root.title("Image Classifier")

# Stylish background image
bg_path = r"C:\Users\vinay\Downloads\pexels-jessica-lewis-583847.jpg"
bg_image = Image.open(bg_path)
bg_image = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(root, image=bg_image)
bg_label.place(relwidth=1, relheight=1)

# Style Configuration
style = ttk.Style()
style.theme_use("clam")  # Choose a minimalist theme

# Custom Styles (Stylish color scheme)
style.configure("TButton", padding=10, font=('Helvetica', 14), foreground="white", background="#3498db", borderwidth=0)  # Blue button
style.map("TButton", background=[("active", "#2980b9")])
style.configure("TLabel", font=('Helvetica', 16), foreground="#ecf0f1")  # Light gray text

# Create UI components
classify_button = ttk.Button(root, text="Classify Image", command=classify_image)
classify_button.pack(pady=20)

result_label = ttk.Label(root, text="", font=("Helvetica", 20, "bold"))
result_label.pack(pady=20)

image_label = ttk.Label(root)
image_label.pack()

# Start the GUI event loop
root.mainloop()
