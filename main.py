import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from PIL import Image, ImageTk

# Update the file path to the downloaded SMS Spam Collection dataset
dataset_path = r"spam.csv"

df = pd.read_csv(dataset_path, encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['text'] = df['text'].apply(lambda x: re.sub(r'\W', ' ', str(x).lower()))

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

text_clf.fit(X_train, y_train)

def classify_sms():
    sms_text = preprocess_text(text_entry.get())
    if not sms_text:
        messagebox.showwarning("Input Error", "Please enter a valid SMS.")
        return

    prediction = text_clf.predict([sms_text])
    display_result(prediction[0])

def preprocess_text(text):
    processed_text = re.sub(r'\W', ' ', text.lower())
    return processed_text

def display_result(prediction):
    result_label.config(text=f"Prediction: {'Non-Spam' if prediction == 'ham' else 'Spam'}", foreground="#3498db" if prediction == 'ham' else "#e74c3c")
    feedback_text.config(text=feedback_messages.get(prediction, "No feedback available for this prediction."), foreground="#2c3e50")

def clear_input():
    text_entry.delete(0, tk.END)
    result_label.config(text="")
    feedback_text.config(text="")

root = tk.Tk()
root.title("SMS Spam Detection")

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
style.configure("TButton", padding=5, font=('Helvetica', 12), foreground="white", background="#3498db", borderwidth=0)  # Blue button
style.map("TButton", background=[("active", "#2980b9")])
style.configure("TLabel", font=('Helvetica', 12), foreground="black")  # Black text color
style.configure("TEntry", font=('Helvetica', 12), background="#34495e", foreground="black", borderwidth=0, relief=tk.FLAT)  # Dark gray background with black text

# Layout Configuration
input_frame = ttk.Frame(root, padding=20)
input_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

text_label = ttk.Label(input_frame, text="Enter SMS:")
text_label.grid(column=0, row=0, padx=5, pady=5, sticky=tk.W)

text_entry = ttk.Entry(input_frame, width=40)
text_entry.grid(column=1, row=0, padx=5, pady=5, sticky=tk.W)

classify_button = ttk.Button(input_frame, text="Classify", command=classify_sms)
classify_button.grid(column=2, row=0, padx=5, pady=5, sticky=tk.W)

clear_button = ttk.Button(input_frame, text="Clear", command=clear_input)
clear_button.grid(column=3, row=0, padx=5, pady=5, sticky=tk.W)

result_label = ttk.Label(root, text="", font=("Helvetica", 18, "bold"))
result_label.grid(column=0, row=1, pady=20, sticky=tk.W)

feedback_messages = {
    'ham': 'This message is legitimate and poses no threat to your communication security.',
    'spam': 'Caution! This message appears to be unsolicited and may contain potentially harmful content.'
}
feedback_text = ttk.Label(root, text="", wraplength=500, justify=tk.LEFT, font=('Helvetica', 12))
feedback_text.grid(column=0, row=2, pady=20, sticky=tk.W)

root.mainloop()
