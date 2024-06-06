import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import joblib
import matplotlib.pyplot as plt

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        self.canvas = tk.Canvas(self.root, width=200, height=200, bg="white")
        self.canvas.pack()

        self.image = Image.new("L", (200, 200), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.button_clear = tk.Button(self.root, text="Clear", command=self.clear)
        self.button_clear.pack()
        self.button_predict = tk.Button(self.root, text="Predict", command=self.predict)
        self.button_predict.pack()
        
        print("Loading model...")
        self.model = joblib.load('mnist_svm.pkl')
        print("Model loaded successfully.")

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
        self.draw.line([x1, y1, x2, y2], fill="black", width=5)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, 200, 200), fill="white")

    def predict(self):
        self.image.save("drawing.png")
        img = self.preprocess_image("drawing.png")
        print("Image preprocessed for prediction.")
        pred = self.model.predict(img)
        print(f"Prediction: {pred[0]}")
        messagebox.showinfo("Prediction", f"Digit drawn: {pred[0]}")

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert("L")
        img = ImageOps.invert(img)
        img = img.resize((8, 8), Image.ANTIALIAS)  # Resize to 8x8 pixels
        img = np.array(img)
        img = img / 16.0  # Normalize pixel values to match training data
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)  # Standardize data to zero mean and unit variance
        img = img.flatten().reshape(1, -1)  # Flatten and reshape for the model
        print(f"Preprocessed image shape: {img.shape}")
        
        # Visualize the preprocessed image
        plt.imshow(img.reshape(8, 8), cmap='gray')
        plt.title('Preprocessed Image')
        plt.show()

        return img

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
