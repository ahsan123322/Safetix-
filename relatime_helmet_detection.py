import cv2
import torch
from ultralytics import YOLO
import winsound
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from ttkthemes import ThemedTk

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load helmet detection model
helmet_model_path = r"./models/helmet.pt"
helmet_model = YOLO(helmet_model_path)
helmet_model.to(device)

def detect_helmets(frame):
    results_helmet = helmet_model(frame, device=device)[0]
    for result in results_helmet.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > 0.76 and helmet_model.names[int(class_id)] == 'helmet':
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'Helmet: {score:.2f}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            winsound.Beep(1000, 500)  # Beep alarm for helmet detection
    return frame

def smooth_frame(frame, kernel_size=(5, 5), sigma=0):
    """
    Apply Gaussian blur to smooth the frame.
    
    :param frame: Input frame
    :param kernel_size: Size of the Gaussian kernel (must be odd numbers)
    :param sigma: Standard deviation in X and Y directions
    :return: Smoothed frame
    """
    return cv2.GaussianBlur(frame, kernel_size, sigma)

def run_program(image_label):
    cap = cv2.VideoCapture(0)
    
    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror the frame
            frame = cv2.resize(frame, (640, 480))
            
            # Apply Gaussian smoothing
            frame = smooth_frame(frame, kernel_size=(5, 5), sigma=0)
            
            frame = detect_helmets(frame)
            
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            image_label.imgtk = imgtk
            image_label.configure(image=imgtk)
            
            image_label.after(10, update_frame)
        else:
            cap.release()

    update_frame()

def setup_gui():
    root = ThemedTk(theme="arc")
    root.title("HELMET DETECTION SYSTEM **PEGASUS HELMETGUARD**")
    root.geometry("800x700")

    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.quit()
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    image_frame = tk.Frame(root, bg="black")
    image_label = tk.Label(image_frame, bg="black")
    image_label.pack()
    image_frame.pack(padx=10, pady=10)

    title_label = ttk.Label(root, text="HELMET DETECTION SYSTEM **PEGASUS HELMETGUARD**", font=("Helvetica", 16))
    title_label.pack(pady=10)

    instructions_label = ttk.Label(root, text="Click 'Run Program' to start the helmet detection system.", font=("Helvetica", 12))
    instructions_label.pack(pady=10)

    run_program_button = ttk.Button(root, text="Run Program", command=lambda: run_program(image_label))
    run_program_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    setup_gui()