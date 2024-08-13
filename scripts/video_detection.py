import cv2
import torch
from ultralytics import YOLO
import winsound
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
from ttkthemes import ThemedTk
import os
import sys
import requests

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Load helmet detection model
helmet_model_name =  ('D:/StudyMat/Projects/SAM/Safetix-/models/helmet.pt')
helmet_model = YOLO(helmet_model_name)
helmet_model.to(device)

#Loading Glasses Model
glasses_model_name = ('D:/StudyMat/Projects/SAM/Safetix-/models/glasses_best.pt')
glasses_model = YOLO(glasses_model_name)
glasses_model.to(device)


def send_notification(detection_type, confidence):
    server_url = "http://localhost:5000/api/notifications"  
    payload = {
        "detection_type": detection_type,
        "confidence": confidence
    }
    try:
        response = requests.post(server_url, json=payload)
        if response.status_code == 200:
            print(f"Notification sent successfully: {detection_type}")
        else:
            print(f"Failed to send notification: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error sending notification: {e}")

#process models results.
def process_result(result, frame, model):
    x1, y1, x2, y2, score, class_id = result
    
    if model.model_name == helmet_model_name:
        if score > 0.50 and (model.names[int(class_id)] == 'helmet' or model.names[int(class_id)] == 'head'):
            if model.names[int(class_id)] == 'head':
                label_name = 'without_helmet' 
            else:
                label_name = model.names[int(class_id)] 
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        label = f'{label_name}: {score:.2f}'
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        winsound.Beep(1000, 500)
        if label_name == 'without_helmet':
            send_notification(label_name, float(score))  # Use the manually set label

    elif model.model_name == glasses_model_name:
        if score > 0.50 and (model.names[int(class_id)] == 'glasses' or model.names[int(class_id)] == 'sunglasses'):
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'{model.names[int(class_id)]}: {score:.2f}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            winsound.Beep(1000, 500)  # Beep alarm for glasses detection
            send_notification(model.names[int(class_id)], float(score))  # Send notification to server
    
    return None

def detect_helmets(frame):
    results_helmet = helmet_model(frame)[0]
    #threads = []
    
    for result in results_helmet.boxes.data.tolist():
        thread = threading.Thread(target=process_result, args=(result, frame,helmet_model))
        #threads.append(thread)
        thread.start()

    # # Optionally wait for all threads to complete
    # for thread in threads:
    #     thread.join()

    return frame

def detect_glasses(frame):

    results_glasses = glasses_model(frame)[0]
    #threads = []
    
    for result in results_glasses.boxes.data.tolist():
        thread = threading.Thread(target=process_result, args=(result, frame,glasses_model))
        #threads.append(thread)
        thread.start()

    # # Optionally wait for all threads to complete
    # for thread in threads:
    #     thread.join()

    return frame

# Process whole Frame.
def process_frame(frame):

    #Main thread to run both helmets and glasses_model
    helmet_thread = threading.Thread(target=detect_helmets,args=(frame,))
    glasses_thread = threading.Thread(target=detect_glasses,args=(frame,))

    helmet_thread.start()
    glasses_thread.start()

    #Join Both Threads..
    helmet_thread.join()
    glasses_thread.join()

    return frame

#Smothing out the Frame.
def smooth_frame(frame, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(frame, kernel_size, sigma)

#Process Real-time video.
def process_video(video_path, image_label):
    cap = cv2.VideoCapture(video_path)
    
    #Update frames.
    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            
            # Apply Gaussian smoothing
            frame = smooth_frame(frame, kernel_size=(5, 5), sigma=0)
            
            # frame = detect_helmets(frame=frame)
            frame = process_frame(frame)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            image_label.imgtk = imgtk
            image_label.configure(image=imgtk)
            
            #Update frame after every 10 ms.
            image_label.after(10, update_frame)
        else:
            cap.release()

    update_frame()

def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if file_path:
        return file_path
    return None

def setup_gui():
    root = ThemedTk(theme="arc")
    root.title("HELMET AND GLASSES DETECTION SYSTEM")
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

    title_label = ttk.Label(root, text="HELMET AND GLASSES DETECTION SYSTEM **PEGASUS HELMETGUARD**", font=("Helvetica", 16))
    title_label.pack(pady=10)

    instructions_label = ttk.Label(root, text="Upload a video and click 'Process Video' to start detection.", font=("Helvetica", 12))
    instructions_label.pack(pady=10)

    def upload_and_process():
        video_path = upload_video()
        if video_path:
            process_video(video_path, image_label)

    upload_button = ttk.Button(root, text="Upload Video", command=upload_and_process)
    upload_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    setup_gui()