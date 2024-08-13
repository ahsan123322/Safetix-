import cv2
import torch
from ultralytics import YOLO
import winsound
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
from ttkthemes import ThemedTk

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load helmet detection model
helmet_model_path = r"./models/helmet.pt"
helmet_model = YOLO(helmet_model_path)
helmet_model.to(device)

def process_result(result,frame):
    x1, y1, x2, y2, score, class_id = result
    if score > 0.65 and helmet_model.names[int(class_id)] == 'helmet':
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f'Helmet: {score:.2f}'
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # winsound.Beep(1000, 500)  # Beep alarm for helmet detection
    return None    

def detect_helmets(frame):
    results_helmet = helmet_model(frame)[0]
    threads = []
    
    for result in results_helmet.boxes.data.tolist():
        thread = threading.Thread(target=process_result, args=(result, frame))
        threads.append(thread)
        thread.start()

    # # Optionally wait for all threads to complete
    # for thread in threads:
    #     thread.join()

    return frame

def smooth_frame(frame, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(frame, kernel_size, sigma)

def process_video(video_path, image_label):
    cap = cv2.VideoCapture(video_path)
    
    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            
            # Apply Gaussian smoothing
            frame = smooth_frame(frame, kernel_size=(5, 5), sigma=0)
            
            frame = detect_helmets(frame=frame)
            
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            image_label.imgtk = imgtk
            image_label.configure(image=imgtk)
            
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

    instructions_label = ttk.Label(root, text="Upload a video and click 'Process Video' to start helmet detection.", font=("Helvetica", 12))
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