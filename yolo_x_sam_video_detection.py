import cv2,time,torch,queue
import numpy as np
from ultralytics import YOLO
from ultralytics import FastSAM
import winsound
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
from ttkthemes import ThemedTk

#Set Up FPS Lock And Segmentation after Interval
FPS=30
FRAME_INTERVAL=2

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load helmet detection model
helmet_model_path = "./models/helmet.pt"
helmet_model = YOLO(helmet_model_path)

#optionally fuse model for faster inference, but increased load time
#helmet_model.fuse()
helmet_model.to(device)

#Loading Glasses Model
glasses_model_path = "./models/glasses_best.pt"
glasses_model = YOLO(glasses_model_path)

#optionally fuse model for faster inference, but increased load time
#glasses_model.fuse()
glasses_model.to(device)

#Load SAM Model and Move to GPU.
sam_model = FastSAM("./models/FastSAM-s.pt")
#sam_model.model.image_encoder.img_size = (480, 640)  # Set the new input size
sam_model = sam_model.to(device)

sam_model.compile()

def empty_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break
#process models results.
def process_result(result,frame,model):
    x1, y1, x2, y2, score, class_id = result
    print(f'Model Name:{model.model_name}')
    #helmet model detections
    if model.model_name == helmet_model_path:
        
        if score > 0.60 and model.names[int(class_id)] == 'helmet':
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'Helmet: {score:.2f}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #glasses model detections
    elif model.model_name == glasses_model_path:

        if score > 0.50 and (model.names[int(class_id)] == 'glasses' or model.names[int(class_id)] == 'sunglasses'):
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'{model.names[int(class_id)]}: {score:.2f}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    return None

# segment the detected models result.
def segment_result(model_bboxes,frame,frame_count,curr_mask,mask_queue,sam_m):
    #x1, y1, x2, y2, score, class_id = result

    if frame_count % FRAME_INTERVAL == 1 or curr_mask is None:
        # print(f'SAM Model:{sam_m}')

        print(f"Is SAM model on GPU: {next(sam_m.parameters()).is_cuda}")
        #disable optimizers since we are doing only forward pass

        if len(model_bboxes) != 0:
            with torch.no_grad():
                seg_results = sam_m(frame, bboxes=model_bboxes)
                print(f'Segmentation Results Length {len(seg_results)}:')
            # Process Segmentation results
            for seg_result in seg_results:
                if seg_result.masks is not None:
                    # Convert mask to numpy array
                    mask_image = seg_result.masks.data[0].cpu().numpy()
                    
                    # Ensure mask_image is 2D and contains values 0 or 1
                    mask_image = (mask_image > 0.5).astype(np.uint8) * 255
                    
                    # Resize mask to match the original frame size
                    mask_image = cv2.resize(mask_image, (frame.shape[1], frame.shape[0]))
                    
                    #Current Mask
                    curr_mask = mask_image
                    mask_queue.put(curr_mask)
    
    return None

def detect_helmets(frame,frame_count,curr_mask,mask_queue,sam_m):
    results_helmet = helmet_model(frame)[0]
    helmet_bboxes = []

    #for each result from helmet model make a bounding box and pick its co-ordinates for segmentation.
    for result in results_helmet.boxes.data.tolist():
        h_thread = threading.Thread(target=process_result, args=(result, frame,helmet_model))
        h_thread.start()
        x1,y1,x2,y2,score,class_id = result
        if(helmet_model.names[int(class_id)] == 'helmet' and score > 0.60):
            helmet_bboxes.append([x1,y1,x2,y2])

    #print(f'helmet Boxes:{helmet_bboxes}')        
    seg_thread = threading.Thread(target=segment_result,args=(helmet_bboxes,frame,frame_count,curr_mask,mask_queue,sam_m))
    seg_thread.start()

    #Optionally Join the threads.
    seg_thread.join()

    return frame

def detect_glasses(frame,frame_count,curr_mask,mask_queue,sam_m):
    results_glasses = glasses_model(frame)[0]
    glasses_bboxes = []

    for result in results_glasses.boxes.data.tolist():
        g_thread = threading.Thread(target=process_result, args=(result, frame,glasses_model))
        g_thread.start()
        x1,y1,x2,y2,score,class_id = result
        if((glasses_model.names[int(class_id)] == 'glasses' or glasses_model.names[int(class_id)] == 'sunglasses') and score>0.50):
            glasses_bboxes.append([x1,y1,x2,y2])

    # print(f'Glasses Boxes:{glasses_bboxes}')    
    seg_thread = threading.Thread(target=segment_result,args=(glasses_bboxes,frame,frame_count,curr_mask,mask_queue,sam_m))
    seg_thread.start()

    # seg_thread.join()

    return frame

# Process whole Frame.
def process_frame(frame,frame_count,current_mask,mask_queue,sam_m):

    #after every frame interval, free the previous segmentations.
    if frame_count % FRAME_INTERVAL == 1:
        empty_queue(mask_queue)

    #Main thread to run both helmets and glasses_model
    helmet_thread = threading.Thread(target=detect_helmets,args=(frame,frame_count,current_mask,mask_queue,sam_m))
    glasses_thread = threading.Thread(target=detect_glasses,args=(frame,frame_count,current_mask,mask_queue,sam_m))

    helmet_thread.start()
    glasses_thread.start()

    #Join Both Threads..
    helmet_thread.join()
    glasses_thread.join()

    #get mask result
    if not mask_queue.empty():
        for mask in mask_queue.queue:
                current_mask = mask
                # Create a colored mask overlay
                colored_mask = cv2.merge([np.zeros_like(current_mask), current_mask, np.zeros_like(current_mask)])
                # Overlay the mask on the original frame
                frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

    return frame , current_mask

#Smothing out the Frame.
def smooth_frame(frame, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(frame, kernel_size, sigma)

#Process Real-time video.
def process_video(video_path,image_label,sam_m):
    
    cap = cv2.VideoCapture(video_path)
    mask_queue = queue.Queue(maxsize=50)
    # Use mutable objects to store state
    state = {
        'frame_count': 0,
        'temp_mask': None
    }

    #Update frames.
    def update_frame():
        start_time = time.time()
        ret, frame = cap.read()
        state['frame_count']= state['frame_count'] + 1

        print(f"Frame Number: {state['frame_count']}")
        if ret:
            frame = cv2.resize(frame, (640, 480))
            
            # Apply Gaussian smoothing
            frame = smooth_frame(frame, kernel_size=(5, 5), sigma=0)
            
            frame , state['temp_mask'] = process_frame(frame,state['frame_count'],state['temp_mask'],mask_queue,sam_m)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            image_label.imgtk = imgtk
            image_label.configure(image=imgtk)

            processing_time = time.time() - start_time
            print(f'Processing Time: {processing_time}')
            #Lock IN of 30 FPS
            wait_time = max(1, int((1/FPS - processing_time) * 1000))
            
            #Update frame to make 30 consistent FPS.
            image_label.after(wait_time,update_frame)
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
    root.title("HELMET AND GLASSES DETECTION SYSTEM **PEGASUS HELMETGUARD**")
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
            process_video(video_path, image_label,sam_model)

    upload_button = ttk.Button(root, text="Upload Video", command=upload_and_process)
    upload_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    setup_gui()