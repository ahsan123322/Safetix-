import cv2
import torch
import numpy as np
from ultralytics import FastSAM

FRAME_INTERVAL = 2

def preprocess_frame(frame):
    # Resize frame to 640x480
    frame = cv2.resize(frame, (640, 480))
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Normalize the frame (0-255 to 0.0-1.0)
    #frame = frame.astype(np.float32) / 255.0
    # Transpose from HWC to CHW format
    frame = np.transpose(frame, (2, 0, 1))
    # Convert to torch tensor and move to GPU
    frame = torch.from_numpy(frame).float().unsqueeze(0).to(device)
    return frame


# Load the SAM model and move it to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FastSAM(model='./models/FastSAM-s.pt')
# model.model.image_encoder.img_size = (480, 640)  # Set the new input size
model = model.to(device)

frame_count = 0
current_mask = None

print(f"Using Device: {device}")
print(f"Is model on CUDA: {next(model.parameters()).is_cuda}")

video_path = r"./sample_videos/1.mp4"
# Initialize video capture (0 for default camera, or use a file path for video)
cap = cv2.VideoCapture(video_path)

# Check if the video capture is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    frame_count+=1

    # frame = preprocess_frame(frame)

    if frame_count % FRAME_INTERVAL == 1 or current_mask is None:

        # Resize frame
        new_size = (640, 480)
        resized_frame = cv2.resize(frame, new_size)
        
        # Perform segmentation with bounding box prompt (example bounding box)
        with torch.no_grad():
            results = model(resized_frame, bboxes=[[100, 70, 200, 200],[280, 25, 360, 100]])

        # Process results
        for result in results:
            if result.masks is not None:
                # Convert mask to numpy array
                mask_image = result.masks.data[0].cpu().numpy()
                
                # Ensure mask_image is 2D and contains values 0 or 1
                mask_image = (mask_image > 0.5).astype(np.uint8) * 255
                
                # Resize mask to match the original frame size
                mask_image = cv2.resize(mask_image, (frame.shape[1], frame.shape[0]))
                
                #Current Mask
                current_mask = mask_image
                print(f'Current Mask:{current_mask}')

    if current_mask is not None:
        # Create a colored mask overlay
        colored_mask = cv2.merge([np.zeros_like(mask_image), mask_image, np.zeros_like(mask_image)])
        
        # Overlay the mask on the original frame
        frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

    # Draw the bounding box on the frame
    cv2.rectangle(frame, (100, 70), (200, 200), (0, 255, 0), 2)
    cv2.rectangle(frame, (280, 25), (360, 100), (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('Frame',frame)

    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()