import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from tkinter import Tk, filedialog
from tqdm import tqdm

def track_ball_movements(video_path, output_csv, output_video_path):
    # Define the color ranges for different balls (in HSV)
    color_ranges = {
        'Red': [(0, 70, 50), (10, 255, 255)],
        'Green': [(36, 50, 50), (89, 255, 255)],
        'Blue': [(90, 50, 50), (128, 255, 255)],
        'Yellow': [(20, 100, 100), (30, 255, 255)],
        'Orange': [(10, 100, 100), (20, 255, 255)],
        'Purple': [(129, 50, 50), (158, 255, 255)],
        'Cyan': [(85, 50, 50), (95, 255, 255)],
        'White': [(0, 0, 200), (180, 20, 255)],
        'Peach': [(5, 50, 50), (15, 255, 255)],
        # Add more colors if needed
    }

    # Capture the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define quadrants (assuming 2x2 grid for simplicity)
    quadrants = [
        (0, 0, width//2, height//2), (width//2, 0, width, height//2),
        (0, height//2, width//2, height), (width//2, height//2, width, height)
    ]

    # Initialize variables
    ball_positions = defaultdict(lambda: None)  # To keep track of the last known quadrant of each ball
    events = []

    # Set up video writer for processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0

    # Progress bar setup
    progress_bar = tqdm(total=total_frames, desc="Processing Frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        current_positions = {}

        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    cx, cy = x + w // 2, y + h // 2
                    current_positions[color] = (cx, cy)
                    for i, (qx1, qy1, qx2, qy2) in enumerate(quadrants):
                        if qx1 <= cx < qx2 and qy1 <= cy < qy2:
                            current_positions[color] = (cx, cy, i + 1)
                            break

        for color, (cx, cy, quadrant) in current_positions.items():
            prev_quadrant = ball_positions[color]
            if prev_quadrant is not None and prev_quadrant != quadrant:
                event_time = frame_count / fps
                events.append((event_time, prev_quadrant, color, 'Exit'))
                events.append((event_time, quadrant, color, 'Entry'))
                cv2.putText(frame, f"{color} Ball Exit Q{prev_quadrant}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"{color} Ball Entry Q{quadrant}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {event_time:.2f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            elif prev_quadrant is None:
                event_time = frame_count / fps
                events.append((event_time, quadrant, color, 'Entry'))
                cv2.putText(frame, f"{color} Ball Entry Q{quadrant}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {event_time:.2f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ball_positions.update(current_positions)
        frame_count += 1

        # Draw bounding boxes and labels for each ball
        for color, (cx, cy, quadrant) in current_positions.items():
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
            cv2.putText(frame, color, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)
        
        # Update progress bar
        progress_bar.update(1)

    cap.release()
    out.release()
    progress_bar.close()

    # Save the events to a CSV file
    df = pd.DataFrame(events, columns=['Time', 'Quadrant Number', 'Ball Colour', 'Type'])
    df.to_csv(output_csv, index=False)

def select_file():
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes("-topmost", True)  # Bring the file dialog to the front
    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    if not video_path:
        print("No file selected.")
        return None, None, None

    # Prompt for output CSV file path
    output_csv = filedialog.asksaveasfilename(title="Save CSV File", defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if not output_csv:
        print("No output file specified.")
        return None, None, None

    # Prompt for output video file path
    output_video_path = filedialog.asksaveasfilename(title="Save Processed Video", defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    if not output_video_path:
        print("No output video file specified.")
        return None, None, None

    return video_path, output_csv, output_video_path

if __name__ == '__main__':
    video_path, output_csv, output_video_path = select_file()
    if video_path and output_csv and output_video_path:
        track_ball_movements(video_path, output_csv, output_video_path)
        print(f"Tracking completed. Events saved to {output_csv}. Processed video saved to {output_video_path}.")
    else:
        print("Operation cancelled or failed.")

