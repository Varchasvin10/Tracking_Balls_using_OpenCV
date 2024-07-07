# Tracking Balls of different colour using OpenCV 


This program tracks colored balls in a video, identifies their movements between predefined quadrants, and records these movements as events in a CSV file. It also saves a processed video that visualizes the tracked movements. Here's a step-by-step explanation of how the program works:

1. **Necessary Libraries**:
   - `cv2` (OpenCV): For video processing and image operations.
   - `numpy` (np): For numerical operations on arrays.
   - `pandas` (pd): For handling data and saving to CSV.
   - `defaultdict` from `collections`: For tracking the last known quadrant of each ball.
   - `Tk`, `filedialog` from `tkinter`: For file dialogs to select input and output files.
   - `tqdm`: For progress bars.

2. **Function `track_ball_movements`**:
   This function is the core of the program. It processes the video, detects the colored balls, tracks their movements, and records events.

   - **Define Color Ranges**: HSV ranges for different colors are defined to detect specific colored balls.
   - **Video Capture**: Opens the video file using OpenCV's `VideoCapture`.
   - **Get Video Properties**: Retrieves FPS, frame width, height, and total number of frames.
   - **Define Quadrants**: Divides the frame into four quadrants for tracking movements.
   - **Initialize Variables**:
     - `ball_positions`: To store the last known position (quadrant) of each ball.
     - `events`: To record the movement events.
   - **Video Writer Setup**: Initializes the video writer to save the processed video.
   - **Frame Processing Loop**:
     - Reads each frame and converts it to HSV color space.
     - For each color, creates a mask to isolate regions of the frame matching the color.
     - Finds contours in the mask to detect colored balls.
     - Calculates the centroid of each detected ball and determines its quadrant.
     - Records entry and exit events when a ball moves between quadrants.
     - Draws visual markers (circles and labels) on the frame for detected balls.
     - Writes the processed frame to the output video.
   - **Release Resources**: Closes the video capture and writer objects.
   - **Save Events to CSV**: Saves the recorded events to a CSV file.

3. **Function `select_file`**:
   This function uses `tkinter` to open file dialogs for selecting the input video file and specifying the output CSV and video file paths.

4. **Main Script Execution**:
   - Calls `select_file` to get the paths for the input video, output CSV, and output video.
   - If valid paths are provided, it calls `track_ball_movements` with these paths.
   - Prints a message indicating the completion of the tracking process and the locations of the saved files.

### Summary of Events Recorded
The program tracks and records the following events:
- **Entry**: When a ball enters a new quadrant.
- **Exit**: When a ball exits a quadrant.

Each event is saved with the following details:
- `Time`: The timestamp of the event (in seconds).
- `Quadrant Number`: The quadrant number (1 to 4).
- `Ball Colour`: The color of the ball.
- `Type`: The type of event (`Entry` or `Exit`).

### Usage
To use this program:
1. Run the script.
2. Select the input video file when prompted.
3. Specify the output CSV file path for saving events.
4. Specify the output video file path for saving the processed video.
5. The script will process the video, track the colored balls, and save the events and processed video to the specified files.


### As I wanted to achive this goal only using OpenCV, this program has few flaws and these flaws can be tackled with the use of Deep Learning Neural Net models like CNN's, Yolov5 or ViT's.
