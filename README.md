# Traffic Flow Analyzer

The **Traffic Flow Analyzer** is a deep learning-powered tool designed for real-time monitoring and analysis of vehicle traffic. Leveraging advanced object detection models, this project detects vehicles, tracks their movement, and provides key insights such as traffic density, speed estimation, and congestion hotspots. This tool is ideal for urban planners, transport authorities, and smart city enthusiasts aiming to improve traffic management and road safety.

---

## Features
- Real-time vehicle detection and tracking.
- Traffic density analysis and speed estimation.
- Identification of congestion hotspots.
- Scalable for various traffic scenarios.

---

## Repository Structure
- `Congestion_detection/`: Contains the main Python script for congestion detection and YOLO model weights.
- `Model yolo11x/`: Stores the training script.
- `train.py`: Script to train the YOLOv11_x object detection model.
- `dataset.txt`: Provides links to traffic datasets and annotations.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/The-Red-Wood-Lab/Smart-Traffic-Flow-Analyzer.git
cd Smart-Traffic-Flow-Analyzer
```

### 2. Install Dependencies
Ensure Python 3.7+ is installed, then run:
```bash
pip install -r requirements.txt
```

---

## Run the Detection

### GPU Requirements
- Running the **main.py** script efficiently requires a GPU.
- The script was tested on **Google Colab** with a **T4 GPU**. You can access the Colab notebook [here](link-to-colab).

### 1. Navigate to the `Congestion_detection` Directory
```bash
cd Congestion_detection
```

### 2. Run the Main Script
```bash
python main.py --weights path/to/weights.pt --input path/to/your_video.mp4 --output path/to/output_video.mp4
```

   **Arguments**:
   - `--weights`: Path to the YOLO model weights (e.g., `model/yolo_11x_traffic.pt`).
   - `--input`: Path to the input video for analysis.
   - `--output`: Path to save the output video (default: `output_video.mp4`).

### Example Usage
```bash
python main.py --weights model/yolo_11x_traffic.pt --input traffic_video.mp4 --output detected_output.mp4
```

---
### Output Example

The output video `output.mp4` is saved in the `Congestion_detection` folder. This video contains annotated frames highlighting detected congestion areas.
## Training the Model

### `train.py` Script
The `train.py` script in the `model/` directory was used to train the **YOLOv11_x** object detection model.

### Training Environment
- The model was originally trained on **Kaggle** using **2x T4 GPUs**.
- The dataset links are provided in the `Dataset links.txt` file for reference and reproducibility.

### Key Training Details
- The model is fine-tuned for vehicle detection in traffic scenarios.
- Pre-trained weights and additional resources can be added to the repository for convenience.

---

## Dataset

The [dataset](https://github.com/tsp1718/Smart-Traffic-Flow-Analyzer/blob/main/Dataset/dataset.txt) file contains essential links to traffic footage and annotations used for training and testing the deep learning models. Ensure you download and use these datasets for further analysis or contributions.

---

### Additional Tips
- **Google Colab**: If you plan to run the project on Colab, ensure you enable GPU in the runtime settings for optimal performance.
- **Custom Input**: Replace the `--input` path with your video file for personalized analysis.

---
