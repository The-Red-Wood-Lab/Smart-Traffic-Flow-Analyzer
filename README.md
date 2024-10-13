# 🚦 Traffic Flow Analyzer

The Traffic Flow Analyzer is a deep learning-powered application designed to monitor and analyze vehicle traffic in real-time. Using object detection models, it identifies vehicles, tracks their movement, and provides insights such as traffic density, speed estimation, and congestion hotspots. This application is useful for urban planners, transport departments, and smart city initiatives aiming to improve road safety and traffic management.

---

## 🔗 Features
- **Real-Time Vehicle Detection**: Detects cars, trucks, and other vehicles in live camera feeds.
- **Traffic Density Estimation**: Measures vehicle density to highlight congestion levels.
- **Speed & Flow Analysis**: Estimates vehicle speed and overall flow, with alerts for slowdowns.
- **Congestion Hotspot Identification**: Highlights high-traffic areas to assist with urban planning.

---

## 📋 Technologies

- **Frontend**:
  - **React.js**: Builds an interactive dashboard to display real-time traffic data.
  - **JavaScript/HTML/CSS**: Powers the video feed display, vehicle count, and other visual elements.

- **Backend**:
  - **FastAPI/Flask**: Hosts the deep learning model, processes incoming video feeds, and returns analyzed data.
  - **OpenCV**: Captures video frames and processes them for traffic flow analysis.
  - **YOLOv5**: An object detection model used to identify and track vehicles in video frames.
  - **Database (PostgreSQL)**: Stores traffic metrics and historical data for analysis.

- **Data & Model**:
  - **Traffic Surveillance Dataset**: Used to train the object detection model for identifying different types of vehicles.
  - **YOLOv5 Model**: Custom-trained for accurate vehicle detection and tracking in urban traffic conditions.

---

## 📦 Setup

### Prerequisites
- Python 3.8+
- Node.js & npm
- PostgreSQL (or another database of choice)
- GPU (optional but recommended for real-time performance)

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/traffic-flow-analyzer.git
   cd traffic-flow-analyzer
   ```

2. **Backend Setup**:
   - **Install Dependencies**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Train/Load the Model**:
     - Train the YOLOv5 model on the traffic surveillance dataset or download a pre-trained model.
     - Place the model file in the `models` directory.

3. **Database Setup**:
   - Set up a PostgreSQL database and create tables for storing traffic data.
   - Add database configuration in the `.env` file:
     ```
     DATABASE_URL=postgresql://user:password@localhost:5432/trafficdb
     ```

4. **Frontend Setup**:
   - **Navigate to Frontend**:
     ```bash
     cd frontend
     ```
   - **Install Dependencies**:
     ```bash
     npm install
     ```
   - **Run Frontend**:
     ```bash
     npm start
     ```

5. **Start the Backend Server**:
   - Go back to the main directory and run:
     ```bash
     python app.py
     ```

6. **Open in Browser**:
   - Go to `http://localhost:3000` to interact with the application.

---

## 🎮 Usage Guide

1. **Input Video Stream**: Connect a live camera feed or upload a video file of traffic footage.
2. **Real-Time Detection**: The app will detect vehicles, track their movement, and provide real-time data on traffic density and flow.
3. **Dashboard Visualization**: View vehicle counts, density heatmaps, and flow rate on the interactive dashboard.

---

## 🚗 Traffic Flow Analysis Breakdown

| Metric                     | Description                                |
|----------------------------|--------------------------------------------|
| Vehicle Count              | Total number of detected vehicles          |
| Traffic Density            | Vehicle count per defined area             |
| Speed Estimation           | Estimated speed of moving vehicles         |
| Congestion Alert           | Alerts when density exceeds set threshold  |

---

## 🛠️ Future Improvements

- **Enhanced Vehicle Classification**: Differentiate between cars, buses, motorcycles, etc.
- **Multi-Camera Integration**: Support multiple camera inputs for broader area coverage.
- **Predictive Analytics**: Forecast traffic congestion trends using historical data.

---

## 🤝 Contributing

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

---

## 📄 License
This project is licensed under the MIT License.

---

## 🌐 Acknowledgements

- [Traffic Surveillance Dataset](https://www.kaggle.com/datasets/) for vehicle detection training.
- [YOLOv5](https://github.com/ultralytics/yolov5) for real-time object detection.
- Inspiration from traffic flow analysis studies and smart city applications.
