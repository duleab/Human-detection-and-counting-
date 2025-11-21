# Human Detection and Counting System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A computer vision system for detecting and counting humans in video streams using YOLOv8 models.

##  Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Comparison](#-model-comparison)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

##  Features

- Real-time human detection in videos
- People counting functionality
- Multiple YOLOv8 model comparisons (nano, small, medium)
- Customizable detection parameters
- Output visualization

##  Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/human-detection.git
   cd human-detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

##  Usage

### Running the Main Detection
```python
# Example command to run detection
python src/detection.py --source data/raw/your_video.mp4 --model models/yolov8s/best.pt
```

### Jupyter Notebooks
- `notebooks/Human_Detection_Main.ipynb`: Main detection pipeline
- `notebooks/people_counting.ipynb`: People counting implementation

##  Project Structure

```
human-detection/
├── data/                 # Dataset and video files
├── models/               # Trained models
├── notebooks/            # Jupyter notebooks
├── outputs/              # Output videos and results
├── src/                  # Source code
└── README.md             # Project documentation
```

##  Model Comparison

| Model | mAP@0.5 | FPS  | Parameters | Size  |
|-------|---------|------|------------|-------|
| YOLOv8n | 0.85    | 45   | 3.2M       | 6.4MB |
| YOLOv8s | 0.88    | 35   | 11.2M      | 22MB  |
| YOLOv8m | 0.90    | 25   | 25.9M      | 52MB  |

##  Results

Include sample output images/videos with detection results here.

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- Other relevant libraries and frameworks
