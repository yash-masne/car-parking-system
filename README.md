# Car Parking System using YOLOv5

## Overview

This project implements an **automated car parking detection system** using **YOLOv5 object detection**. It uses a **pretrained YOLOv5n model** (`yolov5n.pt`) for detecting cars in a parking lot and helps monitor available parking spaces.

The system is built in **Python** and can be extended for live camera feeds or static images.

---

## Features

* Real-time car detection in parking lots.
* Uses **YOLOv5n pretrained model** for fast inference.
* Supports both **images** and **video feeds**.
* Easily extendable to **web-based dashboards** or **IoT integrations**.

---

## Repository Structure

```
project/
├── yolovenv/                   # Python virtual environment (ignored in git)
├── data/                       # Optional: sample images or video
│   ├── images/
│   └── videos/
├── models/
│   └── yolov5n.pt              # Pretrained YOLOv5n weights
├── scripts/                    # Python scripts for inference
│   └── detect.py               # Main detection script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # To ignore venv, logs, etc.
```

---

## Prerequisites

* Python 3.10+
* pip
* Git

---

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/yash-masne/car-parking-system.git
cd car-parking-system
```

2. **Create a virtual environment**

```bash
python -m venv yolovenv
```

3. **Activate the virtual environment**

* Windows:

```bash
yolovenv\Scripts\activate
```

* Linux / Mac:

```bash
source yolovenv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## YOLOv5 Setup

1. Download the **pretrained YOLOv5n weights**:

```bash
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt -P models/
```

> Note: Already included in `models/` folder.

2. Ensure your **dataset or images** are in the `data/` folder.

---

## Running the Detection Script

```bash
python scripts/detect.py --weights models/yolov5n.pt --source data/images
```

* `--weights` → path to YOLOv5 model weights.
* `--source` → folder of images, videos, or live camera feed.
* Outputs are saved in `runs/detect/` folder.

---

## Requirements.txt

Example dependencies:

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python
numpy
matplotlib
pillow
PyYAML
tqdm
```

> Use `pip freeze > requirements.txt` to generate a full list from your environment.

---

## Notes

* **Virtual Environment**: Do not commit `yolovenv/` to GitHub — it's ignored in `.gitignore`.
* **Large Files**: Torch library files are very large; only `.pt` model weights are included.
* **YOLOv5n**: Nano model is used for faster inference with minimal GPU usage.

---

## References

* [YOLOv5 Official Repo](https://github.com/ultralytics/yolov5)
* [YOLOv5n pretrained weights](https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt)

---

This README is **ready-to-use** for GitHub and clearly explains your project, setup, and YOLOv5 usage.

---

