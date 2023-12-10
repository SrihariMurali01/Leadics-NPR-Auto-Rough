# License Plate Detection using YOLOv8

This project implements License Plate Detection using YOLOv8, leveraging the Ultralytics YOLO framework.

> This project is coordinated with LEADICS Inc.
## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

License Plate Detection is a computer vision project that utilizes YOLOv8 for accurate and efficient license plate detection. The system is designed to identify license plates in images and provide bounding box coordinates along with the extracted text using OCR.

## Installation

### Prerequisites

Before you begin, ensure you have the following installed:

- [Python](https://www.python.org/) (>= 3.6)
- [Git](https://git-scm.com/)
- There are more prequirements, but covered within the cloned git repository.

### Setup

1. Clone the repository (by Arijit1810):

   ```bash
   git clone https://github.com/Arijit1080/Licence-Plate-Detection-using-YOLO-V8.git
   cd /content/LPR
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   pip install roboflow
   ```

## Usage

Follow these steps to train the model and make predictions:

### Training the Model (Running in Google Colab for GPU utils)

```bash
python /content/LPR/ultralytics/yolo/v8/detect/train.py model=yolov8n.pt data=/content/License-Plate-Detector-2/data.yaml epochs=30
```

### Making Predictions (Demo Predictions)

```bash
python /content/LPR/ultralytics/yolo/v8/detect/predict.py model=/content/runs/detect/train/weights/best.pt source=/content/LPR/demo2.jpeg
```

## Results

A pretrained set of Results can be visualised in the [Results](/Results) folder.

The above code can be executed in Google Colab for on-site visualization too.

## Upgradations

In the [predict.py](/predict.py) file, we also have added the functionality for reading
the number plate and telling us the data. This is acheived via tesseract module, which can be installed in the client system
and also you need to run the following after installing tesseract [Linked Here](/Setup/tesseract-ocr-w64-setup-5.3.3.20231005.exe)

## Code execution

After prerequisites, run the following:

```bash
python predict.py model=<ModelnameWithPath> > source=<imageName with path>
```
<br>

#### Citations:

- [Arijit1080's Github Repo](https://github.com/Arijit1080/Licence-Plate-Detection-using-YOLO-V8.git)

#### My Github: [Srihari Murali](https://github.com/SrihariMurali01)