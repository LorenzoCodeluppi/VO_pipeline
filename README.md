# Visual Odometry Pipeline

## Overview

The goal of the project is to develop a monocular Visual Odometry pipeline,
incorporating key features such as the initialization of 3D landmarks, keypoint
tracking, pose estimation from 2D ↔ 3D correspondences, and the triangulation
of new landmarks

### Performance

The implemented pipeline demonstrates good quality, exhibiting robustness and effective functionality across various datasets and scenarios.
Additionally, our pipeline achieves notable speed, attaining a performance rate of around 30 fps.

### 1. KITTI Dataset

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/hbj-bSUG1PM/0.jpg)](https://www.youtube.com/watch?v=hbj-bSUG1PM)

### 2. MALAGA Dataset

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/ACydJjL3Eh8/0.jpg)](https://www.youtube.com/watch?v=ACydJjL3Eh8)


### 3. PARKING Dataset

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/a09cD8XpePI/0.jpg)](https://www.youtube.com/watch?v=a09cD8XpePI)


## Setup Instructions

### 1. Download Dataset

Download the dataset from the [(VAMR) website](https://rpg.ifi.uzh.ch/teaching.html).

### 2. Organize Dataset

Place the downloaded dataset in the `data` directory at the root of the project.

### 3. Create Virtual Environment

#### 3.1 Pipenv

To create a virtual environment for the project with Pipenv do the following in the root of this project:

```
pipenv shell
```

Install the required libraries from the `requirements.txt`:

```
pip install -r requirements.txt
```

#### 3.2 Anaconda

To create a virtual environment for the project with Anaconda do the following in the root of this project:

```
conda env create --file=environment.yml
```

The command above will create the virtual environment and install all the necessary libraries.
To activate the environment, type:

```
conda activate vamr
```
### 4. Run the code

Now you can execute the code by typing:
```
python3 src/main.py
```

### 5. Change dataset
To change dataset or any other visualization options, modify the following lines in the ```main.py``` file
```
  # SELECT DATASET
  dataset = Dataset.PARKING
  # IF YOU WANT TO PLOT JUST THE LOCAL TRAJECTORY SET performance_booster = True
  performance_booster = False
  # IF YOU WANT TO COMPARE THE TRAJECTORY WITH THE GROUND TRUTH SET ground_truth_mode = True
  ground_truth_mode = True
  # IF YOU WANT TO PLOT THE FINAL COMPARISON SET final_comparison = True
  # - WARNING: IT WILL NOT PLOT THE TEMPORARY RESULTS
  # - FOR THE MALAGA DATASET NO ground_truth IS PROVIDED
  final_comparison = False
```

