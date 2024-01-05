# vamr_project

## Setup Instructions

### 1. Download Dataset

Download the dataset from the [(VAMR) website](https://rpg.ifi.uzh.ch/teaching.html).

### 2. Organize Dataset

Place the downloaded dataset in the `data` directory at the root of the project.

### 3. Create Virtual Environment

Use `Pipenv` to create a virtual environment for the project:

```
pipenv shell
```

Install the required libraries from the `requirements.txt`:

```
pip install -r requirements.txt
```
### 4. Run the code

Now you can read the code by typing:
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

