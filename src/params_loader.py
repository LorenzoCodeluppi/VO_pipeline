from parameters.kitti import kitti_params_map
from parameters.malaga import malaga_params_map
from parameters.parking import parking_params_map
from structures import Dataset



def load_parameters(dataset: Dataset):
  global params
  if dataset == Dataset.KITTI:
    params = kitti_params_map
  if dataset == Dataset.MALAGA:
    params = malaga_params_map
  if dataset == Dataset.PARKING:
    params = parking_params_map
