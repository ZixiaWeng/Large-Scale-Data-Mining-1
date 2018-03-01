import csv
import numpy as np
from surprise import Dataset
from surprise import Reader
import pandas as pd

class Regression:
    def __init__(self):
        self.data = pd.read_csv('network_backup_dataset.csv')
        print self.data['Week #']


