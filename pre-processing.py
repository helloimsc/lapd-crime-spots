import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split   
from scipy import stats
from sklearn.cluster import DBSCAN
        
class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)
        self.train = None
        self.test = None
    
    def get_info(self):
        # get the detailed information of the origin dataset
        return self.data.info()
    
    def show_data(self):
        # show the first 5 rows of the origin dataset
        return self.data.head()
    
    def spliting(self):
        # Split the dataset into training and test sets
        self.train, self.test = train_test_split(self.data, test_size = 0.2, random_state = 42)
        
class DataPreprocessor:
    def __init__(self, data):
        self.data = data
    
    def check_duplicated(self):
        # Check for the number of duplicated rows
        self.duplicates = self.data.duplicated().sum()
        return self.duplicates
    
    def drop_duplicated(self):
        # If duplicates found, drop them, keep first occurence
        if self.duplicates > 0:
            self.data.drop_duplicates(inplace = True)
        else:
            pass

    def check_missing_value(self):
        # replace 0 with NaN
        self.data.replace(0, np.nan, inplace = True)
        # Check for missing values in crime data
        self.missing_values = self.data.isnull().sum()
        return self.missing_values
    
    def drop_missing_values(self):
        # delete rows with empty cells
        self.data = self.data.dropna()

    def outlier_detection(self, method = 'clustering'):
        # Check for univariate outliers in one column
        # since most of our features are objects, we only do outlier detection for latitude and longitude
        if method == 'iqr':
            self.lower_bound = self.data[['LAT', 'LON']].quantile(0.25) - 1.5 * self.data[['LAT', 'LON']].std()
            self.upper_bound = self.data[['LAT', 'LON']].quantile(0.75) + 1.5 * self.data[['LAT', 'LON']].std()
            self.outliers_iqr = (self.data[['LAT', 'LON']] < self.lower_bound) | (self.data[['LAT', 'LON']] > self.upper_bound)
            self.outliers = self.data[self.outliers_iqr.any(axis = 1)]
        elif method == 'z':
            self.z_scores = abs(stats.zscore(self.data[['LAT', 'LON']]))
            self.threshold = 3
            self.outliers_z = (self.z_scores > self.threshold).any(axis = 1)
            self.outliers = self.data[self.outliers_z]
        elif method == 'clustering':
            # Fit DBSCAN
            dbscan = DBSCAN(eps = 0.1, min_samples = 2).fit(self.data[['LAT','LON']])
            # Add cluster labels to the DataFrame
            self.data.loc[:, 'outlier_cluster'] = dbscan.labels_
            # Identify outliers (labeled as -1)
            self.outliers = self.data[self.data['outlier_cluster'] == -1]
        else:
            print('Invalid method')

        return self.outliers

    def datetime_covertion(self):
        # convert 'Date Rptd' and 'DATE OCC' into datetime
        self.data['Date Rptd'] = pd.to_datetime(self.data['Date Rptd'], format = '%m/%d/%Y %I:%M:%S %p')
        self.data['DATE OCC'] = pd.to_datetime(self.data['DATE OCC'], format = '%m/%d/%Y %I:%M:%S %p')

        # extract year, month and day of the week from datetime
        self.data['Year OCC'] = self.data['DATE OCC'].dt.year
        self.data['Month OCC'] = self.data['DATE OCC'].dt.month
        self.data['Day OCC'] = self.data['DATE OCC'].dt.day_name()
        
# Load dataset
file_path = "/Users/daisyoung/Downloads/Crime_Data_from_2020_to_Present.csv"
dataloader = DataLoader(file_path)
dataloader.spliting()
train = dataloader.train
test = dataloader.test

# Pre-processing dataset
preprocessor = DataPreprocessor(train)
preprocessor.check_duplicated()
preprocessor.check_missing_value()
preprocessor.drop_missing_values()
preprocessor.outlier_detection()
preprocessor.datetime_covertion()