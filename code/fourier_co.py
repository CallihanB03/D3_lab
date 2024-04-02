""""
Fourier Coefficient Script

This script applies Lasso Regressiona on data decomposed with
the Fourier Series to find the Fourier Coefficients. It takes
raw data from a JSON file and saves the Fourier Coefficients 
in another JSON file to the device.

Usage:
    python fourier_co.py data_path [--output directory OUTPUT_DIRECTORY]
    
Arguments:
    data_path (str): Path to the JSON data file containing raw data
    --output_directory (optional1, str): Directory to save the Fourer Coefficient JSON file
    
Dependencies:
    - matplotlib.pyplot
    - sklearn.linear_model.Lasso
    - sklearn.preprocessing.StandardScaler
    - numpy
    - argparse
    - json
    - os
    
How to Run:
    python fourier_co.py data_path 
    OR
    python fourier_co.py data_path --output_directory /path/to/output
    
Author:
    Callihan Bertley
"""

# Package Configuration
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse
import json
import os



def create_json(file_path):
    """"
    Loads json data file into python
    
    Dependencies:
    - json
    
    Parameters:
    - file_path (str): json file path
    
    Returns:
    - json_data (dict): returns dictionary with json file data
    """
    with open(file_path, "r") as file:
        json_data = json.load(file)
    return json_data


# Find Fourier Coefficients
def fourier_transform(time, data, n_terms=50, alpha=0.1, max_iter=10000, plot=False):
  """
  Applies a Lasso Regression on data decomposed with the Fourier Series.

  Dependencies:
  - numpy
  - sklearn.linear_model.Lasso
  - sklearn.preprocessing.StandardScalar

  Parameters:
  - time (numpy.ndarray): time associated with data.
  - data (numpy.ndarray): data to transform.
  - n_terms (int): number of terms in fourier series.
  - alpha (float): Constant that multiplies the L1 term, controlling regularization strength. alpha must be a non-negative float i.e. in [0, inf)
  - max_iter (int): Maximum number of iterations for Lasso regression
  - plot (boolean): if True, will produce plot of the true data vs Lasso_Fourier prediction

  Returns:
  - time (numpy.ndarray): time associated with data
  - predicted data (numpy.ndarray): data predicted with lasso fourier algorithm
  - coefficients (numpy.ndarray): coefficient terms a_0, a_n, and b_n for n ∈ [1, 50]
  """
  # create numpy arrays for fourier transformation
  time = np.array(time)
  data = np.array(data)

  # Number of sine/cosine pairs
  fourier_features = np.zeros((len(time), 2 * n_terms))
  for i in range(1, n_terms+1):
    fourier_features[:,i-1] = np.cos(2 * np.pi * i * time/20)
    fourier_features[:,i+n_terms-1] = np.sin(2 * np.pi * i * time/ 20)

  # Standardize features for better performance with Lasso
  scaler = StandardScaler()
  fourier_features_scaled = scaler.fit_transform(fourier_features)


  # Apply Lasso regression to the Fourier features
  lasso_fourier = Lasso(alpha=alpha, max_iter=max_iter)
  lasso_fourier.fit(fourier_features_scaled, data)     # a_n, and b_n calculated for n ∈ [1, 50]

  # Generate 1001 new points for prediction
  t_new = np.linspace(time.min(), time.max(), 1001)

  # Manually create Fourier features for the new points
  fourier_features_new = np.zeros((len(t_new), 2 * n_terms))
  for i in range(1, n_terms + 1):
    fourier_features_new[:, i - 1] = np.cos(2 * np.pi * i * t_new / 20)
    fourier_features_new[:, i + n_terms-1] = np.sin(2 * np.pi * i * t_new / 20)

  # Standardize the new features using the same scaler as before
  fourier_features_new_scaled = scaler.transform(fourier_features_new)

  # Predict using the Lasso model on the new points
  y_pred_new = lasso_fourier.predict(fourier_features_new_scaled)

  # Find the a_0
  a_0 = np.array(np.mean(data))
  fourier_coefficents = np.hstack((a_0, lasso_fourier.coef_))

  if plot:
    # Plotting the original data and the predictions on the new points
    plt.figure(figsize=(10, 6))
    plt.plot(time, data, color='blue', label='Original Data')
    plt.plot(t_new, y_pred_new,'red', label='Lasso Fourier Prediction', alpha=0.7)
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.title('Lasso Fourier Series Prediction on New Points')
    plt.legend()
    plt.show()
  return t_new, y_pred_new, fourier_coefficents


def save_fs_json(json_data, save=True, directory=os.getcwd()):
    """
    Creates new JSON ile with fourier coefficients
    
    Dependencies:
    - json
    
    Paramters: 
    - json_data (dict): JSON file with raw data
    
    Returns:
    - json_coefficients (dict): JSON file with corresponding fourier coefficients
    
    """
    # Create JSON file
    json_coefficients = {}
    for key in json_data.keys():
        time = json_data[key][0]
        data = json_data[key][-1]
        _, _, fourier_coefficents = fourier_transform(time, data)
        json_coefficients[key] = list(fourier_coefficents)
    
    if save:
        # Save JSON file
        file_name = os.path.join(directory, "fourier_coefficients.json")
        with open(file_name, "w") as file:
            json.dump(json_coefficients, file)
        print("Fourier Coefficients saved to " + file_name)
    return json_coefficients


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="Takes raw data as JSON file and saves fourier coefficients to a new JSON file")
    parser.add_argument("data_path", help="Path to the JSON data with raw data")
    parser.add_argument("--output_directory", help="Directory to save the fourier coefficient JSON file")
    args = parser.parse_args()
    
    # Create JSON file with new data
    json_data = create_json(args.data_path)
    
    # Saves new JSON file
    if args.output_directory:
        _ = save_fs_json(json_data, directory=args.output_directory)
    
    else:
        
        _ = save_fs_json(json_data)

if __name__ == "__main__":
    main()




