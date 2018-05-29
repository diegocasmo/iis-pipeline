import json
import sys
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import normalize

from constants import get_all_emotions

def predict_emotion(points, directory='./', debug=False):
  # Sample data, do not ship
  # data_df = pd.read_csv(r'./reference.csv')
  # data_df = data_df.dropna(axis=1, how='any')
  # features = [x != 'Label' for x in data_df.columns.values]
  # all_values = data_df.loc[:, features].values
  # r = all_values[0]
  points = np.array(points).flatten()
  if debug:
    print(points)

  # Normalize
  all_values = normalize([points], axis=1)

  # Load PCA model and transform input
  pca = joblib.load(directory + 'pca-model.pkl')
  principal_components = pca.transform(all_values)
  if debug:
    print(principal_components)

  # Load SVM model and predict probability
  svm = joblib.load(directory + 'svm-model.pkl')
  predicted_labels = svm.predict_proba(principal_components)[0]

  # Assign each score to its label
  confidence = {}
  labels = get_all_emotions()
  for index,emotion in enumerate(labels):
    confidence[emotion] = predicted_labels[index]

  if debug:
    print(confidence)
  return (confidence)
  # Output to json file.
  # with open('emotion-output.json', 'w') as outfile:
    # json.dump(confidence, outfile)

if __name__ == '__main__':
  predict_emotion(None, debug=True)
