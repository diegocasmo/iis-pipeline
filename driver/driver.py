import sys
sys.path.append('../vision')
sys.path.append('../ml')
sys.path.append('../emo')

import numpy as np

import feat_detect as fd # CV
from predict_emotion import predict_emotion #ML

def handle_features(pos, debug=False):
  if debug:
    for i in range(len(pos)):
      print("Feature " + str(i) + " at " + str(pos[i]) + ".")
  pos = A.avrg_frames(pos, depth=25)

  if debug:
    A.dots(pos)
  return pos

if __name__ == '__main__':
  debug = False
  if len(sys.argv) > 1 and sys.argv[1] == '1':
    print('Debug ON')
    debug = True

  A = fd.FeatureReader()
  try:
    while True:
      # fetch camera images and tag landmarks
      pos = A.frame()
      if not np.array_equal(pos[0],[0,0,0]):
        pos = handle_features(pos, debug=debug)

      if debug:
        A.displayimage()

      # predict emotions
      emotion_predictions = predict_emotion(pos, directory='../ml/', debug=debug)
      if debug:
        print(emotion_predictions)

      # create emotions
      # something furnet...

  except KeyboardInterrupt:
    A.finalize()
    print("\nShutting down -- Good Bye")
