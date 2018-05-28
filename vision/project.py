import feat_detect as fd
import numpy as np

A = fd.FeatureReader()

def handle_features(pos):
	for i in range(len(pos)):
		print("Feature " + str(i) + " at " + str(pos[i]) + ".")
	pos = A.avrg_frames(pos, depth=25)
	#pos = A.median_frames(pos, depth=25)
	A.dots(pos)
	
try:
	while True:
		pos = A.frame()
		if not np.array_equal(pos[0],[0,0,0]):
			handle_features(pos)
		A.displayimage()
except KeyboardInterrupt:
	A.finalize()
	print("\nShutting down -- Good Bye")
