import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
	# Your code here
	#Imagine you have a stack of photographs (our channels), where each photograph is a grid of pixel intensities. Instead of analyzing individual pixels, you calculate the overall average brightness of each photograph, boiling down each image into a single scalar value. This dramatically reduces the number of parameters in a network while retaining global spatial context.
	return np.mean(x, axis=(0, 1))
	pass