import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	if not a or not a[0]:
		return []
	#convert each row into list in a list
	flat = [item for row in a for item in row]
	#this list comprehention iterates each ele through row and column
	numr, numc = new_shape
	if len(flat) != numr * numc :
		return []
	
	return [flat[i * numc : (i + 1) * numc] for i in range(numr)]