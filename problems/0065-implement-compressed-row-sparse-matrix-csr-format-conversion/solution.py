import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

	:param dense_matrix: 2D list representing a dense matrix
	:return: A tuple containing (values array, column indices array, row pointer array)
	"""
	ROWS , COLS = len(dense_matrix), len(dense_matrix[0]) 
	row_ptr = [0] #list that grows based on N+1 where N = len(dense_matrix)
	values = [] #is the array with non - elements
	col_idx = []
	for r in range(ROWS):
		for c in range(COLS):
			if dense_matrix[r][c] != 0:
				values.append(dense_matrix[r][c])
				col_idx.append(c)
		row_ptr.append(len(values))
	return values, col_idx, row_ptr
			
	pass
