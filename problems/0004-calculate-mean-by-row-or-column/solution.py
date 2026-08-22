def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	if mode == 'column':
		column_mean = [sum(cols) / len(cols) for cols in zip(*matrix)]
		means = column_mean
	else:
		row_mean = [sum(row) / len(row) for row in matrix]
		means = row_mean
	
	return means