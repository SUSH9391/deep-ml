def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	n_features, n_observations = len(vectors), len(vectors[0])
	covarience_matrix = [[0 for _ in range(n_features)] for _ in range(n_features)]
	mean = [sum(feature)/n_observations for feature in vectors]
	for i in range(n_features):
		for j in range(n_features):
			covarience = sum((vectors[i][k] - mean[i]) * (vectors[j][k] - mean[i]) for k in range(n_observations))/(n_observations-1)
			covarience_matrix[i][j] = covarience
			covarience_matrix[j][i] = covarience
	return covarience_matrix