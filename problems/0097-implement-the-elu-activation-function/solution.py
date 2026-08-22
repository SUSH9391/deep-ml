import math
def elu(x: float, alpha: float = 1.0) -> float:
	"""
	Compute the ELU activation function.

	Args:
		x (float): Input value
		alpha (float): ELU parameter for negative values (default: 1.0)

	Returns:
		float: ELU activation value
	"""
	# Your code here
	# if the x the input is -ve or is equal to 0 then smoothens slowly down to lower bound of -alpha usually alpha defaults down to 1.0
	#for +ve values its x=y where y is the otput in this case its x = val
	#for -ve values its x = alpha(ex - 1)
	# Apply ELU formula: x if x > 0 else alpha * (exp(x) - 1)
	
	pass
	return round((x if x > 0 else alpha * (math.exp(x) - 1)), 4)
	