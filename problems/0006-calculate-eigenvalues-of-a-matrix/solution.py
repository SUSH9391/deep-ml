import math

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    # Unpack the nested list
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]
    
    trace = a + d
    determinant = (a * d) - (b * c)
    discriminant = (trace ** 2) - (4 * determinant)
    
    if discriminant >= 0:
        root_disc = math.sqrt(discriminant)
        lambda1 = (trace + root_disc) / 2
        lambda2 = (trace - root_disc) / 2
        
        eigenvalues = [lambda1, lambda2]
        return eigenvalues
    else:
        # Use the built-in complex() function, no math. prefix needed
        root_disc = complex(0, math.sqrt(-discriminant)) 
        lambda1 = (trace + root_disc) / 2
        lambda2 = (trace - root_disc) / 2
        
        eigenvalues = [lambda1, lambda2]
        return eigenvalues