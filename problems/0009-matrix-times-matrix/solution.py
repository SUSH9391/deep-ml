def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:

    row_a = len(a)
    row_b = len(b)
    col_a = len(a[0])
    col_b = len(b[0])
    if col_a != row_b:
        return -1
    #this is the result matrix and initialize the result matrix with all 0
    c = [[0 for _ in range(col_b)] for _ in range(row_a)]
    for i in range(row_a):
        for j in range(col_b):
            for k in range(col_a):
                c[i][j] += a[i][k] * b[k][j]
                
    return c