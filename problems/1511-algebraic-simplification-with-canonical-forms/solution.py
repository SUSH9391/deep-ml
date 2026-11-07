def simplify(expr):
    def is_const(e):
        return isinstance(e, (int, float))

    def step(e):
        if not isinstance(e, tuple):
            return e
        
        op = e[0]
        
        if op == "neg":
            a = step(e[1])
            if isinstance(a, tuple) and a[0] == "neg":
                return a[1]
            if is_const(a):
                return -a
            return ("neg", a)
            
        elif op == "add":
            a = step(e[1])
            b = step(e[2])
            
            # Constant folding
            if is_const(a) and is_const(b):
                return a + b
                
            # Identity: x + 0 -> x, 0 + x -> x
            if b == 0:
                return a
            if a == 0:
                return b
                
            # Canonical constants on the right: c + x -> x + c
            if is_const(a) and not is_const(b):
                return ("add", b, a)
                
            # Chained constants: (x + c1) + c2 -> x + (c1 + c2)
            if isinstance(a, tuple) and a[0] == "add" and is_const(a[2]) and is_const(b):
                return ("add", a[1], a[2] + b)
                
            # Associativity toward the right: x + (y + c) -> (x + y) + c
            if isinstance(b, tuple) and b[0] == "add" and is_const(b[2]) and not is_const(a):
                y, c = b[1], b[2]
                return ("add", ("add", a, y), c)
                
            if isinstance(a, tuple) and a[0] == "add" and is_const(a[2]) and not is_const(b):
                x, c = a[1], a[2]
                return ("add", ("add", x, b), c)
                
            return ("add", a, b)
            
        elif op == "mul":
            a = step(e[1])
            b = step(e[2])
            
            # Constant folding
            if is_const(a) and is_const(b):
                return a * b
                
            # Identity/Zero rules: x * 1 -> x, x * 0 -> 0
            if b == 1:
                return a
            if a == 1:
                return b
            if b == 0 or a == 0:
                return 0
                
            # Canonical constants on the right: c * x -> x * c
            if is_const(a) and not is_const(b):
                return ("mul", b, a)
                
            # Chained constants: (x * c1) * c2 -> x * (c1 * c2)
            if isinstance(a, tuple) and a[0] == "mul" and is_const(a[2]) and is_const(b):
                return ("mul", a[1], a[2] * b)
                
            # Distribution of a constant over a sum: (a + b) * c -> a * c + b * c
            if isinstance(a, tuple) and a[0] == "add" and is_const(b):
                x, y = a[1], a[2]
                return ("add", ("mul", x, b), ("mul", y, b))
            if isinstance(b, tuple) and b[0] == "add" and is_const(a):
                x, y = b[1], b[2]
                return ("add", ("mul", x, a), ("mul", y, a))
                
            return ("mul", a, b)
            
        return e

    # Apply bottom-up transformations repeatedly until a fixed point is reached
    current = expr
    while True:
        def traverse(node):
            if not isinstance(node, tuple):
                return node
            op = node[0]
            new_children = tuple(traverse(c) for c in node[1:])
            return step((op,) + new_children)
            
        next_expr = traverse(current)
        if next_expr == current:
            break
        current = next_expr
        
    return current


def size(expr):
    if not isinstance(expr, tuple):
        return 1
    return 1 + sum(size(child) for child in expr[1:])