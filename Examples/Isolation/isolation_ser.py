import sys
import itertools

if __name__ == "__main__":
    # check command line arguments
    if len(sys.argv) != 4:
        print("Usage: python isolation_ser.py <transactions> <keys> <values>")
        sys.exit(1)

    T = int(sys.argv[1])  # number of transactions
    K = int(sys.argv[2])  # number of keys
    V = int(sys.argv[3])  # number of values

    print("MODULE main")
    print("FROZENVAR")

    for t in range(T):
        for k in range(K):
            print(f"    reads_{t}_{k} : 0..{V};")
            print(f"    writes_{t}_{k} : 0..{V};")

    print("VAR")

    for t in range(T):
        print(f"    installed_{t} : boolean;")
    
    for k in range(K):
        print(f"    value_{k} : 0..{V};")

    print("INIT")

    constraints = []

    for t in range(T):
        constraints.append(" | ".join([f"writes_{t}_{k} > 0" for k in range(K)]))

    for k in range(K):
        for t1,t2 in itertools.combinations(range(T),2):
            constraints.append(f"(writes_{t1}_{k} = 0 | writes_{t2}_{k} = 0 | writes_{t1}_{k} != writes_{t2}_{k})")

    for t in range(T):
        constraints.append(f"!installed_{t}")
    for k in range(K):
        constraints.append(f"value_{k} = 0")

    print("    "+" &\n    ".join(constraints))

    print("TRANS")

    def stutter():
        stmts = []
        for t in range(T):
            stmts.append(f"next(installed_{t}) = installed_{t}")
        for k in range(K):
            stmts.append(f"next(value_{k}) = value_{k}")
        return "("+ " & ".join(stmts) + ")"
    
    def install(t):
        stmts = []
        stmts.append(f"!installed_{t}")

        stmts.append("(" + " & ".join([f"(reads_{t}_{k} = 0 | reads_{t}_{k} = value_{k})" for k in range(K)]) + ")")

        stmts.append(f"next(installed_{t}) = TRUE")
        for t2 in range(T):
            if t2 != t:
                stmts.append(f"next(installed_{t2}) = installed_{t2}")
        for k in range(K):
            stmts.append(f"((writes_{t}_{k} = 0 & next(value_{k}) = value_{k}) | (writes_{t}_{k} > 0 & next(value_{k}) = writes_{t}_{k}))")
        return "("+ " & ".join(stmts) + ")"
    
    print("    " + " |\n    ".join([stutter()] + [install(t) for t in range(T)]))

