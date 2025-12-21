import sys

if __name__ == "__main__":
    # check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python cms_any_paper.py <reviewers> <articles>")
        sys.exit(1)

    R = int(sys.argv[1])  # number of reviewers
    A = int(sys.argv[2])  # number of articles

    print("MODULE main")
    print("FROZENVAR")

    
    for a in range(A):
        for r in range(R):
            print(f"    assigns_{a}_{r} : boolean;")

    print("VAR")
    
    # possible reviews: 0 (no review), 1 (reject), 2 (major revision), 3 (accept)
    
    for a in range(A):
        for r in range(R):
            print(f"    review_{a}_{r} : 0..3;")
    
    for a in range(A):
        print(f"    decision_{a} : 0..3;")

    print("INIT")

    constraints = []
    for a in range(A):
        constraints.append("(" + " | ".join([f"assigns_{a}_{r}" for r in range(R)]) + ")")
    for r in range(R):
        constraints.append("(" + " | ".join([f"assigns_{a}_{r}" for a in range(A)]) + ")")
    for a in range(A):
        for r in range(R):
            constraints.append(f"review_{a}_{r} = 0")
    for a in range(A):
        constraints.append(f"decision_{a} = 0")

    print("    "+" &\n    ".join(constraints))

    print("TRANS")

    def stutter():
        stmts = []
        for a in range(A):
            for r in range(R):
                stmts.append(f"next(review_{a}_{r}) = review_{a}_{r}")
        for a in range(A):
            stmts.append(f"next(decision_{a}) = decision_{a}")
        return "("+ " & ".join(stmts) + ")"
    
    def review(a,r):
        stmts = []
        stmts.append(f"assigns_{a}_{r}")
        stmts.append(f"review_{a}_{r} = 0")
        stmts.append(f"next(review_{a}_{r}) != 0")
        for a2 in range(A):
            for r2 in range(R):
                if a2 != a or r2 != r:
                    stmts.append(f"next(review_{a2}_{r2}) = review_{a2}_{r2}")
        for a2 in range(A):
            stmts.append(f"next(decision_{a2}) = decision_{a2}")
        return "(" + " & ".join(stmts) + ")"
    
    def decide(a):
        stmts = []
        stmts.append(f"decision_{a} = 0")
        stmts.append("(" + " & ".join([f"(assigns_{a}_{r} -> review_{a}_{r} != 0)" for r in range(R)]) + ")")
        stmts.append(f"next(decision_{a}) != 0")
        stmts.append("(" + " | ".join([f"next(decision_{a}) = review_{a2}_{r}" for a2 in range(A) for r in range(R)]) + ")")
        for a2 in range(A):
            if a2 != a:
                stmts.append(f"next(decision_{a2}) = decision_{a2}")
        for a2 in range(A):
            for r2 in range(R):
                stmts.append(f"next(review_{a2}_{r2}) = review_{a2}_{r2}")
        return "(" + " & ".join(stmts) + ")"

    print("    " + " |\n    ".join([stutter()] + [review(a,r) for a in range(A) for r in range(R)] + [decide(a) for a in range(A)]))
