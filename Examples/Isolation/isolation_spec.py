import sys
import itertools

if __name__ == "__main__":
    # check command line arguments
    if len(sys.argv) != 4:
        print("Usage: python isolation_spec.py <transactions> <keys> <values>")
        sys.exit(1)

    T = int(sys.argv[1])  # number of transactions
    K = int(sys.argv[2])  # number of keys
    V = int(sys.argv[3])  # number of values

    def same_transactions(A,B):
        stmts = []
        for t in range(T):
            for k in range(K):
                stmts.append(f"(reads_{t}_{k}[{A}] = reads_{t}_{k}[{B}] & writes_{t}_{k}[{A}] = writes_{t}_{k}[{B}])")
        return "(" + " & ".join(stmts) + ")"
    
    def all_installed(A):
        stmts = []
        for t in range(T):
            stmts.append(f"installed_{t}[{A}]")
        return "F (" + " & ".join(stmts) + ")"
    
    print("Forall A . Exists B . !(" + all_installed("A") + " & (" + same_transactions("A","B") + " -> !(" + all_installed("B") + ")))" )