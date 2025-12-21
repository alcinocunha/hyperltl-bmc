import sys

if __name__ == "__main__":
    # check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python cms_gni.py <reviewers> <articles>")
        sys.exit(1)

    R = int(sys.argv[1])  # number of reviewers
    A = int(sys.argv[2])  # number of articles

    def same_assigns(T,U):
        stmts = []
        for a in range(A):
            for r in range(R):
                stmts.append(f"assigns_{a}_{r}[{T}] = assigns_{a}_{r}[{U}]")
        return "(" + " & ".join(stmts) + ")"

    def aligned(r,T,U):
        stmts = []
        for a in range(A):
            stmts.append(f"(assigns_{a}_{r}[{T}] -> (decision_{a}[{T}] = 0 <-> decision_{a}[{U}] = 0) & " + " & ".join([f"(review_{a}_{r2}[{T}] = 0 <-> review_{a}_{r2}[{U}] = 0)" for r2 in range(R)]) + ")")
        return "G (" + " & ".join(stmts) + ")"

    def all_decided(T):
        stmts = []
        for a in range(A):
            stmts.append(f"decision_{a}[{T}] != 0")
        return "F (" + " & ".join(stmts) + ")"

    def same_reviews_decisions(r,T,U):
        stmts = []
        for a in range(A):
            stmts.append(f"(assigns_{a}_{r}[{T}] -> (decision_{a}[{T}] = decision_{a}[{U}] & " + " & ".join([f"review_{a}_{r2}[{T}] = review_{a}_{r2}[{U}]" for r2 in range(R)]) + "))")
        return "G (" + " & ".join(stmts) + ")"
    
    def same_reviews_others(r,T,U):
        stmts = []
        for a in range(A):
            stmts.append(f"(!assigns_{a}_{r}[{T}] -> (" + " & ".join([f"review_{a}_{r2}[{T}] = review_{a}_{r2}[{U}]" for r2 in range(R)]) + "))")
        return "G (" + " & ".join(stmts) + ")"

    print("Forall A . Forall B . Exists C .\n    (" + same_assigns("A","B") + " & " + aligned(0,"A","B") + " & " + all_decided("A") + ")\n    ->\n    (" + same_assigns("B","C") + " & " + same_reviews_decisions(0,"A","C") + " & " + same_reviews_others(0,"B","C") + ")")