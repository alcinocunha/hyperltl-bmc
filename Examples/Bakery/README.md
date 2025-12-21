# Bakery

This example illustrates how hyper-properties can be used to verify symmetry in a concurrent algorithm. In particular, we consider the Bakery algorithm by Lamport for mutual exclusion, first introduced in [A New Solution of Dijkstra's Concurrent Programming Problem](https://lamport.azurewebsites.net/pubs/bakery.pdf). This algorithm is not symmetric by design, as each process has a unique identifier that is used to determine the order in which processes can enter their critical section.

This was one of the examples used in the evaluation of the [HyperQB bounded model checker](https://doi.org/10.1007/978-3-030-72016-2_6), but we changed a bit the SMV model to make it closer to the original algorithm description. Given the desired number of processes `N`, script `bakery.py` generates a declarative SMV model using `INIT` and `TRANS`, and script `bakery_assigns.py` generates an equivalent model using explicit assignments. In both models the ticker number is limited to the number of processes, so a process may stutter indefinitely computing its ticket number. Script `equivalence.py` generates a HyperLTL property that can be used to check the equivalence between the declarative and explicit assignment models.

Given the desired number of processes `N`, script `symmetry.py` generates a HyperLTL property that checks a particular symmetry, namely that for any execution of the algorithm, there exists another execution where process `(i + 1) % N` behaves like process `i`. 
To detect the counter-example, a trace bound of at least 8 is needed.

## Example analyses

| Property     | Bound | Models | Result   |
|--------------|-------|--------|----------|
| `symmetry3.hq` | 7 | `bakery3.smv` | UNSAT |
| `symmetry3.hq` | 8 | `bakery3.smv` | SAT |
| `symmetry5.hq` | 8 | `bakery_assigns5.smv` | SAT |
| `equivalence2.hq` | 5 | `bakery2.smv` `bakery_assigns2.smv` | UNSAT |