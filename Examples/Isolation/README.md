# Isolation

This example illustrates how hyper-properties can be used to verify if two transactional isolation levels are equivalent.

Given a desired scope for the number of transactions, the number of database keys, and the number of possible values, scripts `isolation_rc.py` and `isolation_ser.py` generate declarative SMV models that describe the behavior of a database working in the *Read Committed* and *Serializability* isolation level, respectively. Both these models formalize these isolation levels according to the definitions of [Seeing is Believing: A Client-Centric Specification of Database Isolation](https://doi.org/10.1145/3087801.3087802). Scripts `isolation_rc_assigns.py` and `isolation_ser_assigns.py` generates equivalent models using explicit assignments.

Given a similar scope, script `isolation_stronger.py` generates a HyperLTL property that checks if an isolation level is stronger that another. The generated property can only be checked on declarative SMV models. For models with explicit assignments, the script `isolation_stronger_assigns.py` should be used instead.

In particular, for all executions satisfying one isolation level where all transactions are eventually committed, there exists at least one execution of another isolation level with the same transactions where all are eventually committed. For example, by running `python HyperLasso.py isolation_3x2x2.hq 4 isolation_rc_3x2x2.smv isolation_ser_3x2x2.smv` we will get a set of 3 transactions defined over 2 keys and 2 values that is possible to commit under *Read Committed*, but not possible to commit under *Serializability*. With a scope of `T` transactions we need to run the model checker with a trace length of at least `T+1` states, otherwise the property will be trivially valid because no set of `T` transactions can commit with less that `T+1` states. Of course, since *Serializability* is strictly stronger than *Read Committed*, checking the property in the opposite direction, for example `python HyperLasso.py isolation_3x2x2.hq 4 isolation_ser_3x2x2.smv isolation_rc_3x2x2.smv`, will never produce a counter-example irrespective of the trace length parameter.

Script `equivalence.py` generates a HyperLTL property that can be used to check the equivalence between the declarative and explicit assignment models.

## Example analyses

| Property     | Bound | Models | Resul   |
|--------------|-------|--------|---------|
| `isolation_3x2x2.hq` | 3 | `isolation_rc_3x2x2.smv isolation_ser_3x2x2.smv` | SAT |
| `isolation_3x2x2.hq` | 4 | `isolation_rc_3x2x2.smv isolation_ser_3x2x2.smv` | SAT |
| `isolation_3x2x2.hq` | 4 | `isolation_ser_3x2x2.smv isolation_rc_3x2x2.smv` | UNSAT |
| `isolation_assigns_3x2x2.hq` | 3 | `isolation_rc_assigns_3x2x2.smv isolation_ser_assigns_3x2x2.smv` | SAT |
| `isolation_assigns_3x2x2.hq` | 4 | `isolation_rc_assigns_3x2x2.smv isolation_ser_assigns_3x2x2.smv` | SAT |
| `isolation_assigns_3x2x2.hq` | 4 | `isolation_ser_assigns_3x2x2.smv isolation_rc_assigns_3x2x2.smv` | UNSAT |
| `equivalence_3x2x2.hq` | 4 | `isolation_rc_3x2x2.smv isolation_rc_assigns_3x2x2.smv` | UNSAT |
| `equivalence_3x2x2.hq` | 4 | `isolation_ser_3x2x2.smv isolation_ser_assigns_3x2x2.smv` | UNSAT |