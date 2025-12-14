# HyperLoop: a Bounded Model Checker for HyperLTL

## Description

HyperLoop considers only infinite traces that can be represented in the lasso form: a finite prefix followed by an infinitely repeated lasso. This is the same semantics that complete (non-bounded) model checkers assume for normal LTL.

The user specifies the exact length of the traces to be checked. Since we assume traces to be infinite, for a given length all shorter lasso traces are also considered, so there is no need to iterate the length for completeness.

HyperLoop is implemented in Python and translates the model and the HyperLTL property into a set of constraints using quantifiers that are solved using the Z3 SMT solver. Unlike other existing bounded model checkers for HyperLTL (namely [HyperRUSTY](https://github.com/HyperQB/HyperRUSTY/)), HyperLoop fully supports loop conditions so the results are complete for the given trace length.

The model should be defined in the [SMV language](https://nusmv.fbk.eu) and the HyperLTL property should be specified in a new `HLTLSPEC` section of the SMV file. For the moment, HyperLoop only support declarative models defined using the `INIT`, `TRANS`, and `INVAR` sections. For most examples, this actually results in more concise models than using explicit models defined with the `ASSIGN` section.

## Usage

To run HyperLoop, use the following command:

```bash
python HyperLoop.py <model_file.smv> <trace_length>
```

Replace `<model_file.smv>` with the path to your SMV model file and `<trace_length>` with the exact length of the traces that will be checked.

See example SMV files in the `examples/` directory.