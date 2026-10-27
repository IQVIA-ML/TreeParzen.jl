# Differences from Hyperopt

TreeParzen.jl is a pure Julia port of Hyperopt's tree-structured Parzen estimator optimizer. The projects differ in a few important areas:

- Hyperopt supports storing trial results in MongoDB; TreeParzen.jl does not.
- Hyperopt includes simulated annealing optimization; TreeParzen.jl does not.
- Hyperopt supports parallel and distributed optimization. TreeParzen.jl's optimizer is currently single-threaded and runs in a single instance. Its MLJTuning integration can distribute model evaluations, but not the optimization process itself.
- Hyperopt includes plotting functions. TreeParzen.jl does not; its `Vector` of `Trial` objects can be inspected to visualize optimization progress.

See [Getting Started](usage.md) and [MLJ integration](examples/simple_mlj_demo/simple_mlj_demo.md) for the supported workflows.