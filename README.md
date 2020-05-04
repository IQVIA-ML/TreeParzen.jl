# TreeParzen.jl

A pure Julia hyperparameter optimiser.

![](https://github.com/IQVIA-ML/TreeParzen.jl/workflows/build/badge.svg)![Licence](https://img.shields.io/badge/License-BSD%203--Clause-lime.svg?style=flat)

## Introduction

TreeParzen.jl is a pure Julia port of the Hyperopt Python library.

> *Hyperopt is a Python library for serial and parallel optimization over awkward search spaces, which may include real-valued, discrete, and conditional dimensions.* - [Hyperopt: Distributed Asynchronous Hyper-parameter Optimization](http://hyperopt.github.io/hyperopt) ([GitHub](https://github.com/hyperopt/hyperopt))

TreeParzen.jl is a black-box optimiser based on the tree-parzen estimator method. You can find the original paper that describes this method [here, see section 4 on page 4](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf). It searches for the minimum of a function by manipulating the input parameters. The input parameters can be continuous, discrete or choices between options.

## Differences with hyperopt

Differences between hyperopt and TreeParzen.jl:

- hyperopt supports connection to a MongoDB database for storing the results of trials, TreeParzen.jl does not.
- hyperopt also supports optimisation using annealing, TreeParzen.jl does not.
- hyperopt supports parallelism and distributed computing on top of the IPython engine, TreeParzen.jl is currently single-threaded and single instance. However, TreeParzen.jl comes with MLJTuning integration, which can handle distribution of function evaluations (the expensive part in hyperparameteroptimisation), but not distribution of optimisation itself (which should be *relatively* cheap anyway).
- hyperopt has built-in plotting functions. TreeParzen.jl does not. If you want to visualise what the optimiser is doing you will need to investigate the `Vector` of `Trial` objects.

## Usage

The entry point of TreeParzen.jl is the `fmin` function, [currently found in the `API.jl` file](src/API.jl#L216). You can supply to `fmin` a function to be optimised, a space of possible parameters to explore, and the number of iterations to attempt for.

`fmin` will return a `Dict` of parameters that reflect the lowest output it found during the optimisation iterations.

The function to be optimised should return a `Float64`, which the algorithm will attempt to minimise. If your function actually needs to be maximised and you cannot change it, you can wrap it in another function to modify its output, for example:

``` julia
invert_output(params...) = 1 - actual_function(params...)
```

### Spaces

The space is a `Dict` that describes the parameter ranges and choices that can be made. These can be expressed using a family of functions from [the `HP` module](src/HP.jl).

Each function needs to be given the name again as the first parameter, and then further arguments as relevant to the function. [Instructions are available](docs/hyperparams.md).

The dictionary key should be the name of the parameter as a string. Elements of the space can be nested inside each other. Here is an example:

```julia
space = Dict(
    :num_leaves => hp_quniform(:num_leaves, 1, 1_024, 1),
    :max_depth => hp_choice(:max_depth, vcat(-1, 1:12)),
    :min_data_in_leaf => hp_quniform(:min_data_in_leaf, 20, 2_000, 1),
    :max_bin => hp_qlognormal(:max_bin, log(255), 0.5, 1),
    :learning_rate => hp_loguniform(:learning_rate, log(0.005), log(0.2)),
    :is_unbalance => hp_choice(
        :is_unbalance,
        [
            Dict(:is_unbalance => true),
            Dict(
                :is_unbalance => false,
                :scale_pos_weight => hp_quniform(:scale_pos_weight, 1, 10, 1)
            )
        ]
    )
)
```

### fmin sample usage

Here is an example call of `fmin` using the items described above:

```julia
using TreeParzen

best = fmin(
    invert_output, # The function to be optimised.
    space,         # The space over which the optimisation should take place.
    20,          # The number of iterations to take.
)

println(best)
```

For more examples, please see [the unit tests](test/fmin/points.jl).

### Config object

The optimiser itself has a couple of parameters, which are specified in a `Config` object, or alternatively, as keyword arguments to `fmin`.

- `threshold::Float64` : A value between `0` and `1`, which controls the probability threshold at
    which expected improvement criteria is modeled.
- `linear_forgetting::Int` : A positive value which controls the number of historic points which
    are used for probabilistic modelling, and older points beyond this are linearly
    de-weighted.
- `draws::Int` : A positive value which controls the number of samples to draw when making a
    recommendation for next optimisation candidate.
- `random_trials::Int` : A positive value which controls the number of trials of randomly
    generated candidate points before TreeParzen optimisation is used.
- `prior_weight::Float64` : A value between `0` and `1`, which controls importance of user specified
    probabilistic parameters vs the history of trials.

## Development

A custom diagnostic function called `inside()` has been added. It can be called on any `AbstractDelayed` or `History` object, and the contents of the object will be printed to the console in a tree view.

### About the unique identifiers:

Python dictionaries can store multiple classes with the same content as different keys, where Julia will make them equivalent and thus collapse the keys. To get around this, we are using a random number inside every child of `AbstractDelayed`. They are not shown when using `inside()`.

## Unit tests

To run the unit tests:

```bash
julia --project -e "using Pkg; Pkg.test()"
```

