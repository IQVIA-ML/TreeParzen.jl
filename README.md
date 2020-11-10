# TreeParzen.jl **Beta release**

A pure Julia hyperparameter optimiser.

![](https://github.com/IQVIA-ML/TreeParzen.jl/workflows/build/badge.svg)![Licence](https://img.shields.io/badge/License-BSD%203--Clause-lime.svg?style=flat)

This is a beta release, the package is registered in the general registry.

## Introduction

TreeParzen.jl is a pure Julia port of the Hyperopt Python library, with an interface to [MLJ](https://github.com/alan-turing-institute/MLJ.jl) for use as a hyperparameter tuning strategy.

> *Hyperopt is a Python library for serial and parallel optimization over awkward search spaces, which may include real-valued, discrete, and conditional dimensions.* - [Hyperopt: Distributed Asynchronous Hyper-parameter Optimization](http://hyperopt.github.io/hyperopt) ([GitHub](https://github.com/hyperopt/hyperopt))

TreeParzen.jl is a black-box optimiser based on the tree-parzen estimator method. You can find the original paper that describes this method [here, see section 4 on page 4](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf). It searches for the minimum of a function by manipulating the input parameters. The input parameters can be continuous, discrete or choices between options.

## Differences with hyperopt

Differences between hyperopt and TreeParzen.jl:

- hyperopt supports connection to a MongoDB database for storing the results of trials, TreeParzen.jl does not.
- hyperopt also supports optimisation using annealing, TreeParzen.jl does not.
- hyperopt supports parallelism and distributed computing on top of the IPython engine, TreeParzen.jl is currently single-threaded and single instance. However, TreeParzen.jl comes with MLJTuning integration, which can handle distribution of function evaluations (the expensive part in hyperparameteroptimisation), but not distribution of optimisation itself (which should be *relatively* cheap anyway).
- hyperopt has built-in plotting functions. TreeParzen.jl does not. If you want to visualise what the optimiser is doing you will need to investigate the `Vector` of `Trial` objects.

## Installation

You can install TreeParzen.jl from the REPL with:

```
] add TreeParzen
```

Then use it like this:

```julia
using TreeParzen
```

## Usage

### fmin

The entry point of TreeParzen.jl is the `fmin` function, [currently found in the `API.jl` file](src/API.jl#L216). You can supply to `fmin` a function to be optimised, a space of possible parameters to explore, and the number of iterations to attempt for.

`fmin` will return a `Dict` of parameters that reflect the lowest output it found during the optimisation iterations.

The function to be optimised should return a `Float64`, which the algorithm will attempt to minimise. If your function actually needs to be maximised and you cannot change it, you can wrap it in another function to modify its output, for example:

``` julia
invert_output(params...) = - actual_function(params...)
```

### Spaces

The space is a collection which describes the parameter ranges and choices that can be made. These can be expressed using a family of functions from [the `HP` module](src/HP.jl).

Each `HP.*` function needs to be given the name again as the first parameter, and then further arguments as relevant to the function. [Instructions are available](docs/hyperparams.md).

If using a dictionary form, the key is the what be the name of the parameter. Additionall, elements of the space can be nested inside each other. Here is an example:

```julia
using TreeParzen

space = Dict(
    :num_leaves => HP.QuantUniform(:num_leaves, 1., 1_024., 1.),
    :max_depth => HP.Choice(:max_depth, Float64.(vcat(-1, 1:12))),
    :min_data_in_leaf => HP.QuantUniform(:min_data_in_leaf, 20., 2_000., 1.),
    :max_bin => HP.LogQuantNormal(:max_bin, log(255), 0.5, 1.),
    :learning_rate => HP.LogUniform(:learning_rate, log(0.005), log(0.2)),
    :is_unbalance => HP.Choice(
        :is_unbalance,
        [
            Dict(:is_unbalance => true),
            Dict(
                :is_unbalance => false,
                :scale_pos_weight => HP.QuantUniform(:scale_pos_weight, 1., 10., 1.)
            )
        ]
    )
)
```

Other examples of valid spaces include:

```julia
space = HP.Choice(:a_scalar_sampler, [1, 2]) # will select from 1,2
space = [HP.Choice(:firstel, [10, 100]), HP.Uniform(:seconel, 5., 10.)] # first element will be selected from 10,100 and 2nd element uniformly from 5-10
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


### Ask/Tell API
TreeParzen.jl also supports tuning via an `ask` and `tell!` interface, where the user is afforded
a lot of control on what they can do and just need to ask the optimiser for suggestions, and tell
it about the results.

This allows users to do advanced things such as wrapping up objectives in
a more complex way, using callbacks, controlling termination, optimising after N suggestions,
continuing iterating if solution is not satisfactory, and so on.

One should call `Graph.checkspace(space)` prior to using `ask` -- to avoid inefficiency of repeatedly
checking the space is valid for each ask the user is required to do this themselves.

A basic example:
```julia
using TreeParzen
config = Config()
trialhist = TreeParzen.Trials.Trial[]

space = Dict(:x => HP.Uniform(:x, -5., 5.))
TreeParzen.Graph.checkspace(space)

for i in 1:100

    trial1 = ask(space, trialhist, config)
    tell!(trialhist, trial1, trial1.hyperparams[:x] ^ 2)

end

@show provide_recommendation(trialhist)
```

### MLJTuning
TreeParzen.jl has integration with [MLJTuning](https://github.com/alan-turing-institute/MLJTuning.jl), for which an [example](docs/examples/simple_mlj_demo/simple_mlj_demo.md) is provided.


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

## Contributors ✨

The list of our Contributors can be found [here](CONTRIBUTORS.md).
Please don't hesitate to add yourself when you contribute.
