# Getting Started

Load the package and define a search space using functions from `TreeParzen.HP`. The first argument to each `HP` function is a unique label for that parameter. See the [hyperparameter reference](hyperparams.md) for the full list of distributions.

```julia
using TreeParzen

space = Dict(
    :learning_rate => HP.LogUniform(:learning_rate, log(0.005), log(0.2)),
    :max_depth => HP.Choice(:max_depth, Float64.(vcat(-1, 1:12))),
)
```

## Optimize with `fmin`

`fmin` takes an objective, a search space, and the number of iterations. The objective receives a single `Dict` of parameter values and returns a `Float64` loss to minimize. `fmin` returns the best parameter values found.

```julia
using TreeParzen

objective(params) = (params[:x] - 1.5)^2
space = Dict(:x => HP.Uniform(:x, -5., 5.))

best = fmin(objective, space, 20)
```

If you need to maximize a score, wrap it in an objective that negates the score:

```julia
negative_score(params) = -score(params)
```

Spaces can be dictionaries, arrays, or nested choices. Nested choices can define conditional parameters that only appear for a selected option:

```julia
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
                :scale_pos_weight => HP.QuantUniform(:scale_pos_weight, 1., 10., 1.),
            ),
        ],
    ),
)
```

Other valid space shapes include a scalar choice or an array of parameters:

```julia
scalar_space = HP.Choice(:a_scalar_sampler, [1, 2])
array_space = [
    HP.Choice(:firstel, [10, 100]),
    HP.Uniform(:seconel, 5., 10.),
]
```

For more `fmin` examples, see the [unit tests](https://github.com/IQVIA-ML/TreeParzen.jl/blob/master/test/fmin/points.jl). The [hyperparameter reference](hyperparams.md) documents all available distributions and constraints.

## Ask/Tell API

Use `ask` and `tell!` when you need to control objective evaluation or the optimization loop yourself. Validate the space once with `TreeParzen.Graph.checkspace` before requesting trials; unlike `fmin`, the ask/tell API does not check it automatically.

```julia
using TreeParzen

config = Config()
trialhist = TreeParzen.Trials.Trial[]
space = Dict(:x => HP.Uniform(:x, -5., 5.))
TreeParzen.Graph.checkspace(space)

for _ in 1:100
    trial = ask(space, trialhist, config)
    tell!(trialhist, trial, trial.hyperparams[:x]^2)
end

best = provide_recommendation(trialhist)
```

For model tuning through MLJ, see the [MLJ integration example](examples/simple_mlj_demo/simple_mlj_demo.md). Optimizer settings are described in [Configuration](configuration.md).