module TestMLJIntegration

using MLJBase, MLJDecisionTreeInterface, MLJTuning, TreeParzen
X, y = @load_iris

import MLJDecisionTreeInterface.DecisionTreeClassifier

space = (Dict(
    :min_purity_increase => HP.Uniform(:min_purity_increase, 0.0, 1.0),
    :merge_purity_threshold => HP.Uniform(:merge_purity_threshold, 0.0, 1.0),
    :pdf_smoothing => HP.Uniform(:pdf_smoothing, 0.0, 1.0),
))

dtc = DecisionTreeClassifier()

tm = TunedModel(
    model=dtc,
    ranges=space,
    tuning=MLJTreeParzenTuning(;max_simultaneous_draws=100),
    n=100,
    resampling=CV(nfolds=3, rng=4141),
    measure=cross_entropy
)

mach = machine(tm, X, y)

@info("lazy space")
fit!(mach)


suggestion = Dict(
    :min_purity_increase => 0.6,
    :merge_purity_threshold => 0.6,
    :pdf_smoothing => 0.6,
)

mljspace = MLJTreeParzenSpace(space, suggestion)

tm = TunedModel(
    model=dtc,
    ranges=mljspace,
    tuning=MLJTreeParzenTuning(;max_simultaneous_draws=100),
    n=100,
    resampling=CV(nfolds=3, rng=4141),
    measure=cross_entropy
)

mach = machine(tm, X, y)

@info("single suggestion")
fit!(mach)


suggestions = [
    Dict(
        :min_purity_increase => 0.25,
        :merge_purity_threshold => 0.50,
        :pdf_smoothing => 0.75,
    ),
    Dict(
        :min_purity_increase => 0.75,
        :merge_purity_threshold => 0.25,
        :pdf_smoothing => 0.50,
    ),
    Dict(
        :min_purity_increase => 0.50,
        :merge_purity_threshold => 0.75,
        :pdf_smoothing => 0.25,
    ),
]

mljspace = MLJTreeParzenSpace(space, suggestions)

tm = TunedModel(
    model=dtc,
    ranges=mljspace,
    tuning=MLJTreeParzenTuning(;max_simultaneous_draws=100),
    n=100,
    resampling=CV(nfolds=3, rng=4141),
    measure=cross_entropy
)

mach = machine(tm, X, y)

@info("three suggestions")
fit!(mach)


end # module
true
