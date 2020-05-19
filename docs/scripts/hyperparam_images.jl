# import Pkg
# Pkg.add("Gadfly")
#
# ^ you have to fight with system at the moment because of Compact conflict.
# I edited Manifest to remove specific reference to Compat version then added
# Gadfly, which downgrades Compat to 2.2

using TreeParzen

import Gadfly


######### PCHOICE #############
example_space = Dict(
    :example => HP.PChoice(
        :example,
        [
            Prob(0.1, 0),
            Prob(0.2, 1),
            Prob(0.7, 2),
        ]
    )
)

trials = [ask(example_space) for i in 1:1000]
samples = getindex.(getproperty.(trials, :hyperparams), :example)
vals = sort(unique(samples))
counts = sum(samples .== vals'; dims=1)
probs = dropdims(counts'/sum(counts), dims=2)
p = Gadfly.plot(x=vals, y=probs, Gadfly.Geom.hair, Gadfly.Geom.point, Gadfly.Scale.y_continuous(minvalue=0.0), Gadfly.Guide.xticks(ticks=vals), Gadfly.Guide.yticks(ticks=Float64.(0:0.1:1)));

p |> Gadfly.SVGJS("$(@__DIR__)/../hp_images/pchoice.svg")
######### UNIFORM #############
example_space = Dict(
    :example => HP.Uniform(:example, 0.0, 1.0),
)

trials = [ask(example_space) for i in 1:1000]
samples = getindex.(getproperty.(trials, :hyperparams), :example)
p = Gadfly.plot(x=samples, Gadfly.Stat.density(bandwidth=0.05), Gadfly.Geom.polygon(fill=true, preserve_order=true));

p |> Gadfly.SVGJS("$(@__DIR__)/../hp_images/uniform.svg")
######### QUNIFORM #############
example_space = Dict(
    :example => HP.QuantUniform(:example, 0.0, 10.0, 2.0),
)

trials = [ask(example_space) for i in 1:1000]
samples = getindex.(getproperty.(trials, :hyperparams), :example)
vals = sort(unique(samples))
counts = sum(samples .== vals'; dims=1)
probs = dropdims(counts'/sum(counts), dims=2)
p = Gadfly.plot(x=vals, y=probs, Gadfly.Geom.hair, Gadfly.Geom.point, Gadfly.Scale.y_continuous(minvalue=0.0), Gadfly.Guide.xticks(ticks=vals));

p |> Gadfly.SVGJS("$(@__DIR__)/../hp_images/quniform.svg")
######### NORMAL #############
example_space = Dict(
    :example => HP.Normal(:example, 4.0, 5.0),
)

trials = [ask(example_space) for i in 1:1000]
samples = getindex.(getproperty.(trials, :hyperparams), :example)
p = Gadfly.plot(x=samples, Gadfly.Stat.density(bandwidth=1), Gadfly.Geom.polygon(fill=true, preserve_order=true));

p |> Gadfly.SVGJS("$(@__DIR__)/../hp_images/normal.svg")
######### QNORMAL #############
example_space = Dict(
    :example => HP.QuantNormal(:example, 2., 0.5, 1.0),
)

trials = [ask(example_space) for i in 1:1000]
samples = getindex.(getproperty.(trials, :hyperparams), :example)
vals = sort(unique(samples))
counts = sum(samples .== vals'; dims=1)
probs = dropdims(counts'/sum(counts), dims=2)
p = Gadfly.plot(x=vals, y=probs, Gadfly.Geom.hair, Gadfly.Geom.point, Gadfly.Scale.y_continuous(minvalue=0.0), Gadfly.Guide.xticks(ticks=vals));

p |> Gadfly.SVGJS("$(@__DIR__)/../hp_images/qnormal.svg")
######### LOGNORMAL #############
example_space = Dict(
    :example => HP.LogNormal(:example, log(3.0), 1.0),
)

trials = [ask(example_space) for i in 1:1000]
samples = getindex.(getproperty.(trials, :hyperparams), :example)
p = Gadfly.plot(x=samples, Gadfly.Stat.density(bandwidth=1.0), Gadfly.Geom.polygon(fill=true, preserve_order=true), Gadfly.Scale.x_log10, Gadfly.Guide.xlabel("x (log)"));

p |> Gadfly.SVGJS("$(@__DIR__)/../hp_images/lognormallog.svg")

p = Gadfly.plot(x=samples, Gadfly.Stat.density(bandwidth=0.5), Gadfly.Geom.polygon(fill=true, preserve_order=true));

p |> Gadfly.SVGJS("$(@__DIR__)/../hp_images/lognormal.svg")
######### QLOGNORMAL #############
example_space = Dict(
    :example => HP.LogQuantNormal(:example, log(3.0), 0.5, 2.0),
)

trials = [ask(example_space) for i in 1:1000]
samples = getindex.(getproperty.(trials, :hyperparams), :example)
vals = sort(unique(samples))
counts = sum(samples .== vals'; dims=1)
probs = dropdims(counts'/sum(counts), dims=2)
p = Gadfly.plot(x=vals, y=probs, Gadfly.Geom.hair, Gadfly.Geom.point, Gadfly.Scale.y_continuous(minvalue=0.0), Gadfly.Guide.xticks(ticks=vals));

p |> Gadfly.SVGJS("$(@__DIR__)/../hp_images/qlognormal.svg")
######### LOGUNIFORM #############
example_space = Dict(
    :example => HP.LogUniform(:example, log(1.0), log(5.0)),
)

trials = [ask(example_space) for i in 1:1000]
samples = getindex.(getproperty.(trials, :hyperparams), :example)
p = Gadfly.plot(x=samples, Gadfly.Stat.density(bandwidth=0.25), Gadfly.Geom.polygon(fill=true, preserve_order=true));

p |> Gadfly.SVGJS("$(@__DIR__)/../hp_images/loguniform.svg")
######### QLOGUNIFORM #############
example_space = Dict(
    :example => HP.LogQuantUniform(:example, log(1.01), log(20.0), float(i)),
 )
trials = [ask(example_space) for i in 1:1000]
samples = getindex.(getproperty.(trials, :hyperparams), :example)
vals = sort(unique(samples))
counts = sum(samples .== vals'; dims=1)
probs = dropdims(counts'/sum(counts), dims=2)
p = Gadfly.plot(x=vals, y=probs, Gadfly.Geom.hair, Gadfly.Geom.point, Gadfly.Scale.y_continuous(minvalue=0.0), Gadfly.Guide. xticks(ticks=vals));
p |> Gadfly.SVGJS("$(@__DIR__)/../hp_images/qloguniform.svg")
######### THEEND #############
