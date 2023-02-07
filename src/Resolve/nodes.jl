# Entrypoint (only for TPE, not for random search)
"""
$(TYPEDSIGNATURES)

Resolves a posterior inference space by depth-first iteration, replacing prior random-
variables with new posterior distributions that make use of observations.
"""
function node(
    space::Types.SPACE_TYPE, trials::Vector{Trials.Trial}, config::Config
)::Tuple{Trials.ValsDict, Any}

    vals = Trials.ValsDict()

    hyperparams = node(space, vals, Symbol(), trials, config)

    return vals, hyperparams
end


# Node resolvers. For each node type there is first a random-search resolver and then a TPE
# resolver. They are distinguished by the TPE methods requiring extra items.
"""
$(TYPEDSIGNATURES)

Resolves random search AbstractParam nodes and places parameter results in the vals dictionary.
"""
function node(item::Delayed.AbstractParam, vals::Trials.ValsDict)::Union{IndexObjects.IndexInt, Real}
    if haskey(vals, item.label)
        throw(KeyError("Key $(item) already present in $(vals[item.label])"))
    end

    obj = node(item.obj, vals)
    vals[item.label] = IndexObjects.getval(obj)

    return obj
end
"""
$(TYPEDSIGNATURES)

Resolves random search AbstractParam nodes and places parameter results in the vals dictionary.
"""
function node(
    item::Delayed.AbstractParam, vals::Trials.ValsDict, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Union{IndexObjects.IndexInt, Real}
    if haskey(vals, item.label)
        throw(KeyError("Key $(item.label) already present $(vals[item.label])"))
    end

    # TODO check vals name is the same as hyperparameter name
    # but currently there are many unit tests which have different names.
    # if nid == Symbol()
    #     nid = item.label
    # elseif nid != item.label
    #     throw(KeyError("Hyperparameter $(item.label) is inconsistent with vals name $(nid)"))
    # end

    obj = node(item.obj, vals, item.label, trials, config)
    vals[item.label] = IndexObjects.getval(obj)

    return obj
end


function node(item::Delayed.BinaryOperator, vals::Trials.ValsDict)::Real

    left = node(item.left, vals)
    right = node(item.right, vals)

    return item.operator(left, right)
end
function node(
    item::Delayed.BinaryOperator, vals::Trials.ValsDict, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Float64

    left = node(item.left, vals, nid, trials, config)
    right = node(item.right, vals, nid, trials, config)

    return item.operator(left, right)
end


function node(item::Delayed.UnaryOperator, vals::Trials.ValsDict)::Real

    operand = node(item.operand, vals)

    return item.operator(operand)
end
function node(
    item::Delayed.UnaryOperator, vals::Trials.ValsDict, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Float64

    operand = node(item.operand, vals, nid, trials, config)

    return item.operator(operand)
end


function node(item::Delayed.CategoricalIndex, vals::Trials.ValsDict)::IndexObjects.IndexInt
    return IndexObjects.IndexInt(first(Delayed.categoricalindex(item.probabilities, 1).v))
end
function node(
    item::Delayed.CategoricalIndex, vals::Trials.ValsDict,
    nid::Symbol, trials::Vector{Trials.Trial},
    config::Config
)::IndexObjects.IndexInt

    return Resolve.posterior(item, item.probabilities, nid, trials, config)
end


function node(item::Dict{Symbol, T} where T, vals::Trials.ValsDict)::Dict{Symbol, Any}

    result = Dict{Symbol, Any}()
    # Cannot use a dictionary comprehension because the values will be evaluated twice.
    for (k, v) in item
        result[k] = node(v, vals)
    end

    return result
end
function node(
    item::Dict{Symbol, T} where T, vals::Trials.ValsDict,
    nid::Symbol, trials::Vector{Trials.Trial}, 
    config::Config
)::Dict{Symbol, Any}

    result = Dict{Symbol, Any}()
    # Cannot use a dictionary comprehension because the values will be evaluated twice.
    for (k, v) in item
        result[k] = node(v, vals, k, trials, config)
    end

    return result
end


function node(item::Delayed.Float, vals::Trials.ValsDict)::Float64

    return float(node(item.arg, vals))
end
function node(
    item::Delayed.Float, vals::Trials.ValsDict, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Float64

    return float(node(item.arg, vals, nid, trials, config))
end


function node(item::Delayed.LogNormal, vals::Trials.ValsDict)::Float64
    mu = node(item.mu, vals)
    sigma = node(item.sigma, vals)

    return Delayed.lognormal(mu, sigma)
end
function node(
    item::Delayed.LogNormal, vals::Trials.ValsDict,
    nid::Symbol, trials::Vector{Trials.Trial},
    config::Config
)::Real

    mu = node(item.mu, vals, nid, trials, config)
    sigma = node(item.sigma, vals, nid, trials, config)

    return Resolve.posterior(item, mu, sigma, nid, trials, config)
end


function node(item::Delayed.LogNormalQuantDist, vals::Trials.ValsDict)::Float64
    mu = node(item.mu, vals)
    sigma = node(item.sigma, vals)
    q = node(item.q, vals)

    return Delayed.lognormalquant(item, mu, sigma, q)
end
function node(
    item::Delayed.LogNormalQuantDist, vals::Trials.ValsDict,
    nid::Symbol, trials::Vector{Trials.Trial},
    config::Config
)::Real

    mu = node(item.mu, vals, nid, trials, config)
    sigma = node(item.sigma, vals, nid, trials, config)
    q = node(item.q, vals, nid, trials, config)

    return Resolve.posterior(item, mu, sigma, q, nid, trials, config)
end


function node(item::Delayed.Normal, vals::Trials.ValsDict)::Float64
    mu = node(item.mu, vals)
    sigma = node(item.sigma, vals)

    return Delayed.normal(mu, sigma)
end
function node(
    item::Delayed.Normal, vals::Trials.ValsDict, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Real

    mu = node(item.mu, vals, nid, trials, config)
    sigma = node(item.sigma, vals, nid, trials, config)

    return Resolve.posterior(item, mu, sigma, nid, trials, config)
end

function node(item::Delayed.QuantNormal, vals::Trials.ValsDict)::Float64
    mu = node(item.mu, vals)
    sigma = node(item.sigma, vals)
    q = node(item.q, vals)

    return Delayed.quantnormal(mu, sigma, q)
end
function node(
    item::Delayed.QuantNormal, vals::Trials.ValsDict, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Real

    mu = node(item.mu, vals, nid, trials, config)
    sigma = node(item.sigma, vals, nid, trials, config)
    q = node(item.q, vals, nid, trials, config)

    return Resolve.posterior(item, mu, sigma, q, nid, trials, config)
end


function node(item::Delayed.RandIndex, vals::Trials.ValsDict)::IndexObjects.IndexInt

    upper = node(item.upper, vals)
    if upper < 1
        throw(ArgumentError("upper will be used as index so must be greater than 0"))
    end

    return IndexObjects.IndexInt(rand(1:upper))
end
function node(
    item::Delayed.RandIndex, vals::Trials.ValsDict, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::IndexObjects.IndexInt

    upper = node(item.upper, vals, nid, trials, config)
    if upper < 1
        throw(ArgumentError("upper will be used as index so must be greater than 0"))
    end

    return Resolve.posterior(item, upper, nid, trials, config)
end


function node(item::Delayed.AbstractSwitch, vals::Trials.ValsDict)

    # Randomly generate a number to indicate which index of the options will be returned
    choice = node(item.choice, vals)

    return node(item.options[choice.v], vals)
end
function node(
    item::Delayed.AbstractSwitch, vals::Trials.ValsDict, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)
    choice = node(item.choice, vals, nid, trials, config)

    return node(item.options[choice.v], vals, nid, trials, config)
end


function node(item::Delayed.Uniform, vals::Trials.ValsDict)::Float64
    low = node(item.low, vals)
    high = node(item.high, vals)

    return Delayed.uniform(low, high)
end
function node(
    item::Delayed.Uniform, vals::Trials.ValsDict, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Real

    low = node(item.low, vals, nid, trials, config)
    high = node(item.high, vals, nid, trials, config)

    return Resolve.posterior(item, low, high, nid, trials, config)
end

function node(item::Delayed.QuantUniform, vals::Trials.ValsDict)::Float64
    low = node(item.low, vals)
    high = node(item.high, vals)
    q = node(item.q, vals)

    return Delayed.quantuniform(low, high, q)
end
function node(
    item::Delayed.QuantUniform, vals::Trials.ValsDict, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Real

    low = node(item.low, vals, nid, trials, config)
    high = node(item.high, vals, nid, trials, config)
    q = node(item.q, vals, nid, trials, config)

    return Resolve.posterior(item, low, high, q, nid, trials, config)
end

function node(item::Delayed.LogUniform, vals::Trials.ValsDict)::Float64

    low = node(item.low, vals)
    high = node(item.high, vals)

    return Delayed.loguniform(low, high)
end

function node(
    item::Delayed.LogUniform, vals::Trials.ValsDict,
    nid::Symbol, trials::Vector{Trials.Trial},
    config::Config
)::Real

    low = node(item.low, vals, nid, trials, config)
    high = node(item.high, vals, nid, trials, config)

    return Resolve.posterior(item, low, high, nid, trials, config)
end

function node(item::Delayed.LogUniformQuantDist, vals::Trials.ValsDict)::Float64
    low = node(item.low, vals)
    high = node(item.high, vals)
    q = node(item.q, vals)

    return Delayed.loguniformquant(item, low, high, q)
end
function node(
    item::Delayed.LogUniformQuantDist, vals::Trials.ValsDict,
    nid::Symbol, trials::Vector{Trials.Trial},
    config::Config
)::Real

    low = node(item.low, vals, nid, trials, config)
    high = node(item.high, vals, nid, trials, config)
    q = node(item.q, vals, nid, trials, config)

    return Resolve.posterior(item, low, high, q, nid, trials, config)
end

function node(items::Vector, vals::Trials.ValsDict)::Vector{<: Any}
    return [
        node(item, vals::Trials.ValsDict)
            for item in items
    ]
end
function node(
    items::Vector, vals::Trials.ValsDict, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Vector{<: Any}

    return [
        node(item, vals, nid, trials, config)
        for item in items
    ]
end


function node(items::Tuple, vals::Trials.ValsDict)::Tuple
    return tuple((
        node(item, vals::Trials.ValsDict)
            for item in items
    )...)
end
function node(
    items::Tuple, vals::Trials.ValsDict, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::Tuple

    return tuple((
        node(item, vals, nid, trials, config)
            for item in items
    )...)
end


function node(item::T, vals::Trials.ValsDict)::T where T <: Union{Real, Symbol, String}
    return item
end
function node(
    item::T, vals::Trials.ValsDict, nid::Symbol,
    trials::Vector{Trials.Trial}, config::Config
)::T where T <: Union{Real, Symbol, String}

    return item
end
