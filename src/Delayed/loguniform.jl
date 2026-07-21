"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct LogUniform{L, H} <: AbstractDistDelayed
    low::L
    high::H
end

function loguniform(low::Float64, high::Float64)::Float64
    return exp(uniform(low, high))
end
