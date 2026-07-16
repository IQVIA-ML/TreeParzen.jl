"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct Normal{M, S} <: AbstractDistDelayed
    mu::M
    sigma::S
end

function normal(mu::Float64, sigma::Float64)::Float64
    return only(rand(Distributions.Normal(mu, sigma), 1))
end
