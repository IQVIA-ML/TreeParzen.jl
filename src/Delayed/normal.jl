"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct Normal <: AbstractDistDelayed
    mu::NestedFloat
    sigma::NestedFloat
end

function normal(mu::Float64, sigma::Float64)::Float64
    return first(rand(Distributions.Normal(mu, sigma), 1))
end
