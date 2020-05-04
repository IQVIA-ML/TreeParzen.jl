"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct LogNormal <: AbstractDistDelayed
    mu::NestedFloat
    sigma::NestedFloat
end

lognormal(mu::Float64, sigma::Float64)::Float64 = exp(normal(mu, sigma))
