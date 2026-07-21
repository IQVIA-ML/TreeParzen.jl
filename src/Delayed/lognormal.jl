"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct LogNormal{M, S} <: AbstractDistDelayed
    mu::M
    sigma::S
end

lognormal(mu::Float64, sigma::Float64)::Float64 = exp(normal(mu, sigma))
