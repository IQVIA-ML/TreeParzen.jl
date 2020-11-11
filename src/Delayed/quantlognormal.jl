"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct QuantLogNormal <: LogNormalQuantDist
    mu::NestedFloat
    sigma::NestedFloat
    q::NestedFloat
end

function quantlognormal(mu::Float64, sigma::Float64, q::Float64)::Float64
    return round(exp(normal(mu, sigma)) / q) * q
end

lognormalquant(x::QuantLogNormal, mu, sigma, q) = quantlognormal(mu, sigma, q)
