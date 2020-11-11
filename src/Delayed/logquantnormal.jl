"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct LogQuantNormal <: LogNormalQuantDist
    mu::NestedFloat
    sigma::NestedFloat
    q::NestedFloat
end

function logquantnormal(mu::Float64, sigma::Float64, q::Float64)::Float64
    return exp(quantnormal(mu, sigma, q))
end

lognormalquant(x::LogQuantNormal, mu, sigma, q) = logquantnormal(mu, sigma, q)
