"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct LogQuantNormal{M, S, Q} <: LogNormalQuantDist
    mu::M
    sigma::S
    q::Q
end

function logquantnormal(mu::Float64, sigma::Float64, q::Float64)::Float64
    return exp(quantnormal(mu, sigma, q))
end

lognormalquant(x::LogQuantNormal, mu, sigma, q) = logquantnormal(mu, sigma, q)
