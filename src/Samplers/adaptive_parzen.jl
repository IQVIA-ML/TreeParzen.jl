"""
$(TYPEDSIGNATURES)

A heuristic estimator for the mu and sigma values of a GMM.
"""
function adaptive_parzen_normal(
    mus::Vector{Float64}, prior_mu::Float64, prior_sigma::Float64, config::Config
)::NTuple{3, Vector{Float64}}

    if prior_sigma <= 0
        throw(DimensionMismatch("prior_sigma: $(prior_sigma) is less than or equal to 0"))
    end

    srtd_mus = []
    sigma = []
    prior_pos = 1
    # sortperm must be used here because order is applied to unsorted_weights below
    order = sortperm(mus)

    if isempty(mus)
        srtd_mus = [prior_mu]
        sigma = [prior_sigma]
    elseif length(mus) == 1
        if prior_mu < first(mus)
            srtd_mus = [prior_mu, first(mus)]
            sigma = [prior_sigma, prior_sigma * 0.5]
        else
            prior_pos = 2
            srtd_mus = [first(mus), prior_mu]
            sigma = [prior_sigma * 0.5, prior_sigma]
        end
    elseif length(mus) >= 2
        # Create new_mus, which is sorted, and in which the prior has been inserted
        prior_pos = searchsortedfirst(mus[order], prior_mu)
        srtd_mus = mus[order]
        splice!(srtd_mus, prior_pos:(prior_pos - 1), prior_mu)
        sigma = zero(srtd_mus)
        sigma[2:end-1] = max.(
            srtd_mus[2:end-1] - srtd_mus[1:end-2],
            srtd_mus[3:end] - srtd_mus[2:end-1]
        )
        lsigma = srtd_mus[2] - srtd_mus[1]
        usigma = srtd_mus[end] - srtd_mus[end-1]
        sigma[1] = lsigma
        sigma[end] = usigma
    end

    if config.linear_forgetting < length(mus)
        unsorted_weights = ForgettingWeights.forgetting_weights(
            length(mus), config.linear_forgetting
        )
        if length(unsorted_weights) + 1 != length(srtd_mus)
            throw(DimensionMismatch(string(
                "length(unsorted_weights) + 1: ", length(unsorted_weights) + 1,
                " doesn't equal length(srtd_mus): ", length(srtd_mus)
            )))
        end
        sorted_weights = unsorted_weights[order]
        splice!(sorted_weights, prior_pos:(prior_pos - 1), config.prior_weight)
    else
        sorted_weights = ones(length(srtd_mus))
        sorted_weights[prior_pos] = config.prior_weight
    end

    # Magic formula
    maxsigma = prior_sigma
    minsigma = prior_sigma / min(100.0, (1.0 + length(srtd_mus)))
    clamp!(sigma, minsigma, maxsigma)
    sigma[prior_pos] = prior_sigma

    if all(sigma .<= 0)
        throw(DimensionMismatch(string(
            "minimum(sigma): ", minimum(sigma),
            " minsigma: ", minsigma,
            " maxsigma: ", maxsigma
        )))
    end

    sorted_weights /= sum(sorted_weights)

    return sorted_weights, srtd_mus, sigma
end
