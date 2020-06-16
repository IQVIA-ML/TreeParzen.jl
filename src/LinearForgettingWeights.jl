module LinearForgettingWeights

using Compat

function linear_forgetting_weights(N::Int, lf::Int)::Vector{Float64}
    if N < 0 throw(KeyError("linear_forgetting_weights: $(N) below 0")) end

    if N < lf
        return ones(N)
    end

    ramp = if N - lf == 1
        1.0 / N
    else
        @compat range(1.0 / N, stop = 1.0, length = N - lf)
    end
    return vcat(ramp, ones(lf))

end
end # module LinearForgettingWeights
