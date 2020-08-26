module ForgettingWeights

using Compat

function forgetting_weights(N::Int, lf::Int)::Vector{Float64}
    if N <= 0 throw(KeyError("forgetting_weights: $(N) below 0")) end

    if N < lf
        return ones(N)
    end

    ramp = @compat range(0., stop = 1., length = N + 2)[2:end-1]

    return vcat(ramp, ones(lf))

end
end # module ForgettingWeights
