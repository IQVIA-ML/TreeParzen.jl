module ForgettingWeights

using Compat

function forgetting_weights(N::Int, lf::Int)::Vector{Float64}
    if N < 0 throw(ArgumentError("forgetting_weights: $(N) below 0")) end

    if N < lf
        return ones(N)
    end

    ramp = @compat range(0., stop = 1., length = N - lf + 2)[2:end-1]

    output = vcat(ramp, ones(lf))

    if length(output) != N
        @error "output is not the requested length" N output
        throw(ErrorException("output is not the requested length"))
    end

    return output
end

end # module ForgettingWeights
