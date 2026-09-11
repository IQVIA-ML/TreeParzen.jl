module ForgettingWeights

import ..ConstrainedVectors: ObsWeights

"""
    forgetting_weights(N, lf) -> ObsWeights

Linear forgetting weights in `(0, 1]`: the most recent `lf` observations get weight
`1.0`; older ones ramp linearly toward 0. Empty when `N == 0`.
"""
function forgetting_weights(N::Int, lf::Int)::ObsWeights
    if N < 0 throw(ArgumentError("forgetting_weights: $(N) below 0")) end

    if N < lf
        return ObsWeights(ones(N))
    end

    ramp = range(0., stop = 1., length = N - lf + 2)[2:end-1]

    output = vcat(ramp, ones(lf))

    if length(output) != N
        @error "output is not the requested length" N output
        throw(ErrorException("output is not the requested length"))
    end

    return ObsWeights(output)
end

end # module ForgettingWeights
