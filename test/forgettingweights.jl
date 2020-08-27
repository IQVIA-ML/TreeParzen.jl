module TestForgettingWeights

using Compat
import Distributions
using Test

using TreeParzen

# When run with a single trial, ApFilterTrials.ap_filter_trials produces an obs_above and
# an obs_below, one of which will contain a single value and the other of which will contain
# nothing. However, due to the condition of `config.linear_forgetting < length(mus)` in
# Samplers.adaptive_parzen_normal, N can never be 0 even if config.linear_forgetting is 0.

# It also cannot be the case in HP.PChoice because although forgetting_weights is called in
# Samplers.categoricalindex, the user cannot submit 0 choices due to the PChoice's inner
# constructor.

@testset "Test a variety of N and lf" begin

    # N is a number of observations/mus, lf is based on config and indicates a positive linear
    # forgetting value beyond which older points are de-weighted, e.g. if lf == 25, only
    # the last 25 values will be weighted full. The rest will be linearly de-weighted.
    history_len = 50
    for N in 0:history_len, lf in 0:history_len
        case = TreeParzen.ForgettingWeights.forgetting_weights(N, lf)
        @test length(case) == N
        if !isempty(case)
            # There is no point putting in a history item that is to be completely forgotten
            @test minimum(case) > 0
            @test maximum(case) <= 1
        end

        # Test the fully-remembered part
        ones = [x for x in case if x == 1]
        # If don't you want to forget anything
        if N - lf <= 0
            @test length(ones) == N
        else
            @test length(ones) == lf
        end

        if !isempty(ones)
            @test unique(ones) == [1]
        end

        # Test the ramp
        ramp = [x for x in case if x != 1]
        # If don't you want to forget anything
        if N - lf <= 0
            @test isempty(ramp)
        else
            @test length(ramp) == N - lf
        end

        if !isempty(ramp)
            full_ramp = vcat(0, ramp, 1)
            # Check that the ramp is linear
            ramp_diff = unique(round.(diff(full_ramp), digits = 3))
            @test length(ramp_diff) == 1
        end
    end
end

end # module TestForgettingWeights
true
