module RandomSeed

using Test
using TreeParzen
using Random

SEED = 12355

objective(params) = params[:x]^2
space = Dict(:x => HP.Uniform(:x, -10.0, 10.0))
nsamples = 10

# Figure out what the best parameter is with this seed.
# 
# Note: this is not hard-coded on purpose since setting a random seed only guarantes you get the random state if
# and only if all other extraneous variables (i.e. bug fixes and speed improvements may change results) did not change.
Random.seed!(SEED)
best = fmin(objective, space, nsamples)
best_x = best[:x]

# Ensure equivalent calls to fmin, returns the same value.
for i in 1:10
    Random.seed!(SEED)
    best = fmin(objective, space, nsamples)
    @test isequal(best[:x], best_x)
end

end
