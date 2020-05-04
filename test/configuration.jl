module TestConfiguration

using Test
using TreeParzen

# threshold
@test_throws ArgumentError Config(-1.0, 25, 24, 20, 1.0)
@test_throws ArgumentError Config(2.0, 25, 24, 20, 1.0)

# linear_forgetting
@test_throws ArgumentError Config(0.25, -1, 24, 20, 1.0)

# draws
@test_throws ArgumentError Config(0.25, 25, -1, 20, 1.0)

# random_trials
@test_throws ArgumentError Config(0.25, 25, 24, -1, 1.0)

# prior_weight
@test_throws ArgumentError Config(0.25, 25, 24, -1, -1.0)
@test_throws ArgumentError Config(0.25, 25, 24, -1, 2.0)

end
true
