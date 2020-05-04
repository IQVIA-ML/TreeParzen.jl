module TestGraph

using Test
import TreeParzen: Delayed, Graph, HP

# Delayed objects can be nested
a = HP.Normal(:a, 10.0, 2.0)
b = HP.Uniform(:b, a, 7.0)
ab = Dict(:x => [a, b])
ba = Dict(:x => [b, a])

@test typeof.(Graph.dfs(ab)) == [
    Delayed.Normal,
    HP.Normal,
    Delayed.Normal,
    HP.Normal,
    Delayed.Uniform,
    HP.Uniform,
]
@test typeof.(Graph.dfs(ba)) == [
    Delayed.Normal,
    HP.Normal,
    Delayed.Uniform,
    HP.Uniform,
    Delayed.Normal,
    HP.Normal,
]

# Using the same label more than once is not allowed
space = Dict(
    :x => HP.Normal(:a, 1.0, 2.0),
    :y => HP.Normal(:a, 1.0, 2.0),
)
@test_throws DomainError Graph.checkspace(space)

end
true
