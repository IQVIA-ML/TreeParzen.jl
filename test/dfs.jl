"""
Check that dfs can find AbstractDelayed objects nested inside non-AbstractDelayed objects
"""
module TestDFS

using Test
using TreeParzen
import TreeParzen: Delayed, Graph

a = Dict(
    :c => 11,
    :d => HP.Normal(:test, 1.0, 2.0),
)
order = Graph.dfs(a)
@test all(
    isa.(
        order,
        [
            Delayed.Normal,
            Delayed.AbstractParam,
        ]
    )
)
end
true
