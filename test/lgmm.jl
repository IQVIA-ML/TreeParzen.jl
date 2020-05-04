module TestLGMM

using Test
import TreeParzen: LogGMM

# LGMM1_lpdf dimenstion mismatch
@test_throws DimensionMismatch LogGMM.LGMM1_lpdf(
    [[1.0 2.0] [3.0 4.0]], [1.0, 0.0], [0.0, 1.0, 2.0], [10.0]
)

end
true
