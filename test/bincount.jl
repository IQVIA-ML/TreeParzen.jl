module TestBincount

using Test
import TreeParzen: Bincounts
import TreeParzen.ConstrainedVectors: ObsWeights

@test_throws DimensionMismatch Bincounts.bincount([1], ObsWeights([1.0, 1.0]), 3)

@test Bincounts.bincount([1], 3) == [1.0, 0.0, 0.0]
@test_throws MethodError Bincounts.bincount([1], [1.0], 3)

end
true
