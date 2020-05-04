module TestBincount

using Test
import TreeParzen: Bincounts

@test_throws DimensionMismatch Bincounts.bincount([1], [1.0, 2.0], 3)

@test Bincounts.bincount([1], 3) == [1.0, 0.0, 0.0]

end
true
