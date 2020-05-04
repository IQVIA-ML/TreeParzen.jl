module TestHP

using Distributions
using Test
using TreeParzen
import TreeParzen: Delayed

# Test that weights sum to 1

@test_throws ArgumentError HP.PChoice(:a, [
    Prob(0.1, 0),
    Prob(0.2, 1),
])

# Test that the space is constructed correctly

space = HP.PChoice(:a, [
    Prob(0.1, 0),
    Prob(0.2, 1),
    Prob(0.7, 2),
])
@test isa(Multinomial(1, space.choice.obj.probabilities), Multinomial)
@test isa(space, Delayed.AbstractSwitch)
@test length(space.options) == 3
@test isa(space.choice, Delayed.AbstractParam)
@test space.choice.label == :a
@test isa(space.choice.obj, Delayed.CategoricalIndex)
@test space.options == [0, 1, 2]

space = HP.PChoice(:b, [Prob(0.1, false), Prob(0.9, true)])
@test isa(space, Delayed.AbstractSwitch)
@test length(space.options) == 2
@test isa(space.choice, Delayed.AbstractParam)
@test space.choice.label == :b
@test isa(space.choice.obj, Delayed.CategoricalIndex)
@test space.options == [false, true]

@test_throws ArgumentError HP.PChoice(:c, Prob[])
@test_throws ArgumentError HP.Choice(:d, [])

end
true
