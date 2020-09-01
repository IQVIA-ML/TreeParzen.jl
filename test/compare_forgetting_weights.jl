using Test
using TreeParzen

function objective_min(space)
    x = space[:x]
    y = space[:y]
    z = space[:z]
    a = space[:a]
    b = space[:b]
    c = space[:c]

    return ((x^2 + y^2 + z^15 * 75a) * b^12) + exp(c)
end

space = Dict(
    :x => HP.Uniform(:x, -5.0, 5.0),
    :y => HP.Uniform(:y, -5.0, 5.0),
    :z => HP.Uniform(:z, -5.0, 5.0),
    :a => HP.LogUniform(:a, -5.0, 5.0),
    :b => HP.LogNormal(:b, -5.0, 5.0),
    :c => HP.QuantUniform(:c, -5.0, 5.0, 7.4),
)

answers = [
    fmin(objective_min, space, 100) |> objective_min
    for _ in 1:100
]

println(answers)
