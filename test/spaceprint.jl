module TestSpacePrint

using Test
using TreeParzen

# Print out a space for visual inspection

space = Dict(
    :args => HP.PChoice(:probchoice, [
        Prob(
            0.1,
            Dict(
                :symbol => :test,
                :string => "test",
                :int => 1,
                :float => 1.0,
                :nothing => nothing,
            )
        ),
        Prob(
            0.2,
            Dict(
                :dists => HP.Choice(:dists, [
                    HP.Uniform(:uniform, -10.0, 10.0) + HP.Normal(:normal, -10.0, 10.0),
                    float(HP.LogNormal(:lognormal, -10.0, 10.0)),
                ]),
            )
        ),
        Prob(
            0.7,
            Dict(
                :dictmiddle => Dict(
                    :tuple => (0.1, 0.3, 0.6),
                    :set => Set([4, 5, 6]),
                    :vector => [7, 8, 9],
                ),
                :dictend => Dict(:a => 1, :b => 4),
            )
        ),
    ]),
)
spaceprint(space)

end # module TestSpacePrint
true
