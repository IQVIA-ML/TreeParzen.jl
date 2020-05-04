"""
Based on
https://github.com/bjkomer/hyperopt-tutorial/blob/master/Function-Fitting-Example.ipynb
"""
module TestFunctionFitting

using Compat
import Distributions
using Test

using TreeParzen

# Define possible functions to fit

function sinusoid(domain, amplitude, frequency, offset, phase)
    return amplitude .* sin.(frequency .* domain .+ phase) .+ offset
end

function polynomial(domain, coefficients)
    result = zeros(length(domain))
    for (p, c) in enumerate(coefficients)
        result .+= (c * domain .^ p)
    end
    return result
end

# Define the objective function

function objective_ff(params)
    args = params[:args]

    # Generate set of data points from a sinusoid
    num_points = 20
    x = @compat range(0, 10; length = num_points)
    data = 1.5sin.(x .+ 1) .+ 2

    # Add noise to the data
    data += rand(Distributions.Normal(0, 0.2), num_points)
    estimate = if args[:type] == :sinusoid
        sinusoid(x, args[:amplitude], args[:frequency], args[:offset], args[:phase])
    elseif args[:type] == :polynomial
        polynomial(x, args[:coefficients])
    end
    # sum of squared error
    return (data - estimate) .^ 2 |> sum
end

# Define the search space with sinusoid and polynomial
space = Dict(
    :args => HP.Choice(:function, [
        Dict(
            :type => :sinusoid,
            :amplitude => HP.Uniform(:amplitude, 0.1, 2.0),
            :frequency => HP.Uniform(:frequency, 0.1, 2.0),
            :offset => HP.Normal(:offset, 0.0, 2.0),
            :phase => HP.Normal(:phase, 0.0, 2.0),
        ),
        Dict(
            :type => :polynomial,
            :coefficients => HP.Choice(:degree, [
                [HP.Normal(:d0_c0, 0.0, 5.0)],
                [HP.Normal(:d1_c0, 0.0, 5.0), HP.Normal(:d1_c1, 0.0, 5.0)],
                [
                    HP.Normal(:d2_c0, 0.0, 5.0), HP.Normal(:d2_c1, 0.0, 5.0),
                    HP.Normal(:d2_c2, 0.0, 5.0),
                ],
                [
                    HP.Normal(:d3_c0, 0.0, 5.0), HP.Normal(:d3_c1, 0.0, 5.0),
                    HP.Normal(:d3_c2, 0.0, 5.0), HP.Normal(:d3_c3, 0.0, 5.0),
                ],
            ]),
        ),
    ]),
)

# Run the search for the specified number of evaluations
best = fmin(objective_ff, space, 1_000)

@test isapprox(best[:args][:amplitude], 1.4762631349533661; rtol=1e1)
@test isapprox(best[:args][:frequency], 0.98526529964042; rtol=1e1)
@test isapprox(best[:args][:offset], 2.010532261653186; rtol=1e1)
@test isapprox(best[:args][:phase], 0.8694539754903817; rtol=1e1)

# Define the search space with polynomial alone
space = Dict(
    :args => HP.Choice(:function, [
        Dict(
            :type => :polynomial,
            :coefficients => HP.Choice(:degree, [
                [HP.Normal(:d0_c0, 0.0, 5.0)],
                [HP.Normal(:d1_c0, 0.0, 5.0), HP.Normal(:d1_c1, 0.0, 5.0)],
                [
                    HP.Normal(:d2_c0, 0.0, 5.0), HP.Normal(:d2_c1, 0.0, 5.0),
                    HP.Normal(:d2_c2, 0.0, 5.0),
                ],
                [
                    HP.Normal(:d3_c0, 0.0, 5.0), HP.Normal(:d3_c1, 0.0, 5.0),
                    HP.Normal(:d3_c2, 0.0, 5.0), HP.Normal(:d3_c3, 0.0, 5.0),
                ],
            ]),
        ),
    ]),
)

best = fmin(objective_ff, space, 1_000)
@test best[:args][:type] == :polynomial
@test isa(best[:args][:coefficients], Vector{Float64})

space = Dict(
    :args => HP.PChoice(:function, [
        HP.Prob(0.7, Dict(
            :type => :sinusoid,
            :amplitude => HP.Uniform(:amplitude, 0.1, 2.0),
            :frequency => HP.Uniform(:frequency, 0.1, 2.0),
            :offset => HP.Normal(:offset, 0.0, 2.0),
            :phase => HP.Normal(:phase, 0.0, 2.0),
        )),
        HP.Prob(0.3, Dict(
            :type => :polynomial,
            :coefficients => HP.Choice(:degree, [
                [HP.Normal(:d0_c0, 0.0, 5.0)],
                [HP.Normal(:d1_c0, 0.0, 5.0), HP.Normal(:d1_c1, 0.0, 5.0)],
                [
                    HP.Normal(:d2_c0, 0.0, 5.0), HP.Normal(:d2_c1, 0.0, 5.0),
                    HP.Normal(:d2_c2, 0.0, 5.0),
                ],
                [
                    HP.Normal(:d3_c0, 0.0, 5.0), HP.Normal(:d3_c1, 0.0, 5.0),
                    HP.Normal(:d3_c2, 0.0, 5.0), HP.Normal(:d3_c3, 0.0, 5.0),
                ],
            ]),
        )),
    ]),
)

# Test that evaluation completes
fmin(objective_ff, space, 1_000)

end # module TestFunctionFitting
true
