module HP

using DocStringExtensions

import ..Delayed

export Prob

"""
$(TYPEDEF)
$(TYPEDFIELDS)

State the weighted probability of an option, for use with PChoice.
"""
struct Prob
    probability::Float64
    option
end


struct PChoice <: Delayed.AbstractSwitch
    choice::Delayed.Param
    options::Vector

    @doc """
    $(TYPEDSIGNATURES)

    Choose from a list of options with weighted probabilities

    # Arguments
    - `label`               : Label
    - `probability_options` : Array of Prob objects

    Example:

    ```julia
    HP.PChoice(:a, [
        Prob(0.1, 0),
        Prob(0.2, 1),
        Prob(0.7, 2),
    ])
    ```

    Note that the Prob probability weights must sum to 1.
    """
    function PChoice(label::Symbol, probability_options::Vector{Prob})
        if isempty(probability_options)
            throw(ArgumentError("$(label): you must give at least one choice"))
        end
        probabilities = [o.probability for o in probability_options]
        if sum(probabilities) != 1
            throw(ArgumentError("Prob probability weights must sum to 1"))
        end
        options = [o.option for o in probability_options]

        return new(Delayed.Param(label, Delayed.CategoricalIndex(probabilities)), options)
    end
end


struct Choice <: Delayed.AbstractSwitch
    choice::Delayed.Param
    options::Vector


    @doc """
    $(TYPEDSIGNATURES)

    Randomly choose which option will be extracted, and also supply the list of options.
    The elements of options can themselves be [nested] stochastic expressions. In this case, the
    stochastic choices that only appear in some of the options become conditional parameters.

    Example:

    ```julia
    Dict(
        :example => HP.Choice(:example, [1.0, 0.9, 0.8, 0.7]),
    )
    ```

    Example of conditional paramaters:

    ```julia
    Dict(
        :example => HP.Choice(:example, [
            (:case1, HP.Uniform(:param1, 0.0, 1.0)),
            (:case2, HP.Uniform(:param2, -10.0, 10.0)),
        ]),
    )
    ```

    ':param1' and ':param2' are examples of conditional parameters. Each of ':param1' and
    ':param2' only features in the returned sample for a particular value of ':example'. If
    ':example' is 0, then ':param1' is used but not ':param2'. If ':example' is 1, then
    ':param2' is used but not ':param1'.

    Example with nested arrays of different lengths:

    ```julia
    Dict(
        :example => HP.Choice(:example, [
            [HP.Normal(:d0_c0, 0.0, 5.0)],
            [HP.Normal(:d1_c0, 0.0, 5.0), HP.Normal(:d1_c1, 0.0, 5.0)],
            [
                HP.Normal(:d2_c0, 0.0, 5.0), HP.Normal(:d2_c1, 0.0, 5.0),
                HP.Normal(:d2_c2, 0.0, 5.0),
            ],
        ]),
    )
    ```

    Note that all labels (the symbol given as the first parameter to all the HP.* functions)
    must be unique. These labels identify the parts of the space that the optimiser learns from
    over iterations.
    """
    function Choice(label::Symbol, options::Vector)
        if isempty(options)
            throw(ArgumentError("$(label): you must give at least one choice"))
        end

        return new(Delayed.Param(label, Delayed.RandIndex(length(options))), options)
    end
end


struct Uniform <: Delayed.AbstractParam
    label::Symbol
    obj::Delayed.Uniform

    @doc """
    $(TYPEDSIGNATURES)

    Returns a value with uniform probability from between `low` and `high`.
    When optimising, this variable is constrained to a two-sided interval.

    ```julia
    Dict(
        :example => HP.Uniform(:example, 0.0, 1.0),
    )
    ```

    where `label` is the parameter and the returned value is uniformly distributed between
    `low` at 0.0 and `high` at 1.0
    """
    function Uniform(label::Symbol, low::Delayed.NestedFloat, high::Delayed.NestedFloat)
        return new(label, Delayed.Uniform(low, high))
    end
end


struct QuantUniform <: Delayed.AbstractParam
    label::Symbol
    obj::Delayed.QuantUniform

    @doc """
    $(TYPEDSIGNATURES)

    Returns a value uniformly between low and high, with a quantisation.
    When optimising, this variable is constrained to a two-sided interval.

    ```julia
    Dict(
        :example => HP.QuantUniform(:example, 0.0, 10.0, 2.0),
    )
    ```

    where `label` is the parameter and the returned value is uniformly distributed between
    `low` at 0.0 and `high` at 10.0, with the `q`uantisation set at 2.0.
    Valid sampled values would be 0.0, 2.0, 4.0, 6.0, 8.0 and 10.0.
    """
    function QuantUniform(
        label::Symbol, low::Delayed.NestedFloat, high::Delayed.NestedFloat,
        q::Delayed.NestedFloat
    )
        return new(label, Delayed.QuantUniform(low, high, q))
    end
end


struct Normal <: Delayed.AbstractParam
    label::Symbol
    obj::Delayed.Normal

    @doc """
    $(TYPEDSIGNATURES)

    Returns a real value that's normally-distributed with mean mu and standard deviation sigma.
    When optimizing, this is an unconstrained variable.

    ```julia
    Dict(
        :example => HP.Normal(:example, 4.0, 5.0),
    )
    ```
    """
    function Normal(label::Symbol, mu::Delayed.NestedFloat, sigma::Delayed.NestedFloat)
        return new(label, Delayed.Normal(mu, sigma))
    end
end


struct QuantNormal <: Delayed.AbstractParam
    label::Symbol
    obj::Delayed.QuantNormal

    @doc """
    $(TYPEDSIGNATURES)

    Returns a real value that's normally-distributed with mean mu and standard deviation sigma, with a quantisation.
    When optimizing, this is an unconstrained variable.

    ```julia
    Dict(
        :example => HP.Normal(:example, 2., 0.5, 1.0),
    )
    ```

    In this example, the values are sampled normally first, and then `q`uantised in rounds of 1.0, so one
    only observes 1.0, 2.0, 3.0, etc, centered around 2.0.

    N.B. that due to rounding, the observed values will not follow exactly normal distribution, particularly
    if sigma is much smaller than quantisation.
    """
    function QuantNormal(
        label::Symbol, mu::Delayed.NestedFloat, sigma::Delayed.NestedFloat,
        q::Delayed.NestedFloat
    )
        return new(label, Delayed.QuantNormal(mu, sigma, q))
    end
end


struct LogNormal <: Delayed.AbstractParam
    label::Symbol
    obj::Delayed.LogNormal

    @doc """
    $(TYPEDSIGNATURES)

    Returns a value drawn according to exp(normal(mu, sigma)) so that the logarithm of the
    sampled value is normally distributed. When optimising, this variable is constrained to be
    positive.

    ```julia
    Dict(
        :example => HP.LogNormal(:example, log(3.0), 0.5),
    )
    ```

    In this example, the log normal distribution will be centred around 3. The distribution is
    not truncated.
    """
    function LogNormal(label::Symbol, mu::Delayed.NestedFloat, sigma::Delayed.NestedFloat)
        return new(label, Delayed.LogNormal(mu, sigma))
    end
end


struct QuantLogNormal <: Delayed.AbstractParam
    label::Symbol
    obj::Delayed.QuantLogNormal

    @doc """
    $(TYPEDSIGNATURES)

    Returns a value drawn according to exp(normal(mu, sigma)), with a quantisation, so that the
    logarithm of the sampled value is normally distributed. When optimising, this variable is
    constrained to be positive.

    ```julia
    Dict(
        :example => HP.LogNormal(:example, log(3.0), 0.5, 2.0),
    )
    ```

    In this example, the log normal distribution will be centred around 3. The distribution is
    not truncated. The values with be quantised to multiples of 2, i.e. 2.0, 4.0, 6.0, etc.
    """
    function QuantLogNormal(
        label::Symbol, mu::Delayed.NestedFloat, sigma::Delayed.NestedFloat,
        q::Delayed.NestedFloat
    )
        return new(label, Delayed.QuantLogNormal(mu, sigma, q))
    end
end


struct LogUniform <: Delayed.AbstractParam
    label::Symbol
    obj::Delayed.LogUniform

    @doc """
    $(TYPEDSIGNATURES)

    Returns a value drawn according to exp(uniform(low, high)) such that the logarithm
    of the return value is uniformly distributed.
    When optimizing, this variable is constrained to the interval [exp(low), exp(high)].

    ```julia
    Dict(
        :example => HP.LogUniform(:example, log(3.0), 1.0),
    )
    ```
    """
    function LogUniform(label::Symbol, low::Delayed.NestedFloat, high::Delayed.NestedFloat)
        return new(label, Delayed.LogUniform(low, high))
    end
end


struct QuantLogUniform <: Delayed.AbstractParam
    label::Symbol
    obj::Delayed.QuantLogUniform

    @doc """
    $(TYPEDSIGNATURES)

    Returns a value drawn according to exp(uniform(low, high)), with a quantisation, such that
    the logarithm of the return value is uniformly distributed.

    Suitable for a discrete variable with respect to which the objective is "smooth" and gets
    smoother with the size of the value, but which should be bounded both above and below.

    ```julia
    Dict(
        :example => HP.QuantLogUniform(:example, log(3.0), 1.0, 2.0),
    )
    ```
    """
    function QuantLogUniform(
        label::Symbol, low::Delayed.NestedFloat, high::Delayed.NestedFloat,
        q::Delayed.NestedFloat
    )
        return new(label, Delayed.QuantLogUniform(low, high, q))
    end
end

end # module HP
