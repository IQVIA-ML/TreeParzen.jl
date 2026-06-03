"""
$(TYPEDEF)
$(TYPEDFIELDS)

A delayed random integer node that samples values in `0:(upper - 1)`.
"""
struct RandInt <: AbstractDistDelayed
    upper::NestedInt

    RandInt(upper::Types.AbstractDelayed) = new(upper)
    function RandInt(upper::Int)
        if upper < 1
            throw(ArgumentError("upper must be greater than 0"))
        end

        return new(upper)
    end
end

function randint(upper::Vector{Int}, sample_size::Int)::Vector{Int}
    if length(upper) != sample_size
        throw(ArgumentError("randint: length(upper) must equal sample_size"))
    end
    iszero(sample_size) && return Int[]

    return Int[randint(uu) for uu in upper]
end
function randint(upper::Vector{Int}, sample_size::Vector{Int})::Vector{Int}
    if length(sample_size) != 1
        throw(ArgumentError("randint: length(sample_size) must be 1"))
    end

    return randint(upper, first(sample_size))
end
function randint(upper::Int, sample_size::Int)::Vector{Int}
    if upper < 1
        throw(ArgumentError("upper must be greater than 0"))
    end
    iszero(sample_size) && return Int[]
    sample_size == 1 && return Int[randint(upper)]

    return randint(repeat([upper], sample_size), sample_size)
end
function randint(upper::Int)::Int
    if upper < 1
        throw(ArgumentError("upper must be greater than 0"))
    end

    return rand(0:(upper - 1))
end
