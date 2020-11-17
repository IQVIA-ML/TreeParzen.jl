module Types

abstract type AbstractDelayed end

const SPACE_TYPE = Union{
    Dict{Symbol, T} where T,
    AbstractDelayed,
    AbstractVector{<: AbstractDelayed},
    AbstractVector{Dict{Symbol, T} where T},
}

end # module
