module Types

abstract type AbstractDelayed end

const SPACE_ELEMENT = {

const SPACE_TYPE = Union{
    Dict{Symbol, T} where T,
    AbstractDelayed,
    AbstractVector,
}

end # module
