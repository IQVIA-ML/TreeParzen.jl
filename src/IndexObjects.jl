"""
Wrap around data that will be used as indexes and enforce >= 1
"""
module IndexObjects

struct IndexVector
    v::Vector{Int}
    function IndexVector(v::AbstractVector{<:Real})
        values = convert(Vector{Int}, v)
        if !isempty(values) && minimum(values) < 1
            throw(ArgumentError(string("v will be used as index so must be greater than or equal to 1: ", unique(values))))
        end

        return new(values)
    end
end

struct IndexInt
    v::Int
    function IndexInt(v::Int)
        if v < 1
            throw(ArgumentError(string("v will be used as index so must be greater than 1", unique(v))))
        end

        return new(v)
    end
end

getval(obj::IndexInt) = obj.v
getval(obj::Real) = obj

end # module IndexObjects
