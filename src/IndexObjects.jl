"""
Wrap around data that will be used as indexes and enforce >= 1
"""
module IndexObjects

struct IndexVector
    v::Vector{Int}
    function IndexVector(v::Vector{Int})
        if isempty(v)
            return new(v)
        end
        if minimum(v) < 1
            throw(ArgumentError(string("v will be used as index so must be greater than 1", unique(v))))
        end

        return new(v)
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
