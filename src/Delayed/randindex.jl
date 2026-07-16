"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct RandIndex{U} <: AbstractDistDelayed
    upper::U
end

function RandIndex(upper::Types.AbstractDelayed)
    return RandIndex{typeof(upper)}(upper)
end
function RandIndex(upper::Int)
    if upper < 1
        throw(ArgumentError("upper will be used as index so must be greater than 0"))
    end

    return RandIndex{Int}(upper)
end
