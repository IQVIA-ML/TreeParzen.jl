"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct RandIndex{U} <: AbstractDistDelayed
    upper::U

    RandIndex(upper::Types.AbstractDelayed) = new{typeof(upper)}(upper)
    function RandIndex(upper::Int)
        if upper < 1
            throw(ArgumentError("upper will be used as index so must be greater than 0"))
        end

        return new{Int}(upper)
    end
end
