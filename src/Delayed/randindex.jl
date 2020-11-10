"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct RandIndex <: AbstractDistDelayed
    upper::NestedInt

    RandIndex(upper::Types.AbstractDelayed) = new(upper)
    function RandIndex(upper::Int)
        if upper < 1
            throw(ArgumentError("upper will be used as index so must be greater than 0"))
        end

        return new(upper)
    end
end
