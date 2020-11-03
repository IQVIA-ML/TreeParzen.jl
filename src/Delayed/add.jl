"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct Add <: Types.AbstractDelayed
    left::NestedReal
    right::NestedReal
end

Base.:+(left::Types.AbstractDelayed, right::Types.AbstractDelayed) = Add(left, right)
Base.:+(left::Types.AbstractDelayed, right::Real) = Add(left, right)
Base.:+(left::Real, right::Types.AbstractDelayed) = Add(left, right)
