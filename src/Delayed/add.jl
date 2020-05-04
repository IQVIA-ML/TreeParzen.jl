"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct Add <: AbstractDelayed
    left::NestedReal
    right::NestedReal
end

Base.:+(left::AbstractDelayed, right::AbstractDelayed) = Add(left, right)
Base.:+(left::AbstractDelayed, right::Real) = Add(left, right)
Base.:+(left::Real, right::AbstractDelayed) = Add(left, right)
