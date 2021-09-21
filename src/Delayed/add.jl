"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct Operator <: Types.AbstractDelayed
    left::NestedReal
    right::NestedReal
    operator::Function
end

Base.:+(left::Types.AbstractDelayed, right::Types.AbstractDelayed) = Operator(left, right, +)
Base.:+(left::Types.AbstractDelayed, right::Real) = Operator(left, right, +)
Base.:+(left::Real, right::Types.AbstractDelayed) = Operator(left, right, +)

Base.:-(left::Types.AbstractDelayed, right::Types.AbstractDelayed) = Operator(left, right, -)
Base.:-(left::Types.AbstractDelayed, right::Real) = Operator(left, right, -)
Base.:-(left::Real, right::Types.AbstractDelayed) = Operator(left, right, -)

Base.:*(left::Types.AbstractDelayed, right::Types.AbstractDelayed) = Operator(left, right, *)
Base.:*(left::Types.AbstractDelayed, right::Real) = Operator(left, right, *)
Base.:*(left::Real, right::Types.AbstractDelayed) = Operator(left, right, *)

Base.:/(left::Types.AbstractDelayed, right::Types.AbstractDelayed) = Operator(left, right, /)
Base.:/(left::Types.AbstractDelayed, right::Real) = Operator(left, right, /)
Base.:/(left::Real, right::Types.AbstractDelayed) = Operator(left, right, /)

Base.:^(left::Types.AbstractDelayed, right::Types.AbstractDelayed) = Operator(left, right, ^)
Base.:^(left::Types.AbstractDelayed, right::Real) = Operator(left, right, ^)
Base.:^(left::Real, right::Types.AbstractDelayed) = Operator(left, right, ^)
