
abstract type Operator <: Types.AbstractDelayed end

"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct BinaryOperator <: Types.AbstractDelayed
    left::NestedReal
    right::NestedReal
    operator::Function
end

Base.:+(left::Types.AbstractDelayed, right::Types.AbstractDelayed) = BinaryOperator(left, right, +)
Base.:+(left::Types.AbstractDelayed, right::Real) = BinaryOperator(left, right, +)
Base.:+(left::Real, right::Types.AbstractDelayed) = BinaryOperator(left, right, +)

Base.:-(left::Types.AbstractDelayed, right::Types.AbstractDelayed) = BinaryOperator(left, right, -)
Base.:-(left::Types.AbstractDelayed, right::Real) = BinaryOperator(left, right, -)
Base.:-(left::Real, right::Types.AbstractDelayed) = BinaryOperator(left, right, -)

Base.:*(left::Types.AbstractDelayed, right::Types.AbstractDelayed) = BinaryOperator(left, right, *)
Base.:*(left::Types.AbstractDelayed, right::Real) = BinaryOperator(left, right, *)
Base.:*(left::Real, right::Types.AbstractDelayed) = BinaryOperator(left, right, *)

Base.:/(left::Types.AbstractDelayed, right::Types.AbstractDelayed) = BinaryOperator(left, right, /)
Base.:/(left::Types.AbstractDelayed, right::Real) = BinaryOperator(left, right, /)
Base.:/(left::Real, right::Types.AbstractDelayed) = BinaryOperator(left, right, /)

Base.:^(left::Types.AbstractDelayed, right::Types.AbstractDelayed) = BinaryOperator(left, right, ^)
Base.:^(left::Types.AbstractDelayed, right::Real) = BinaryOperator(left, right, ^)
Base.:^(left::Real, right::Types.AbstractDelayed) = BinaryOperator(left, right, ^)

Base.:%(left::Types.AbstractDelayed, right::Types.AbstractDelayed) = BinaryOperator(left, right, %)
Base.:%(left::Types.AbstractDelayed, right::Real) = BinaryOperator(left, right, %)
Base.:%(left::Real, right::Types.AbstractDelayed) = BinaryOperator(left, right, %)
