
abstract type Operator <: Types.AbstractDelayed end

"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct BinaryOperator{L,R,F} <: Types.AbstractDelayed
    left::L
    right::R
    operator::F
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct UnaryOperator{O,F} <: Types.AbstractDelayed
    operand::O
    operator::F
end

function Base.:+(left::Types.AbstractDelayed, right::Types.AbstractDelayed)
    return BinaryOperator(left, right, +)
end
Base.:+(left::Types.AbstractDelayed, right::Real) = BinaryOperator(left, right, +)
Base.:+(left::Real, right::Types.AbstractDelayed) = BinaryOperator(left, right, +)

function Base.:-(left::Types.AbstractDelayed, right::Types.AbstractDelayed)
    return BinaryOperator(left, right, -)
end
Base.:-(left::Types.AbstractDelayed, right::Real) = BinaryOperator(left, right, -)
Base.:-(left::Real, right::Types.AbstractDelayed) = BinaryOperator(left, right, -)
Base.:-(operand::Types.AbstractDelayed) = UnaryOperator(operand, -)

function Base.:*(left::Types.AbstractDelayed, right::Types.AbstractDelayed)
    return BinaryOperator(left, right, *)
end
Base.:*(left::Types.AbstractDelayed, right::Real) = BinaryOperator(left, right, *)
Base.:*(left::Real, right::Types.AbstractDelayed) = BinaryOperator(left, right, *)

function Base.:/(left::Types.AbstractDelayed, right::Types.AbstractDelayed)
    return BinaryOperator(left, right, /)
end
Base.:/(left::Types.AbstractDelayed, right::Real) = BinaryOperator(left, right, /)
Base.:/(left::Real, right::Types.AbstractDelayed) = BinaryOperator(left, right, /)

function Base.:^(left::Types.AbstractDelayed, right::Types.AbstractDelayed)
    return BinaryOperator(left, right, ^)
end
Base.:^(left::Types.AbstractDelayed, right::Real) = BinaryOperator(left, right, ^)
Base.:^(left::Real, right::Types.AbstractDelayed) = BinaryOperator(left, right, ^)

function Base.:%(left::Types.AbstractDelayed, right::Types.AbstractDelayed)
    return BinaryOperator(left, right, %)
end
Base.:%(left::Types.AbstractDelayed, right::Real) = BinaryOperator(left, right, %)
Base.:%(left::Real, right::Types.AbstractDelayed) = BinaryOperator(left, right, %)

Base.:round(operand::Types.AbstractDelayed) = UnaryOperator(operand, round)
