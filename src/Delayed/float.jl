"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct Float{T <: Types.AbstractDelayed} <: Types.AbstractDelayed
    arg::T
end

Base.float(arg::Types.AbstractDelayed) = Float(arg)
