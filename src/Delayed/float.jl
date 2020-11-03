"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct Float <: Types.AbstractDelayed
    arg::Types.AbstractDelayed
end

Base.float(arg::Types.AbstractDelayed) = Float(arg)
