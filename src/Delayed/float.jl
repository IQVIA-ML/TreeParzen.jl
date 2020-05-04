"""
$(TYPEDEF)
$(TYPEDFIELDS)

"""
struct Float <: AbstractDelayed
    arg::AbstractDelayed
end

Base.float(arg::AbstractDelayed) = Float(arg)
