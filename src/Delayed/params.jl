abstract type AbstractParam <: Types.AbstractDelayed end

"""
$(TYPEDSIGNATURES)

A node of the space used to assign parameter labels. Only Param objects are extracted into
the vals dictionary for output. Therefore, every parameter of the user's function must have
a corresponding Param object in the space.
"""
struct Param <: AbstractParam
    label::Symbol
    obj::Types.AbstractDelayed
end
