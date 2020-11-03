module Delayed

import Base
import Distributions
using DocStringExtensions

import ..IndexObjects
import ..SpacePrint
import ..Types


"""DistDelayed objects represent functions that draw output from distributions"""
abstract type AbstractDistDelayed <: Types.AbstractDelayed end
const NestedFloat = Union{Types.AbstractDelayed, Float64}

"""Switch objects represent a choice between options"""
abstract type AbstractSwitch <: Types.AbstractDelayed end

"""Allow Delayed objects to contain other Delayed objects"""
const NestedFloat = Union{Types.AbstractDelayed, Float64}
const NestedInt = Union{Types.AbstractDelayed, Int}
const NestedReal = Union{Types.AbstractDelayed, Real}

include("add.jl")
include("categoricalindex.jl")
include("float.jl")
include("lognormal.jl")
include("logquantnormal.jl")
include("normal.jl")
include("quantnormal.jl")
include("params.jl")
include("randindex.jl")
include("uniform.jl")
include("loguniform.jl")
include("quantuniform.jl")
include("logquantuniform.jl")

function SpacePrint.spaceprint(
    item::Types.AbstractDelayed; index::Int = 1, tab::String = "", corner::String = "",
    final::Bool = true
)::Nothing
    println(tab, corner, index, ": ", typeof(item))
    for (i, propertyname) in enumerate(propertynames(item))
        finalitem = i == length(propertynames(item))
        println(tab, SpacePrint.fillerchar(final), "├", propertyname)
        SpacePrint.spaceprint(
            getproperty(item, propertyname); index = 1,
            tab = string(tab, SpacePrint.fillerchar(final)),
            corner = SpacePrint.cornerchar(finalitem), final = finalitem
        )
    end

    return nothing
end

end # module Delayed
