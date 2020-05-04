module Delayed

import Base
import Distributions
using DocStringExtensions

import ..IndexObjects
import ..SpacePrint

"""Delayed objects sit in the space representing simple functions to be run later"""
abstract type AbstractDelayed end

"""DistDelayed objects represent functions that draw output from distributions"""
abstract type AbstractDistDelayed <: AbstractDelayed end
const NestedFloat = Union{AbstractDelayed, Float64}

"""Switch objects represent a choice between options"""
abstract type AbstractSwitch <: AbstractDelayed end

"""Allow Delayed objects to contain other Delayed objects"""
const NestedFloat = Union{AbstractDelayed, Float64}
const NestedInt = Union{AbstractDelayed, Int}
const NestedReal = Union{AbstractDelayed, Real}

include("add.jl")
include("categoricalindex.jl")
include("float.jl")
include("lognormal.jl")
include("quantlognormal.jl")
include("normal.jl")
include("quantnormal.jl")
include("params.jl")
include("randindex.jl")
include("uniform.jl")
include("loguniform.jl")
include("quantuniform.jl")
include("quantloguniform.jl")

function SpacePrint.spaceprint(
    item::AbstractDelayed; index::Int = 1, tab::String = "", corner::String = "",
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
