module Graph

using DocStringExtensions

import ..Types
import ..Delayed

"""
$(TYPEDSIGNATURES)

Given an delayed object, return an array of argument values.
"""
function delayedproperties(item::Types.AbstractDelayed)::Vector

    return [
        getproperty(item, propertyname)
            for propertyname in propertynames(item)
    ]

end
delayedproperties(a::Delayed.AbstractSwitch) = [a.choice, a.options...]
delayedproperties(a::Delayed.CategoricalIndex) = a.probabilities

"""
$(TYPEDSIGNATURES)

Depth-first search
Unrolls a graph into an array of all the nodes.
"""
dfs(space::Dict{Symbol, T} where T)::Vector = dfs!(Types.AbstractDelayed[], space)

"""
$(TYPEDSIGNATURES)

Unrolls a nested space into a vector of all the Types.AbstractDelayed nodes.
"""
function dfs!(seq::Vector{Types.AbstractDelayed}, item::Types.AbstractDelayed)::Vector
    # For every input (arg) of the object, add those to the list too.
    for prop in delayedproperties(item)
        dfs!(seq, prop)
    end
    push!(seq, item)

    return seq
end
function dfs!(seq::Vector{Types.AbstractDelayed}, item::Dict)::Vector
    for v in values(item)
        dfs!(seq, v)
    end

    return seq
end
function dfs!(seq::Vector{Types.AbstractDelayed}, item::Union{Tuple, Vector})::Vector
    for v in item
        dfs!(seq, v)
    end

    return seq
end
dfs!(seq::Vector{Types.AbstractDelayed}, item::Any)::Vector = seq

function checklabel!(labels::Vector{Symbol}, node::Delayed.AbstractParam)::Nothing
    if node.label in labels
        throw(DomainError(node.label, "Your space has a duplicate label :$(node.label)"))
    end
    push!(labels, node.label)

    return nothing
end
checklabel!(labels::Vector{Symbol}, node::T where T)::Nothing = nothing

"""
$(TYPEDSIGNATURES)

Ensure the user hasn't submitted any duplicate labels in their space.
"""
function checkspace(space::Dict{Symbol, T})::Dict{Symbol, T} where T
    labels = Symbol[]
    for node in Graph.dfs(space)
        checklabel!(labels, node)
    end

    return space
end

end # module Graph
