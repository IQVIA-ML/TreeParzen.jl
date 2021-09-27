"""
spaceprint() functions can draw a tree of AbstractDelayed objects in the terminal.
"""
module SpacePrint

using Dates
using DocStringExtensions

export spaceprint

function cornerchar(final::Bool)::String
    if final
        return "└"
    end

    return "├"
end
function fillerchar(final::Bool)::String
    if final
        return " "
    end

    return "│"
end

"""
$(TYPEDSIGNATURES)
Show what is inside Delayed objects.

tab - what happens before this row. Grows with each index.
final - whether or not this is the final item in the list. Affects the
        corner style. If nothing, no corner or filler.
"""
function spaceprint(
    item; index::Int = 1, tab::String = "", corner::String = "", final::Bool = true
)::Nothing
    println(tab, corner, " ", string(item))

    return nothing
end
function spaceprint(
    item::Nothing; index::Int = 1, tab::String = "", corner::String = "", final::Bool = true
)::Nothing
    println(tab, corner, index, ": nothing")

    return nothing
end
function spaceprint(
    item::String; index::Int = 1, tab::String = "", corner::String = "", final::Bool = true
)::Nothing
    println(tab, corner, index, ": \"", item, '"')

    return nothing
end
function spaceprint(
    item::Symbol; index::Int = 1, tab::String = "", corner::String = "", final::Bool = true
)::Nothing
    println(tab, corner, index, ": :", item)

    return nothing
end
function spaceprint(
    item::Union{Dates.DateTime, Real, Type, UnitRange}; index::Int = 1, tab::String = "",
    corner::String = "", final::Bool = true
)::Nothing
    println(tab, corner, index, ": ", item)

    return nothing
end
function spaceprint(
    item::Union{Set, Tuple, Vector}; index::Int = 1, tab::String = "", corner::String = "",
    final::Bool = true
)::Nothing
    println(tab, corner, index, ": ", typeof(item), " of ", length(item), " items")
    for (i, v) in enumerate(item)
        finalitem = i == length(item)
        spaceprint(
            v; index = i, tab = string(tab, fillerchar(final)),
            corner = cornerchar(finalitem), final = finalitem
        )
    end

    return nothing
end
function spaceprint(
    item::Pair; index::Int = 1, tab::String = "", corner::String = "", final::Bool = true
)::Nothing
    k, v = item
    println(tab, '├', k)
    spaceprint(v; index = index, tab = tab, corner = corner, final = final)

    return nothing
end
function spaceprint(
    item::Dict; index::Int = 1, tab::String = "", corner::String = "", final::Bool = true
)::Nothing
    println(tab, corner, index, ": ", typeof(item), " of ", length(item), " items")
    for (i, kv) in enumerate(item)
        finalitem = i == length(item)
        spaceprint(
            kv; index = i, tab = string(tab, fillerchar(final)),
            corner = cornerchar(finalitem), final = finalitem
        )
    end

    return nothing
end

end # module SpacePrint
