module TesSpaces

using Test
using TreeParzen


space = Dict(
    :a => HP.Choice(:a, [:a, :b, :c]),
    :b => HP.Choice(:b, ["a", "b", "c"])
)

sample = ask(space).hyperparams

@test sample[:a] in (:a, :b, :c)
@test sample[:b] in ("a", "b", "c")

end
true
