module TreeParzen

# NOTE: The files must be included in the right order - if the functions in a
#       file depend on another file, the dependency must be loaded FIRST.

include("Types.jl")
include("SpacePrint.jl")
include("Configuration.jl")
include("IndexObjects.jl")
include("Trials.jl")
include("ApFilterTrials.jl")
include("Bincounts.jl")
include("ForgettingWeights.jl")
include("Delayed/Delayed.jl")
include("GMM.jl")
include("LogGMM.jl")
include("Samplers/Samplers.jl")
include("Graph.jl")
include("Resolve/Resolve.jl")
include("API.jl")
include("HP.jl")
include("MLJTreeParzen.jl")

using .API
using .Configuration
using .HP
using .SpacePrint
using .MLJTreeParzen

export ask
export Config
export fmin
export HP
export MLJTreeParzen, MLJTreeParzenTuning, MLJTreeParzenSpace
export Prob
export provide_recommendation
export spaceprint
export tell!

end
