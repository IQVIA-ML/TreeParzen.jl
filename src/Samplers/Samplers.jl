module Samplers

using DocStringExtensions

import ..Bincounts
using ..Configuration
import ..Delayed
import ..GMM
import ..IndexObjects
import ..ForgettingWeights
import ..LogGMM

include("adaptive_parzen.jl")
include("normal.jl")
include("quantnormal.jl")
include("lognormal.jl")
include("logquantnormal.jl")
include("randindex.jl")
include("uniform.jl")
include("quantuniform.jl")
include("categoricalindex.jl")
include("logquantuniform.jl")
include("loguniform.jl")

end # module Samplers
