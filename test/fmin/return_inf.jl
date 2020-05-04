module TestReturnInf

using Test
using TreeParzen

function objective(params)
    if params[:x] < 0
        return (params[:x] - 3)^2
    end

    return Inf
end
best = fmin(objective, Dict(:x => HP.Uniform(:x, -5.0, 5.0)), 50)
@test best[:x] < 0

end
true
