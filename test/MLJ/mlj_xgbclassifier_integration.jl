module MLJ_tell_signature

using MLJBase, MLJXGBoostInterface, MLJTuning, TreeParzen, Test

import MLJXGBoostInterface.XGBoostClassifier

X, y = @load_iris
xgbclassifier = XGBoostClassifier()

space = Dict(
    :gamma => HP.LogNormal(:gamma, log(0.5), 3.),
    :max_depth => HP.QuantUniform(:max_depth, 2., 50., 1.0),
    :min_child_weight => HP.QuantUniform(:min_child_weight, 2., 10., 1.),
    :max_delta_step => HP.LogNormal(:max_delta_step, log(1.), 5.),
    :lambda => HP.LogNormal(:lambda, log(0.05), .2),
    :alpha => HP.LogNormal(:alpha, log(0.05), .2),
    :max_leaves => HP.Choice(:max_leaves, [0, 2, 4, 8, 16, 32]),
    )

num_cv_folds = 4
rand_trials = 3
simultaneous_draws = 2
StrCV = CV(nfolds=num_cv_folds)
num_models_trained = 25
metrics = LogLoss()

tuned_classifier_model = MLJTuning.TunedModel(
                                            model=xgbclassifier,
                                            tuning=TreeParzen.MLJTreeParzen.MLJTreeParzenTuning(
                                                    ;random_trials=rand_trials,
                                                    max_simultaneous_draws=simultaneous_draws,
                                                    linear_forgetting=simultaneous_draws*25
                                            ),
                                            resampling=StrCV,
                                            repeats=5,
                                            n=num_models_trained,
                                            range=space,
                                            measure=metrics,
                                            )

tuned_classifier = machine(tuned_classifier_model, X, y)
fitted_classifier = fit!(tuned_classifier, verbosity=1)
# this is just a simple property output testing to make sure the fitting above completes with no errors
@test length(propertynames(fitted_classifier)) == 13

end # module
true
