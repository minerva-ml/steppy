Default logic of the `fit_transform()`:
1. execute it on the last Step.
1. Step is fitted and transformed. Any data or models will be overridden (default setup).
1. If `force_fitting` is not obligatory, then look for cache:
    1. if output is cached, then use it. In such situation `fit_tranform()` was just taking output from cache.
    1. If output is not cached -> steppy looks for persisted (saved to disk) output. If exist, `fit_tranform()` was just loading output from the project directory.
