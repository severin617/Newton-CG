# Newton-CG


## How to install/use the cg-approximated second order optimizer for tensorflow


```pip install -i https://test.pypi.org/simple/ newton-cg==0.0.2 ```

and use in code like


```
import newton_cg as es

    optimizer = es.EHNewtonOptimizer(
        learning_rate,
        tau=FLAGS.eso_tau,
        cg_tol=FLAGS.eso_cg_tol,
        max_iter=FLAGS.eso_max_iter)
```

