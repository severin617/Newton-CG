# Newton-CG


## How to install/use the cg-approximated second order optimizer for tensorflow


```pip install -i https://test.pypi.org/simple/ newton-cg==0.0.4 ```

and use in code like


```
import newton_cg as es

    optimizer = es.EHNewtonOptimizer(
        learning_rate,
        tau=FLAGS.eso_tau,
        cg_tol=FLAGS.eso_cg_tol,
        max_iter=FLAGS.eso_max_iter)
```
 and use the learning rate scheduler like: 

```
# lr scheduler
from clr_callback import CyclicLR
clr_trig2 = CyclicLR(mode='exp_range', base_lr=0.00001, max_lr=0.0001, step_size=1, gamma=0.5) #gamma=0.9994)

model.fit(data_X, data_Y,  epochs=1, callbacks=[clr_trig2])

```

