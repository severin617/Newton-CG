# TODO License
"""
A rewrite of efficient_second, deriving from `tf.python.keras.optimizer_v2`
Not "distribute-aware".

Stitches all variables first and computes the update step
for all values at once. This reduces the memory overhead of
computing gradients inside `while_v2.while_loop`.

Efficient Hessian-vector multiplication based, CG-approximated Newton
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
from tensorflow.python import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import while_v2
from tensorflow.python.framework import dtypes
from tensorflow.python.eager import backprop  # Not used, but should be.
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras import backend_config
from tensorflow.python.util import nest
from tensorflow.python.ops import clip_ops
# distribution stuff
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.eager import context
# Keras export
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.optimizers.EfficientHessianOptimizer')
class EHNewtonOptimizer(optimizer_v2.OptimizerV2):
  r"""Optimizer that implements the CG-approximated nonlinear Newton algorithm.

  Conjugate Gradient is used to approximate a solution to Newton's equation.
  Computation of the Hessian is avoided, via Pearlmutter's trick.

  See [Pearlmutter, 1993](https://doi.org/10.1162/neco.1994.6.1.147)
  ([pdf](https://www.mitpressjournals.org/doi/pdf/10.1162/neco.1994.6.1.147)).
  """
  
  def __init__(self,
               learning_rate=0.001,
               tau=1e-5,
               cg_tol=1e-5,
               max_iter=20,
               epsilon=None,
               name='EHNewton',
               **kwargs):
    r"""Construct a new EHNewton optimizer.
    Args:
      learning_rate: A `Tensor` or a floating point value. The learning rate.
      tau: A constant for Tikhonov regularization.
      cg_tol: Tolerance for convergence during Conjugate Gradients.
      max_iter: Maximum number of iterations during Conjugate Gradients.
      use_locking: If True use locks for update operations.
      epsilon: A small constant for numerical stability.
      name: Optional name for the operations created when applying gradients.
        Defaults to "EfficientHessian".
        @compatibility(eager) When eager execution is
        enabled, `learning_rate` and `tau` can each be a
        callable that takes no arguments and returns the actual value to use.
        This can be useful for changing these values across different
        invocations of optimizer functions. @end_compatibility
    """
    
    # Backwards compatibility with other keras optimizers.
    kwargs['decay'] = kwargs.pop('schedule_decay', 0.004)
    learning_rate = kwargs.get('lr', learning_rate)
    if isinstance(learning_rate, learning_rate_schedule.LearningRateSchedule):
      raise ValueError('The EHNewton optimizer does not support '
                       'tf.keras.optimizers.LearningRateSchedules as the '
                       'learning rate.')
    
    super(EHNewtonOptimizer, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('tau', tau)
    self._set_hyper('cg_tol', cg_tol)
    self._set_hyper('max_iter', max_iter)
    self.epsilon = epsilon or backend_config.epsilon()
  
  def _create_slots(self, var_list):
    pass
  
  def _prepare_local(self, var_device, var_dtype, apply_state):
    lr_t = array_ops.identity(self._get_hyper('learning_rate', var_dtype))
    tau_t = array_ops.identity(self._get_hyper('tau', var_dtype))
    cg_tol_t = array_ops.identity(self._get_hyper('cg_tol', var_dtype))
    max_iter_t = array_ops.identity(self._get_hyper('max_iter', dtypes.int32))
    
    apply_state[(var_device, var_dtype)] = dict(
        lr_t=lr_t,
        tau_t=tau_t,
        cg_tol_t=cg_tol_t,
        max_iter_t=max_iter_t,
        epsilon_t=ops.convert_to_tensor(self.epsilon, var_dtype),
    )
  
  def _resource_apply_dense(self, step, var, apply_state=None):
    """NOTE: The purpose of this method changed.
    Whereas the step computation happens in this method in all other
    TF optimizers, here we just apply the step to var.
    Our step computation happens in `_resource_compute_dense()`.

    The reason for the change: `resouce_apply_dense()` is called in an `update`
    context, while we need a `replica` context to compute second-order
    derivatives during the step computation.
    """
    # Just add the step to var
    return state_ops.assign_add(var, step, use_locking=self._use_locking).op
  
  def _resource_compute_dense(self, grad, var, apply_state=None):
    print("eso._resource_compute_dense() called.")
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))
    
    step = self._newton_step(grad, var, coefficients)
    scaled_step = math_ops.multiply(step, coefficients['lr_t'])
    print("eso._resource_compute_dense() FINISHED.")
    return scaled_step
  
  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    # raise NotImplementedError("_resource_apply_sparse t.b.d.")
    return state_ops.assign_add(var, grad, use_locking=self._use_locking).op
  def _resource_compute_sparse(self, grad, var, apply_state=None):
    print("eso._resource_compute_sparse() called.")
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))
    
    step = self._newton_step(grad, var, coefficients)
    scaled_step = math_ops.multiply(step, coefficients['lr_t'])
    print("eso._resource_compute_dense() FINISHED.")
    return scaled_step


  def get_config(self):
    config = super(EHNewtonOptimizer, self).get_config()
    config.update({
      'learning_rate': self._serialize_hyperparameter('learning_rate'),
      'decay': self._serialize_hyperparameter('decay'),
      'tau': self._serialize_hyperparameter('tau'),
      'cg_tol': self._serialize_hyperparameter('cg_tol'),
      'max_iter': self._serialize_hyperparameter('max_iter'),
      'epsilon': self.epsilon,
    })
    return config
  
  @staticmethod
  def _vv(a, b):
    """Vector-Vector Multiplication for arbitrary shapes.
    Computes `a^T * b`, treating a and b as if they were 1-D vectors.

    Considerably faster than calling `tf.reshape()` and `tf.matmul()`.

    *NOTE*: `Multiply` supports broadcasting. More about broadcasting
    [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    Args:
      a: A `Tensor`
      b: A `Tensor`. Must have same type as `a`.
    Returns:
      A scalar (rank 0) `Tensor` with the value `a^T * b`, same type as `a`.
    """
    elemwise_product = math_ops.multiply(a, b)
    return math_ops.reduce_sum(elemwise_product)
  
  @staticmethod
  def _compute_gradient(loss, var):
    return gradients.gradients(loss, var)[0]
  
  def _pearlmutter_hessian(self, grad, var, s):
    """Hessian-Vector Multiplication.
    Computes `H * s`, where `H` is the Hessian w.r.t. `var`.

    ADAPTED from tensorflow/python/ops/gradients_impl.py

    Avoids computing the Hessian explicitly, via Pearlmutter's Trick.
    See [Pearlmutter, 1993](https://doi.org/10.1162/neco.1994.6.1.147)
    ([pdf](https://www.mitpressjournals.org/doi/pdf/10.1162/neco.1994.6.1.147)).

    Args:
      grad: A `Tensor`. The gradient of `var`.
      var: A `Tensor` or a `Variable`. Must be same type as `grad`.
      s: A `Tensor`. The vector to be multiplied by the Hessian.
    Returns:
      A `Tensor` with the value `H * s`, same type as `grad`.
    """
    dg_dv = self._compute_gradient(grad, var)  # first backprop
    elemwise = math_ops.multiply(dg_dv, array_ops.stop_gradient(s))
    return self._compute_gradient(elemwise, var)  # second backprop
  
  def _cg_solve(self, Ax, b, cg_tol_t, max_iter_t):
    """Conjugate Gradient solver for `Ax = b`.

    Computes the approximate solution (within tolerance `tol`).
    Stops after at most `max_iter` iterations.
    Works completely statically, adding ops to the computational graph.

    Params:
      Ax: A callable function that accepts `x` as argument and returns `Ax`.
      b: A `Tensor` with the same type and shape as returned by `Ax(..)`.
      cg_tol_t: A `Tensor` containing the convergence criterion.
      max_iter_t: A `Tensor` containing the maximum number of iterations.
      x_last: Initial value of `x`. Will contain new value of `x` after CG.

    Returns:
      A `Tensor` holding the approx. solution to `Ax = b`, same type as `grad`.
    """
    
    def run_loop(x_0, r_0):
      p_0 = array_ops.identity(r_0)  # deep copy
      
      def _cond(_x, r, _p):  # while loop cond: |r| > tolerance
        return math_ops.greater(linalg_ops.norm(r), cg_tol_t)
      
      def _body(x, r, p):  # while loop body: compute one CG iteration
        rtr = self._vv(r, r)
        axp = Ax(p)
        alpha = math_ops.divide(rtr, self._vv(p, axp))
        x_ret = math_ops.add(x, math_ops.multiply(p, alpha))
        r_ret = math_ops.subtract(r, math_ops.multiply(axp, alpha))
        rtr_new = self._vv(r_ret, r_ret)
        beta = math_ops.divide(rtr_new, rtr)
        p_ret = math_ops.add(r_ret, math_ops.multiply(p, beta))
        return x_ret, r_ret, p_ret
      
      return while_v2.while_loop(
          _cond, _body, (x_0, r_0, p_0),
          parallel_iterations=1,
          maximum_iterations=max_iter_t
      )
    
    x_0 = tf.zeros(shape=tf.shape(b), dtype=b.dtype)
    r_0 = array_ops.identity(b)
    x_n, _, _ = run_loop(x_0, r_0)
    return x_n
  
  def _newton_step(self, grad, var, coefficients):
    """One step of the nonlinear Newton's Algorithm.

    Approximates the solution to the Newton equation using Conjgate Gradients.
    See `_cg_solve(self, Ax, b, x_init=None)` above.

    Args:
      grad: A `Tensor`. The gradient of `var`.
      var: A `Tensor` or a `Variable`. Must be same type as `grad`.
      coefficients: A dict containing `Tensor` coefficients of this algorithm.

    Returns:
      A `Tensor` holding the next Newton step, same type as `grad`.
    """
    
    def tikhonov(s_itr):
      hess_s = self._pearlmutter_hessian(grad, var, s_itr)
      reg_term = math_ops.multiply(s_itr, coefficients['tau_t'])
      return math_ops.add(hess_s, reg_term)
    
    d = self._cg_solve(Ax=tikhonov,
                       b=-grad,
                       cg_tol_t=coefficients['cg_tol_t'],
                       max_iter_t=coefficients['max_iter_t'])
    cnd = math_ops.greater_equal(self._vv(grad, d), -coefficients['tau_t'])
    return control_flow_ops.case([(cnd, lambda: -grad)], default=lambda: d)
  
  def stitch_hack(self, var_list):
    stitch_indices = []
    split_indices = []
    shapes = dict()
    counter = tf.zeros((), dtype=dtypes.int32)
    j = 0
    for v in var_list:
      size = tf.size(v)
      stitch_indices.append(array_ops.reshape(
          math_ops.range(size, dtype=dtypes.int32),
          array_ops.shape(v)))
      split_indices.append(array_ops.reshape(
          array_ops.zeros(size, dtype=dtypes.int32),
          array_ops.shape(v)))
      stitch_indices[-1] = math_ops.add(stitch_indices[-1], counter)
      split_indices[-1] = math_ops.add(split_indices[-1], j)
      shapes[v] = array_ops.shape(v)
      counter = math_ops.add(counter, size)
      j += 1
    
    num_partitions = len(var_list)
    stacked_indices = tf.concat([tf.reshape(i, [1, -1])
                                 for i in split_indices],
                                axis=1)
    stacked_indices = array_ops.reshape(stacked_indices, [-1])

    stitched_vars = tf.dynamic_stitch(indices=stitch_indices,
                                      data=[v for v in var_list],
                                      name="stitch_the_vars")
    split_stitched_vars = tf.dynamic_partition(stitched_vars,
                                               stacked_indices,
                                               num_partitions,
                                               name="resplit_the_vars")
    # assign split_stitched_vars to vars in var list
    assigns = [state_ops.assign(var, array_ops.reshape(spstvar, shapes[var]))
               for var, spstvar in zip(var_list, split_stitched_vars)]
    return assigns, stitched_vars
  
  #######################################################
  # NEW Update step computation within replica context. #
  #######################################################
  def _compute_step(self, grads_and_vars, apply_state):
    """ADDED: Run update step computation within a replica context."""
    
    def compute_step_to_update_var(var, grad):
      """Compute the step to update var (using grad)."""
      apply_kwargs = {}
      if isinstance(grad, ops.IndexedSlices):
        # TODO handle sparse case
        if var.constraint is not None:
          raise RuntimeError(
              "Cannot use a constraint function on a sparse variable.")
        if "apply_state" in self._sparse_apply_args:
          apply_kwargs["apply_state"] = apply_state
        return self._resource_apply_sparse_duplicate_indices(
            grad.values, var, grad.indices, **apply_kwargs)
      
      if "apply_state" in self._dense_apply_args:
        apply_kwargs["apply_state"] = apply_state
      return self._resource_compute_dense(grad, var, **apply_kwargs)
    
    steps = [compute_step_to_update_var(v, grad) for grad, v in grads_and_vars]
    var_list = [v for _, v in grads_and_vars]
    steps_and_vars = list(zip(steps, var_list))
    
    return steps_and_vars
  
  ######################################################################
  # OVERRIDE METHOD FROM optimizer_v2 (COPIED from the TF v2.2.0 code) #
  # MODIFIED apply_gradients and _distributed_apply to include the new #
  # update logic (computed in _compute_step, just applied after that). #
  ######################################################################
  
  def compute_gradients(self, loss, var_list, grad_loss=None):
    """Compatibility for users expecting tf.train.Optimizer signature"""
    return self._compute_gradients(loss, var_list, grad_loss=grad_loss)
  
  def _compute_gradients(self, loss, var_list, grad_loss=None):
    """COPIED and MODIFIED from super class
    Compute gradients of `loss` for the variables in `var_list`.

    This is the first part of `minimize()`.  It returns a list
    of (gradient, variable) pairs where "gradient" is the gradient
    for "variable".  Note that "gradient" can be a `Tensor`, an
    `IndexedSlices`, or `None` if there is no gradient for the
    given variable.

    Args:
      loss: A callable taking no arguments which returns the value to minimize.
      var_list: list or tuple of `Variable` objects to update to minimize
        `loss`, or a callable returning the list or tuple of `Variable` objects.
        Use callable when the variable list would otherwise be incomplete before
        `minimize` and the variables are created at the first time when `loss`
        is called.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

    Returns:
      A list of (gradient, variable) pairs. Variable is always present, but
      gradient can be `None`.

    Raises:
      TypeError: If `var_list` contains anything else than `Variable` objects.
      ValueError: If some arguments are invalid, or var_list is None.
    """
    
    assigns, stitched_vars = self.stitch_hack(
        var_list() if callable(var_list) else var_list)
    
    if not callable(loss):
      with backend.get_graph().as_default(), backend.name_scope(self._name +
                                                                "/gradients"):
        with ops.control_dependencies(assigns):
          grads = gradients.gradients(loss, stitched_vars)
        for grad, param in zip(grads, [stitched_vars]):
          if grad is None:
            raise ValueError("Variable {} has `None` for gradient. "
                             "Please make sure that all of your ops have a "
                             "gradient defined (i.e. are differentiable). "
                             "Common ops without gradient: "
                             "K.argmax, K.round, K.eval.".format(param))
        if hasattr(self, "clipnorm"):
          grads = [clip_ops.clip_by_norm(g, self.clipnorm) for g in grads]
        if hasattr(self, "clipvalue"):
          grads = [
            clip_ops.clip_by_value(g, -self.clipvalue, self.clipvalue)
            for g in grads
          ]
      return list(zip(grads, [stitched_vars]))
    
    # TODO(josh11b): Test that we handle weight decay in a reasonable way.
    with backprop.GradientTape() as tape:
      if not callable(stitched_vars):
        tape.watch(stitched_vars)
      with ops.control_dependencies(assigns):
        loss_value = loss()
    if callable(stitched_vars):
      stitched_vars = stitched_vars()
    stitched_vars = nest.flatten(stitched_vars)
    with backend.name_scope(self._name + "/gradients"):
      grads = tape.gradient(loss_value, stitched_vars, grad_loss)
      
      if hasattr(self, "clipnorm"):
        grads = [clip_ops.clip_by_norm(g, self.clipnorm) for g in grads]
      if hasattr(self, "clipvalue"):
        grads = [
          clip_ops.clip_by_value(g, -self.clipvalue, self.clipvalue)
          for g in grads
        ]
    
    grads_and_vars = list(zip(grads, stitched_vars))
    self._assert_valid_dtypes([
      v for g, v in grads_and_vars
      if g is not None and v.dtype != dtypes.resource
    ])
    
    return grads_and_vars
  
  def apply_gradients(self,
                      grads_and_vars,
                      name=None,
                      experimental_aggregate_gradients=True):
    """MODIFIED to support second-order step computation.
    Apply gradients to variables.

    This is the second part of `minimize()`. It returns an `Operation` that
    applies gradients.

    The method sums gradients from all replicas in the presence of
    `tf.distribute.Strategy` by default. You can aggregate gradients yourself by
    passing `experimental_aggregate_gradients=False`.

    Example:

    ```python
    grads = tape.gradient(loss, vars)
    grads = tf.distribute.get_replica_context().all_reduce('sum', grads)
    # Processing aggregated gradients.
    optimizer.apply_gradients(zip(grads, vars),
        experimental_aggregate_gradients=False)

    ```

    Args:
      grads_and_vars: List of (gradient, variable) pairs.
      name: Optional name for the returned operation. Default to the name passed
        to the `Optimizer` constructor.
      experimental_aggregate_gradients: Whether to sum gradients from different
        replicas in the presense of `tf.distribute.Strategy`. If False, it's
        user responsibility to aggregate the gradients. Default to True.

    Returns:
      An `Operation` that applies the specified gradients. The `iterations`
      will be automatically increased by 1.

    Raises:
      TypeError: If `grads_and_vars` is malformed.
      ValueError: If none of the variables have gradients.
    """
    grads_and_vars = optimizer_v2._filter_grads(grads_and_vars)
    var_list = [v for (_, v) in grads_and_vars]
    
    with backend.name_scope(self._name):
      # Create iteration if necessary.
      with ops.init_scope():
        self._create_all_weights(var_list)
      
      if not grads_and_vars:
        # Distribution strategy does not support reducing an empty list of
        # gradients
        return control_flow_ops.no_op()
      
      if distribute_ctx.in_cross_replica_context():
        raise RuntimeError(
            "`apply_gradients() cannot be called in cross-replica context. "
            "Use `tf.distribute.Strategy.run` to enter replica "
            "context.")
      
      strategy = distribute_ctx.get_strategy()
      if (not experimental_aggregate_gradients and strategy and isinstance(
          strategy.extended,
          parameter_server_strategy.ParameterServerStrategyExtended)):
        raise NotImplementedError(
            "`experimental_aggregate_gradients=False is not supported for "
            "ParameterServerStrategy and CentralStorageStrategy")
      
      apply_state = self._prepare(var_list)
      if experimental_aggregate_gradients:
        reduced_grads = self._aggregate_gradients(grads_and_vars)
        var_list = [v for _, v in grads_and_vars]
        grads_and_vars = list(zip(reduced_grads, var_list))
      # MODIFIED: compute first (in replica context), then apply the result.
      steps_and_vars = self._compute_step(grads_and_vars, apply_state=apply_state)
      return distribute_ctx.get_replica_context().merge_call(
          functools.partial(self._distributed_apply, apply_state=apply_state),
          args=(steps_and_vars,),
          kwargs={
            "name": name,
          })
  
  def _aggregate_gradients(self, grads_and_vars):
    """Returns all-reduced gradients.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.

    Returns:
      A list of all-reduced gradients.
    """
    grads_and_vars = list(grads_and_vars)
    filtered_grads_and_vars = optimizer_v2._filter_grads(grads_and_vars)
    
    def all_reduce_fn(distribution, grads_and_vars):
      return distribution.extended.batch_reduce_to(
          ds_reduce_util.ReduceOp.SUM, grads_and_vars)
    
    # We switch to a cross-replica context since there is a bug which causes
    # IndexedSlices to be converted to dense tensors when all-reduced in a
    # replica context.
    # TODO(b/150507409): Do not switch to a cross-replica context once the bug
    # is fixed.
    if filtered_grads_and_vars:
      reduced = distribute_ctx.get_replica_context().merge_call(
          all_reduce_fn, args=(filtered_grads_and_vars,))
    else:
      reduced = []
    # Copy 'reduced' but add None gradients back in
    reduced_with_nones = []
    reduced_pos = 0
    for g, _ in grads_and_vars:
      if g is None:
        reduced_with_nones.append(None)
      else:
        reduced_with_nones.append(reduced[reduced_pos])
        reduced_pos += 1
    assert reduced_pos == len(reduced), "Failed to add all gradients"
    return reduced_with_nones
  
  def _distributed_apply(self, distribution, steps_and_vars, name, apply_state):
    """MODIFIED: apply the precomputed step
    `apply_gradients` using a `DistributionStrategy`.
    """
    
    def apply_step_to_update_var(var, step):
      """Apply gradient to variable."""
      if isinstance(var, ops.Tensor):
        raise NotImplementedError("Trying to update a Tensor ", var)
      
      apply_kwargs = {}
      if isinstance(step, ops.IndexedSlices):
        if var.constraint is not None:
          raise RuntimeError(
              "Cannot use a constraint function on a sparse variable.")
        if "apply_state" in self._sparse_apply_args:
          apply_kwargs["apply_state"] = apply_state
        return self._resource_apply_sparse_duplicate_indices(
            step.values, var, step.indices, **apply_kwargs)
      
      if "apply_state" in self._dense_apply_args:
        apply_kwargs["apply_state"] = apply_state
      update_op = self._resource_apply_dense(step, var, **apply_kwargs)
      if var.constraint is not None:
        with ops.control_dependencies([update_op]):
          return var.assign(var.constraint(var))
      else:
        return update_op
    
    eagerly_outside_functions = ops.executing_eagerly_outside_functions()
    update_ops = []
    with ops.name_scope(name or self._name):  # REMOVED: TF2 specific, skip_on_eager=True):
      for step, var in steps_and_vars:
        # TODO(crccw): It's not allowed to assign PerReplica value to
        # MirroredVariable.  Remove this after we relax this restriction.
        
        # def _assume_mirrored(step):
        #   if isinstance(step, ds_values.PerReplica):
        #     return ds_values.Mirrored(step.values)
        #   return step
        #
        # step = nest.map_structure(_assume_mirrored, step)
        
        # Colocate the update with variables to avoid unnecessary communication
        # delays. See b/136304694.
        with distribution.extended.colocate_vars_with(var):
          with ops.name_scope("update" if eagerly_outside_functions else
                              "update_" + var.op.name):  # REMOVED: TF 2 specific, skip_on_eager=True):
            update_ops.extend(distribution.extended.update(
                var, apply_step_to_update_var, args=(step,), group=False))
      
      any_symbolic = any(isinstance(i, ops.Operation) or
                         tf_utils.is_symbolic_tensor(i) for i in update_ops)
      if not context.executing_eagerly() or any_symbolic:
        # If the current context is graph mode or any of the update ops are
        # symbolic then the step update should be carried out under a graph
        # context. (eager updates execute immediately)
        with ops._get_graph_from_inputs(update_ops).as_default():  # pylint: disable=protected-access
          with ops.control_dependencies(update_ops):
            return self._iterations.assign_add(1, read_value=False)
      
      return self._iterations.assign_add(1)
  
  def _create_all_weights(self, var_list):
    """Creates all weights, including iterations, hyperparameters and slot vars.
    This will add newly created variables to `optimizer.weights`.
    New variables are only created when this method is called the first time, or
    when called with different variables in the var_list.
    Args:
      var_list: list or tuple of `Variable` objects that will be minimized
        using this optimizer.
    """
    
    _ = self.iterations
    self._create_hypers()
    self._create_slots(var_list)
