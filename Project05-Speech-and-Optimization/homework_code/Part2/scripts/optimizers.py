"""SGD, Adam, and RMSProp optimizers implemented from scratch.

All three optimizers subclass ``torch.optim.Optimizer`` so that they
follow the standard PyTorch optimizer interface (``step(closure=None)``,
``zero_grad()``, ``param_groups``, per-parameter ``state`` dicts, etc.)
while still implementing the update rules by hand.

Usage example::

    params = torch.tensor([x0, y0], dtype=torch.float32, requires_grad=True)
    opt = SGDOptimizer([params], lr=0.1)
    params.grad = torch.tensor([gx, gy])
    opt.step()
    opt.zero_grad()
"""

import torch


class SGDOptimizer(torch.optim.Optimizer):
    """Plain Stochastic Gradient Descent, implemented from scratch.

    Subclasses :class:`torch.optim.Optimizer` so that it follows the
    standard PyTorch optimizer interface while implementing the vanilla
    SGD update rule by hand:

    .. math:: \\theta_{t+1} = \\theta_t - \\text{lr} \\cdot \\nabla f(\\theta_t)
    """

    def __init__(self, params, lr):
        """Store the learning rate as a default hyperparameter.

        Args:
            params (iterable): Iterable of tensors to optimize.
            lr (float): Step size used to update the parameters.
        """
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one SGD update.

        Reads ``p.grad`` for every parameter and updates ``p.data``
        in-place following the vanilla SGD rule.

        Args:
            closure (callable, optional): A closure that recomputes
                the loss and returns it. Not used here but required by
                the :class:`torch.optim.Optimizer` interface.

        Returns:
            The loss returned by ``closure``, or ``None`` if no closure
            was provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.add_(p.grad, alpha=-lr)

        return loss


class RMSPropOptimizer(torch.optim.Optimizer):
    """RMSProp optimizer, implemented from scratch.

    Subclasses :class:`torch.optim.Optimizer` and maintains a moving
    average of squared gradients in the per-parameter ``state`` dict,
    following the standard RMSProp update rule:

    .. math::
        s_t      &= \\beta \\, s_{t-1} + (1-\\beta) \\, g_t^2 \\\\
        \\theta_{t+1} &= \\theta_t - \\frac{\\text{lr} \\cdot g_t}
                                           {\\sqrt{s_t} + \\epsilon}
    """

    def __init__(self, params, lr, beta=0.99, epsilon=1e-8):
        """Store the hyperparameters as defaults.

        Args:
            params (iterable): Iterable of tensors to optimize.
            lr (float): Step size used to update the parameters.
            beta (float): Decay rate for the moving average of squared
                gradients. Defaults to 0.99 (same default PyTorch uses).
            epsilon (float): Small constant added for numerical stability.
        """
        defaults = dict(lr=lr, beta=beta, epsilon=epsilon)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one RMSProp update.

        Reads ``p.grad`` for every parameter, updates the per-parameter
        moving average of squared gradients, and updates ``p.data``
        in-place.

        Args:
            closure (callable, optional): A closure that recomputes
                the loss and returns it.

        Returns:
            The loss returned by ``closure``, or ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            epsilon = group["epsilon"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if "square_avg" not in state:
                    state["square_avg"] = torch.zeros_like(p)

                square_avg = state["square_avg"]
                square_avg.mul_(beta).addcmul_(p.grad, p.grad, value=1 - beta)

                update = lr * p.grad / (torch.sqrt(square_avg) + epsilon)
                p.data.sub_(update)

        return loss


class AdamOptimizer(torch.optim.Optimizer):
    """Adam optimizer, implemented from scratch.

    Subclasses :class:`torch.optim.Optimizer` and maintains first and
    second moment estimates in the per-parameter ``state`` dict,
    following the standard Adam update rule:

    .. math::
        m_t &= \\beta_1 \\, m_{t-1} + (1-\\beta_1) \\, g_t \\\\
        v_t &= \\beta_2 \\, v_{t-1} + (1-\\beta_2) \\, g_t^2 \\\\
        \\hat{m}_t &= m_t / (1 - \\beta_1^t) \\\\
        \\hat{v}_t &= v_t / (1 - \\beta_2^t) \\\\
        \\theta_{t+1} &= \\theta_t - \\frac{\\text{lr} \\cdot \\hat{m}_t}
                                           {\\sqrt{\\hat{v}_t} + \\epsilon}
    """

    def __init__(self, params, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Store the hyperparameters as defaults.

        Args:
            params (iterable): Iterable of tensors to optimize.
            lr (float): Step size used to update the parameters.
            beta1 (float): Decay rate for the first moment (mean of
                gradients). Defaults to 0.9.
            beta2 (float): Decay rate for the second moment (uncentered
                variance of gradients). Defaults to 0.999.
            epsilon (float): Small constant added for numerical stability.
        """
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one Adam update.

        Reads ``p.grad`` for every parameter, updates the first and
        second moment estimates with bias correction, and updates
        ``p.data`` in-place.

        Args:
            closure (callable, optional): A closure that recomputes
                the loss and returns it.

        Returns:
            The loss returned by ``closure``, or ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            epsilon = group["epsilon"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["step"] = 0

                state["step"] += 1
                t = state["step"]
                m = state["m"]
                v = state["v"]

                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                # Bias correction, since m and v start at zero and are
                # biased towards zero during the first few steps.
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                update = lr * m_hat / (torch.sqrt(v_hat) + epsilon)
                p.data.sub_(update)

        return loss
