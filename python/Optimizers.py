import torch
class PAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, lambda_p=1e-2, p_norm=1, *args, **kwargs):
        super(PAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=0, *args, **kwargs)
        self.p_norm = p_norm
        self.lambda_p = lambda_p

    @torch.no_grad()
    def step(self, closure=None):
        # Store the old params
        old_params = []
        for group in self.param_groups:
            old_params.append({param: param.data.clone() for param in group['params'] if param.grad is not None})

        # Perform the standard Adam step
        loss = super(PAdam, self).step(closure)

        # Perform the PAdam step
        for group, old_group in zip(self.param_groups, old_params):
            for param in group['params']:
                if param.grad is None:
                    continue

                # Use old parameters in the decay factor
                param_old = old_group[param]
                X = param_old.abs()**(2 - self.p_norm)
                update_term = X / (X + self.p_norm * group['lr'] * self.lambda_p)

                # Update the parameters
                param.data.mul_(update_term)

        return loss



class AdamP(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 lambda_p=0, amsgrad=False, p_norm=2):
        # Initialize the parent class (Adam) with weight_decay set to 0
        super(AdamP, self).__init__(params, lr=lr, betas=betas, eps=eps,
                                     weight_decay=0, amsgrad=amsgrad)
        self.lambda_p = lambda_p
        self.p_norm = p_norm

    def step(self, closure=None):
        # Compute the loss using the parent class's step method if a closure is provided
        loss = None
        if closure is not None:
            loss = closure()

        # Apply custom Lp^p regularization to the gradients
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                # Apply the general Lp^p regularization
                if self.lambda_p != 0:
                    lp_grad = (param.data.abs()**(self.p_norm - 2)) * param.data
                    param.grad.data.add_(lp_grad, alpha=self.p_norm * self.lambda_p)

        # Perform the optimization step using the parent class's logic
        super(AdamP, self).step(closure)

        return loss
