from torch import optim


class SharedAdam(optim.adam):
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False):

        super(SharedAdam, self).__init__(params,
                                         lr=lr,
                                         betas=betas,
                                         eps=eps,
                                         weight_decay=weight_decay,
                                         amsgrad=amsgrad)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
