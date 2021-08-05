from torch import optim
import torch


class SharedRMSProp(optim.RMSprop):
    def __init__(self,
                 params,
                 lr=1e-2,
                 alpha=0.99,
                 eps=1e-8,
                 weight_decay=0,
                 momentum=0,
                 centered=False):

        super(SharedRMSProp, self).__init__(params,
                                            lr=lr,
                                            alpha=alpha,
                                            eps=eps,
                                            weight_decay=weight_decay,
                                            momentum=momentum,
                                            centered=centered)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)[0]
                state['square_avg'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['square_avg'].share_memory_()
