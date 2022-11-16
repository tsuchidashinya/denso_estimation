def print_current_losses(self, phase, epoch, i, losses):
    """ prints train loss on terminal / file """
    message = '(phase: %s, epoch: %d, iters: %d) loss: %.3f ' %(phase, epoch, i, losses)
    print(message)
    