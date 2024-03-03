#Reference:

#1. C.W. Park, M. Kornbluth, J. Vandermause, C. Wolverton, B. Kozinsky, J.P. Mailoa, Accurate and scalable graph neural network force field and molecular dynamics with direct force architecture, npj Comput. Mater. 7(1) (2021) 73. 
#This code is extended from https://github.com/ken2403/gnnff.git, which has the MIT License.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.


from schnetpack.train.hooks import Hook


__all__ = ["NaNStoppingHook"]


class NaNStopError(Exception):
    pass


class NaNStoppingHook(Hook):
    def on_batch_end(self, trainer, train_batch, result, loss):
        if loss.isnan().any():
            trainer._stop = True
            raise NaNStopError(
                "The value of training loss has become nan! Stop training."
            )
