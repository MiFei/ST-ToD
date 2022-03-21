import torch

from tqdm import tqdm

from utils.Interpret.saliency_interpreter import SaliencyInterpreter


class VanillaGradient(SaliencyInterpreter):
    """
    Interprets the prediction using vanilla gradient (no modifications)
    Registered as a `SaliencyInterpreter` with name "vanilla-gradient".
    """
    def __init__(self,
                 model,
                 tokenizer,
                 show_progress=True,
                 **kwargs):
        super().__init__(model, tokenizer, show_progress, **kwargs)

    def saliency_interpret(self, data, use_truth=True):

        instances_with_grads = []
        self.batch_output = []
        self._vanilla_grads(data, use_truth)
        batch_output = self.update_output()
        instances_with_grads.extend(batch_output)

        return instances_with_grads

    def _register_forward_hook(self):
        """
        Register a forward hook on the embedding layer.
        Used for one term in the SmoothGrad sum.
        """

        def forward_hook(module, inputs, output):
            pass

        # Register the hook
        encoder = self.kwargs.get("encoder")
        if encoder:
            embedding_layer = self.model.__getattr__(encoder).embeddings
        else:
            embedding_layer = self.model.utterance_encoder.embeddings
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle

    def _vanilla_grads(self, batch, use_truth):

        handle = self._register_forward_hook()
        grads = self._get_gradients(batch, use_truth)
        handle.remove()

        self.batch_output.append(grads)
