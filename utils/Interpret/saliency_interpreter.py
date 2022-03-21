import torch
from torch.nn.functional import softmax

import matplotlib
import matplotlib.pyplot as plt

SPECIAL_TOKENS = ['[CLS]', '[SEP]']


class SaliencyInterpreter:
    def __init__(self,
                 model,
                 tokenizer,
                 show_progress=True,
                 **kwargs):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.show_progress = show_progress
        self.kwargs = kwargs
        # to save outputs in saliency_interpret
        self.batch_output = None

    def _get_gradients(self, batch, use_truth):
        # set requires_grad to true for all parameters, but save original values to
        # restore them later
        original_param_name_to_requires_grad_dict = {}
        for param_name, param in self.model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
            param.requires_grad = True
        embedding_gradients = []
        hooks = self._register_embedding_gradient_hooks(embedding_gradients)

        self.model.zero_grad()

        # feed data and calculate gradient
        self.forward_step(batch, use_truth)

        for hook in hooks:
            hook.remove()

        # restore the original requires_grad values of the parameters
        for param_name, param in self.model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]

        return embedding_gradients[0]

    def _register_embedding_gradient_hooks(self, embedding_gradients):
        """
        Registers a backward hook on the
        Used to save the gradients of the embeddings for use in get_gradients()
        When there are multiple inputs (e.g., a passage and question), the hook
        will be called multiple times. We append all the embeddings gradients
        to a list.
        """

        def hook_layers(module, grad_in, grad_out):
            embedding_gradients.append(grad_out[0])

        backward_hooks = []
        encoder = self.kwargs.get("encoder")
        if encoder:
            embedding_layer = self.model.__getattr__(encoder).embeddings
        else:
            embedding_layer = self.model.utterance_encoder.embeddings
        backward_hooks.append(embedding_layer.register_backward_hook(hook_layers))
        return backward_hooks

    def forward_step(self, batch, use_truth):
        outputs = self.model(data=batch, evaluate_gradient=True, use_truth=use_truth)
        self.batch_output = [batch["context"], outputs['logits']]

    def update_output(self):

        input_ids, outputs, grads = self.batch_output

        probs = softmax(outputs, dim=-1)
        probs, labels = torch.max(probs, dim=-1)

        tokens = [
            self.tokenizer.convert_ids_to_tokens(input_ids_)
            for input_ids_ in input_ids
        ]

        embedding_grads = grads.sum(dim=2)
        # norm for each sequence
        norms = torch.norm(embedding_grads, dim=1, p=1)
        # normalizing
        for i, norm in enumerate(norms):
            embedding_grads[i] = torch.abs(embedding_grads[i]) / norm

        batch_output = []
        for example_tokens, example_prob, example_grad, example_label in zip(tokens,
                                                                             probs,
                                                                             embedding_grads,
                                                                             labels):
            example_dict = dict()
            # as we do it by batches we has a padding so we need to remove it
            example_tokens = [t for t in example_tokens if t != '[PAD]']
            example_dict['tokens'] = example_tokens
            example_dict['grad'] = example_grad.cpu().tolist()[:len(example_tokens)]
            example_dict['label'] = example_label.item()
            example_dict['prob'] = example_prob.item()
            batch_output.append(example_dict)
        return batch_output

