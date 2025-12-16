import torch
import torch.nn as nn
def train_one_round_buff(classifier, loader, device, optimizer, scheduler=None,
                         steps_per_round=200, max_grad_norm=1.0):
    """Online incremental training: run fixed number of SGD steps each round"""

    classifier.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    ### >>> UPDATED: remove optimizer redefinition – keep existing optimizer persistent
    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [...]
    # (removed)
    ### <<< UPDATED

    step = 0
    loader_iter = iter(loader)

    while step < steps_per_round:

        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = {k: v.to(device) for k, v in batch.items() if k != 'text'}

        logits = classifier(batch['encoding'])
        loss = criterion(logits, batch['label'])

        loss.backward()
        nn.utils.clip_grad_norm_(classifier.parameters(), max_grad_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        ### >>> UPDATED
        if scheduler is not None:
            scheduler.step()
        ### <<< UPDATED

        step += 1