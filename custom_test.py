import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F


def validation(model, criterion, val_loader, converter, CFG, device):
    model.eval()
    n_correct = 0
    length_of_data = 0
    val_loss = []
    with torch.no_grad():
        for image_batch, labels in tqdm(iter(val_loader)):
            batch_size = image_batch.size(0)
            image = image_batch.to(device)
            labels = list(labels)
            length_of_data = length_of_data + batch_size

            # For max length prediction
            length_for_pred = torch.IntTensor([CFG['batch_max_length']] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, CFG['batch_max_length'] + 1).fill_(0).to(device)

            text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=CFG['batch_max_length'])

            if 'CTC' in CFG['Prediction']:
                preds = model(image, text_for_pred)

                # Calculate evaluation loss for CTC deocder.
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                # permute 'preds' to use CTCloss format
                loss = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss.cpu(), preds_size, length_for_loss)

                # Select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(image, text_for_pred, is_train=False)

                preds = preds[:, :text_for_loss.shape[1] - 1, :]
                target = text_for_loss[:, 1:]  # without [GO] Symbol
                loss = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)
                labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

            val_loss.append(loss.item())

            # calculate accuracy & confidence score
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            confidence_score_list = []
            for gt, pred in zip(labels, preds_str):
                if 'Attn' in CFG['Prediction']:
                    gt = gt[:gt.find('[s]')]
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])

                if pred == gt:
                    n_correct += 1

    _val_loss = np.mean(val_loss)
    accuracy = n_correct / float(length_of_data) * 100

    return _val_loss, accuracy, preds_str, labels, length_of_data


def inference(model, test_loader, converter, CFG, device):
    model.eval()
    preds_lst = []
    with torch.no_grad():
        for image_batch in tqdm(iter(test_loader)):
            batch_size = image_batch.size(0)
            image_batch = image_batch.to(device)

            # For max length prediction
            length_for_pred = torch.IntTensor([CFG['batch_max_length']] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, CFG['batch_max_length'] + 1).fill_(0).to(device)

            if 'CTC' in CFG['Prediction']:
                preds = model(image_batch, text_for_pred)

                # Calculate evaluation loss for CTC deocder.
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)

                # Select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(image_batch, text_for_pred, is_train=False)

                preds = preds[:, :text_for_pred.shape[1], :]

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            preds_lst.extend(preds_str)

        for i, pred in enumerate(preds_lst):
            if 'Attn' in CFG['Prediction']:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                preds_lst[i] = pred

    return preds_lst