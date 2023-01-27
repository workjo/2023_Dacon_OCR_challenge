import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from config import now
import custom_test
from torch.utils.tensorboard import SummaryWriter


def train(model, optimizer, criterion, train_loader, val_loader, converter, CFG, device):
    model.to(device)

    best_accuracy = -1
    best_model = None
    loss_dict = {}
    loss_dict['train_loss_lst'] = []
    loss_dict['val_loss_lst'] = []
    loss_dict['accuracy'] = []
    writer = SummaryWriter(f'./saved_models/{CFG["exp_name"]}/')

    for epoch in range(1, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []
        for image_batch, labels in tqdm(iter(train_loader)):

            image_batch = image_batch.to(device)
            labels = list(labels)
            text, length = converter.encode(labels, batch_max_length=CFG['batch_max_length'])
            batch_size = image_batch.size(0)

            if 'CTC' in CFG['Prediction']:
                preds = model(image_batch, text)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.log_softmax(2).permute(1, 0, 2)
                loss = criterion(preds, text.cpu(), preds_size, length)

            else:
                preds = model(image_batch, text[:, :-1])  # align with Attention.forward
                target = text[:, 1:]  # without [GO] Symbol
                loss = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])  # gradient clipping with 5 (Default)
            optimizer.step()

            train_loss.append(loss.item())

        _train_loss = np.mean(train_loss)
        writer.add_scalar("Loss/train", _train_loss, epoch)

        with open(f'./saved_models/{CFG["exp_name"]}/log_train.txt', 'a') as log:
            _val_loss, current_accuracy, preds, labels, length_of_data = custom_test.validation(
                model, criterion, val_loader, converter, CFG, device)

            writer.add_scalar("Loss/valid", _val_loss, epoch)
            writer.add_scalar("acc/valid", current_accuracy, epoch)
            loss_dict['train_loss_lst'].append(_train_loss)
            loss_dict['val_loss_lst'].append(_val_loss)
            loss_dict['accuracy'].append(current_accuracy)
            print(f'Epoch : [{epoch}] Train CTC Loss : [{_train_loss:.5f}] Val CTC Loss : [{_val_loss:.5f}]')

            loss_log = f'[{epoch}/{CFG["EPOCHS"]}] Train loss: {_train_loss:0.5f}, Valid loss: {_val_loss:0.5f}'

            current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}'

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_model = model
                torch.save(model.state_dict(), f'./saved_models/{CFG["exp_name"]}/best_accuracy.pth')

            best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}'

            loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
            print(loss_model_log)
            log.write(loss_model_log + '\n')

            # show some predicted results
            dashed_line = '-' * 80
            head = f'{"Ground Truth":25s} | {"Prediction":25s} | T/F'
            predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
            for gt, pred in zip(labels[:5], preds[:5]):
                if 'Attn' in CFG['Prediction']:
                    gt = gt[:gt.find('[s]')]
                    pred = pred[:pred.find('[s]')]

                predicted_result_log += f'{gt:25s} | {pred:25s} | {str(pred == gt)}\n'
            predicted_result_log += f'{dashed_line}'
            print(predicted_result_log)
            log.write(predicted_result_log + '\n')

    writer.flush()
    writer.close()

    loss_df = pd.DataFrame(loss_dict)
    loss_df.to_csv('./loss/train_val_loss{}.csv'.format(now))

    return best_model
