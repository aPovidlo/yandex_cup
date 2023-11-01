import argparse
import os

import h5py
import lightning as L
import numpy as np
import torch.utils.data as data
import torch
import tqdm

from data.datasets import RadarDataset
from models.baseline_models import ConvLSTMModel, PersistantModel
from src.models.IAM4VP import ImplicitStackedAutoregressiveForVideoPrediction
from src.models.SimVP import SimVPModel

torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')


def prepare_data_loaders(data_name='intensity', train_batch_size=1, valid_batch_size=1, test_batch_size=1):
    path_to_train_data = '../data/raw/train'
    path_to_test_data = '../data/raw/test'

    train_dataset = RadarDataset(
        data=data_name,
        list_of_files=[f'{path_to_train_data}/2021-01-train.hdf5', f'{path_to_train_data}/2021-03-train.hdf5',
                       # f'{path_to_train_data}/2021-04-train.hdf5', f'{path_to_train_data}/2021-06-train.hdf5',
                       # f'{path_to_train_data}/2021-07-train.hdf5', f'{path_to_train_data}/2021-09-train.hdf5',
                       # f'{path_to_train_data}/2021-10-train.hdf5', f'{path_to_train_data}/2021-12-train.hdf5'
        ],
        with_time=False

    )

    valid_dataset = RadarDataset(
        data=data_name,
        list_of_files=[f'{path_to_train_data}/2021-02-train.hdf5', f'{path_to_train_data}/2021-05-train.hdf5',
                       # f'{path_to_train_data}/2021-08-train.hdf5', f'{path_to_train_data}/2021-11-train.hdf5'
        ],
        with_time=False
    )

    test_dataset = RadarDataset(data=data_name, list_of_files=[f'{path_to_test_data}/2022-test-public.hdf5'],
                                out_seq_len=0, with_time=True
    )

    train_loader = data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=11,
                                   persistent_workers=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=11,
                                   persistent_workers=True)
    test_loader = data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=11,
                                  persistent_workers=True)

    return train_loader, valid_loader, test_loader


def evaluate_on_val(model, valid_loader):
    rmses = np.zeros((12,), dtype=float)

    for item in tqdm.tqdm(valid_loader):
        inputs, target = item
        output = model(inputs)
        rmses += np.sum((
            np.square(target.detach().cpu().numpy() - output.detach().cpu().numpy()))
            * (target.detach().cpu().numpy() != -1), axis=(0, 2, 3, 4))

    rmses /= len(valid_loader)

    return np.mean(np.sqrt(rmses))


def process_test(model, test_loader, output_file='../data/processed/output.hdf5'):
    model.eval()

    for index, item in tqdm.tqdm(enumerate(test_loader)):
        (inputs, last_input_timestamp), _ = item
        output = model(inputs)

        with h5py.File(output_file, mode='a') as f_out:
            for index in range(output.shape[1]):
                timestamp_out = str(int(last_input_timestamp[-1]) + 600 * (index + 1))
                f_out.create_group(timestamp_out)
                f_out[timestamp_out].create_dataset(
                    'intensity',
                    data=output[0, index, 0].detach().cpu().numpy()
                )


def main(model_name, dataset_name, tensorboard_path, chk_path=None):
    train_loader, valid_loader, test_loader = prepare_data_loaders(dataset_name)

    if model_name == 'persistant':
        # score on valid set: 197.64139689523992
        # score on test set: 283.66210850104176
        model = PersistantModel()
        print(evaluate_on_val(model, valid_loader))
        process_test(model, test_loader)

    elif model_name == 'convlstm':
        # score on valid set: 144
        # score on test set: ~172
        if chk_path is not None and os.path.isfile(chk_path):
            print('Model loaded from the checkpoint')
            model = ConvLSTMModel.load_from_checkpoint(chk_path)
            model.eval()
        else:
            model = ConvLSTMModel()

            trainer = L.Trainer(
                logger=L.pytorch.loggers.TensorBoardLogger(save_dir=tensorboard_path),
                accelerator='gpu',
                max_epochs=5,
                precision="bf16",
                enable_checkpointing=True
            )

            trainer.fit(model, train_loader)

        print(evaluate_on_val(model, valid_loader))
        process_test(model, test_loader, output_file=f'../data/processed/{data}_output.hdf5')

    elif model_name == 'simvp':
        if chk_path is not None and os.path.isfile(chk_path):
            print('Model loaded from the checkpoint')
            model = SimVPModel.load_from_checkpoint(chk_path)
            model.eval()
        else:
            model = SimVPModel()
            trainer = L.Trainer(
                logger=L.pytorch.loggers.TensorBoardLogger(save_dir=tensorboard_path),
                accelerator='gpu',
                gpus=-1,
                max_epochs=1,
                precision="bf16",
                enable_checkpointing=True
            )

            trainer.fit(model, train_loader)

        print(evaluate_on_val(model, valid_loader))
        process_test(model, test_loader)

    elif model_name == 'iam4vp':
        # TODO: Required to mod for this problem
        # raise NotImplementedError()

        # score on valid set:
        # score on test set:
        if chk_path is not None and os.path.isfile(chk_path):
            print('Model loaded from the checkpoint')
            model = ImplicitStackedAutoregressiveForVideoPrediction.load_from_checkpoint(chk_path)
            model.eval()
        else:
            model = ImplicitStackedAutoregressiveForVideoPrediction()
            trainer = L.Trainer(
                logger=L.pytorch.loggers.TensorBoardLogger(save_dir=tensorboard_path),
                accelerator='cpu',
                max_epochs=1,
                precision="bf16",
                enable_checkpointing=True
            )

            trainer.fit(model, train_loader)

        print(evaluate_on_val(model, valid_loader))
        process_test(model, test_loader)

    else:
        print('Unknown model name')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='simvp')
    parser.add_argument('--dataset', default='events')
    parser.add_argument('--tensorboard_path', default='../reports/tensorboard')
    parser.add_argument('--chk_path',
                        # default='../reports/tensorboard/lightning_logs/version_0/checkpoints/epoch=4-step=85275.ckpt'
    )
    args = parser.parse_args()
    main(args.model, args.dataset, args.tensorboard_path, args.chk_path)
