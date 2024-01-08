import torch


def load_pretrained_frontend(frontend, pretrained_path):
    print(f'LOADING PRETRAINED RESNET FRONTEND {pretrained_path}...')

    pm = torch.load(pretrained_path, map_location='cpu')

    def copy(p, v):
        p.data.copy_(v)

    # stem
    copy(frontend.frontend3D[0].weight, pm['encoder.frontend.frontend3D.0.weight'])
    for attr in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']: 
        copy(getattr(frontend.frontend3D[1], attr), pm[f'encoder.frontend.frontend3D.1.{attr}'])

    # trunks
    for i in range(1, 5):  # layer 
        for j in range(2):  # block
            # conv_2d_1
            copy(
                getattr(frontend.trunk, f'layer{i}')[j].conv1.weight,
                pm[f'encoder.frontend.trunk.layer{i}.{j}.conv1.weight']
            )

            # batch_norm_2d_1
            for attr in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                copy(
                    getattr(getattr(frontend.trunk, f'layer{i}')[j].bn1, attr),
                    pm[f'encoder.frontend.trunk.layer{i}.{j}.bn1.{attr}']
                )

            # conv_2d_2
            copy(
                getattr(frontend.trunk, f'layer{i}')[j].conv2.weight,
                pm[f'encoder.frontend.trunk.layer{i}.{j}.conv2.weight']
            )

            # batch_norm_2d_1
            for attr in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                copy(
                    getattr(getattr(frontend.trunk, f'layer{i}')[j].bn2, attr),
                    pm[f'encoder.frontend.trunk.layer{i}.{j}.bn2.{attr}']
                )

            # occurs in first block of layers 2-4
            is_downsample = i in [2, 3, 4] and j == 0
            if is_downsample: 
                copy(
                    getattr(frontend.trunk, f'layer{i}')[j].downsample[0].weight,
                    pm[f'encoder.frontend.trunk.layer{i}.{j}.downsample.0.weight']
                )
                for attr in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                    copy(
                        getattr(getattr(frontend.trunk, f'layer{i}')[j].downsample[1], attr),
                        pm[f'encoder.frontend.trunk.layer{i}.{j}.downsample.1.{attr}']
                    )

    return frontend
