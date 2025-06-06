import tensorflow as tf
import numpy as np

from src.client_attacks import Attack
from src.config_cli import get_config
from src.federated_averaging import FederatedAveraging
from src.tf_model import Model
from src.config.definitions import Config

import logging

logger = logging.getLogger(__name__)

def load_model():
    if config.environment.load_model is not None:
        model = tf.keras.models.load_model(config.environment.load_model) # Load with weights
    else:
        model = Model.create_model(
            config.client.model_name, config.server.intrinsic_dimension,
            config.client.model_weight_regularization, config.client.disable_bn)

    # save_model(model)

    return model

def save_model(model):
    weights = np.concatenate([x.flatten() for x in model.get_weights()])
    np.savetxt("resnet18_intrinsic_40k.txt", weights)

def main():
    import torch
    if torch.cuda.is_available():
        torch.cuda.current_device()
        limit_gpu_mem()

    # from src.torch_compat.anticipate_lenet import LeNet
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # m1 = LeNet(10)
    # m2 = m1.to(device)  # try

    models = [load_model()]

    if config.client.malicious is not None:
        config.client.malicious.attack_type = Attack.UNTARGETED.value \
                         if config.client.malicious.objective['name'] == "UntargetedAttack" else Attack.BACKDOOR.value

    server_model = FederatedAveraging(config, models, args.config_filepath)
    server_model.init()
    server_model.fit()

    return

    # if args.hyperparameter_tuning.lower() == "true":
    #     tune_hyper(args, config)
    # elif len(args.permute_dataset) > 0:
    #     # Permute, load single attack
    #     if not Model.model_supported(args.model_name, args.dataset):
    #         raise Exception(
    #             f'Model {args.model_name} does not support {args.dataset}! '
    #             f'Check method Model.model_supported for the valid combinations.')
    #
    #     attack = load_attacks()[0]
    #     amount_eval = 3
    #     amount_select = 80
    #     from itertools import combinations
    #     import random
    #     total_combinations = list(combinations(set(args.permute_dataset), amount_eval))
    #     indices = sorted(random.sample(range(len(total_combinations)), amount_select))
    #     logger.info(f"Running {len(total_combinations)} combinations!")
    #     for i, p in enumerate([total_combinations[i] for i in indices]):
    #         train = list(set(args.permute_dataset) - set(p))
    #         eval = list(p)
    #         attack['backdoor']['train'] = train
    #         attack['backdoor']['test'] = eval
    #         config['attack'] = attack
    #         config['attack_type'] = Attack.UNTARGETED.value \
    #             if attack['objective']['name'] == "UntargetedAttack" else Attack.BACKDOOR.value
    #
    #         logger.info(f"Running backdoor with samples {eval} {train}")
    #
    #         models = [load_model() for i in range(args.workers)]
    #
    #         server_model = FederatedAveraging(config, models, f"attack-{i}")
    #         server_model.init()
    #         server_model.fit()
    # else:
    #     if not Model.model_supported(args.model_name, args.dataset):
    #         raise Exception(
    #             f'Model {args.model_name} does not support {args.dataset}! '
    #             f'Check method Model.model_supported for the valid combinations.')
    #
    #     for i, attack in enumerate(load_attacks()):
    #         config['attack'] = attack
    #         config['attack_type'] = Attack.UNTARGETED.value \
    #             if attack['objective']['name'] == "UntargetedAttack" else Attack.BACKDOOR.value
    #
    #         logger.info(f"Running attack objective {config['attack_type']}"
    #                     f" (evasion: {attack['evasion']['name'] if 'evasion' in attack else None})")
    #
    #         models = [load_model() for i in range(args.workers)]
    #
    #         server_model = FederatedAveraging(config, models, f"attack-{i}")
    #         server_model.init()
    #         server_model.fit()

def limit_gpu_mem():
    limit_mb = config.environment.limit_tf_gpu_mem_mb
    if limit_mb is None:
        return
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit_mb)])  # Notice here
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


if __name__ == '__main__':
    config: Config
    config, args = get_config()
    np.random.seed(config.environment.seed)
    tf.random.set_seed(config.environment.seed)

    main()
