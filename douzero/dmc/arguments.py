import argparse

parser = argparse.ArgumentParser(description='DouZero: PyTorch DouDizhu AI')

# General Settings
parser.add_argument('--xpid', default='douzero',
                    help='Experiment id (default: douzero)')
parser.add_argument('--save_interval', default=30, type=int,
                    help='Time interval (in minutes) at which to save the model')    
parser.add_argument('--objective', default='adp', type=str, choices=['adp', 'wp', 'logadp'],
                    help='Use ADP or WP as reward (default: ADP)')    

# Training settings
parser.add_argument('--actor_device_cpu', action='store_true',
                    help='Use CPU as actor device')
parser.add_argument('--gpu_devices', default='0', type=str,
                    help='Which GPUs to be used for training')
parser.add_argument('--num_actor_devices', default=1, type=int,
                    help='The number of devices used for simulation')
parser.add_argument('--num_actors', default=5, type=int,
                    help='The number of actors for each simulation device')
parser.add_argument('--training_device', default='0', type=str,
                    help='The index of the GPU used for training models. `cpu` means using cpu')
parser.add_argument('--load_model', action='store_true',
                    help='Load an existing model')
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint')
parser.add_argument('--savedir', default='douzero_checkpoints',
                    help='Root dir where experiment data will be saved')
parser.add_argument('--model_type', default='lstm', type=str,
                    choices=['lstm', 'transformer'],
                    help='Model type: "lstm" for original LSTM, "transformer" for full transformer fusion')

# Hyperparameters
parser.add_argument('--total_frames', default=100000000000, type=int,
                    help='Total environment frames to train for')
parser.add_argument('--exp_epsilon', default=0.01, type=float,
                    help='The probability for exploration')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Learner batch size')
parser.add_argument('--unroll_length', default=100, type=int,
                    help='The unroll length (time dimension)')
parser.add_argument('--num_buffers', default=50, type=int,
                    help='Number of shared-memory buffers')
parser.add_argument('--num_threads', default=4, type=int,
                    help='Number learner threads')
parser.add_argument('--max_grad_norm', default=40, type=float,
                    help='Max norm of gradients')


# Optimizer settings
parser.add_argument('--optimizer', default='rmsprop', type=str, choices=['rmsprop', 'adam', 'adamw'],
                    help='Optimizer to use: rmsprop (default), adam, or adamw')
parser.add_argument('--learning_rate', default=0.0001, type=float,
                    help='Learning rate')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum')
parser.add_argument('--epsilon', default=1e-5, type=float,
                    help='RMSProp epsilon')
parser.add_argument('--weight_decay', default=0.01, type=float,
                    help='Weight decay (only applicable for AdamW)')

# Teacher related arguments
parser.add_argument('--teacher_mode', action='store_true',
                    help='Use baseline teacher models for knowledge distillation')
parser.add_argument('--teacher_landlord', type=str, default='baselines/douzero_ADP/landlord.ckpt',
                    help='Path to baseline teacher model for landlord')
parser.add_argument('--teacher_landlord_up', type=str, default='baselines/douzero_ADP/landlord_up.ckpt',
                    help='Path to baseline teacher model for landlord_up')
parser.add_argument('--teacher_landlord_down', type=str, default='baselines/douzero_ADP/landlord_down.ckpt',
                    help='Path to baseline teacher model for landlord_down')
parser.add_argument('--teacher_loss_weight', default=1.0, type=float,
                    help='Initial teacher loss weight')
parser.add_argument('--teacher_loss_final', default=0.0, type=float,
                    help='Final teacher loss weight after decay')
parser.add_argument('--teacher_loss_decay_steps', default=1e9, type=float,
                    help='Number of frames over which teacher loss weight decays linearly')
parser.add_argument('--teacher_temperature', type=int, default=2,
                    help='Knowledge Distillation temperature')



# python train.py --xpid transformer --num_actors 15 --model_type transformer --max_grad_norm 20 --optimizer adamw --teacher_mode --load_model
# python evaluate.py --landlord random --landlord_up baselines/transformer/landlord_up.ckpt --landlord_down baselines/transformer/landlord_down.ckpt
# python evaluate.py --landlord baselines/transformer/landlord.ckpt --landlord_up random --landlord_down random
# python evaluate.py --landlord baselines/douzero_WP/landlord.ckpt --landlord_up baselines/transformer/landlord_up.ckpt --landlord_down baselines/transformer/landlord_down.ckpt
# python evaluate.py --landlord baselines/sl/landlord.ckpt --landlord_up baselines/transformer/landlord_up.ckpt --landlord_down baselines/transformer/landlord_down.ckpt