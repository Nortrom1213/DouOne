# New Features

We have extended the original DouZero codebase with several new features to provide greater flexibility and improved performance:

1. **`--model_type` Argument**  
   - **Options:** `lstm` (default) or `transformer`.  
   - **Usage Example:**  
     ```bash
     python3 train.py --model_type transformer
     ```
   - **Description:**  
     - `lstm`: Uses the original LSTM-based network from DouZero.  
     - `transformer`: Replaces LSTM with a 2-layer Transformer encoder for processing historical states, potentially offering better handling of long-range dependencies.

2. **Optimizers**  
   - **New Argument:** `--optimizer` with choices `rmsprop`, `adam`, and `adamw`.  
   - **Usage Example:**  
     ```bash
     python3 train.py --optimizer adamw --weight_decay 0.01
     ```
   - **Description:**  
     - By default, DouZero uses `rmsprop`. You can now opt for `adam` or `adamw`. If `adamw` is selected, you can adjust `--weight_decay` for L2 regularization.

3. **Knowledge Distillation (Teacher Models)**  
   - **Arguments:**
     - `--teacher_mode`: Enables knowledge distillation during training.
     - `--teacher_landlord`, `--teacher_landlord_up`, `--teacher_landlord_down`: Paths to teacher checkpoints for each position.
     - `--teacher_loss_weight`: Initial distillation loss weight.
     - `--teacher_loss_final`: Final distillation loss weight after linear decay.
     - `--teacher_loss_decay_steps`: Number of frames over which teacher loss decays.
     - `--teacher_temperature`: Temperature parameter for softening teacher logits.
   - **Usage Example:**  
     ```bash
     python3 train.py --teacher_mode \
       --teacher_landlord baselines/douzero_ADP/landlord.ckpt \
       --teacher_landlord_up baselines/douzero_ADP/landlord_up.ckpt \
       --teacher_landlord_down baselines/douzero_ADP/landlord_down.ckpt \
       --teacher_loss_weight 1.0 \
       --teacher_loss_final 0.1 \
       --teacher_loss_decay_steps 50000000
     ```
   - **Description:**  
     - Allows you to train a student model using teacher signals from pretrained DouZero-based models.  
     - The distillation loss weight decreases over time, so the student eventually learns to surpass the teacher’s strategies.

---

Pretrained Transformer Baseline: [https://drive.google.com/file/d/1T8tpnZy4DllAqKhd7LIjcZWUDoTqfssw/view?usp=drive_link](https://drive.google.com/file/d/1DpQMcs0y6P7shjGYDULqyTgLHkH57cf0/view?usp=drive_link)

## Installation
The training code is designed for GPUs. Thus, you need to first install CUDA if you want to train models. You may refer to [this guide](https://docs.nvidia.com/cuda/index.html#installation-guides). For evaluation, CUDA is optional and you can use CPU for evaluation.

First, clone the repo with (if you are in China and Github is slow, you can use the mirror in [Gitee](https://gitee.com/daochenzha/DouZero)):
```
git clone https://github.com/kwai/DouZero.git
```
Make sure you have python 3.6+ installed. Install dependencies:
```
cd douzero
pip3 install -r requirements.txt
```
We recommend installing the stable version of DouZero with
```
pip3 install douzero
```
If you are in China and the above command is too slow, you can use the mirror provided by Tsinghua University:
```
pip3 install douzero -i https://pypi.tuna.tsinghua.edu.cn/simple
```
or install the up-to-date version (it could be not stable) with
```
pip3 install -e .
```
Note that Windows users can only use CPU as actors. See [Issues in Windows](README.md#issues-in-windows) about why GPUs are not supported. Nonetheless, Windows users can still [run the demo locally](https://github.com/datamllab/rlcard-showdown).  

## Training
To use GPU for training, run
```
python3 train.py
```
This will train DouZero on one GPU. To train DouZero on multiple GPUs. Use the following arguments.
*   `--gpu_devices`: what gpu devices are visible
*   `--num_actor_devices`: how many of the GPU deveices will be used for simulation, i.e., self-play
*   `--num_actors`: how many actor processes will be used for each device
*   `--training_device`: which device will be used for training DouZero

For example, if we have 4 GPUs, where we want to use the first 3 GPUs to have 15 actors each for simulating and the 4th GPU for training, we can run the following command:
```
python3 train.py --gpu_devices 0,1,2,3 --num_actor_devices 3 --num_actors 15 --training_device 3
```
To use CPU training or simulation (Windows can only use CPU for actors), use the following arguments:
*   `--training_device cpu`: Use CPU to train the model
*   `--actor_device_cpu`: Use CPU as actors

For example, use the following command to run everything on CPU:
```
python3 train.py --actor_device_cpu --training_device cpu
```
The following command only runs actors on CPU:
```
python3 train.py --actor_device_cpu
```
For more customized configuration of training, see the following optional arguments:
```
--xpid XPID           Experiment id (default: douzero)
--save_interval SAVE_INTERVAL
                      Time interval (in minutes) at which to save the model
--objective {adp,wp}  Use ADP or WP as reward (default: ADP)
--actor_device_cpu    Use CPU as actor device
--gpu_devices GPU_DEVICES
                      Which GPUs to be used for training
--num_actor_devices NUM_ACTOR_DEVICES
                      The number of devices used for simulation
--num_actors NUM_ACTORS
                      The number of actors for each simulation device
--training_device TRAINING_DEVICE
                      The index of the GPU used for training models. `cpu`
                	  means using cpu
--load_model          Load an existing model
--disable_checkpoint  Disable saving checkpoint
--savedir SAVEDIR     Root dir where experiment data will be saved
--total_frames TOTAL_FRAMES
                      Total environment frames to train for
--exp_epsilon EXP_EPSILON
                      The probability for exploration
--batch_size BATCH_SIZE
                      Learner batch size
--unroll_length UNROLL_LENGTH
                      The unroll length (time dimension)
--num_buffers NUM_BUFFERS
                      Number of shared-memory buffers
--num_threads NUM_THREADS
                      Number learner threads
--max_grad_norm MAX_GRAD_NORM
                      Max norm of gradients
--learning_rate LEARNING_RATE
                      Learning rate
--alpha ALPHA         RMSProp smoothing constant
--momentum MOMENTUM   RMSProp momentum
--epsilon EPSILON     RMSProp epsilon
```

## Evaluation
The evaluation can be performed with GPU or CPU (GPU will be much faster). Pretrained model is available at [Google Drive](https://drive.google.com/drive/folders/1NmM2cXnI5CIWHaLJeoDZMiwt6lOTV_UB?usp=sharing) or [百度网盘](https://pan.baidu.com/s/18g-JUKad6D8rmBONXUDuOQ), 提取码: 4624. Put pre-trained weights in `baselines/`. The performance is evaluated through self-play. We have provided pre-trained models and some heuristics as baselines:
*   [random](douzero/evaluation/random_agent.py): agents that play randomly (uniformly)
*   [rlcard](douzero/evaluation/rlcard_agent.py): the rule-based agent in [RLCard](https://github.com/datamllab/rlcard)
*   SL (`baselines/sl/`): the pre-trained deep agents on human data
*   DouZero-ADP (`baselines/douzero_ADP/`): the pretrained DouZero agents with Average Difference Points (ADP) as objective
*   DouZero-WP (`baselines/douzero_WP/`): the pretrained DouZero agents with Winning Percentage (WP) as objective

### Step 1: Generate evaluation data
```
python3 generate_eval_data.py
```
Some important hyperparameters are as follows.
*   `--output`: where the pickled data will be saved
*   `--num_games`: how many random games will be generated, default 10000

### Step 2: Self-Play
```
python3 evaluate.py
```
Some important hyperparameters are as follows.
*   `--landlord`: which agent will play as Landlord, which can be random, rlcard, or the path of the pre-trained model
*   `--landlord_up`: which agent will play as LandlordUp (the one plays before the Landlord), which can be random, rlcard, or the path of the pre-trained model
*   `--landlord_down`: which agent will play as LandlordDown (the one plays after the Landlord), which can be random, rlcard, or the path of the pre-trained model
*   `--eval_data`: the pickle file that contains evaluation data
*   `--num_workers`: how many subprocesses will be used
*   `--gpu_device`: which GPU to use. It will use CPU by default

For example, the following command evaluates DouZero-ADP in Landlord position against random agents
```
python3 evaluate.py --landlord baselines/douzero_ADP/landlord.ckpt --landlord_up random --landlord_down random
```
The following command evaluates DouZero-ADP in Peasants position against RLCard agents
```
python3 evaluate.py --landlord rlcard --landlord_up baselines/douzero_ADP/landlord_up.ckpt --landlord_down baselines/douzero_ADP/landlord_down.ckpt
```
By default, our model will be saved in `douzero_checkpoints/douzero` every half an hour. We provide a script to help you identify the most recent checkpoint. Run
```
sh get_most_recent.sh douzero_checkpoints/douzero/
```
The most recent model will be in `most_recent_model`.

## Issues in Windows
You may encounter `operation not supported` error if you use a Windows system to train with GPU as actors. This is because doing multiprocessing on CUDA tensors is not supported in Windows. However, our code extensively operates on the CUDA tensors since the code is optimized for GPUs. Please contact us if you find any solutions!

## Acknowlegements
*   The demo is largely based on [RLCard-Showdown](https://github.com/datamllab/rlcard-showdown)
*   Code implementation is inspired by [TorchBeast](https://github.com/facebookresearch/torchbeast)
*   Code implementation is based on [DouZero](https://github.com/kwai/DouZero)
