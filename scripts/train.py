import argparse
import time
import datetime
import torch_ac
import tensorboardX
import sys
import multiprocessing
import utils
from utils import device
from model import ACModel
from gym_minigrid.wrappers_dh import FullyObsWrapper
import inspect
import warnings
warnings.filterwarnings("ignore", message="This code base is no longer maintained")
import os

parser = argparse.ArgumentParser()
# action option
parser.add_argument("--action-option", required=True,
                    help="action to use: vlm_action | train_ppo_from_vlm | ppo_baseline | a2c_baseline | train_a2c_from_vlm")
parser.add_argument("--save", default=True,
                    help="model save or not save")

# General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=16, 
                    help="random seed (default: 1)") 
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=1,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=3,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=20000, 
                    help="number of frames of training (default: 1e7)")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,  # None==128
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1, # 4 
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=True,
                    help="add a GRU to the model to handle text input")

if __name__ == "__main__":
    args = parser.parse_args()
    multiprocessing.set_start_method("spawn")
    args.mem = args.recurrence > 1

    # frame_per_progs
    if args.algo=="ppo":
        args.frames_per_proc=128
        print("frames_per_proc: 128 ")
    elif args.algo=="a2c":
        args.frames_per_proc=5
        print("frames_per_proc : 5")
    else:
        print("wrong algo")

    # Set run dir
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.action_option}_{args.env}_{args.algo}_seed{args.seed}_{date}_{args.frames}3"

    model_name = args.model or default_model_name

    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device
    txt_logger.info(f"Device: {device}\n")

    # Load environments
    envs = []
    for i in range(args.procs):
        #envs.append(utils.make_env(args.env, args.seed + 10000 * i)) 
        env = FullyObsWrapper(utils.make_env(args.env, args.seed + 10000 * i) , tile_size=32)
        envs.append(env)
    txt_logger.info("Environments loaded\n")

    # Load training status
    try:
        status = utils.get_status(model_dir)
    except OSError:
        if args.action_option=='vlm_action':
            print("Calculate VLM Success rate..")
            status = {"num_frames": 0, "update": 0}
        else:
            print(" train from 0 frame")
            status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")
    
    # Load model
    acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo
    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss,args.procs, args.action_option,args.env)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss,args.procs, args.action_option,args.env)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model
    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    total_success_rate=0
    #  vlm_action | train_ppo_from_vlm | ppo_baseline 
    if args.action_option!="vlm_action":
        while num_frames < args.frames:
            # Update model parameters
            update_start_time = time.time()
            exps, logs1,rgb_images ,missions= algo.collect_experiences()
            logs2 = algo.update_parameters(exps,rgb_images,missions)
            logs = {**logs1, **logs2}
            status['num_frames'] += logs["num_frames"]
            status['update'] += 1
            update_end_time = time.time()
            num_frames += logs["num_frames"]
            update += 1

            # Print logs
            if update % args.log_interval == 0:
                fps = logs["num_frames"] / (update_end_time - update_start_time)
                duration = int(time.time() - start_time)
                return_per_episode = utils.synthesize(logs["return_per_episode"])
                success_per_episode = utils.synthesize(
                    [1 if r > 0 else 0 for r in logs["return_per_episode"]])
                rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
                num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

                header = ["update", "frames", "FPS", "duration"]
                data = [update, num_frames, fps, duration]
                header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                data += rreturn_per_episode.values()

                header +=["success_rate"]
                data.append(success_per_episode['mean'])
            

                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm", "KL_div"]
                data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"], logs["KL_div"]]
                if args.action_option=="train_ppo_from_vlm":
                    print(f"ppo training from trained ppo seed {args.seed}")
                elif args.action_option=="train_a2c_from_vlm":
                    print(f"a2c training from trained a2c seed {args.seed}")
                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | S {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | KL {:.3f}"
                    .format(*data))

                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()

                if status["num_frames"] == 0:
                    csv_logger.writerow(header)
                csv_logger.writerow(data)
                csv_file.flush()

                for key, value in zip(header, data):
                    tb_writer.add_scalar(key, value, num_frames)

            # Save status
            if args.save==True and args.save_interval > 0 and update % args.save_interval == 0: 
                status = {"num_frames": num_frames, "update": update,
                        "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
                if hasattr(preprocess_obss, "vocab"):
                    status["vocab"] = preprocess_obss.vocab.vocab
                utils.save_status(status, model_dir)
                txt_logger.info("Status saved")
    
    # vlm_action 
    else:
        while num_frames < args.frames:
            # Update model parameters
            update_start_time = time.time()
            exps, logs = algo.collect_experiences()
            status['num_frames'] += logs["num_frames"]
            status['update'] += 1
            update_end_time = time.time()

            num_frames += logs["num_frames"]
            update += 1

            # Print logs
            if update % args.log_interval == 0:
                fps = logs["num_frames"] / (update_end_time - update_start_time)
                duration = int(time.time() - start_time)
                return_per_episode = utils.synthesize(logs["return_per_episode"])
                success_per_episode = utils.synthesize(
                    [1 if r > 0 else 0 for r in logs["return_per_episode"]])
                rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
                num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

                header = ["update", "frames", "FPS", "duration"]
                data = [update, num_frames, fps, duration]
                header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                data += rreturn_per_episode.values()

                header +=["success_rate"]
                data.append(success_per_episode['mean'])

                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
            
                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | S {:.2f} | F:μσmM {:.1f} {:.1f} {} {} "
                    .format(*data))

                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()

                if status["num_frames"] == 0:
                    csv_logger.writerow(header)
                csv_logger.writerow(data)
                csv_file.flush()

                for key, value in zip(header, data):
                    tb_writer.add_scalar(key, value, num_frames)

            # Save status
            if args.save==True and args.save_interval > 0 and update % args.save_interval == 0: 
                status = {"num_frames": num_frames, "update": update,
                        "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
                if hasattr(preprocess_obss, "vocab"):
                    status["vocab"] = preprocess_obss.vocab.vocab
                utils.save_status(status, model_dir)
                txt_logger.info("Status saved")
