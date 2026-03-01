import collections
import json
import os
import pickle
import glob
import re
import sys

import numpy as np
import torch
from tqdm import tqdm

from absl import flags
from absl import app

sys.path.append(os.path.expanduser("~/Desktop/DEM-ML"))
import learned_simulator
import noise_utils
import reading_utils
import data_loader

flags.DEFINE_enum(
    'mode', 'train', ['train', 'valid', 'rollout'],
    help='Train model, validation or rollout evaluation.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_string('model_path', 'models/', help='The path for saving checkpoints of the model.')
flags.DEFINE_string('output_path', 'rollouts/', help='The path for saving outputs.')
flags.DEFINE_string('output_filename', 'rollout', help='Base name for saving the rollout')
flags.DEFINE_string('model_file', None, help='Model filename (.pt) to resume from.')
flags.DEFINE_string('train_state_file', 'train_state.pt', help='Train state filename (.pt) to resume from.')
flags.DEFINE_integer('ntraining_steps', int(2E7), help='Number of training steps.')
flags.DEFINE_integer('validation_interval', None, help='Validation interval.')
flags.DEFINE_integer('nsave_steps', int(5000), help='Number of steps at which to save the model.')
flags.DEFINE_float('lr_init', 1e-4, help='Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', int(5e6), help='Learning rate decay steps.')
flags.DEFINE_integer("cuda_device_number", None, help="CUDA device number.")
flags.DEFINE_integer("n_gpus", 1, help="Number of GPUs.")

FLAGS = flags.FLAGS
Stats = collections.namedtuple('Stats', ['mean', 'std'])
INPUT_SEQUENCE_LENGTH = 6
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3


def rollout(simulator, position, particle_types, material_property,
            n_particles_per_example, nsteps, device):
  initial_positions = position[:, :INPUT_SEQUENCE_LENGTH]
  ground_truth_positions = position[:, INPUT_SEQUENCE_LENGTH:]
  current_positions = initial_positions
  predictions = []
  for step in tqdm(range(nsteps), total=nsteps):
    next_position = simulator.predict_positions(
        current_positions,
        nparticles_per_example=[n_particles_per_example],
        particle_types=particle_types,
        material_property=material_property)
    kinematic_mask = (particle_types == KINEMATIC_PARTICLE_ID).clone().detach().to(device)
    next_position_ground_truth = ground_truth_positions[:, step]
    kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, current_positions.shape[-1])
    next_position = torch.where(kinematic_mask, next_position_ground_truth, next_position)
    predictions.append(next_position)
    current_positions = torch.cat(
        [current_positions[:, 1:], next_position[:, None, :]], dim=1)
  predictions = torch.stack(predictions)
  ground_truth_positions = ground_truth_positions.permute(1, 0, 2)
  loss = (predictions - ground_truth_positions) ** 2
  output_dict = {
      'initial_positions': initial_positions.permute(1, 0, 2).cpu().numpy(),
      'predicted_rollout': predictions.cpu().numpy(),
      'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
      'particle_types': particle_types.cpu().numpy(),
      'material_property': material_property.cpu().numpy() if material_property is not None else None
  }
  return output_dict, loss


def predict(device):
  metadata = reading_utils.read_metadata(FLAGS.data_path, "rollout")
  simulator = _get_simulator(metadata, FLAGS.noise_std, FLAGS.noise_std, device)
  if os.path.exists(FLAGS.model_path + FLAGS.model_file):
    simulator.load(FLAGS.model_path + FLAGS.model_file)
  else:
    raise Exception(f"Model does not exist at {FLAGS.model_path + FLAGS.model_file}")
  simulator.to(device)
  simulator.eval()
  if not os.path.exists(FLAGS.output_path):
    os.makedirs(FLAGS.output_path)
  split = 'test' if FLAGS.mode == 'rollout' else 'valid'
  ds = data_loader.get_data_loader_by_trajectories(path=f"{FLAGS.data_path}{split}.npz")
  material_property_as_feature = len(ds.dataset._data[0]) == 3
  eval_loss = []
  with torch.no_grad():
    for example_i, features in enumerate(ds):
      print(f"Processing example {example_i}")
      positions = features[0].to(device)
      if metadata['sequence_length'] is not None:
        nsteps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
      else:
        nsteps = positions.shape[1] - INPUT_SEQUENCE_LENGTH
      particle_type = features[1].to(device)
      if material_property_as_feature:
        material_property = features[2].to(device)
        n_particles_per_example = torch.tensor([int(features[3])], dtype=torch.int32).to(device)
      else:
        material_property = None
        n_particles_per_example = torch.tensor([int(features[2])], dtype=torch.int32).to(device)
      example_rollout, loss = rollout(simulator, positions, particle_type,
                                      material_property, n_particles_per_example,
                                      nsteps, device)
      example_rollout['metadata'] = metadata
      print(f"Example {example_i} loss: {loss.mean()}")
      eval_loss.append(torch.flatten(loss))
      if FLAGS.mode == 'rollout':
        example_rollout['loss'] = loss.mean()
        filename = f'{FLAGS.output_filename}_ex{example_i}.pkl'
        filename = os.path.join(FLAGS.output_path, filename)
        with open(filename, 'wb') as f:
          pickle.dump(example_rollout, f)
  print(f"Mean rollout loss: {torch.mean(torch.cat(eval_loss))}")


def optimizer_to(optim, device):
  for param in optim.state.values():
    if isinstance(param, torch.Tensor):
      param.data = param.data.to(device)
      if param._grad is not None:
        param._grad.data = param._grad.data.to(device)
    elif isinstance(param, dict):
      for subparam in param.values():
        if isinstance(subparam, torch.Tensor):
          subparam.data = subparam.data.to(device)
          if subparam._grad is not None:
            subparam._grad.data = subparam._grad.data.to(device)


def acceleration_loss(pred_acc, target_acc, non_kinematic_mask):
  loss = (pred_acc - target_acc) ** 2
  loss = loss.sum(dim=-1)
  num_non_kinematic = non_kinematic_mask.sum()
  loss = torch.where(non_kinematic_mask.bool(), loss, torch.zeros_like(loss))
  loss = loss.sum() / num_non_kinematic
  return loss


def save_model_and_train_state(device, simulator, flags, step, epoch, optimizer,
                                train_loss, valid_loss, train_loss_hist, valid_loss_hist):
  simulator.save(flags["model_path"] + 'model-' + str(step) + '.pt')
  train_state = dict(
      optimizer_state=optimizer.state_dict(),
      global_train_state={"step": step, "epoch": epoch,
                          "train_loss": train_loss, "valid_loss": valid_loss},
      loss_history={"train": train_loss_hist, "valid": valid_loss_hist})
  torch.save(train_state, f'{flags["model_path"]}train_state-{step}.pt')


def train(flags, device):
  metadata = reading_utils.read_metadata(flags["data_path"], "train")
  simulator = _get_simulator(metadata, flags["noise_std"], flags["noise_std"], device)
  optimizer = torch.optim.Adam(simulator.parameters(), lr=flags["lr_init"])
  scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

  step = 0
  epoch = 0
  steps_per_epoch = 0
  valid_loss = None
  epoch_train_loss = 0
  train_loss_hist = []
  valid_loss_hist = []
  train_loss = 0

  if flags["model_file"] is not None:
    if flags["model_file"] == "latest":
      fnames = glob.glob(f'{flags["model_path"]}*model*pt')
      max_model_number = 0
      expr = re.compile(r".*model-(\d+).pt")
      for fname in fnames:
        model_num = int(expr.search(fname).groups()[0])
        if model_num > max_model_number:
          max_model_number = model_num
      flags["model_file"] = f"model-{max_model_number}.pt"
      flags["train_state_file"] = f"train_state-{max_model_number}.pt"
    if os.path.exists(flags["model_path"] + flags["model_file"]):
      simulator.load(flags["model_path"] + flags["model_file"])
      train_state = torch.load(flags["model_path"] + flags["train_state_file"],
                               map_location=device, weights_only=False)
      optimizer.load_state_dict(train_state["optimizer_state"])
      optimizer_to(optimizer, device)
      step = train_state["global_train_state"]["step"]
      epoch = train_state["global_train_state"]["epoch"]
      train_loss_hist = train_state["loss_history"]["train"]
      valid_loss_hist = train_state["loss_history"]["valid"]

  simulator.train()
  simulator.to(device)

  dl = data_loader.get_data_loader_by_samples(
      path=f'{flags["data_path"]}train.npz',
      input_length_sequence=INPUT_SEQUENCE_LENGTH,
      batch_size=flags["batch_size"])
  n_features = len(dl.dataset._data[0])

  if flags["validation_interval"] is not None:
    dl_valid = data_loader.get_data_loader_by_samples(
        path=f'{flags["data_path"]}valid.npz',
        input_length_sequence=INPUT_SEQUENCE_LENGTH,
        batch_size=flags["batch_size"])

  try:
    while step < flags["ntraining_steps"]:
      for example in dl:
        steps_per_epoch += 1
        position = example[0][0].to(device)
        particle_type = example[0][1].to(device)
        if n_features == 3:
          material_property = example[0][2].to(device)
          n_particles_per_example = example[0][3].to(device)
        else:
          material_property = None
          n_particles_per_example = example[0][2].to(device)
        labels = example[1].to(device)

        sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
            position, noise_std_last_step=flags["noise_std"]).to(device)
        non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).clone().detach().to(device)
        sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
          pred_acc, target_acc = simulator.predict_accelerations(
              next_positions=labels,
              position_sequence_noise=sampled_noise,
              position_sequence=position,
              nparticles_per_example=n_particles_per_example,
              particle_types=particle_type,
              material_property=material_property if n_features == 3 else None)
          loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask)

        train_loss = loss.item()
        epoch_train_loss += train_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        lr_new = flags["lr_init"] * (flags["lr_decay"] ** (step / flags["lr_decay_steps"]))
        for param in optimizer.param_groups:
          param['lr'] = lr_new

        print(f"epoch={epoch}, step={step}/{flags['ntraining_steps']}, loss={train_loss:.6f}", flush=True)

        if step % flags["nsave_steps"] == 0:
          save_model_and_train_state(device, simulator, flags, step, epoch,
                                     optimizer, train_loss, valid_loss,
                                     train_loss_hist, valid_loss_hist)
        step += 1
        if step >= flags["ntraining_steps"]:
          break

      epoch_train_loss /= max(steps_per_epoch, 1)
      train_loss_hist.append((epoch, epoch_train_loss))
      print(f"Epoch {epoch} train loss: {epoch_train_loss:.6f}")
      epoch_train_loss = 0
      epoch += 1
      steps_per_epoch = 0
      if step >= flags["ntraining_steps"]:
        break

  except KeyboardInterrupt:
    pass

  save_model_and_train_state(device, simulator, flags, step, epoch, optimizer,
                             train_loss, valid_loss, train_loss_hist, valid_loss_hist)


def _get_simulator(metadata, acc_noise_std, vel_noise_std, device):
  normalization_stats = {
      'acceleration': {
          'mean': torch.FloatTensor(metadata['acc_mean']).to(device),
          'std': torch.sqrt(torch.FloatTensor(metadata['acc_std'])**2 +
                            acc_noise_std**2).to(device),
      },
      'velocity': {
          'mean': torch.FloatTensor(metadata['vel_mean']).to(device),
          'std': torch.sqrt(torch.FloatTensor(metadata['vel_std'])**2 +
                            vel_noise_std**2).to(device),
      },
  }
  nnode_in = 37 if metadata['dim'] == 3 else 30
  nedge_in = metadata['dim'] + 1
  simulator = learned_simulator.LearnedSimulator(
      particle_dimensions=metadata['dim'],
      nnode_in=nnode_in,
      nedge_in=nedge_in,
      latent_dim=128,
      nmessage_passing_steps=10,
      nmlp_layers=2,
      mlp_hidden_dim=128,
      connectivity_radius=metadata['default_connectivity_radius'],
      boundaries=np.array(metadata['bounds']),
      normalization_stats=normalization_stats,
      nparticle_types=NUM_PARTICLE_TYPES,
      particle_type_embedding_size=16,
      boundary_clamp_limit=metadata.get("boundary_augment", 1.0),
      device=device)
  return simulator


def validation(simulator, example, n_features, flags, device):
  position = example[0][0].to(device)
  particle_type = example[0][1].to(device)
  if n_features == 3:
    material_property = example[0][2].to(device)
    n_particles_per_example = example[0][3].to(device)
  else:
    material_property = None
    n_particles_per_example = example[0][2].to(device)
  labels = example[1].to(device)
  sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
      position, noise_std_last_step=flags["noise_std"]).to(device)
  non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).clone().detach().to(device)
  sampled_noise *= non_kinematic_mask.view(-1, 1, 1)
  with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
      pred_acc, target_acc = simulator.predict_accelerations(
          next_positions=labels,
          position_sequence_noise=sampled_noise,
          position_sequence=position,
          nparticles_per_example=n_particles_per_example,
          particle_types=particle_type,
          material_property=material_property if n_features == 3 else None)
  return acceleration_loss(pred_acc, target_acc, non_kinematic_mask)


def main(_):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device: {device}")
  myflags = reading_utils.flags_to_dict(FLAGS)
  if FLAGS.mode == 'train':
    if not os.path.exists(FLAGS.model_path):
      os.makedirs(FLAGS.model_path)
    train(myflags, device)
  elif FLAGS.mode in ['valid', 'rollout']:
    predict(device)


if __name__ == '__main__':
  app.run(main)
