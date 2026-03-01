import torch
import learned_simulator
def get_random_walk_noise_for_position_sequence(
        position_sequence: torch.tensor,
        noise_std_last_step):
  """Returns random-walk noise in the velocity applied to the position.
  Args: 
    position_sequence: A sequence of particle positions. Shape is
      (nparticles, 6, dim). Includes current + last 5 positions.
    noise_std_last_step: Standard deviation of noise in the last step.
  """
  velocity_sequence = learned_simulator.time_diff(position_sequence)
  num_velocities = velocity_sequence.shape[1]
  velocity_sequence_noise = torch.randn(
      list(velocity_sequence.shape)) * (noise_std_last_step/num_velocities**0.5)
  velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)
  position_sequence_noise = torch.cat([
      torch.zeros_like(velocity_sequence_noise[:, 0:1]),
      torch.cumsum(velocity_sequence_noise, dim=1)], dim=1)
  return position_sequence_noise
