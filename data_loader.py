import torch
import numpy as np


def load_npz_data(path):
    with np.load(path, allow_pickle=True) as data_file:
        if 'gns_data' in data_file:
            data = data_file['gns_data']
        else:
            data = [item for _, item in data_file.items()]
    return data


class SamplesDataset(torch.utils.data.Dataset):
    def __init__(self, path, input_length_sequence):
        super().__init__()
        self._data = load_npz_data(path)
        self._dimension = self._data[0][0].shape[-1]
        self._input_length_sequence = input_length_sequence
        self._material_property_as_feature = True if len(self._data[0]) >= 3 else False
        if self._material_property_as_feature:
            self._data_lengths = [x.shape[0] - self._input_length_sequence for x, _, _ in self._data]
        else:
            self._data_lengths = [x.shape[0] - self._input_length_sequence for x, _ in self._data]
        self._length = sum(self._data_lengths)
        self._precompute_cumlengths = [sum(self._data_lengths[:x]) for x in range(1, len(self._data_lengths) + 1)]
        self._precompute_cumlengths = np.array(self._precompute_cumlengths, dtype=int)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        trajectory_idx = np.searchsorted(self._precompute_cumlengths - 1, idx, side="left")
        start_of_selected_trajectory = self._precompute_cumlengths[trajectory_idx - 1] if trajectory_idx != 0 else 0
        time_idx = self._input_length_sequence + (idx - start_of_selected_trajectory)
        positions = self._data[trajectory_idx][0][time_idx - self._input_length_sequence:time_idx]
        positions = np.transpose(positions, (1, 0, 2))
        particle_type = np.full(positions.shape[0], self._data[trajectory_idx][1], dtype=int)
        n_particles_per_example = positions.shape[0]
        label = self._data[trajectory_idx][0][time_idx]
        if self._material_property_as_feature:
            material_property = np.full(positions.shape[0], self._data[trajectory_idx][2], dtype=float)
            training_example = ((positions, particle_type, material_property, n_particles_per_example), label)
        else:
            training_example = ((positions, particle_type, n_particles_per_example), label)
        return training_example


def collate_fn(data):
    material_property_as_feature = True if len(data[0][0]) >= 4 else False
    position_list = []
    particle_type_list = []
    if material_property_as_feature:
        material_property_list = []
    n_particles_per_example_list = []
    label_list = []

    if material_property_as_feature:
        for ((positions, particle_type, material_property, n_particles_per_example), label) in data:
            position_list.append(positions)
            particle_type_list.append(particle_type)
            material_property_list.append(material_property)
            n_particles_per_example_list.append(n_particles_per_example)
            label_list.append(label)
    else:
        for ((positions, particle_type, n_particles_per_example), label) in data:
            position_list.append(positions)
            particle_type_list.append(particle_type)
            n_particles_per_example_list.append(n_particles_per_example)
            label_list.append(label)

    if material_property_as_feature:
        collated_data = (
            (
                torch.tensor(np.vstack(position_list)).to(torch.float32).contiguous(),
                torch.tensor(np.concatenate(particle_type_list)).contiguous(),
                torch.tensor(np.concatenate(material_property_list)).to(torch.float32).contiguous(),
                torch.tensor(n_particles_per_example_list).contiguous(),
            ),
            torch.tensor(np.vstack(label_list)).to(torch.float32).contiguous()
        )
    else:
        collated_data = (
            (
                torch.tensor(np.vstack(position_list)).to(torch.float32).contiguous(),
                torch.tensor(np.concatenate(particle_type_list)).contiguous(),
                torch.tensor(n_particles_per_example_list).contiguous(),
            ),
            torch.tensor(np.vstack(label_list)).to(torch.float32).contiguous()
        )
    return collated_data


class TrajectoriesDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()
        self._data = load_npz_data(path)
        self._dimension = self._data[0][0].shape[-1]
        self._length = len(self._data)
        self._material_property_as_feature = True if len(self._data[0]) >= 3 else False

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if self._material_property_as_feature:
            positions, _particle_type, _material_property = self._data[idx]
            positions = np.transpose(positions, (1, 0, 2))
            particle_type = np.full(positions.shape[0], _particle_type, dtype=int)
            material_property = np.full(positions.shape[0], _material_property, dtype=float)
            n_particles_per_example = positions.shape[0]
            trajectory = (
                torch.tensor(positions).to(torch.float32).contiguous(),
                torch.tensor(particle_type).contiguous(),
                torch.tensor(material_property).to(torch.float32).contiguous(),
                n_particles_per_example
            )
        else:
            positions, _particle_type = self._data[idx]
            positions = np.transpose(positions, (1, 0, 2))
            particle_type = np.full(positions.shape[0], _particle_type, dtype=int)
            n_particles_per_example = positions.shape[0]
            trajectory = (
                torch.tensor(positions).to(torch.float32).contiguous(),
                torch.tensor(particle_type).contiguous(),
                n_particles_per_example
            )
        return trajectory


def get_data_loader_by_samples(path, input_length_sequence, batch_size, shuffle=True):
    dataset = SamplesDataset(path, input_length_sequence)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       pin_memory=True, collate_fn=collate_fn)


def get_data_loader_by_trajectories(path):
    dataset = TrajectoriesDataset(path)
    return torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False,
                                       pin_memory=True)
