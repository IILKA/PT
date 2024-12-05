from torch.utils.data import Dataset, DataLoader
import torch
import os 
import numpy as np 
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EHRDataset(Dataset):
    def __init__(self, feat_dict, ids, labels=None, mean_feature=None, noise_std=-1, device=None, K_feature = 100):
        """
        Args:
            feat_dict (dict): feature dictionary, key is ID and value is feature vector.
            ids (list): ID list. admission ID 
            labels (list, optional): label list.
            mean_feature (numpy.ndarray, optional): mean feature vector, used for missing feature imputation.
            noise_std (float): set -1 to disable noise, otherwise add Gaussian noise to features.
            device (torch.device, optional): device to store features.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feat_dict = feat_dict
        self.ids = ids
        self.labels = labels
        self.mean_feature = mean_feature if mean_feature is not None else np.zeros(K_feature)
        self.noise_std = noise_std  # add gaussian noise to features
        self.note_dir = "/mnt/data/home/ldy/mmiv_data/notes_embedding_t"
        self.note_labels = "/mnt/data/home/ldy/mmiv_data/notes_label"  # Note path
        self.note_dim = 1024  # set note dimension
        self.device = device  # device to store features
        # self.notes_mean = torch.load('/mnt/data/home/ldy/mmiv_data/mean_std/note_t/mean.pt', map_location=device)  # 
        # self.notes_std = torch.load('/mnt/data/home/ldy/mmiv_data/mean_std/note_t/std.pt', map_location=device)  #
        self.notes_mean, self.notes_std = self._get_mean_std(self.note_dir)
        self.note_files = os.listdir(self.note_dir)
        self.note_cache = {}
        # self.note_labels = os.listdir(self.note_labels)

    def _get_mean_std(self,note_dir):
        #check if there is mean and std file in the note_dir
        if os.path.exists(os.path.join(note_dir, 'mean.pt')) and os.path.exists(os.path.join(note_dir, 'std.pt')):
            return torch.load(os.path.join(note_dir, 'mean.pt')), torch.load(os.path.join(note_dir, 'std.pt'))
        #else calculate the mean and std
        note_files = os.listdir(note_dir)
        note_feature = []
        for note_file in note_files:
            note_data = torch.load(os.path.join(note_dir, note_file), map_location="cuda")
            note_feature.append(note_data)
        note_feature = torch.stack(note_feature)
        mean = note_feature.mean(dim=0)
        std = note_feature.std(dim=0)
        torch.save(mean, os.path.join(note_dir, 'mean.pt'))
        torch.save(std, os.path.join(note_dir, 'std.pt'))
        mean_note_feature = torch.load(os.path.join(note_dir, 'mean.pt'))
        std_note_feature = torch.load(os.path.join(note_dir, 'std.pt'))
        print("mean and std of note feature are calculated and saved")
        return mean_note_feature, std_note_feature

    
    def get_related_note_file(self, id_):
        subject_id = id_.split('_')[0]
        related_note_files = [] 
        for note_file in self.note_files:
            if subject_id in note_file:
                related_note_files.append(note_file)
        return related_note_files

    def sorted_note_files(self, note_files):
        new_note_files = []
        for note_file in note_files:
            note_seq = self._get_note_seq(os.path.join(self.note_labels, note_file))
            new_note_files.append((note_seq, note_file))
        new_note_files.sort(key=lambda x: x[0])
        return [note_file[1] for note_file in new_note_files]
    
    def read_all_note_files(self, note_files):
        all_notes = []
        for note_file in note_files:
            note_data = torch.load(os.path.join(self.note_dir, note_file), map_location=device)
            all_notes.append(note_data)
        if len(all_notes) == 1:
            #add one more dimension 
            return all_notes[0].unsqueeze(0)
        return torch.stack(all_notes)
    
    def load_notes(self, id_):
        if id_ in self.note_cache:
            return self.note_cache[id_]
        else:
            note_files = self.get_related_note_file(id_)
            # print("current note_files:", note_files)
            note_files = self.sorted_note_files(note_files)
            # print("note_files:", note_files)
            all_notes = self.read_all_note_files(note_files)  
            self.note_cache[id_] = all_notes
        return all_notes
        

    def _get_note_seq(self, note_file):
        note_label = note_file.replace('.pt', '.txt')
        note_label_file = os.path.join(self.note_labels, note_label)
        try:
            with open(note_label_file, 'r') as f:
                cnt = 0
                for line in f:
                    if cnt != 2: 
                        cnt += 1
                    else: 
                        note_seq = int(line.split()[1])
                        return note_seq
        except: 
            print(f"Error reading file {note_label_file}")
            return -1
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        # print(id_)
        features = self.feat_dict.get(id_, self.mean_feature)

        # 添加高斯噪声进行数据增强
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, features.shape)
            features = features + noise

        features = torch.tensor(features, dtype=torch.float)

        # 加载对应的Note文件
        note_file_path = os.path.join(self.note_dir, f"{id_}.pt")
        if os.path.exists(note_file_path):    
            try:
                note_data = self.load_notes(id_)
                note_data = (note_data - self.notes_mean) / self.notes_std #normalize the note data
                # if note_data.shape != (self.note_dim,):
                #     note_data = torch.zeros(self.note_dim) 
            except Exception as e:
                print(f"Error loading file {note_file_path}: {e}")
                note_data = torch.zeros(self.note_dim)  # if error occurs, use zero tensor
        else:
            # print(f"File not found: {note_file_path}")
            note_data = torch.zeros(self.note_dim)  # use zero tensor if note file not found
        # if os.path.exists(note_file_path):
        #     note_data = torch.load(note_file_path) #torch.jit.load
        #     # print(note_data)
        #  
        #     if note_data.shape != (self.note_dim,):
        #         note_data = torch.zeros(self.note_dim)  # 
        # else:
        #     note_data = torch.zeros(self.note_dim)  # 
        # print("first of note_data:", note_data[0])
        note_data = note_data.to(self.device)
        

        if self.labels is not None:
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.float)
            return features, note_data, label, id_
        else:
            return features, note_data, id_

def collate_fn(batch):
    """
    handle batch data to sequences 
    handle the condition there is no label in the test.csv
    Args:
        batch (list): list of samples
    Returns: 
        padded_sequences (torch.Tensor): padded feature sequences
        padded_note_sequences (torch.Tensor): padded note sequences
        src_key_padding_mask (torch.Tensor): mask for feature sequences
        src_key_padding_mask_notes (torch.Tensor): mask for note sequences
        labels (torch.Tensor, optional): labels for samples
        ids (list): list of IDs
    """
    if len(batch[0]) == 4:
        sequences = [item[0] for item in batch]
        notes_seq = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        ids = [item[3] for item in batch]
    elif len(batch[0]) == 3:
        sequences = [item[0] for item in batch]
        notes_seq = [item[1] for item in batch]
        labels = None
        ids = [item[2] for item in batch]
    else:
        raise ValueError("Unexpected batch format in collate_fn.")

    lengths = [seq.size(0) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True)

    lengths_notes = [note.size(0) for note in notes_seq]

    notes_seq = [note.unsqueeze(0) if len(note.size()) == 1 else note for note in notes_seq]
    
    padded_note_sequences = pad_sequence(notes_seq, batch_first=True)

    max_length = padded_sequences.size(1)
    src_key_padding_mask = torch.zeros((len(sequences), max_length), dtype=torch.bool)
    for i, length in enumerate(lengths):
        if length < max_length:
            src_key_padding_mask[i, length:] = True

    max_length_notes = padded_note_sequences.size(1)
    src_key_padding_mask_notes = torch.zeros((len(notes_seq), max_length_notes), dtype=torch.bool)
    for i, length in enumerate(lengths_notes):
        if length < max_length_notes:
            src_key_padding_mask_notes[i, length:] = True

    if labels is not None:
        labels = torch.stack(labels)
        return padded_sequences, padded_note_sequences, src_key_padding_mask, src_key_padding_mask_notes,labels, ids
    else:
        return padded_sequences, padded_note_sequences, src_key_padding_mask, src_key_padding_mask_notes, ids


