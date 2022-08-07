import torchaudio
from torch.utils.data import Dataset
import pandas as pd

class UrbanSoundDataset(Dataset):

    def __init__(self, annotation_path, audio_path):
        self.annotation = pd.read_csv(annotation_path)
        self.audio_dir = audio_path

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        audio_sample_path = self._get_audio_sample_oath(idx)
        label = self._get_audio_sample_label(idx)
        