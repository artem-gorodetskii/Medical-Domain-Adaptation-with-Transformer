from torch.utils.data import Dataset
from pathlib import Path
from sc_wave_rnn import audio
import sc_wave_rnn.hparams as hp
import numpy as np
import torch


class VocoderDataset(Dataset):
    def __init__(self, metadata_fpath: Path, mel_dir: Path, wav_dir: Path, spk_embd_dir: Path): #added

        print("Using inputs from:\n\t%s\n\t%s\n\t%s\n\t%s" % (metadata_fpath, mel_dir, wav_dir, spk_embd_dir)) #added
        
        with metadata_fpath.open("r") as metadata_file:
            metadata = [line.split("|") for line in metadata_file]
        
        gta_fnames = [x[1] for x in metadata if int(x[4])]
        gta_fpaths = [mel_dir.joinpath(fname) for fname in gta_fnames]
        wav_fnames = [x[0] for x in metadata if int(x[4])]
        wav_fpaths = [wav_dir.joinpath(fname) for fname in wav_fnames]
        spk_embd_names = [x[2] for x in metadata if int(x[4])] #added
        spk_embd_fpaths = [spk_embd_dir.joinpath(fname) for fname in spk_embd_names] #added
        self.samples_fpaths = list(zip(gta_fpaths, wav_fpaths, spk_embd_fpaths)) #added
        
        print("Found %d samples" % len(self.samples_fpaths))
    
    def __getitem__(self, index):  
        mel_path, wav_path, spk_embd_path = self.samples_fpaths[index] #added
        
        # Load the mel spectrogram and adjust its range to [-1, 1]
        mel = np.load(mel_path).T.astype(np.float32) / hp.mel_max_abs_value
        
        # Load the wav
        wav = np.load(wav_path)
        if hp.apply_preemphasis:
            wav = audio.pre_emphasis(wav)
        wav = np.clip(wav, -1, 1)

	    # Load the embedding
        spk_embd = np.load(spk_embd_path) #added
        
        # Fix for missing padding   # TODO: settle on whether this is any useful
        r_pad =  (len(wav) // hp.hop_length + 1) * hp.hop_length - len(wav)
        wav = np.pad(wav, (0, r_pad), mode='constant')
        assert len(wav) >= mel.shape[1] * hp.hop_length
        wav = wav[:mel.shape[1] * hp.hop_length]
        assert len(wav) % hp.hop_length == 0
        
        # Quantize the wav
        if hp.voc_mode == 'RAW':
            if hp.mu_law:
                quant = audio.encode_mu_law(wav, mu=2 ** hp.bits)
            else:
                quant = audio.float_2_label(wav, bits=hp.bits)
        elif hp.voc_mode == 'MOL':
            quant = audio.float_2_label(wav, bits=16)
            
        return mel.astype(np.float32), quant.astype(np.int64), spk_embd #added

    def __len__(self):
        return len(self.samples_fpaths)
        
        
def collate_vocoder(batch):
    mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
    max_offsets = [x[0].shape[-1] -2 - (mel_win + 2 * hp.voc_pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hp.voc_pad) * hp.hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    labels = [x[1][sig_offsets[i]:sig_offsets[i] + hp.voc_seq_len + 1] for i, x in enumerate(batch)]

    spk_embd = [x[2] for i, x in enumerate(batch)] #added

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)
    spk_embd = np.stack(spk_embd).astype(np.float32) #added

    mels = torch.tensor(mels)
    labels = torch.tensor(labels).long()
    spk_embd = torch.tensor(spk_embd) #added

    x = labels[:, :hp.voc_seq_len]
    y = labels[:, 1:]

    bits = 16 if hp.voc_mode == 'MOL' else hp.bits

    x = audio.label_2_float(x.float(), bits)

    if hp.voc_mode == 'MOL' :
        y = audio.label_2_float(y.float(), bits)

    return x, y, mels, spk_embd
