from sc_wave_rnn.models.fatchord_version import WaveRNN
from sc_wave_rnn import hparams as hp
import torch


_model = None   # type: WaveRNN

def load_model(weights_fpath, verbose=True):
    global _model, _device
    
    if verbose:
        print("Building SC Wave-RNN")
    _model = WaveRNN(
        rnn_dims=hp.voc_rnn_dims,
        fc_dims=hp.voc_fc_dims,
        bits=hp.bits,
        pad=hp.voc_pad,
        upsample_factors=hp.voc_upsample_factors,
        feat_dims=hp.num_mels,
        compute_dims=hp.voc_compute_dims,
        res_out_dims=hp.voc_res_out_dims,
        res_blocks=hp.voc_res_blocks,
        hop_length=hp.hop_length,
        sample_rate=hp.sample_rate,
        mode=hp.voc_mode
    )

    if torch.cuda.is_available():
        _model = _model.cuda()
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')
    
    if verbose:
        print("Loading model weights at %s" % weights_fpath)
    checkpoint = torch.load(weights_fpath, _device)
    _model.load_state_dict(checkpoint['model_state'])
    _model.eval()


def is_loaded():
    return _model is not None


def infer_waveform(mel, spk_embd, normalize=True,  batched=True, target=8000, overlap=800, 
                   progress_callback=None):
    """

    Infers the waveform of a mel spectrogram output by the synthesizer (the format must match 
    that of the synthesizer!)
    
    :param normalize:  
    :param batched: 
    :param target: 
    :param overlap: 
    :return: 
    """
    if _model is None:
        raise Exception("Please load Wave-RNN in memory before using it")
    
    if normalize:
        mel = mel / hp.mel_max_abs_value

    spk_embd = torch.tensor(spk_embd).unsqueeze(0)

    mel = torch.tensor(mel).unsqueeze(0)

    wav = _model.generate(mel, spk_embd, batched, target, overlap, hp.mu_law)
    return wav
