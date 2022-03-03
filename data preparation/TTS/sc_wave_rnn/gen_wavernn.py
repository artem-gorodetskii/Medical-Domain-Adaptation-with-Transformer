from sc_wave_rnn.models.fatchord_version import  WaveRNN
from sc_wave_rnn.audio import *
import torch


def gen_testset(model: WaveRNN, test_set, samples, batched, target, overlap, save_path):

    k = model.get_step() // 1000

    for i, (m, x, s_e) in enumerate(test_set, 1):
        if i > samples: 
            break

        print('\n| Generating: %i/%i' % (i, samples))

        x = x[0].numpy()

        bits = 16 if hp.voc_mode == 'MOL' else hp.bits

        if hp.mu_law and hp.voc_mode != 'MOL' :
            x = decode_mu_law(x, 2**bits, from_labels=True)
        else :
            x = label_2_float(x, bits)

        save_wav(x, save_path.joinpath("%dk_steps_%d_target.wav" % (k, i)))
        
        batch_str = "gen_batched_target%d_overlap%d" % (target, overlap) if batched else \
            "gen_not_batched"
            
        save_str = save_path.joinpath("%dk_steps_%d_%s.wav" % (k, i, batch_str))

        s_e = torch.tensor(s_e)

        m = torch.tensor(m)

        wav = model.generate(m, s_e, batched, target, overlap, hp.mu_law)
        save_wav(wav, save_str)

