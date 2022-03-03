from sc_wave_rnn.models.fatchord_version import WaveRNN
from sc_wave_rnn.vocoder_dataset import VocoderDataset, collate_vocoder
from sc_wave_rnn.distribution import discretized_mix_logistic_loss
from sc_wave_rnn.display import stream, simple_table
from sc_wave_rnn.gen_wavernn import gen_testset
from torch.utils.data import DataLoader
from pathlib import Path
from torch import optim
import torch.nn.functional as F
import sc_wave_rnn.hparams as hp
import numpy as np
import time
import torch

from sc_wave_rnn import visualizations
import soundfile

import copy

# copy encoder to synthesizer directory
from encoder import inference as encoder ###
from pathlib import Path ###
encoder_weights = Path("./encoder/saved_models/pretrained.pt") ###
encoder.load_model(encoder_weights) ###


def train(run_id: str, syn_dir: Path, voc_dir: Path, syn_val_dir: Path, voc_val_dir: Path, 
          models_dir: Path, ground_truth: bool, save_every: int, backup_every: int, eval_every: int,force_restart: bool):

    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length
    
    # Instantiate the model
    print("Initializing the model...")
    model = WaveRNN(
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
        model = model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')   

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())
    for p in optimizer.param_groups: 
        p["lr"] = hp.voc_lr
    loss_func = F.cross_entropy if model.mode == "RAW" else discretized_mix_logistic_loss

    # Load the weights
    model_dir = models_dir.joinpath(run_id)
    model_dir.mkdir(exist_ok=True)
    weights_fpath = model_dir.joinpath(run_id + ".pt")
    if force_restart or not weights_fpath.exists():
        print("\nStarting the training of WaveRNN from scratch\n")
        model.save(weights_fpath, optimizer)
    else:
        print("\nLoading weights at %s" % weights_fpath)
        model.load(weights_fpath, optimizer)
        print("WaveRNN weights loaded from step %d" % model.step)
    
    # Initialize the dataset
    metadata_fpath = syn_dir.joinpath("train.txt") if ground_truth else \
        voc_dir.joinpath("synthesized.txt")

    val_metadata_fpath = syn_val_dir.joinpath("validation.txt") if ground_truth else \
        voc_val_dir.joinpath("synthesized.txt")

    mel_dir = syn_dir.joinpath("mels") if ground_truth else voc_dir.joinpath("mels_gta")
    val_mel_dir = syn_val_dir.joinpath("mels") if ground_truth else voc_val_dir.joinpath("mels_gta")

    wav_dir = syn_dir.joinpath("audio")
    val_wav_dir = syn_val_dir.joinpath("audio")
    spk_embd_dir = syn_dir.joinpath("embeds")
    val_spk_embd_dir = syn_val_dir.joinpath("embeds")

    dataset = VocoderDataset(metadata_fpath, mel_dir, wav_dir, spk_embd_dir)
    val_dataset = VocoderDataset(val_metadata_fpath, val_mel_dir, val_wav_dir, val_spk_embd_dir)


    # Begin the training
    simple_table([('Batch size', hp.voc_batch_size),
                  ('LR', hp.voc_lr),
                  ('Sequence Len', hp.voc_seq_len)])

    global plotter
    plotter = visualizations.VisdomLinePlotter(env_name='vocoder_training_'+run_id)
    
    for epoch in range(1, 350): # was 350
        

        data_loader = DataLoader(dataset,
                                 collate_fn=collate_vocoder,
                                 batch_size=hp.voc_batch_size,
                                 num_workers=8,
                                 shuffle=True,
                                 pin_memory=True)

        val_data_loader = DataLoader(val_dataset,
                                    collate_fn=collate_vocoder,
                                    batch_size=hp.voc_batch_size,
                                    num_workers=8,
                                    shuffle=True,
                                    pin_memory=True)


        start = time.time()
        running_loss = 0.

        for i, (x, y, m, s_e) in enumerate(data_loader, 1):
            if torch.cuda.is_available():
                x, m, y, spk_embd = x.cuda(), m.cuda(), y.cuda(), s_e.cuda()
            
            # Forward pass
            y_hat = model(x, m, spk_embd)
            if model.mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            elif model.mode == 'MOL':
                y = y.float()
            y = y.unsqueeze(-1)
            
            # Backward pass
            loss = loss_func(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            speed = i / (time.time() - start)
            avg_loss = running_loss / i

            step = model.get_step()
            last_step = copy.deepcopy(step)
            k = step // 1000

            if backup_every != 0 and step % backup_every == 0 :
                model.checkpoint(model_dir, optimizer)
                
            if save_every != 0 and step % save_every == 0 :
                model.save(weights_fpath, optimizer)

            msg = f"| Epoch: {epoch} ({i}/{len(data_loader)}) | " \
                  f"Loss: {avg_loss:.4f} | {speed:.1f} " \
                  f"steps/s | Step: {k}k | "
            stream(msg)

            if step % eval_every == 0:
                plotter.plot('loss', 'train', 'Class Loss', last_step // 1000, avg_loss)

            # Perform evaluation
            if eval_every !=0 and step % eval_every == 0:

                print()
                print()
                print("Evaluation of validation data " )

                model.eval()
                with torch.no_grad():

                    val_losses = []
                    val_similarities = []
                    train_similarities = []
                    for val_i, (val_x, val_y, val_m, val_s_e) in enumerate(val_data_loader, 1):

                        if torch.cuda.is_available():
                            val_x, val_m, val_y, val_spk_embd = val_x.cuda(), val_m.cuda(), val_y.cuda(), val_s_e.cuda()

                        # Forward pass
                        val_y_hat = model(val_x, val_m, val_spk_embd)
                        if model.mode == 'RAW':
                            val_y_hat = val_y_hat.transpose(1, 2).unsqueeze(-1)
                        elif model.mode == 'MOL':
                            val_y = val_y.float()
                        val_y = val_y.unsqueeze(-1)

                        # Calculate losses
                        val_loss = loss_func(val_y_hat, val_y)
                        val_losses.append(val_loss.item())

                     
                        # Generate audio for the first validation batch
                        if val_i == 1:

                            val_loader = DataLoader(val_dataset,
                                                    batch_size=1,
                                                    shuffle=True,
                                                    pin_memory=True)

                            train_loader = DataLoader(dataset,
                                                      batch_size=1,
                                                      shuffle=True,
                                                      pin_memory=True)

                            for j, (source_m, source_y, source_spk_embd) in enumerate(val_loader, 1):

                                if torch.cuda.is_available():
                                    source_mel, source_y, source_spk_embd = source_m.cuda(), source_y.cuda(), source_spk_embd.cuda()
                           
                                #source_mel = source_mel / hp.mel_max_abs_value
                                
                                generated_wav = model.generate(source_mel, source_spk_embd, batched=True, 
                                                                target=8000, overlap=800, mu_law=hp.mu_law)

                                soundfile.write(str(model_dir)+"/val_generated_"+str(j)+'.wav', generated_wav, samplerate=16000)

                                generated_wav = encoder.preprocess_wav(generated_wav)
                                embed, partial_embeds, _ = encoder.embed_utterance(generated_wav, return_partials=True)
                                generated_embeding = np.mean(partial_embeds, axis=0)
                                generated_embeding = generated_embeding / np.linalg.norm(generated_embeding, 2) 
                                val_similarities.append(np.inner(source_spk_embd.cpu().detach().numpy(), generated_embeding))
                                                               
                                if j>hp.voc_batch_size * 3:
                                    break

                            for j, (source_m, source_y, source_spk_embd) in enumerate(train_loader, 1):

                                if torch.cuda.is_available():
                                    source_mel, source_y, source_spk_embd = source_m.cuda(), source_y.cuda(), source_spk_embd.cuda()
                           
                                #source_mel = source_mel / hp.mel_max_abs_value
                                
                                generated_wav = model.generate(source_mel, source_spk_embd, batched=True, 
                                                                target=8000, overlap=800, mu_law=hp.mu_law)

                                soundfile.write(str(model_dir)+"/train_generated_"+str(j)+'.wav', generated_wav, samplerate=16000)

                                generated_wav = encoder.preprocess_wav(generated_wav)
                                embed, partial_embeds, _ = encoder.embed_utterance(generated_wav, return_partials=True)
                                generated_embeding = np.mean(partial_embeds, axis=0)
                                generated_embeding = generated_embeding / np.linalg.norm(generated_embeding, 2) 
                                train_similarities.append(np.inner(source_spk_embd.cpu().detach().numpy(), generated_embeding))
                                                               
                                if j>hp.voc_batch_size * 3:
                                    break
                       

                        if val_i*hp.voc_batch_size>len(val_data_loader):
                            avg_val_similarity = np.mean(val_similarities)
                            avg_train_similarity = np.mean(train_similarities)
                            avg_val_loss = np.mean(val_losses)
                            print()
                            print("Val Loss: " + str(format(avg_val_loss, '.4f')) + " | Val Similarity: " + str(format(avg_val_similarity, '.4f'))+ " | Train Similarity: " + str(format(avg_train_similarity, '.4f')))
                            plotter.plot('loss', 'validation', 'Class Loss', last_step // 1000, avg_val_loss)
                            plotter.plot('similarity', 'validation', 'Class Similarity', last_step // 1000, avg_val_similarity)
                            plotter.plot('similarity', 'train', 'Class Similarity', last_step // 1000, avg_train_similarity)
                            # switch to train mode
                            print()
                            model.train()
                            break


        #gen_testset(model, test_loader, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
                    #hp.voc_target, hp.voc_overlap, model_dir)
        print("")
