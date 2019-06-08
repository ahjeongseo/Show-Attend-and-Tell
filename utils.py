import torch
import os


def save_checkpoint(epoch, epochs_since_improvement, bleu4, encoder, decoder, model_path, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu4': bleu4,
             'encoder_state_dict': encoder.state_dict(),
             'decoder_state_dict': decoder.state_dict()
            }
    filename = 'checkpoint_' + str(epoch) + '.pth.tar'
    torch.save(state, os.path.join(model_path, filename))
    if is_best:
        torch.save(state, os.path.join(model_path, 'BEST_' + filename))

#Todo: calcuate Bleu score
