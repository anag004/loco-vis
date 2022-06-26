import os
import torch
from rsl_rl.modules import ActorCritic, ActorCriticRMARecurrent, ActorCriticRecurrent, ActorCriticRMA
import sys
import argparse

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
     if checkpoint==-1:
         models = [file for file in os.listdir(root) if model_name_include in file]
         models.sort(key=lambda m: '{0:0>15}'.format(m))
         model = models[-1]
     else:
         model = "model_{}.pt".format(checkpoint) 

     load_path = os.path.join(root, model)
     return load_path

def play(args):    
    expert_all = ActorCriticRMA(187 + 47 + 17 + 10 + 15 * 58 * 87 + 15 * 32,
                                187 + 47 + 17 + 10 + 15 * 58 * 87 + 15 * 32,
                                (187 + 47, 187 + 47 + 17), 
                                12,
                                use_depth_backbone=True,
                                backbone_type="mlp_hierarchical",
                                num_input_vis_obs=15 * 58 * 87 + 15 * 32,
                                num_output_vis_obs=100,
                                scandots_compression=[187, 256, 128, 32],
                                actor_hidden_dims = [512, 256, 128],
                                critic_hidden_dims = [512, 256, 128])

    load_path = get_load_path(root=args.load_run)
    expert_all.load_state_dict(torch.load(load_path)['model_state_dict'])
    expert_all = expert_all.cpu()

    try:
        os.mkdir(os.path.join(args.load_run, "traced"))
    except IsADirectoryError:
        pass

    # Save the traced actor
    traced_actor = torch.jit.trace(expert_all.actor, torch.zeros(1, expert_all.actor_input_dim))
    save_path = os.path.join(args.load_run, "traced", "traced_actor.pt")
    traced_actor.save(save_path)
    print("Saved traced_actor at ", save_path)

    # Save scandots compression
    traced_scandots_compression = torch.jit.trace(expert_all.scandots_compression_nn, torch.zeros(1, 187))
    save_path = os.path.join(args.load_run, "traced", "traced_scandots_compression.pt")
    traced_scandots_compression.save(save_path)
    print("Saved traced scandots compression at ", save_path)

    traced_depth_backbone = torch.jit.trace(expert_all.depth_backbone, torch.zeros(1, 15 * 58 * 87 + 15 * 32))
    save_path = os.path.join(args.load_run, "traced", "traced_depth_backbone.pt")
    print("Saving traced_depth_backbone at {}".format(save_path))
    traced_depth_backbone.save(save_path)

    traced_image_compression = torch.jit.trace(expert_all.depth_backbone.image_compression, torch.zeros(1, 1, 58, 87))
    save_path = os.path.join(args.load_run, "traced", "traced_image_compression.pt")
    print("Saving traced_image_compression at {}".format(save_path))
    traced_image_compression.save(save_path)

    traced_depth_prop_processor = torch.jit.trace(expert_all.depth_backbone.mlp, torch.zeros(1, 128 * 15 + 32 * 15))
    save_path = os.path.join(args.load_run, "traced", "traced_depth_prop_processor.pt")
    print("Saving traced_depth_prop_processor at {}".format(save_path))
    traced_depth_prop_processor.save(save_path)

    # load distilled student encoder
    encoder_dir = os.path.join(args.teacher, "dagger")
    runid = -1
    fname = get_load_path(root=encoder_dir, checkpoint=runid, model_name_include="encoder")
    student_encoder = torch.jit.load(fname, map_location="cpu")
    print("Loaded history encoder model from {}".format(fname))

    # Save the encoder to CPU
    student_encoder = student_encoder.cpu()
    save_path = os.path.join(args.load_run, "traced", "encoder_cpu.pt")
    student_encoder.save(save_path)
    print("Saved encoder at ", save_path)
    student_encoder = student_encoder.cuda()
    
if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--load_run", type=str, required=True)
    args = parser.parse_args()

    play(args)
