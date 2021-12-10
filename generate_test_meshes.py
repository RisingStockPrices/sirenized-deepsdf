import argparse
import json
import numpy as np
import os
import torch

import deep_sdf
import deep_sdf.workspace as ws
from deep_sdf.data import remove_nans, unpack_sdf_samples
from deep_sdf.mesh import convert_sdf_samples_to_ply

"""
npz_file = '/home/spock-the-wizard/Desktop/DeepSDF/data/SdfSamples/ShapeNetV2/03001627/1007e20d5e811b308351982a6e40cf41.npz'
N=256


voxel_origin = [-1, -1, -1]
voxel_size = 2.0 / (N - 1)

npz = np.load(npz_file,allow_pickle=True)
pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

sdf = [
        pos_tensor[torch.randperm(pos_tensor.shape[0])],
        neg_tensor[torch.randperm(neg_tensor.shape[0])],
    ]
samples=unpack_sdf_samples()

import pdb; pdb.set_trace()
outfile=npz_file.split('/')[-1].split('.')[0]
import pdb; pdb.set_trace()
convert_sdf_samples_to_ply(sdf,voxel_origin,voxel_size,outfile+'.ply')
"""

def code_to_mesh(experiment_directory, checkpoint, keep_normalized=False):

    specs_filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(experiment_directory, ws.model_params_subdir, checkpoint + ".pth")
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    decoder.eval()

    tmp = '{}/Reconstructions/{}/Codes/ShapeNetV2/03001627/d38129a3301d31350b1fc43ca5e85e.pth'.format(experiment_directory,saved_model_epoch)
    latent_vector = torch.load(tmp) # size is [1,1,256]
    latent_vector = latent_vector.cuda()


    data_source = specs["DataSource"]

    dataset_name='ShapeNetV2'
    class_name='03001627'
    instance_name=tmp.split('/')[-1].split('.')[0]

    mesh_dir = os.path.join(
            experiment_directory,
            ws.training_meshes_subdir,
            str(saved_model_epoch),
            dataset_name,
            class_name,
        )
        
    print(mesh_dir)
    if not os.path.isdir(mesh_dir):
        os.makedirs(mesh_dir)

    mesh_filename = os.path.join(mesh_dir, instance_name)

    offset = None
    scale = None

    if not keep_normalized:

        normalization_params = np.load(
            ws.get_normalization_params_filename(
                data_source, dataset_name, class_name, instance_name
            )
        )
        offset = normalization_params["offset"]
        scale = normalization_params["scale"]

    with torch.no_grad():
        deep_sdf.mesh.create_mesh(
            decoder,
            latent_vector[0],
            mesh_filename,
            N=256,
            max_batch=int(2 ** 18),
            offset=offset,
            scale=scale,
        )


        
if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to generate a mesh given a latent code."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--keep_normalization",
        dest="keep_normalized",
        default=True,
        action="store_true",
        help="If set, keep the meshes in the normalized scale.",
    )
    arg_parser.add_argument(
        "--latent_path",
        dest="latent_path",
        default=None,
        help="If set, keep the meshes in the normalized scale.",
    )
    
    
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    code_to_mesh(args.experiment_directory, args.checkpoint, args.keep_normalized)
