import torch
from argparse import ArgumentParser
import pytorch_lightning as L
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader

from utils.aoj import AspenOpenJets, JetFeatures, sample_from_empirical_masks
from utils.datasets import MultiModalDataset, DataCoupling, data_coupling_collate_fn
from utils.callbacks import FlowGeneratorCallback, EMACallback
from utils.plotting import plot_flavor_feats, flavor_kinematics, plot_kin_feats
from utils.tensorclass import TensorMultiModal
from utils.helpers import load_from_experiment

###############################################################################

def experiment_configs():
    config = ArgumentParser()

    config.add_argument("--num_nodes", "-N", type=int, default=1)
    config.add_argument("--dir", type=str, default='/pscratch/sd/d/dfarough')
    config.add_argument("--project", "-proj", type=str, default='jet_sequences')
    config.add_argument("--experiment_id", "-id", type=str, required=True)
    config.add_argument("--data_files", "-f", type=str, default='RunG_batch0.h5')
    config.add_argument("--continuous_features", "-cont", type=str, nargs='*', default=['pt', 'eta_rel', 'phi_rel'])
    config.add_argument("--discrete_features", "-disc", type=str, default='tokens')
    config.add_argument("--batch_size", "-bs", type=int, default=256)
    config.add_argument("--tag", "-t", type=str, default='')
    config.add_argument("--checkpoint", "-ckpt", type=str, default='best')
    config.add_argument("--num_jets", "-n", type=int, default=100_000)
    config.add_argument("--num_timesteps", "-steps", type=int, nargs='*')
    config.add_argument("--temperature", "-tmp",  type=float, nargs='*')
    config.add_argument("--top_k", type=int, default=None)
    config.add_argument("--top_p", type=float, default=None)
    config.add_argument("--use_final_max_rates", type=bool, default=False)
    config.add_argument("--num_files", type=int, default=1)
    config.add_argument("--make_plots", "-plots", type=bool, default=False)

    config = config.parse_args()

    run_config = load_from_experiment(f"{config.dir}/{config.project}/{config.experiment_id}")
    run_config.continuous_features = config.continuous_features
    run_config.discrete_features = config.discrete_features
    run_config.checkpoint = config.checkpoint
    run_config.data_files = config.data_files
    run_config.num_jets = config.num_jets
    run_config.temperature = config.temperature
    run_config.top_k = config.top_k
    run_config.top_p = config.top_p
    run_config.use_final_max_rates = config.use_final_max_rates
    run_config.num_timesteps = config.num_timesteps
    run_config.batch_size = config.batch_size
    run_config.tag = config.tag
    run_config.num_files = config.num_files
    run_config.make_plots = config.make_plots
    
    return run_config

def _load_pretrained_model(lightning_module, config, temp, num_steps, tag):

    config.tag = f"{tag}_steps_{num_steps}_temp_{temp}"
    config.res_dir = f"{config.dir}/{config.project}/{config.experiment_id}/generation_results_{config.tag}"
    ckpt = f"{config.dir}/{config.project}/{config.experiment_id}/checkpoints/{config.checkpoint}.ckpt" 
    model = lightning_module.load_from_checkpoint(ckpt, map_location="cpu", config=config)
    model.config.num_timesteps = num_steps 
    model.config.temperature = temp

    return model


def _make_source_dataloader(config):

    aoj = AspenOpenJets(data_dir=config.dir + '/aoj', data_files=config.data_files)

    # target for empirical masks:
    test, _ = aoj(num_jets=config.num_jets,
                  download=False,
                  features={"continuous": config.continuous_features, "discrete": config.discrete_features},    
                  pt_order=True,
                  padding='zeros')

    pad_mask = sample_from_empirical_masks(test.mask, config.num_jets, config.max_num_particles)
    noise_continuous = torch.randn_like(test.continuous) * pad_mask
    noise_discrete = torch.randint_like(test.discrete, 1, config.vocab_size) * pad_mask
    t0 = torch.full((len(pad_mask),), config.time_eps)  # (B) t_0=eps

    source = TensorMultiModal(time=t0, continuous=noise_continuous, discrete=noise_discrete,  mask=pad_mask)
    source_data = DataCoupling(source=source, target=TensorMultiModal())
    source_dataset = MultiModalDataset(source_data)

    del aoj, test

    return DataLoader(source_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=data_coupling_collate_fn)
 

def run_experiment(lightning_module, config, temp, num_steps, tag):

    model = _load_pretrained_model(lightning_module, config, temp, num_steps, tag)
    
    callbacks = [FlowGeneratorCallback(config)]
    
    if config.use_ema_weights:
        callbacks += [EMACallback(config)]

    generator = L.Trainer(accelerator="gpu", 
                          devices='auto',
                          strategy="ddp",
                          num_nodes=config.num_nodes, 
                          callbacks=callbacks,
                          )

    dataloader = _make_source_dataloader(config)
    generator.predict(model, dataloaders=dataloader)

    return generator

@rank_zero_only
def eval_metrics(config, generator):
    aoj = AspenOpenJets(data_dir=config.dir + '/aoj', data_files="RunG_batch1.h5")
    test, _ = aoj(num_jets=config.num_jets,
                  download=False,
                  features={"continuous": ['pt', 'eta_rel', 'phi_rel'], "discrete": 'tokens'},
                  pt_order=True,
                  padding='zeros')

    test = test.squeeze(-1, 'discrete')

    sample = TensorMultiModal.load_from(f"{config.res_dir}/generated_sample.h5")
    sample = sample.squeeze(-1, 'discrete')

    # metrics & plots:

    fig0 = plot_flavor_feats(sample, test, path=f"{config.res_dir}/plots_flavor.png")

    gen = JetFeatures(sample)
    aoj = JetFeatures(test) 

    # metrics & plots:

    fig1 = plot_kin_feats(gen, aoj, path=f"{config.res_dir}/plots_kin.png")
    fig2 = flavor_kinematics(gen, aoj, path=f"{config.res_dir}/flavor_kinematics.png")

    generator.logger.experiment.add_figure('plots_flavor', fig0)
    generator.logger.experiment.add_figure('plots_kin', fig1)
    generator.logger.experiment.add_figure('flavor_kinematics', fig2)


if __name__ == "__main__":

    from model.MMF import MultiModalFlowBridge

    config = experiment_configs()  

    temperatures = config.temperature   
    num_steps = config.num_timesteps

    for i in range(0, config.num_files):
        for temp in temperatures:
            for step in num_steps:
                
                if i > 0: suffix = f'_{i}'
                else: suffix = ''

                # run_experiment(MultiModalFlowBridge, config, temp, step, tag)
                generator = run_experiment(MultiModalFlowBridge, config, temp, step, config.tag + suffix)

                if config.make_plots:
                    if generator.global_rank == 0:
                        eval_metrics(config, generator)
                
