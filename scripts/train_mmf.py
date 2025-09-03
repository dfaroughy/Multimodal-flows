import pytorch_lightning as L
from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split

from utils.aoj import AspenOpenJets 
from utils.tensorclass import TensorMultiModal
from utils.helpers import load_from_experiment, set_logger
from utils.datasets import MultiModalDataset, DataCoupling, data_coupling_collate_fn
from utils.callbacks import TrainLoggerCallback, EMACallback, ProgressBarCallback     


def experiment_configs():

    config = ArgumentParser()

    # system
    config.add_argument("--num_nodes", "-N", type=int, default=1)
    config.add_argument("--dir", type=str, default='/pscratch/sd/d/dfarough')
    config.add_argument("--dir_aoj", type=str, default='/pscratch/sd/d/dfarough/aoj')
    config.add_argument("--project", "-proj", type=str, default='aoj_jets')
    config.add_argument("--comet_workspace", type=str, default='dfaroughy')
    config.add_argument("--comet_api_key", type=str, help='insert your personal comet api here')
    config.add_argument("--experiment_id", "-id", type=str, default=None, help='id for resuming or generating data from existing ckpt')
    config.add_argument("--ckpt_path", "-ckpt", type=str, default=None,  help='path to existing ckpt dir')
    config.add_argument("--resume_ckpt", "-resume", type=str, default='last') 
    config.add_argument("--tags", type=str, nargs='*')

    # training
    config.add_argument("--data_files", "-f", type=str, default='RunG_batch0.h5', help='aoj data file for train/val')
    config.add_argument("--num_jets", "-n", type=int, default=1_250_000)
    config.add_argument("--max_num_particles", "-d", type=int, default=150)
    config.add_argument("--batch_size", "-bs", type=int, default=256)
    config.add_argument("--max_epochs", "-epochs", type=int, default=1500)
    config.add_argument("--train_frac", type=float, default=0.8)
    config.add_argument("--lr", type=float, default=5e-4)
    config.add_argument("--lr_final", type=float, default=1e-5)
    config.add_argument("--warmup_epochs", type=int, default=0)
    config.add_argument("--use_ema_weights", "-ema",  type=bool, default=False)
    config.add_argument("--ema_decay", type=float, default=0.9999)

    # model
    config.add_argument("--model", "-nn", type=str, default='ParticleFormer')
    config.add_argument("--continuous_features", "-cont", type=str, nargs='*', default=['pt', 'eta_rel', 'phi_rel'])
    config.add_argument("--discrete_features", "-disc", type=str, default='tokens')
    config.add_argument("--vocab_size", type=int, default=9, help="Number of discrete tokens (1,...,8) plus one mask token (0)")
    config.add_argument("--dim_continuous", type=int, default=3)
    config.add_argument("--n_embd", type=int, default=256)
    config.add_argument("--n_inner", type=int, default=512)
    config.add_argument("--n_layer", type=int, default=5, help='number of layers for kin and flavor transformers')
    config.add_argument("--n_layer_fused", type=int, default=6, help='number of layers for fused transformers')
    config.add_argument("--n_head", type=int, default=4)
    config.add_argument("--dropout", type=float, default=0.0)
    config.add_argument("--qk_layernorm", type=bool, default=True)
    config.add_argument("--bias", type=bool, default=True)
    config.add_argument("--multitask_loss", "-loss", type=str, default='time-weighted')
    config.add_argument("--use_coocurrence", type=bool, default=False)
    
    # dynamics
    config.add_argument("--beta", "-b", type=float, default=0.075, help='Markov jump bridge stochasticity hyper-parameter')
    config.add_argument("--sigma", "-sig", type=float, default=1e-5, help='Flow-matching Gaussian smoothinig hyper-parameter')
    config.add_argument("--time_eps", "-eps", type=float, default=1e-5, help='time endpoints regularizer')

    # sampling
    config.add_argument("--num_timesteps", "-steps", type=int, default=100)
    config.add_argument("--temperature", type=float, default=1.0, help='Temperature scaling hyperparameter for softmax')
    config.add_argument("--top_k", type=int, default=None)
    config.add_argument("--top_p", type=float, default=None)

    config = config.parse_args()

    if config.experiment_id is None: return config
    else: 
        path = f"{config.dir}/{config.project}/{config.experiment_id}"
        run_config = load_from_experiment(path)
        run_config.max_epochs = config.max_epochs
        run_config.lr = config.lr
        run_config.lr_final = config.lr_final
        run_config.resume_ckpt = config.resume_ckpt
        return run_config



def make_dataloaders(config):

    aoj = AspenOpenJets(data_dir=config.dir_aoj, data_files=config.data_files, download=True)

    # target
    jets, metadata = aoj(num_jets=config.num_jets,
                        download=False,
                        features={"continuous": config.continuous_features, "discrete": config.discrete_features},    
                        transform='standardize',
                        pt_order=True,
                        padding='zeros')
               
    config.metadata = metadata

    # source
    noise = TensorMultiModal(mask=jets.mask.clone())

    # source-target coupling
    data = DataCoupling(source=noise, target=jets)
    dataset = MultiModalDataset(data)
    train_size = int(config.train_frac * len(dataset))
    val_size   = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=data_coupling_collate_fn)
    val_dataloader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False, collate_fn=data_coupling_collate_fn)
 
    return train_dataloader, val_dataloader

    
def run_train_experiment(config, lighting_module: L.LightningModule):

    train, val = make_dataloaders(config)

    if config.experiment_id is not None: 
        
        # resume training
        print('INFO: Resuming training from checkpoint: ', config.resume_ckpt)
        config.ckpt_path = f"{config.dir}/{config.project}/{config.experiment_id}/checkpoints/{config.resume_ckpt}.ckpt"
        model = lighting_module.load_from_checkpoint(config.ckpt_path, config=config, map_location="cpu",)

    else: 
        
        # new run
        model = lighting_module(config)

    callbacks = [L.callbacks.ModelCheckpoint(dirpath=None,
                                            monitor="val_loss",
                                            filename="best",
                                            save_top_k=10,
                                            mode="min",
                                            save_last=True,
                                                )]
    
    callbacks += [L.callbacks.ModelCheckpoint(dirpath=None,
                                        monitor="val_loss_mse",
                                        filename="best_mse",
                                        save_top_k=10,
                                        mode="min",
                                            )]

    callbacks += [L.callbacks.ModelCheckpoint(dirpath=None,
                                        monitor="val_loss_ce",
                                        filename="best_ce",
                                        save_top_k=10,
                                        mode="min",
                                            )]

    callbacks += [TrainLoggerCallback(config)]
    callbacks += [ProgressBarCallback()]
    
    if config.use_ema_weights:
        callbacks += [EMACallback(config)]


    logger = set_logger(config)  # comet 

    trainer = L.Trainer(max_epochs=config.max_epochs, 
                        accelerator='gpu', 
                        devices='auto',
                        strategy='ddp',
                        num_nodes=config.num_nodes,
                        callbacks=callbacks,
                        logger=logger,
                        sync_batchnorm=True,
                        gradient_clip_val=1.0,
                        )
    
    trainer.fit(model, train, val, ckpt_path=config.ckpt_path)



if __name__ == "__main__":

    from model.MMF import MultiModalFlowBridge

    config = experiment_configs()
    run_train_experiment(config, lighting_module=MultiModalFlowBridge)

