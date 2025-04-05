import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from utils.debugger import MyDebugger
from utils.meter import Meter
from models.network import SparseComposer
from models.module.diffusion_network import MyUNetModel, UNetModel
from models.module.dwt import DWTInverse3d_Laplacian, DWTForward3d_Laplacian
from models.module.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, mean_flat
from models.module.resample import UniformSampler, LossSecondMomentResampler, LossAwareSampler
from data_utils.my_data import MaterialVoxelDataset
from tqdm import tqdm
import os
import argparse


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class Trainer(object):

    def __init__(self, config, debugger):
        self.debugger = debugger
        self.config = config
        self.material_channels = self.config.material_channels
        self.feature_channels = self.config.feature_channels


    def print_dict(self, idx, record_dict : dict):

        str = f'Epoch {idx} : '
        for key, item in record_dict.items():
            str += f'{key} : {item} '

        print(str)

def train_network(self):
        ### create dataset
        samples = MaterialVoxelDataset(
            data_files=self.config.data_files,
            feature_files=self.config.feature_files
        )
        
        data_loader = DataLoader(dataset=samples,
                                batch_size=self.config.batch_size,
                                num_workers=self.config.data_worker,
                                shuffle=True,
                                drop_last=True)
        
        ### initialize network 
        network = UNetModel(
            in_channels=self.material_channels + self.feature_channels,  # Combined input channels
            model_channels=self.config.unet_model_channels,
            out_channels=self.material_channels,  # Output only material channels
            num_res_blocks=self.config.unet_num_res_blocks,
            channel_mult=self.config.unet_channel_mult,
            attention_resolutions=self.config.attention_resolutions,
            dropout=0,
            dims=3,
            activation=self.config.unet_activation if hasattr(self.config, 'unet_activation') else None
        )
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Use {torch.cuda.device_count()} GPUS!")
            network = nn.DataParallel(network)
        


        ### diffusion setting
        betas = get_named_beta_schedule(
            self.config.diffusion_beta_schedule, 
            self.config.diffusion_step, 
            self.config.diffusion_scale_ratio
        )
        
        diffusion_module = GaussianDiffusion(
            betas=betas,
            model_var_type=self.config.diffusion_model_var_type,
            model_mean_type=self.config.diffusion_model_mean_type,
            loss_type=self.config.diffusion_loss_type,
            rescale_timesteps=self.config.diffusion_rescale_timestep if hasattr(self.config, 'diffusion_rescale_timestep') else False
        )
        
        # Setup sampler for training
        ## how to sample the timestep for training. UniformSampler
        ## samples the timestep uniformly. LossSecondMomentResampler
        ## prioritizes the timesteps that has higher losses.
        if self.config.diffusion_sampler == 'uniform':
            sampler = UniformSampler(diffusion_module)
        elif self.config.diffusion_sampler == 'second-order':
            sampler = LossSecondMomentResampler(diffusion_module)
        else:
            raise Exception("Unknown Sampler...")



        ## only convert all
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Use {torch.cuda.device_count()} GPUS!")
            network = nn.DataParallel(network)

        network = network.to(device)

        ## reload the network if needed
        if self.config.network_resume_path is not None:
            ### remove something that is not needed
            network_state_dict = torch.load(self.config.network_resume_path)
            new_state_dict = network.state_dict()
            for key in list(network_state_dict.keys()):
                if key not in new_state_dict:
                    del network_state_dict[key]
            network.load_state_dict(network_state_dict)
            network.train()
            print(f"Reloaded thenetwork from {self.config.network_resume_path}")


        log_meter = Meter()
        log_meter.add_attributes('mse_loss')
        log_meter.add_attributes('total_loss')
        mse_fuction = self.config.loss_function

        if hasattr(self.config, 'optimizer') and self.config.optimizer:
            optimizer = self.config.optimizer(network.parameters(), lr = self.config.lr,
                                              betas=(self.config.beta1, self.config.beta2))
        else:
            optimizer = torch.optim.Adam(network.parameters(), lr=self.config.lr,
                                         betas=(self.config.beta1, self.config.beta2)
                                         )

        if self.config.lr_decay:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=self.config.lr_decay_rate)
        else:
            scheduler = None

        if self.config.optimizer_resume_path is not None:
            optimizer_state_dict = torch.load(self.config.optimizer_resume_path)
            new_state_dict = optimizer.state_dict()
            for key in list(optimizer_state_dict.keys()):
                if key not in new_state_dict:
                    del optimizer_state_dict[key]
            optimizer.load_state_dict(optimizer_state_dict)
            print(f"Reloaded the optimizer from {self.config.optimizer_resume_path}")
            self.config.optimizer_resume_path = None

        # mixed precision training
        if hasattr(self.config, 'mix_precision') and self.config.mix_precision:
            scaler = GradScaler()

            if hasattr(self.config, 'scaler_resume_path') and self.config.scaler_resume_path is not None:
                scaler_state_dict = torch.load(self.config.scaler_resume_path)
                scaler.load_state_dict(scaler_state_dict)



        for idx in range(self.config.starting_epoch, self.config.training_epochs + 1):
            if scheduler is not None:
                scheduler.step(idx)

            with tqdm(data_loader, unit='batch') as tepoch:
                tepoch.set_description(f'Epoch {idx}')

                network.train()
                ## main training loop
                for material_grid, feature_grid in tepoch:
                    ## remove gradient
                    optimizer.zero_grad()
                    loss = 0
                    ###
                    mse_loss = 0.0
                    if hasattr(self.config, 'mix_precision') and self.config.mix_precision:
                        with autocast():
                            mse_loss = self.conditional_diffusion_loss(diffusion_module, material_grid, feature_grid, network, sampler)
                    else:
                        mse_loss = self.conditional_diffusion_loss(diffusion_module, material_grid, feature_grid, network, sampler)


                    log_meter.add_data('mse_loss', mse_loss.item())
                    loss = loss + mse_loss
                    log_meter.add_data('total_loss', loss.item())

                    if hasattr(self.config, 'mix_precision') and self.config.mix_precision:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if self.config.use_gradient_clip:
                        torch.nn.utils.clip_grad_norm_(network.parameters(), self.config.gradient_clip_value)

                    if hasattr(self.config, 'mix_precision') and self.config.mix_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()


                    log_dict = log_meter.return_avg_dict()

                    tepoch.set_postfix(**log_dict)

                self.print_dict(idx, log_dict)
                log_meter.clear_data()


    def conditional_diffusion_loss(self, diffusion_module, material_grid, feature_grid, network, sampler):
        """
        Compute the diffusion loss with conditioning on feature_grid
        
        Args:
            diffusion_module: The diffusion process module
            material_grid: Target material grid [B, M, N, N, N]
            feature_grid: Conditioning feature grid [B, F, N, N, N]
            network: The UNet model
            t: Timesteps
            weights: Importance weights for the timesteps
        """
        t, weights = sampler.sample(material_grid.size(0), device=device)
        # Add noise to material grid according to timestep t
        noise = torch.randn_like(material_grid)

        ## only noise the target material grid
        noisy_materials = diffusion_module.q_sample(material_grid, t, noise=noise)
        
        # Concatenate noisy materials with conditioning features along channel dimension
        # model learns to predict the noise conditioned on the feature grid
        model_input = torch.cat([noisy_materials, feature_grid], dim=1)
        
        # Get model prediction
        model_output = network(model_input, t)
        
        # Calculate loss based on prediction type (noise or direct)
        if self.config.diffusion_model_mean_type == "EPSILON":  # Predict noise
            target = noise
        else:  # Predict denoised
            target = material_grid
        
        iterative_loss = mean_flat((target - model_output) ** 2)

        if isinstance(sampler, LossAwareSampler):
            sampler.update_with_local_losses(t, iterative_loss) ## per-sample loss
            ## iterative_loss is a tensor of shape [B, N, N, N], each for a sample.
            ## t is a tensor of shape [B]

        mse_loss = mse_loss + torch.mean(iterative_loss * weights)

        return mse_loss


if __name__ == '__main__':
    import importlib
    #torch.multiprocessing.set_start_method('spawn')  # good solution !!!!

    ## additional args for parsing
    optional_args = [("network_resume_path", str), ("optimizer_resume_path", str) ,("starting_epoch", int), ('starting_stage', int),
                     ("special_symbol", str), ("resume_path", str), ("scaler_resume_path", str)]
    parser = argparse.ArgumentParser()
    for optional_arg, arg_type in optional_args:
        parser.add_argument(f"--{optional_arg}", help=f"{optional_arg}", type=arg_type)

    args = parser.parse_args()
    ## Resume setting
    resume_path = None

    ## resume from path if needed
    if args.resume_path is not None:
        resume_path = args.resume_path

    if resume_path is None:
        from configs import config
        resume_path = os.path.join('configs', 'config.py')
    else:
        ## import config here
        spec = importlib.util.spec_from_file_location('*', resume_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

    for optional_arg, arg_type in optional_args:
        if args.__dict__.get(optional_arg, None) is not None:
            locals()['config'].__setattr__(optional_arg, args.__dict__.get(optional_arg, None))


    debugger = MyDebugger(f'MatDiffusion-Training-experiment{"-" + config.special_symbol if len(config.special_symbol) > 0 else config.special_symbol}', is_save_print_to_file = True, config_path = resume_path)
    trainer = Trainer(config = config, debugger = debugger)
    trainer.train_network()
