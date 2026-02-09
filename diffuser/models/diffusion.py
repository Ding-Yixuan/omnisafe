import numpy as np
import torch
from torch import nn
import pdb
from torch.autograd import Variable
from qpth.qp import QPFunction, QPSolvers
import einops

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.norm_mins = 0
        self.norm_maxs = 0
        self.safe1 = 0
        self.safe2 = 0

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):

        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):

        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()   #only for sampling
    def Shield(self, x0, xp10):  #Truncate method

        x = x0.clone()
        xp1 = xp10.clone()

        xp1 = xp1.squeeze(0)

        nBatch = xp1.shape[0]

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        b = ((xp1[:,2:3] - off_y)/yr)**2 + ((xp1[:,3:4] - off_x)/xr)**2 - 1

        for k in range(nBatch):
            if b[k, 0] < 0: 
                theta = torch.atan2((xp1[k,2:3] - off_y)/yr, (xp1[k,3:4] - off_x)/xr)
                xp1[k,2] = yr*torch.sin(theta) + off_y
                xp1[k,3] = xr*torch.cos(theta) + off_x

        b = ((xp1[:,2:3] - off_y)/yr)**2 + ((xp1[:,3:4] - off_x)/xr)**2 - 1

         #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b2 = ((xp1[:,2:3] - off_y)/yr)**4 + ((xp1[:,3:4] - off_x)/xr)**4 - 1

        self.safe1 = torch.min(b[:,0])
        self.safe2 = torch.min(b2[:,0])

        xp1 = xp1.unsqueeze(0)
        return xp1
    
    @torch.no_grad()   #only for sampling
    def GD(self, x0, xp10):    #classifier guidance or potential-based method

        x = x0.clone()
        xp1 = xp10.clone()

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        b = ((xp1[:,2:3] - off_y)/yr)**2 + ((xp1[:,3:4] - off_x)/xr)**2 - 1

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b2 = ((xp1[:,2:3] - off_y)/yr)**4 + ((xp1[:,3:4] - off_x)/xr)**4 - 1

        for k in range(nBatch):
            if b[k, 0] < 0.1:  # 0, 0.2
                u1 = 0.2/(2*((xp1[k,2:3] - off_y)/yr)/yr)
                u2 = 0.2/(2*((xp1[k,3:4] - off_x)/xr)/xr)
                xp1[k,2] = xp1[k,2] + u1*0.001  #note no 0.1/0.01 for GD, but has for potential
                xp1[k,3] = xp1[k,3] + u2*0.001
            elif b2[k, 0] < 0.1:  # 0, 0.2
                u1 = 0.2/(4*((xp1[k,2:3] - off_y)/yr)**3/yr)
                u2 = 0.2/(4*((xp1[k,3:4] - off_x)/xr)**3/xr)
                xp1[k,2] = xp1[k,2] + u1*0.001
                xp1[k,3] = xp1[k,3] + u2*0.001
            # else:
            #     x[k,2] = xp1[k,2]
            #     x[k,3] = xp1[k,3]

        self.safe1 = torch.min(b[:,0])
        self.safe2 = torch.min(b2[:,0])

        xp1 = xp1.unsqueeze(0)
        return xp1


    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        xp1 = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        # Note:  choose any one of the below
        #---------------------------------------start--------------------------------------------------#
        ####################### original diffuser only
        x = xp1      
        self.safe1 = torch.tensor(0.0, device=device)  # 或者 device='cuda' / x.device
        self.safe2 = torch.tensor(0.0, device=device)
        # xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        # yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        # off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        # off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        # b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1
        # self.safe1 = torch.min(b[:,0])
        # xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        # yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        # off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        # off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        # b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1
        # self.safe2 = torch.min(b[:,0])

        ####################### truncate (shield) and GD (classifier-guidance/potential-based)
        # x = self.Shield(x, xp1)
        # x = self.GD(x, xp1)

        ####################### SafeDiffusers 
        # x = xp1 # for training only
        # x = self.invariance(x, xp1)    # RoS
        # x = self.invariance_cf(x, xp1)  # RoS closed form
        # x = self.invariance_relax(x, xp1, t) # ReS
        # x = self.invariance_relax_cf(x, xp1, t)   #ReS closed form    
        # x = self.invariance_time(x, xp1, t)   # TVS
        # x = self.invariance_time_cf(x, xp1, t)  # TVS closed form
        # x = self.invariance_relax_narrow(x, xp1, t)  # narrow passage case

        ####################### Applying SafeDiffusers to only the last 10 steps
        # if t <= 10:  #10
        #     # x = self.invariance_relax(x, xp1, t)  #done
        #     # x = self.invariance_relax_narrow(x, xp1, t)

        #     x = self.GD(x, xp1)
        # else:
        #     x = xp1
        #     xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        #     yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        #     off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        #     off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        #     b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1
        #     self.safe1 = torch.min(b[:,0])
        #     xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        #     yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        #     off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        #     off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        #     b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1
        #     self.safe2 = torch.min(b[:,0])

        
        ###################### umaze case
        # x = self.invariance_umaze(x, xp1)   #umaze
        # x = self.invariance_umaze_relax(x, xp1, t)   #umaze
        #-----------------------------------------end--------------------------------------------------#


        return x

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        safe1, safe2 = [], []
        for i in reversed(range(0, self.n_timesteps)):  #-50 change here for the number of diffusion steps,
            if i < 0:
                i = 0
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps)
            x = apply_conditioning(x, cond, self.action_dim)
            safe1.append(self.safe1.unsqueeze(0))
            safe2.append(self.safe2.unsqueeze(0))
            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)
        
        self.safe1 = torch.cat(safe1, dim=0)
        self.safe2 = torch.cat(safe2, dim=0)

        progress.close()
        # pdb.set_trace()
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, *args, horizon=None, return_diffusion = True, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, return_diffusion= return_diffusion, *args, **kwargs)   ## debug

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        # import pdb; pdb.set_trace()
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        # return self.p_losses(x, cond, t)
        # 1. 计算原始 Loss (保持原样)
        loss, info = self.p_losses(x, cond, t)
        
        # 2. 🔥【植入监控】每 100 个 Batch 抽查一次
        # 这样不会拖慢训练速度，但能让你实时看到模型是不是傻了
        if torch.rand(1) < 0.01: 
            with torch.no_grad():
                # 重新模拟一次前向传播，只为了看输出
                noise = torch.randn_like(x)
                x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
                
                # 让模型预测 (可能是预测噪声，也可能是预测动作，取决于 predict_epsilon)
                model_output = self.model(x_noisy, cond, t)
                
                # 提取动作部分的输出 (假设动作在最后几维)
                # output shape: (Batch, Horizon, 28) -> 取最后 2 维
                pred_actions = model_output[:, :, -self.action_dim:] 
                
                # 计算方差 (Standard Deviation)
                # 如果这个值是 0.00xx，说明模型都在输出同一个数（复读机）
                # 如果这个值 > 0.1，说明模型开始有想法了
                act_std = pred_actions.std()
                
                print(f"📉 Step Loss: {loss.item():.4f} | 📊 Model Activity (Std): {act_std.item():.4f}")
                
                # 额外诊断：如果你在做 predict_epsilon=True
                # 标准差应该接近 1.0 (因为噪声的标准差是1)
                # 如果只有 0.0x，说明它连噪声都预测不出来
                if self.predict_epsilon and act_std.item() < 0.1:
                    print("   ⚠️ 警告: 预测方差过低！模型可能发生了均值坍塌 (Mean Collapse)")

        return loss, info


    def _vb_terms_bpd(
        self, x_start, conditions, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        # batch_size = x_start.shape(0)
        # conditions = self._format_conditions(conditions, batch_size)

        true_mean, _, true_log_variance_clipped = self.q_posterior(
            x_start=x_start, x_t=x_t, t=t
        )
        
        mean, _, log_variance = self.p_mean_variance(
             x_t, conditions, t)
        kl = normal_kl(
            true_mean, true_log_variance_clipped, mean, log_variance
        )
        kl = mean_flat(kl) / np.log(2.0)

        # import pdb; pdb.set_trace()
        # decoder_nll = -discretized_gaussian_log_likelihood(
        #     x_start, means=mean, log_scales=0.5 * log_variance
        # )
        
        # assert decoder_nll.shape == x_start.shape
        # decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))

        # output = torch.where((t == 0), decoder_nll, kl)

        return kl


    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

