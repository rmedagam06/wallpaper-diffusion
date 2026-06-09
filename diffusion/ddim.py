import torch


class DDIMSampler:
    def __init__(self, scheduler, num_steps=50):
        self._scheduler = scheduler
        T = scheduler.T
        self.timesteps = list(range(0, T, T // num_steps))[::-1]

    @property
    def scheduler(self):
        return self._scheduler

    @torch.no_grad()
    def sample(self, model, shape, context, cfg_scale=7.5, device="cpu"):
        x = torch.randn(shape, device=device)
        null_context = torch.zeros_like(context)

        for i, t_cur in enumerate(self.timesteps):
            t_prev = self.timesteps[i + 1] if i + 1 < len(self.timesteps) else 0
            t_batch = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)

            eps_cond   = model(x, t_batch, context)
            eps_uncond = model(x, t_batch, null_context)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            ab_cur  = self._scheduler.alpha_bar[t_cur].to(device)
            ab_prev = self._scheduler.alpha_bar[t_prev].to(device)
            x0_pred = (x - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)
            x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps

        return x
