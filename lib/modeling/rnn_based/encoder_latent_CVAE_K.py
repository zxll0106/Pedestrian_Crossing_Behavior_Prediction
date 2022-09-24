import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder_latent_CVAE_K(nn.Module):
    def __init__(self, cfg=None, hidden_size=128, latent_dim=32):
        super(Encoder_latent_CVAE_K, self).__init__()

        self.cfg = cfg

        self.K=self.cfg.MODEL.K

        self.p_z_x = nn.Sequential(nn.Linear(hidden_size,
                                             64),
                                   nn.ReLU(),
                                   nn.Linear(64, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, latent_dim * 2))

        self.latent_dim = latent_dim

        self.initial_h = nn.Sequential(nn.Linear(self.cfg.DATASET.NUM_ACTION, hidden_size),
                                       nn.ReLU())  # self.cfg.DATASET.NUM_ACTION

        self.future_action_encoder = nn.GRU(input_size=1,
                                            hidden_size=hidden_size,
                                            bidirectional=True,
                                            batch_first=True)  # self.cfg.DATASET.NUM_ACTION

        self.q_z_xy = nn.Sequential(nn.Linear(hidden_size * 2 + hidden_size,
                                              128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.latent_dim * 2))

        self.hz_to_decoder_input = nn.Sequential(nn.Linear(hidden_size + self.latent_dim, hidden_size), nn.ReLU())

    def forward(self, enc_hx, cur_action, target_future_action=None):
        batch_size = enc_hx.shape[0]

        z_mu_logvar_p = self.p_z_x(enc_hx)
        z_mu_p = z_mu_logvar_p[:, :self.latent_dim]
        z_logvar_p = z_mu_logvar_p[:, self.latent_dim:]

        if target_future_action is not None:
            initial_h = self.initial_h(cur_action)
            initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=initial_h.device)], dim=0)

            _, target_action_h = self.future_action_encoder(target_future_action.unsqueeze(-1).float(), initial_h)
            target_action_h = target_action_h.permute(1, 0, 2)
            target_action_h = target_action_h.reshape(batch_size, -1)

            z_mu_logvar_q = self.q_z_xy(torch.cat((enc_hx, target_action_h), dim=-1))
            z_mu_q = z_mu_logvar_q[:, :self.latent_dim]
            z_logvar_q = z_mu_logvar_q[:, self.latent_dim:]

            Z_mu = z_mu_q
            Z_logvar = z_logvar_q

            KLD = 0.5 * ((z_logvar_q.exp() / z_logvar_p.exp()) + \
                         (z_mu_p - z_mu_q).pow(2) / z_logvar_p.exp() - \
                         1 + \
                         (z_logvar_p - z_logvar_q))
            KLD = KLD.sum(dim=-1).mean()
            KLD = torch.clamp(KLD, min=0.001)

        else:
            Z_mu = z_mu_p
            Z_logvar = z_logvar_p
            KLD = torch.as_tensor(0.0, device=Z_logvar.device)


        eps = torch.normal(mean=0, std=1.5, size=(batch_size,self.K,self.latent_dim)).to(Z_logvar.device)
        Z_std = torch.exp(0.5 * Z_logvar)
        Z = Z_mu.unsqueeze(1).repeat(1,self.K,1) + eps * Z_std.unsqueeze(1).repeat(1,self.K,1)

        decoder_input = self.hz_to_decoder_input(torch.cat((enc_hx.unsqueeze(1).repeat(1,self.K,1), Z), dim=-1))

        return decoder_input, KLD


if __name__ == '__main__':
    enc_hx = torch.randn((128, 128))
    cur_action = torch.randn((128, 7))
    target_future_action = torch.randn((128, 5))

    encoder_latent = Encoder_latent_CVAE()
    decoder_input, KLD = encoder_latent(enc_hx, cur_action, target_future_action)
