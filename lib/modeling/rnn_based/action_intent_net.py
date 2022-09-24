'''
The action net take stack of observed image features
and detect the observed actions (and predict the futrue actions)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder_net import Encoder
from .encoder_latent_CVAE_K import Encoder_latent_CVAE_K
from thop import profile,clever_format

import pdb

class ActionIntentNet(nn.Module):
    def __init__(self, cfg, x_visual_extractor=None):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = self.cfg.MODEL.HIDDEN_SIZE
        self.pred_len = self.cfg.MODEL.PRED_LEN
        # The encoder RNN to encode observed image features
        # NOTE: there are two ways to encode the feature
        self.enc_drop = nn.Dropout(self.cfg.MODEL.DROPOUT)
        self.recurrent_drop = nn.Dropout(self.cfg.MODEL.RECURRENT_DROPOUT)

        self.x_visual_extractor = nn.Sequential(nn.Dropout2d(0.4),
                                                nn.AvgPool2d(kernel_size=[7, 7], stride=(1, 1)),
                                                nn.Flatten(start_dim=1, end_dim=-1),
                                                nn.Linear(512, 128),
                                                nn.ReLU())

        if self.cfg.DATASET.NAME=='PIE':
            self.encoder=Encoder(cfg,x_bbox_dim=16,x_visual_dim=128,x_ego_dim=4,future_inputs_dim=128,hidden_size=128)
        else:
            self.encoder = Encoder(cfg,x_bbox_dim=16, x_visual_dim=128, x_ego_dim=1, future_inputs_dim=128, hidden_size=128)

        self.zdim=32


        self.encoder_latent=Encoder_latent_CVAE_K(cfg=self.cfg,hidden_size=self.hidden_size,latent_dim=self.zdim)

        # The decoder RNN to predict future actions
        self.dec_drop = nn.Dropout(self.cfg.MODEL.DROPOUT)

        self.dec_input_linear = nn.Sequential(nn.Linear(self.cfg.DATASET.NUM_ACTION, self.hidden_size),
                                              nn.ReLU())
        self.future_linear = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                           nn.ReLU())
        self.dec_cell = nn.GRUCell(self.hidden_size, self.hidden_size)

        self.cur_action_to_dec_input=nn.Sequential(nn.Linear(self.cfg.DATASET.NUM_ACTION+1, self.hidden_size),
                                           nn.ReLU())
        # The classifier layer
        self.action_classifier = nn.Linear(self.hidden_size, self.cfg.DATASET.NUM_ACTION)
        self.intent_classifier = nn.Linear(self.hidden_size, 1) #self.cfg.DATASET.NUM_INTENT

        self.action_classifier_pred=nn.Linear(self.hidden_size*2, self.cfg.DATASET.NUM_ACTION)

    def enc_step(self, x_visual, enc_hx, x_bbox=None, x_pose=None, x_ego=None, x_traffic=None, enc_h_ego=None, future_inputs=None):

        batch_size = x_visual.shape[0]
        t=x_visual.shape[1]

        x_visual = self.x_visual_extractor(x_visual.reshape(-1,x_visual.shape[2],x_visual.shape[3],x_visual.shape[4])) #128,128
        x_visual=x_visual.reshape(batch_size,t,-1)

        fusion_input={}
        fusion_input['x_bbox']=x_bbox
        fusion_input['x_visual']=x_visual

        if self.cfg.MODEL.WITH_POSE:
            fusion_input['x_pose']=x_pose
        if self.cfg.MODEL.WITH_EGO:

            fusion_input['x_ego']=x_ego
        if self.cfg.MODEL.WITH_TRAFFIC:

            fusion_input['x_traffic']=x_traffic
        if future_inputs is not None:

            fusion_input['future_inputs']=future_inputs

        enc_hx=self.encoder(fusion_input)
        enc_act_score = self.action_classifier(self.enc_drop(enc_hx)) #128,7
        enc_int_score = self.intent_classifier(self.enc_drop(enc_hx)) #128,1

        return enc_hx, enc_act_score, enc_int_score, enc_h_ego

    def single_decoder_K(self, enc_hx,intent, dec_inputs=None):
        '''
        Run decoder for pred_len step to predict future actions
            enc_hx: last hidden state of encoder
            dec_inputs: decoder inputs
        '''

        K = self.cfg.MODEL.K
        batch_size = enc_hx.shape[0]
        dec_hx = enc_hx[-1] if isinstance(enc_hx, list) else enc_hx  # 128,K,128
        dec_scores = []

        dec_hx = dec_hx.reshape(-1, self.hidden_size)
        dec_inputs = dec_inputs.reshape(-1, self.hidden_size)
        if self.cfg.MODEL.FUTURE_INPUT_FUSE == 0:
            future_inputs = dec_hx.new_zeros(batch_size,
                                             self.hidden_size) if 'trn' in self.cfg.MODEL.INTENT_NET else None  # 128,128
        elif self.cfg.MODEL.FUTURE_INPUT_FUSE == 1:
            future_inputs = [] if 'trn' in self.cfg.MODEL.INTENT_NET else None
        for t in range(self.pred_len):
            dec_hx = self.dec_cell(self.dec_drop(dec_inputs),  # 128*K,128or 128*K,192
                                   self.recurrent_drop(dec_hx))

            dec_score = self.action_classifier(self.dec_drop(dec_hx))  # 128*K,7

            dec_scores.append(dec_score.reshape(batch_size, K, -1))

            if self.cfg.MODEL.INTENT_TO_DEC_INPUT==True:
                dec_inputs = self.cur_action_to_dec_input(torch.cat((dec_score,intent.unsqueeze(1).repeat(1,K,1).reshape(-1,1)),dim=-1))  # 128*K,128
            else:
                dec_inputs = self.dec_input_linear(dec_score)

            if self.cfg.MODEL.FUTURE_INPUT_FUSE == 0:
                future_inputs = future_inputs + self.future_linear(dec_hx.reshape(batch_size, K, -1)).mean(
                    dim=1) if future_inputs is not None else None  # 128,128
            elif self.cfg.MODEL.FUTURE_INPUT_FUSE == 1:
                future_inputs.append(self.future_linear(dec_hx.reshape(batch_size, K, -1)).mean(
                    dim=1)) if future_inputs is not None else None
        if self.cfg.MODEL.FUTURE_INPUT_FUSE == 0:
            future_inputs = future_inputs / self.pred_len if future_inputs is not None else None  # average
        elif self.cfg.MODEL.FUTURE_INPUT_FUSE == 1:
            future_inputs = torch.stack(future_inputs, dim=1)
        elif self.cfg.MODEL.FUTURE_INPUT_FUSE==2:
            future_inputs=None
        # dec_scores=torch.stack(dec_scores, dim=1)
        return torch.stack(dec_scores, dim=1), future_inputs

    
    def step(self, x_visual, enc_hx, x_bbox=None, x_pose=None, x_ego=None, x_traffic=None, enc_h_ego=None, future_inputs=None, dec_inputs=None,target_future_action=None):
        '''
        Directly call step when run inferencing.
        x_visual: (batch, 512, 7, 7)
        enc_hx: (batch, hidden_size)
        '''
        # 1. encoder
        enc_hx, enc_act_scores, enc_int_scores, enc_h_ego = self.enc_step(x_visual, enc_hx, 
                                                                            x_bbox=x_bbox,
                                                                            x_pose=x_pose,
                                                                            x_ego=x_ego, 
                                                                            x_traffic=x_traffic,
                                                                            enc_h_ego=enc_h_ego, 
                                                                            future_inputs=future_inputs)

        decoder_inputs,KLD=self.encoder_latent(enc_hx,enc_act_scores,target_future_action)


        dec_inputs=self.cur_action_to_dec_input(torch.cat((enc_act_scores,enc_int_scores),dim=-1))
        dec_inputs=dec_inputs.unsqueeze(1).repeat(1,self.cfg.MODEL.K,1)


        dec_scores,future_inputs=self.single_decoder_K(decoder_inputs,enc_int_scores,dec_inputs=dec_inputs)

            
        return enc_act_scores, enc_int_scores, enc_hx, dec_scores, future_inputs, enc_h_ego, KLD

