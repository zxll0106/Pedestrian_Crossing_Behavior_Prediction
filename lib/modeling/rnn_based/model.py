'''
main function of our action-intention detection model
Action head
Intention head
'''
import torch
import torch.nn as nn
from .action_intent_net import ActionIntentNet
from .Spatial_Temporal_Heterogeneous_Graph import Spatial_Temporal_Heterogeneous_Graph
from thop import profile,clever_format


class ActionIntentionDetection(nn.Module):
    def __init__(self, cfg, parameter_scheduler=None):
        super().__init__()
        self.cfg = cfg
        self.parameter_scheduler = parameter_scheduler
        self.hidden_size = self.cfg.MODEL.HIDDEN_SIZE
        
        self.bbox_embedding = nn.Sequential(nn.Linear(4, 16),
                                            nn.ReLU())
        self.pose_embedding= nn.Sequential(nn.Linear(36, 128),
                                            nn.ReLU())

        self.x_visual_extractor = None

        self.action_intent_model = ActionIntentNet(cfg, x_visual_extractor=self.x_visual_extractor)

        self.Spatial_Temporal_Heterogeneous_Graph=Spatial_Temporal_Heterogeneous_Graph(cfg,sub_layers=self.cfg.MODEL.VECTORNET_LAYER)


    def _init_hidden_states(self, x, net_type='gru', task_exists=True):
        batch_size = x.shape[0]
        if not task_exists:
            return None
        elif 'convlstm' in net_type:
            return [x.new_zeros(batch_size, self.cfg.MODEL.CONVLSTM_HIDDEN, 6, 6),
                      x.new_zeros(batch_size, self.cfg.MODEL.CONVLSTM_HIDDEN, 6, 6),
                      x.new_zeros(batch_size, self.hidden_size)]
        elif 'gru' in net_type:
            return x.new_zeros(batch_size, self.hidden_size)
        else:
            raise ValueError(net_type)

    def forward(self, 
                x_visual, 
                x_bbox=None,
                x_pose=None,
                x_ego=None, 
                x_traffic=None,
                dec_inputs=None, 
                local_bboxes=None, 
                masks=None,
                target_future_action=None):
        

        return self.forward_single_stream(x_visual,
                                          x_bbox=x_bbox,
                                          x_pose=x_pose,
                                          x_ego=x_ego,
                                          x_traffic=x_traffic,
                                          dec_inputs=dec_inputs,
                                          target_future_action=target_future_action)



    def forward_single_stream(self, x_visual, x_bbox=None, x_pose=None, x_ego=None, x_traffic=None, dec_inputs=None,target_future_action=None):
        '''
        NOTE: Action and Intent net share the same encoder network, but different classifiers
        '''
        seg_len = x_visual.shape[1]
        # initialize inputs and hidden states for encoders
        if self.cfg.MODEL.FUTURE_INPUT_FUSE ==0:
            future_inputs = x_visual.new_zeros(x_visual.shape[0], self.hidden_size) if 'trn' in self.cfg.MODEL.INTENT_NET else None #128,128
        elif self.cfg.MODEL.FUTURE_INPUT_FUSE==1:
            future_inputs=torch.zeros(x_visual.shape[0], self.cfg.MODEL.PRED_LEN,self.hidden_size).to(x_visual.device) if 'trn' in self.cfg.MODEL.INTENT_NET else None
        elif self.cfg.MODEL.FUTURE_INPUT_FUSE==2:
            future_inputs=None

        enc_hx = self._init_hidden_states(x_visual, net_type=self.cfg.MODEL.ACTION_NET,  task_exists=True) #128,128
        enc_h_ego = x_visual.new_zeros(x_visual.shape[0], 32) if self.cfg.MODEL.WITH_EGO else None #128,32
        action_detection_scores, intent_detection_scores, action_prediction_scores = [], [], []
        all_attentions = []


        total_KLD=0.0
        for t in range(seg_len):
            # Run one step of action detector/predictor
            if target_future_action != None:
                target_future_action_t=target_future_action[:,t]
            else:
                target_future_action_t=None

            x_ego_input = x_ego if x_ego is not None else None
            x_traffic_input, traffic_attentions = None, None
            if isinstance(x_traffic, torch.Tensor):
                x_traffic_input = x_traffic[:, t]
            elif isinstance(x_traffic, dict):

                x_traffic_input,traffic_attentions=self.Spatial_Temporal_Heterogeneous_Graph(x_traffic,x_bbox,t)

            x_pose_input=None

            ret = self.step_one_stream(x_visual[:, :t+1], #128,512,7,7
                                       enc_hx, #128,128
                                       x_bbox=x_bbox[:, :t+1], #128,4
                                       x_pose=x_pose_input,
                                       x_ego=x_ego_input, #128,1
                                       x_traffic=x_traffic_input, #128,6*32 or 128,64
                                       enc_h_ego=enc_h_ego, #None
                                       future_inputs=future_inputs, #128,128
                                       dec_inputs=dec_inputs,
                                       target_future_action=target_future_action_t)
            enc_act_scores, enc_int_scores, enc_hx, dec_act_scores, future_inputs, enc_h_ego, KLD = ret
            total_KLD=total_KLD+KLD
            action_detection_scores.append(enc_act_scores)
            intent_detection_scores.append(enc_int_scores)
            all_attentions.append(traffic_attentions)
            
            if dec_act_scores is not None:
                action_prediction_scores.append(dec_act_scores)
            
        action_detection_scores = torch.stack(action_detection_scores, dim=1) if action_detection_scores else None
        action_prediction_scores = torch.stack(action_prediction_scores, dim=1) if action_prediction_scores else None
        intent_detection_scores = torch.stack(intent_detection_scores, dim=1) if intent_detection_scores else None
        return action_detection_scores, action_prediction_scores, intent_detection_scores, all_attentions, total_KLD/seg_len
        
    def step_one_stream(self, x_visual, enc_hx, x_bbox=None, x_pose=None, x_ego=None, x_traffic=None, enc_h_ego=None, future_inputs=None, dec_inputs=None,target_future_action=None):
        if x_bbox is not None: #128,t,4
            x_bbox = self.bbox_embedding(x_bbox) #128,t,16
        if x_pose is not None:
            x_pose=self.pose_embedding(x_pose)
        ret = self.action_intent_model.step(x_visual, 
                                            enc_hx, 
                                            x_bbox=x_bbox,
                                            x_pose=x_pose,
                                            x_ego=x_ego,
                                            x_traffic=x_traffic,
                                            enc_h_ego=enc_h_ego,
                                            future_inputs=future_inputs, 
                                            dec_inputs=dec_inputs,
                                            target_future_action=target_future_action)
        enc_act_scores, enc_int_scores, enc_hx, dec_scores, future_inputs, enc_h_ego, KLD = ret
        return enc_act_scores, enc_int_scores, enc_hx, dec_scores, future_inputs, enc_h_ego, KLD
    
