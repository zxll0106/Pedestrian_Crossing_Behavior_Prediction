import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,cfg,x_bbox_dim=16,x_pose_dim=128,x_visual_dim=128,x_ego_dim=4,future_inputs_dim=128,hidden_size=128):
        super(Encoder, self).__init__()

        self.cfg=cfg
        self.hidden_size=hidden_size

        # self.lstm_bbox=nn.LSTM(input_size=x_bbox_dim,hidden_size=hidden_size,batch_first=True,dropout=0.1)
        # self.lstm_visual=nn.LSTM(input_size=x_visual_dim,hidden_size=hidden_size,batch_first=True,dropout=0.1)
        # self.lstm_ego=nn.LSTM(input_size=x_ego_dim,hidden_size=hidden_size,batch_first=True,dropout=0.1,bidirectional=True)
        # self.lstm_ego_h0=nn.Linear(in_features=x_ego_dim,out_features=hidden_size)
        # self.lstm_ego_c0=nn.Linear(in_features=x_ego_dim,out_features=hidden_size)
        # self.lstm_future=nn.LSTM(input_size=future_inputs_dim,hidden_size=hidden_size,batch_first=True,dropout=0.1,bidirectional=True)

        self.gru_bbox=nn.GRU(input_size=x_bbox_dim,hidden_size=hidden_size,batch_first=True,dropout=0.1)
        self.gru_pose = nn.GRU(input_size=x_pose_dim, hidden_size=hidden_size, batch_first=True, dropout=0.1)
        self.gru_visual=nn.GRU(input_size=x_visual_dim,hidden_size=hidden_size,batch_first=True,dropout=0.1)
        self.gru_ego=nn.GRU(input_size=x_ego_dim,hidden_size=hidden_size,batch_first=True,dropout=0.1,bidirectional=True)
        self.gru_ego_h0=nn.Linear(in_features=x_ego_dim,out_features=hidden_size)
        self.gru_future=nn.GRU(input_size=future_inputs_dim,hidden_size=hidden_size,batch_first=True,dropout=0.1,bidirectional=True)

        if self.cfg.MODEL.WITH_IMAGE==True:
            size=hidden_size*2
        else:
            size=hidden_size

        if self.cfg.MODEL.FUTURE_INPUT_FUSE==0:
            size=size+hidden_size
        elif self.cfg.MODEL.FUTURE_INPUT_FUSE==1:
            size=size+hidden_size*2
        elif self.cfg.MODEL.FUTURE_INPUT_FUSE==2:
            size=size+0

        if cfg.MODEL.WITH_POSE:
            size=size+hidden_size

        if self.cfg.MODEL.WITH_EGO:
            size=size+hidden_size*2


        if self.cfg.MODEL.WITH_TRAFFIC:
            if self.cfg.DATASET.NAME == 'JAAD':
                size=size+hidden_size*2

            else:
                size = size + 64*len(self.cfg.MODEL.TRAFFIC_KEYS)

        self.linear = nn.Linear(size, 128)

    def forward(self,fusion_input):
        batch_size=fusion_input['x_bbox'].shape[0]

        x_bbox_encoded,x_visual_encoded,x_ego_encoded,x_future_encoded=None,None,None,None

        all_features=[]

        # x_bbox_encoded,_=self.lstm_bbox(fusion_input['x_bbox']) #128,t,hidden_size
        x_bbox_encoded, _ = self.gru_bbox(fusion_input['x_bbox'])  # 128,t,hidden_size
        all_features.append(x_bbox_encoded[:,-1].squeeze(1))

        if self.cfg.MODEL.WITH_IMAGE==True:
            # x_visual_encoded,_=self.lstm_visual(fusion_input['x_visual']) #128,t,hidden_size
            x_visual_encoded, _ = self.gru_visual(fusion_input['x_visual'])  # 128,t,hidden_size
            all_features.append(x_visual_encoded[:,-1].squeeze(1))

        if 'x_pose' in fusion_input:
            x_pose_encoded,_=self.gru_pose(fusion_input['x_pose'])
            all_features.append(x_pose_encoded[:,-1].squeeze(1))

        if 'x_ego' in fusion_input:
            # ego_h0=self.lstm_ego_h0(fusion_input['x_ego'][:,0,:]) #128,hidden_size
            # ego_c0=self.lstm_ego_c0(fusion_input['x_ego'][:,0,:]) #128,hidden_size
            ego_h0 = self.gru_ego_h0(fusion_input['x_ego'][:, 0, :])  # 128,hidden_size

            ego_h0=torch.cat((ego_h0.unsqueeze(1),torch.zeros((batch_size,1,self.hidden_size),device=fusion_input['x_bbox'].device)),dim=1)
            # ego_c0=torch.cat((ego_c0.unsqueeze(1),torch.zeros((batch_size,1,self.hidden_size),device=fusion_input['x_bbox'].device)),dim=1)

            ego_h0=ego_h0.transpose(0,1).contiguous()
            # ego_c0=ego_c0.transpose(0,1).contiguous()

            # _,x_ego_encoded=self.lstm_ego(fusion_input['x_ego'],(ego_h0,ego_c0))
            _, x_ego_encoded = self.gru_ego(fusion_input['x_ego'], ego_h0)
            # x_ego_encoded=unpack_RNN_state(x_ego_encoded)
            x_ego_encoded=x_ego_encoded.transpose(0,1)
            x_ego_encoded=x_ego_encoded.reshape(batch_size,-1)
            all_features.append(x_ego_encoded)

        if 'x_traffic' in fusion_input:
            all_features.append(fusion_input['x_traffic'])

        all_features_no_future=torch.cat(all_features,dim=1)

        if 'future_inputs' in fusion_input:

            # _,x_future_encoded=self.lstm_future(fusion_input['future_inputs'])
            # x_future_encoded=unpack_RNN_state(x_future_encoded)
            if self.cfg.MODEL.FUTURE_INPUT_FUSE==0:
                all_features.append(fusion_input['future_inputs'])
            elif self.cfg.MODEL.FUTURE_INPUT_FUSE==1:
                _,x_future_encoded=self.gru_future(fusion_input['future_inputs'])
                x_future_encoded=x_future_encoded.transpose(0,1)
                x_future_encoded=x_future_encoded.reshape(batch_size,-1)
                all_features.append(x_future_encoded)



        all_features=torch.cat(all_features,dim=1)
        all_features=self.linear(all_features)


        return all_features


def unpack_RNN_state(state_tuple):
    # PyTorch returned LSTM states have 3 dims:
    # (num_layers * num_directions, batch, hidden_size)

    state = torch.cat(state_tuple, dim=0).permute(1, 0, 2)
    # Now state is (batch, 2 * num_layers * num_directions, hidden_size)

    state_size = state.size()
    return torch.reshape(state, (-1, state_size[1] * state_size[2]))



if __name__=='__main__':
    fusion_input={}
    fusion_input['x_visual']=torch.randn((128,10,128))
    fusion_input['x_bbox']=torch.randn((128,10,16))
    fusion_input['x_ego']=torch.randn((128,10,4))
    fusion_input['future_inputs']=torch.randn((128,5,128))
    cfg=1
    encoder=Encoder(cfg)
    encoder(fusion_input)