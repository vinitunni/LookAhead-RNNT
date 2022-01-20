"""Transducer joint network implementation."""

import torch

from espnet.nets.pytorch_backend.nets_utils import get_activation


class JointNetwork(torch.nn.Module):
    """Transducer joint network module.

    Args:
        joint_output_size: Joint network output dimension
        encoder_output_size: Encoder output dimension.
        decoder_output_size: Decoder output dimension.
        joint_space_size: Dimension of joint space.
        joint_activation_type: Type of activation for joint network.

    """

    def __init__(
        self,
        joint_output_size: int,
        encoder_output_size: int,
        decoder_output_size: int,
        joint_space_size: int,
        joint_activation_type: int,
        eta_mixing: bool = False,
        eta_mixing_type: str = "linear",
        future_context_lm = False,
        future_context_lm_kernel = 10,
        future_context_lm_type = 'linear',
        future_context_lm_linear_layers=1,
        future_context_lm_units=256,
        la_embed_size=128,
        la_window=4,
        la_greedy_scheduled_sampling_probability=0.2,
        la_teacher_forcing_dist_threshold=0.10,
    ):
        """Joint network initializer."""
        super().__init__()

        self.lin_enc = torch.nn.Linear(encoder_output_size, joint_space_size)
        self.lin_dec = torch.nn.Linear(
            decoder_output_size, joint_space_size, bias=False
        )

        self.lin_out = torch.nn.Linear(joint_space_size, joint_output_size)

        self.joint_activation = get_activation(joint_activation_type)
        self.eta_mixing = eta_mixing
        self.eta_mixing_type = eta_mixing_type
        if self.eta_mixing:
            if self.eta_mixing_type=="linear":
                # implies concatenation of audio and text embeddings passed to a linear layer
                self.eta_network=torch.nn.Linear(2*joint_space_size,1)
            elif self.eta_mixing_type=="state_based_linear":
                # implies concatenation of audio and text embeddings and the prev eta passed to a inear layer
                self.eta_network=torch.nn.Linear(2*joint_space_size+1,1)
        self.future_context_lm = future_context_lm
        self.future_context_lm_kernel = future_context_lm_kernel
        self.future_context_lm_type = future_context_lm_type
        if self.future_context_lm:
            if self.future_context_lm_type.lower() == 'linear':
                self.future_context_conv_network = torch.nn.Conv1d(encoder_output_size, encoder_output_size, self.future_context_lm_kernel, padding=0)
                if future_context_lm_linear_layers == 1:
                    self.future_context_combine_network = torch.nn.Linear(decoder_output_size+encoder_output_size , decoder_output_size)
                else:
                    future_context_linear_list = []
                    future_context_linear_list.append(torch.nn.Linear(decoder_output_size+encoder_output_size , future_context_lm_units))
                    future_context_linear_list.append(torch.nn.Tanh())
                    for i in range(future_context_lm_linear_layers-2):
                        future_context_linear_list.append(torch.nn.Linear(future_context_lm_units , future_context_lm_units))
                        future_context_linear_list.append(torch.nn.Tanh())
                    future_context_linear_list.append(torch.nn.Linear(future_context_lm_units , decoder_output_size))
                    self.future_context_combine_network = torch.nn.Sequential(*future_context_linear_list)
                        
            elif self.future_context_lm_type.lower() == 'lstm':
                self.future_context_conv_network = torch.nn.Conv1d(encoder_output_size, encoder_output_size, self.future_context_lm_kernel, padding=0)
               # print('Nothing to do here as conv am is combined at decoder stage') 
            elif self.future_context_lm_type == 'greedy_lookahead_aligned' or  self.future_context_lm_type == 'greedy_lookahead_aligned_lev_dist'or  self.future_context_lm_type == 'greedy_lookahead_aligned_rapidfuzz':
                self.la_embed_size=la_embed_size
                self.la_window=la_window
                self.la_greedy_scheduled_sampling_probability=la_greedy_scheduled_sampling_probability   # With this probability, use the ground truth
                self.la_teacher_forcing_dist_threshold = la_teacher_forcing_dist_threshold
                self.embed_la = torch.nn.Embedding(joint_output_size, self.la_embed_size, padding_idx=0)
                if future_context_lm_linear_layers == 1:
                    self.future_context_combine_network = torch.nn.Linear(decoder_output_size+(self.la_window*self.la_embed_size) , decoder_output_size)
                else:
                    future_context_linear_list = []
                    future_context_linear_list.append(torch.nn.Linear(decoder_output_size+(self.la_window*self.la_embed_size) , future_context_lm_units))
                    future_context_linear_list.append(torch.nn.Tanh())
                    for i in range(future_context_lm_linear_layers-2):
                        future_context_linear_list.append(torch.nn.Linear(future_context_lm_units , future_context_lm_units))
                        future_context_linear_list.append(torch.nn.Tanh())
                    future_context_linear_list.append(torch.nn.Linear(future_context_lm_units , decoder_output_size))
                    self.future_context_combine_network = torch.nn.Sequential(*future_context_linear_list)
            elif self.future_context_lm_type == 'greedy_lookahead_acoustic_aligned':
                self.la_embed_size=la_embed_size
                self.la_window=la_window
                self.la_greedy_scheduled_sampling_probability=la_greedy_scheduled_sampling_probability   # With this probability, use the ground truth
                if future_context_lm_linear_layers == 1:
                    self.future_context_combine_network = torch.nn.Linear(decoder_output_size+(self.la_window*encoder_output_size) , decoder_output_size)
                else:
                    future_context_linear_list = []
                    future_context_linear_list.append(torch.nn.Linear(decoder_output_size+(self.la_window*encoder_output_size) , future_context_lm_units))
                    future_context_linear_list.append(torch.nn.Tanh())
                    for i in range(future_context_lm_linear_layers-2):
                        future_context_linear_list.append(torch.nn.Linear(future_context_lm_units , future_context_lm_units))
                        future_context_linear_list.append(torch.nn.Tanh())
                    future_context_linear_list.append(torch.nn.Linear(future_context_lm_units , decoder_output_size))
                    self.future_context_combine_network = torch.nn.Sequential(*future_context_linear_list)
                

    def forward(
        self,
        enc_out: torch.Tensor,
        dec_out: torch.Tensor,
        target:torch.Tensor = torch.zeros(1),
        implicit_lm: bool = False,
        implicit_am: bool = False,
        is_aux: bool = False,
        quantization: bool = False,
        char_list: list=[]
    ) -> torch.Tensor:
        """Joint computation of encoder and decoder hidden state sequences.

        Args:
            enc_out: Expanded encoder output state sequences (B, T, 1, D_enc)
            dec_out: Expanded decoder output state sequences (B, 1, U, D_dec)
            is_aux: Whether auxiliary tasks in used.
            quantization: Whether dynamic quantization is used.

        Returns:
            joint_out: Joint output state sequences. (B, T, U, D_out)

        """
        if self.future_context_lm :  #Added self.training to the condition as in beam search, a single state is passed along
            if self.future_context_lm_type == 'linear' and len(enc_out.shape)>2 and not (implicit_lm or implicit_am):
                u_len = dec_out.shape[2]
                t_len = enc_out.shape[1]
                zero_pad = torch.nn.ConstantPad1d((0,self.future_context_lm_kernel-1),0)
                convolved_am = self.future_context_conv_network(zero_pad(enc_out.squeeze(2).transpose(1,2))).transpose(1,2).unsqueeze(2)
                gu_temp = self.future_context_combine_network(torch.cat((dec_out.expand(-1,t_len,-1,-1),convolved_am.expand(-1,-1,u_len,-1)),dim=-1))
                dec_out = gu_temp
            elif self.future_context_lm_type == 'greedy_lookahead_aligned' and len(enc_out.shape)>2 and not implicit_am:
                am_outs = self.lin_out(self.lin_enc(enc_out)).argmax(dim=-1).squeeze(-1)  # after this, the size is B x T
                B, T = am_outs.shape
                U = dec_out.shape[2]
                am_outs = torch.cat([am_outs,torch.zeros([B,1],dtype=am_outs.dtype,device=am_outs.device)],dim=-1)
                la_tokens = torch.zeros(B,T,self.la_window,dtype=am_outs.dtype,device=am_outs.device)
                for b in range(am_outs.shape[0]):
                    for t in range(T):
                        la_tokens[b,t] = torch.cat([am_outs[b,t+1:][am_outs[b,t+1:]!=0][:self.la_window],torch.zeros(self.la_window,device=enc_out.device,dtype=am_outs.dtype)])[:self.la_window]
                la_tokens =  la_tokens.unsqueeze(-2).expand(-1,-1,U,-1)   # Shape here is B x T x U x embed*num_tokens
                if self.training:  # Perform scheduled sampling only during training
                    target = torch.cat([target,torch.zeros([B,1],device=am_outs.device,dtype=target.dtype)],dim=-1)
                    sched_samp = torch.zeros([B,U,self.la_window],dtype=am_outs.dtype,device=am_outs.device)
                    for b in range(B):
                        for u in range(U):
                            sched_samp[b,u] = torch.cat([target[b,u:][:self.la_window],torch.zeros(self.la_window,device=enc_out.device,dtype=am_outs.dtype)])[:self.la_window]
                    sched_samp = sched_samp.unsqueeze(1).expand(-1,T,-1,-1)
                    # sched_samp_rand = torch.rand(sched_samp.shape,device=la_tokens.device)   # Commented to enable teacher forcing at substring level instead of token level
                    sched_samp_rand = torch.rand([B,T,U,1],device=la_tokens.device).expand(-1,-1,-1,self.la_window)
                    la_tokens = la_tokens * (sched_samp_rand > self.la_greedy_scheduled_sampling_probability).to(int) + sched_samp * (sched_samp_rand <= self.la_greedy_scheduled_sampling_probability).to(int)
                la_tokens = self.embed_la(la_tokens).reshape(B,T,U,-1)
                dec_out = dec_out.expand(-1,T,-1,-1)
                dec_out = torch.cat([dec_out,la_tokens],dim=-1)
                dec_out = self.future_context_combine_network(dec_out)
            elif self.future_context_lm_type == 'greedy_lookahead_acoustic_aligned' and len(enc_out.shape)>2 and not implicit_am:
                am_outs = self.lin_out(self.lin_enc(enc_out)).argmax(dim=-1).squeeze(-1)  # after this, the size is B x T
                B, T = am_outs.shape
                U = dec_out.shape[2]
                am_outs = torch.cat([am_outs,torch.zeros([B,1],dtype=am_outs.dtype,device=am_outs.device)],dim=-1)
                la_tokens = torch.zeros([B,T,self.la_window,1,enc_out.shape[3]],dtype=am_outs.dtype,device=am_outs.device)
                enc_outs_temp = torch.cat([enc_out,torch.zeros([B,1,1,enc_out.shape[3]],device=enc_out.device)],dim=1)
                for b in range(am_outs.shape[0]):
                    for t in range(T):
                        la_tokens[b,t] = torch.cat([enc_outs_temp[b,t+1:][am_outs[b,t+1:]!=0][:self.la_window],torch.zeros([self.la_window,1,enc_out.shape[3]],device=enc_out.device,dtype=am_outs.dtype)],dim=0)[:self.la_window]
                la_tokens = la_tokens.reshape(B,T,-1)
                la_tokens =  la_tokens.unsqueeze(-2).expand(-1,-1,U,-1)   # Shape here is B x T x U x embed*num_tokens
                dec_out = dec_out.expand(-1,T,-1,-1)
                dec_out = torch.cat([dec_out,la_tokens],dim=-1)
                dec_out = self.future_context_combine_network(dec_out)
            elif self.future_context_lm_type == 'greedy_lookahead_aligned_lev_dist' and len(enc_out.shape)>2 and not implicit_am:
                am_outs = self.lin_out(self.lin_enc(enc_out)).argmax(dim=-1).squeeze(-1)  # after this, the size is B x T
                B, T = am_outs.shape
                U = dec_out.shape[2]
                am_outs = torch.cat([am_outs,torch.zeros([B,1],dtype=am_outs.dtype,device=am_outs.device)],dim=-1)
                la_tokens = torch.zeros(B,T,self.la_window,dtype=am_outs.dtype,device=am_outs.device)
                la_tokens_temp = torch.zeros(B,T,U,self.la_window,dtype=am_outs.dtype,device=am_outs.device)
                import numpy as np, panphon.distance
                dst = panphon.distance.Distance()
                # import epitran
                # epi=epitran.Epitran('eng-Latn')
                for b in range(am_outs.shape[0]):
                    for t in range(T):
                        la_tokens[b,t] = torch.cat([am_outs[b,t+1:][am_outs[b,t+1:]!=0][:self.la_window],torch.zeros(self.la_window,device=enc_out.device,dtype=am_outs.dtype)])[:self.la_window]
                if self.training:  # Perform scheduled sampling only during training
                    target = torch.cat([target,torch.zeros([B,1],device=am_outs.device,dtype=target.dtype)],dim=-1)
                    sched_samp = torch.zeros([B,U,self.la_window],dtype=am_outs.dtype,device=am_outs.device)
                    for b in range(B):
                        for u in range(U):
                            sched_samp[b,u] = torch.cat([target[b,u:][:self.la_window],torch.zeros(self.la_window,device=enc_out.device,dtype=am_outs.dtype)])[:self.la_window]
                    sched_samp_rand = torch.rand([B,T,U,1],device=la_tokens.device)
                    for b in range(am_outs.shape[0]):
                        set_la = set([tuple(x) for x in la_tokens[b].detach().tolist()])
                        set_gd = set([tuple(x) for x in sched_samp[b].detach().tolist()])
                        dict_la = {}
                        for key in set_la:
                            # dict_la[key]=epi.transliterate(''.join([char_list[x_tmp] for x_tmp in key]))
                            dict_la[key]=''.join([char_list[x_tmp] for x_tmp in key])
                        dict_gd = {}
                        for key in set_gd:
                            # dict_gd[key]=epi.transliterate(''.join([char_list[x_tmp] for x_tmp in key]))
                            dict_gd[key]=''.join([char_list[x_tmp] for x_tmp in key])
                        for t, tmp_chars_t in enumerate(la_tokens[b].detach().tolist()):
                            for u, tmp_chars_u in enumerate(sched_samp[b].detach().tolist()):
                                if sched_samp_rand[b,t,u] > self.la_greedy_scheduled_sampling_probability:
                                    la_tokens_temp[b,t,u] = la_tokens[b,t]
                                # elif dst.dolgo_prime_distance_div_maxlen(epi.transliterate(''.join([char_list[c_tmp] for c_tmp in tmp_chars_t])),epi.transliterate(''.join([char_list[c_tmp] for c_tmp in tmp_chars_u]))) <=0.5:
                                else:
                                    if dst.fast_levenshtein_distance_div_maxlen(dict_la[tuple(tmp_chars_t)],dict_gd[tuple(tmp_chars_u)]) <= self.la_teacher_forcing_dist_threshold:
                                        la_tokens_temp [b,t,u] = sched_samp[b,u]
                                    else:
                                        la_tokens_temp[b,t,u] = la_tokens[b,t]
                                
                del la_tokens
                la_tokens =  la_tokens_temp
                if self.training:  # Perform scheduled sampling only during training
                    sched_samp = sched_samp.unsqueeze(1).expand(-1,T,-1,-1)
                    # sched_samp_rand = torch.rand(sched_samp.shape,device=la_tokens.device)   # Commented to enable teacher forcing at substring level instead of token level
                la_tokens = self.embed_la(la_tokens).reshape(B,T,U,-1)
                dec_out = dec_out.expand(-1,T,-1,-1)
                dec_out = torch.cat([dec_out,la_tokens],dim=-1)
                dec_out = self.future_context_combine_network(dec_out)
            elif self.future_context_lm_type == 'greedy_lookahead_aligned_rapidfuzz' and len(enc_out.shape)>2 and not implicit_am:
                from rapidfuzz import process, string_metric
                am_outs = self.lin_out(self.lin_enc(enc_out)).argmax(dim=-1).squeeze(-1)  # after this, the size is B x T
                B, T = am_outs.shape
                U = dec_out.shape[2]
                am_outs = torch.cat([am_outs,torch.zeros([B,1],dtype=am_outs.dtype,device=am_outs.device)],dim=-1)
                la_tokens = torch.zeros(B,T,self.la_window,dtype=am_outs.dtype,device=am_outs.device)
                for b in range(am_outs.shape[0]):
                    for t in range(T):
                        la_tokens[b,t] = torch.cat([am_outs[b,t+1:][am_outs[b,t+1:]!=0][:self.la_window],torch.zeros(self.la_window,device=enc_out.device,dtype=am_outs.dtype)])[:self.la_window]
                if self.training:  # Perform scheduled sampling only during training
                    target = torch.cat([target,torch.zeros([B,1],device=am_outs.device,dtype=target.dtype)],dim=-1)
                    # sched_samp = torch.zeros([B,U,self.la_window],dtype=am_outs.dtype,device=am_outs.device)
                    sets_tgt=[set() for _ in range(B)]
                    dicts_tgt=[{} for _ in range(B)]
                    for b in range(B):
                        for u in range(U):
                            temp_tokens = torch.cat([target[b,u:][:self.la_window],torch.zeros(self.la_window,device=enc_out.device,dtype=am_outs.dtype)])[:self.la_window]
                            temp_chars = ''.join([char_list[x_tmp] for x_tmp in temp_tokens])
                            if temp_chars not in dicts_tgt[b]:
                                dicts_tgt[b][temp_chars]=temp_tokens
                            sets_tgt[b].add(temp_chars)
                    sched_samp_rand = torch.rand([B,T,U],device=la_tokens.device)
                    la_tokens_temp = la_tokens.unsqueeze(-2).repeat(1,1,U,1)
                    import numpy as np
                    for b, t, u in np.nonzero(sched_samp_rand < self.la_greedy_scheduled_sampling_probability):
                        temp_chars = ''.join([char_list[x_tmp] for x_tmp in la_tokens[b,t]])
                        la_tokens_temp[b,t,u] = dicts_tgt[b][process.extract(temp_chars,sets_tgt[b],scorer=string_metric.normalized_levenshtein,limit=1)[0][0]]
                    # for b in range(B):
                    #     for t in range(T):
                    #         for u in range(U):
                    #             if sched_samp_rand[b,t,u] < self.la_greedy_scheduled_sampling_probability:
                                    # temp_chars = ''.join([char_list[x_tmp] for x_tmp in la_tokens[b,t]])
                                    # la_tokens_temp[b,t,u] = dicts_tgt[b][process.extract(temp_chars,sets_tgt[b],scorer=string_metric.normalized_levenshtein,limit=1)[0][0]]
                del la_tokens
                la_tokens =  la_tokens_temp
                la_tokens = self.embed_la(la_tokens).reshape(B,T,U,-1)
                dec_out = dec_out.expand(-1,T,-1,-1)
                dec_out = torch.cat([dec_out,la_tokens],dim=-1)
                dec_out = self.future_context_combine_network(dec_out)
                    
        if is_aux:
            joint_out = self.joint_activation(enc_out + self.lin_dec(dec_out))
        elif quantization:
            joint_out = self.joint_activation(
                self.lin_enc(enc_out.unsqueeze(0)) + self.lin_dec(dec_out.unsqueeze(0))
            )

            return self.lin_out(joint_out)[0]
        elif self.eta_mixing:
            #TODO Eta mixing while decoding
            u_len = dec_out.shape[2]
            t_len = enc_out.shape[1]
            if self.eta_mixing_type == "linear":
                etas = torch.sigmoid(self.eta_network(torch.cat((self.lin_enc(enc_out).expand(-1,-1,u_len,-1),self.lin_dec(dec_out).expand(-1,t_len,-1,-1)),dim=-1)))
            elif eta_mixing_type == "state_based_linear":
                #TODO
                raise
            joint_out = self.joint_activation(etas * self.lin_enc(enc_out).expand(-1,-1,u_len,-1)+(1-etas)*self.lin_dec(dec_out).expand(-1,t_len,-1,-1))
        elif implicit_lm and "greedy" in self.future_context_lm_type:
            joint_out = self.joint_activation(self.lin_dec(dec_out).expand(-1,enc_out.shape[1],-1,-1))
        else:
            joint_out = self.joint_activation(
                self.lin_enc(enc_out) + self.lin_dec(dec_out)
            )
        joint_out = self.lin_out(joint_out)

        return joint_out
