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
                self.future_context_combine_network = torch.nn.Linear(decoder_output_size+encoder_output_size , decoder_output_size)
            elif self.future_context_lm_type.lower() == 'lstm':
                self.future_context_conv_network = torch.nn.Conv1d(encoder_output_size, encoder_output_size, self.future_context_lm_kernel, padding=0)
               # print('Nothing to do here as conv am is combined at decoder stage') 

    def forward(
        self,
        enc_out: torch.Tensor,
        dec_out: torch.Tensor,
        is_aux: bool = False,
        quantization: bool = False,
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
        if self.future_context_lm and self.training:  #Added self.training to the condition as in beam search, a single state is passed along
            if self.future_context_lm_type == 'linear':
                u_len = dec_out.shape[2]
                t_len = enc_out.shape[1]
                zero_pad = torch.nn.ConstantPad1d((0,self.future_context_lm_kernel-1),0)
                convolved_am = self.future_context_conv_network(zero_pad(enc_out.squeeze(2).transpose(1,2))).transpose(1,2).unsqueeze(2)
                gu_temp = self.future_context_combine_network(torch.cat((dec_out.expand(-1,t_len,-1,-1),convolved_am.expand(-1,-1,u_len,-1)),dim=-1))
                dec_out = gu_temp
        if is_aux:
            joint_out = self.joint_activation(enc_out + self.lin_dec(dec_out))
        elif quantization:
            joint_out = self.joint_activation(
                self.lin_enc(enc_out.unsqueeze(0)) + self.lin_dec(dec_out.unsqueeze(0))
            )

            return self.lin_out(joint_out)[0]
        elif self.eta_mixing:
            u_len = dec_out.shape[2]
            t_len = enc_out.shape[1]
            if self.eta_mixing_type == "linear":
                etas = torch.sigmoid(self.eta_network(torch.cat((self.lin_enc(enc_out).expand(-1,-1,u_len,-1),self.lin_dec(dec_out).expand(-1,t_len,-1,-1)),dim=-1)))
            elif eta_mixing_type == "state_based_linear":
                #TODO
                raise
            joint_out = self.joint_activation(etas * self.lin_enc(enc_out).expand(-1,-1,u_len,-1)+(1-etas)*self.lin_dec(dec_out).expand(-1,t_len,-1,-1))
        else:
            joint_out = self.joint_activation(
                self.lin_enc(enc_out) + self.lin_dec(dec_out)
            )
        joint_out = self.lin_out(joint_out)

        return joint_out
