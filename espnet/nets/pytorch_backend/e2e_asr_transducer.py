"""Transducer speech recognition model (pytorch)."""

from argparse import ArgumentParser
from argparse import Namespace
from dataclasses import asdict
import logging
import math
import numpy
from typing import List

import chainer
import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.beam_search_transducer import BeamSearchTransducer
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transducer.arguments import (
    add_auxiliary_task_arguments,  # noqa: H301
    add_custom_decoder_arguments,  # noqa: H301
    add_custom_encoder_arguments,  # noqa: H301
    add_custom_training_arguments,  # noqa: H301
    add_decoder_general_arguments,  # noqa: H301
    add_encoder_general_arguments,  # noqa: H301
    add_rnn_decoder_arguments,  # noqa: H301
    add_rnn_encoder_arguments,  # noqa: H301
    add_transducer_arguments,  # noqa: H301
)
from espnet.nets.pytorch_backend.transducer.custom_decoder import CustomDecoder
from espnet.nets.pytorch_backend.transducer.custom_encoder import CustomEncoder
from espnet.nets.pytorch_backend.transducer.error_calculator import ErrorCalculator
from espnet.nets.pytorch_backend.transducer.initializer import initializer
from espnet.nets.pytorch_backend.transducer.rnn_decoder import RNNDecoder
from espnet.nets.pytorch_backend.transducer.rnn_encoder import encoder_for
from espnet.nets.pytorch_backend.transducer.transducer_tasks import TransducerTasks
from espnet.nets.pytorch_backend.transducer.utils import get_decoder_input
from espnet.nets.pytorch_backend.transducer.utils import valid_aux_encoder_output_layers
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.utils.fill_missing_args import fill_missing_args


class Reporter(chainer.Chain):
    """A chainer reporter wrapper for Transducer models."""

    def report(
        self,
        loss: float,
        loss_trans: float,
        loss_ctc: float,
        loss_aux_trans: float,
        loss_symm_kl_div: float,
        loss_lm: float,
        loss_ilm: float,
        loss_iam: float,
        cer: float,
        wer: float,
    ):
        """Instantiate reporter attributes.

        Args:
            loss: Model loss.
            loss_trans: Main Transducer loss.
            loss_ctc: CTC loss.
            loss_aux_trans: Auxiliary Transducer loss.
            loss_symm_kl_div: Symmetric KL-divergence loss.
            loss_lm: Label smoothing loss.
            cer: Character Error Rate.
            wer: Word Error Rate.

        """
        chainer.reporter.report({"loss": loss}, self)
        chainer.reporter.report({"loss_trans": loss_trans}, self)
        chainer.reporter.report({"loss_ctc": loss_ctc}, self)
        chainer.reporter.report({"loss_lm": loss_lm}, self)
        chainer.reporter.report({"loss_aux_trans": loss_aux_trans}, self)
        chainer.reporter.report({"loss_symm_kl_div": loss_symm_kl_div}, self)
        chainer.reporter.report({"cer": cer}, self)
        chainer.reporter.report({"wer": wer}, self)

        logging.info("loss:" + str(loss))


class E2E(ASRInterface, torch.nn.Module):
    """E2E module for Transducer models.

    Args:
        idim: Dimension of inputs.
        odim: Dimension of outputs.
        args: Namespace containing model options.
        ignore_id: Padding symbol ID.
        blank_id: Blank symbol ID.
        training: Whether the model is initialized in training or inference mode.

    """

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add arguments for Transducer model."""
        E2E.encoder_add_general_arguments(parser)
        E2E.encoder_add_rnn_arguments(parser)
        E2E.encoder_add_custom_arguments(parser)

        E2E.decoder_add_general_arguments(parser)
        E2E.decoder_add_rnn_arguments(parser)
        E2E.decoder_add_custom_arguments(parser)

        E2E.training_add_custom_arguments(parser)
        E2E.transducer_add_arguments(parser)
        E2E.auxiliary_task_add_arguments(parser)

        return parser

    @staticmethod
    def encoder_add_general_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add general arguments for encoder."""
        group = parser.add_argument_group("Encoder general arguments")
        group = add_encoder_general_arguments(group)

        return parser

    @staticmethod
    def encoder_add_rnn_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add arguments for RNN encoder."""
        group = parser.add_argument_group("RNN encoder arguments")
        group = add_rnn_encoder_arguments(group)

        return parser

    @staticmethod
    def encoder_add_custom_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add arguments for Custom encoder."""
        group = parser.add_argument_group("Custom encoder arguments")
        group = add_custom_encoder_arguments(group)

        return parser

    @staticmethod
    def decoder_add_general_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add general arguments for decoder."""
        group = parser.add_argument_group("Decoder general arguments")
        group = add_decoder_general_arguments(group)

        return parser

    @staticmethod
    def decoder_add_rnn_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add arguments for RNN decoder."""
        group = parser.add_argument_group("RNN decoder arguments")
        group = add_rnn_decoder_arguments(group)

        return parser

    @staticmethod
    def decoder_add_custom_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add arguments for Custom decoder."""
        group = parser.add_argument_group("Custom decoder arguments")
        group = add_custom_decoder_arguments(group)

        return parser

    @staticmethod
    def training_add_custom_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add arguments for Custom architecture training."""
        group = parser.add_argument_group("Training arguments for custom archictecture")
        group = add_custom_training_arguments(group)

        return parser

    @staticmethod
    def transducer_add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add arguments for Transducer model."""
        group = parser.add_argument_group("Transducer model arguments")
        group = add_transducer_arguments(group)

        return parser

    @staticmethod
    def auxiliary_task_add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Add arguments for auxiliary task."""
        group = parser.add_argument_group("Auxiliary task arguments")
        group = add_auxiliary_task_arguments(group)

        return parser

    @property
    def attention_plot_class(self):
        """Get attention plot class."""
        return PlotAttentionReport

    def get_total_subsampling_factor(self) -> float:
        """Get total subsampling factor."""
        if self.etype == "custom":
            return self.encoder.conv_subsampling_factor * int(
                numpy.prod(self.subsample)
            )
        else:
            return self.enc.conv_subsampling_factor * int(numpy.prod(self.subsample))

    def __init__(
        self,
        idim: int,
        odim: int,
        args: Namespace,
        ignore_id: int = -1,
        blank_id: int = 0,
        training: bool = True,
    ):
        """Construct an E2E object for Transducer model."""
        torch.nn.Module.__init__(self)

        args = fill_missing_args(args, self.add_arguments)

        self.is_transducer = True

        self.use_auxiliary_enc_outputs = (
            True if (training and args.use_aux_transducer_loss) else False
        )

        self.subsample = get_subsample(
            args, mode="asr", arch="transformer" if args.etype == "custom" else "rnn-t"
        )

        if self.use_auxiliary_enc_outputs:
            n_layers = (
                ((len(args.enc_block_arch) * args.enc_block_repeat) - 1)
                if args.enc_block_arch is not None
                else (args.elayers - 1)
            )

            aux_enc_output_layers = valid_aux_encoder_output_layers(
                args.aux_transducer_loss_enc_output_layers,
                n_layers,
                args.use_symm_kl_div_loss,
                self.subsample,
            )
        else:
            aux_enc_output_layers = []

        if args.etype == "custom":
            if args.enc_block_arch is None:
                raise ValueError(
                    "When specifying custom encoder type, --enc-block-arch"
                    "should be set in training config."
                )

            self.encoder = CustomEncoder(
                idim,
                args.enc_block_arch,
                args.custom_enc_input_layer,
                repeat_block=args.enc_block_repeat,
                self_attn_type=args.custom_enc_self_attn_type,
                positional_encoding_type=args.custom_enc_positional_encoding_type,
                positionwise_activation_type=args.custom_enc_pw_activation_type,
                conv_mod_activation_type=args.custom_enc_conv_mod_activation_type,
                aux_enc_output_layers=aux_enc_output_layers,
                input_layer_dropout_rate=args.custom_enc_input_dropout_rate,
                input_layer_pos_enc_dropout_rate=(
                    args.custom_enc_input_pos_enc_dropout_rate
                ),
            )
            encoder_out = self.encoder.enc_out
        else:
            self.enc = encoder_for(
                args,
                idim,
                self.subsample,
                aux_enc_output_layers=aux_enc_output_layers,
            )
            encoder_out = args.eprojs

        try:
            ILM_loss=args.ILM_loss
        except:
            ILM_loss=False
        try:
            self.IAM_loss=args.IAM_loss
        except:
            self.IAM_loss=False
        try:
            ILM_loss_weight=args.ILM_loss_weight
        except:
            ILM_loss_weight=0
        try:
            self.IAM_loss_weight=args.IAM_loss_weight
        except:
            self.IAM_loss_weight=0
        try:
            eta_mixing=args.eta_mixing
        except:
            eta_mixing=False
        try:
            eta_mixing_type=args.eta_mixing_type
        except:
            eta_mixing_type='linear'
        try:
            future_context_lm=args.future_context_lm
        except:
            future_context_lm=False
        try:
            future_context_lm_kernel=args.future_context_lm_kernel
        except:
            future_context_lm_kernel=0
        try:
            self.future_context_lm_type=args.future_context_lm_type
        except:
            self.future_context_lm_type='linear'
        try:
            self.future_context_lm_linear_layers=args.future_context_lm_linear_layers
        except:
            self.future_context_lm_linear_layers=1
        try:
            self.future_context_lm_linear_units=args.future_context_lm_linear_units
        except:
            self.future_context_lm_linear_units=256
        try:
           self.la_embed_size=args.la_embed_size
        except:
           self.la_embed_size=128
        try:
           self.la_window=args.la_window
        except:
           self.la_window=4
        try:
           self.la_window_left=args.la_window_left
        except:
           self.la_window_left=0
        try:
           self.la_greedy_scheduled_sampling_probability=args.la_greedy_scheduled_sampling_probability
        except:
           self.la_greedy_scheduled_sampling_probability=0.2
        self.char_list = args.char_list
        try:
            self.la_teacher_forcing_dist_threshold = args.la_teacher_forcing_dist_threshold
        except:
            self.la_teacher_forcing_dist_threshold = 0.1

        try:
            self.acoustic_warm_start = args.acoustic_warm_start
            self.acoustic_warm_start_epoch = args.acoustic_warm_start_epoch
        except:
            self.acoustic_warm_start = False
            self.acoustic_warm_start_epoch = 2
        try:
            self.topK = args.topK
        except:
            self.topK = 5
        try:
            self.ctc_charVocab_file=args.ctc_charVocab_file
            self.ctc_type=args.ctc_type
            self.IAM_loss_type=args.IAM_loss_type
        except:
            self.ctc_charVocab_file=''
            self.ctc_type='default'
            self.IAM_loss_type='default'

        if args.dtype == "custom":
            if args.dec_block_arch is None:
                raise ValueError(
                    "When specifying custom decoder type, --dec-block-arch"
                    "should be set in training config."
                )

            self.decoder = CustomDecoder(
                odim,
                args.dec_block_arch,
                args.custom_dec_input_layer,
                repeat_block=args.dec_block_repeat,
                positionwise_activation_type=args.custom_dec_pw_activation_type,
                input_layer_dropout_rate=args.dropout_rate_embed_decoder,
                blank_id=blank_id,
            )
            decoder_out = self.decoder.dunits
        else:
            self.dec = RNNDecoder(
                odim,
                args.dtype,
                args.dlayers,
                args.dunits,
                args.dec_embed_dim,
                dropout_rate=args.dropout_rate_decoder,
                dropout_rate_embed=args.dropout_rate_embed_decoder,
                blank_id=blank_id,
                future_context_lm_type=self.future_context_lm_type,
                encoder_out=encoder_out,
            )
            decoder_out = args.dunits

        self.transducer_tasks = TransducerTasks(
            encoder_out,
            decoder_out,
            args.joint_dim,
            odim,
            joint_activation_type=args.joint_activation_type,
            transducer_loss_weight=args.transducer_weight,
            ctc_loss=args.use_ctc_loss,
            ctc_loss_weight=args.ctc_loss_weight,
            ctc_loss_dropout_rate=args.ctc_loss_dropout_rate,
            lm_loss=args.use_lm_loss,
            lm_loss_weight=args.lm_loss_weight,
            lm_loss_smoothing_rate=args.lm_loss_smoothing_rate,
            aux_transducer_loss=args.use_aux_transducer_loss,
            aux_transducer_loss_weight=args.aux_transducer_loss_weight,
            aux_transducer_loss_mlp_dim=args.aux_transducer_loss_mlp_dim,
            aux_trans_loss_mlp_dropout_rate=args.aux_transducer_loss_mlp_dropout_rate,
            symm_kl_div_loss=args.use_symm_kl_div_loss,
            symm_kl_div_loss_weight=args.symm_kl_div_loss_weight,
            fastemit_lambda=args.fastemit_lambda,
            blank_id=blank_id,
            ignore_id=ignore_id,
            training=training,
            ILM_loss=ILM_loss,
            IAM_loss=self.IAM_loss,
            ILM_loss_weight=ILM_loss_weight,
            IAM_loss_weight=self.IAM_loss_weight,
            eta_mixing=eta_mixing,
            eta_mixing_type=eta_mixing_type,
            future_context_lm=future_context_lm,
            future_context_lm_kernel=future_context_lm_kernel,
            future_context_lm_type=self.future_context_lm_type,
            future_context_lm_linear_layers=self.future_context_lm_linear_layers,
            future_context_lm_linear_units=self.future_context_lm_linear_units,
            la_embed_size=self.la_embed_size,
            la_window=self.la_window,
            la_window_left=self.la_window_left,
            la_greedy_scheduled_sampling_probability=self.la_greedy_scheduled_sampling_probability,
            la_teacher_forcing_dist_threshold = self.la_teacher_forcing_dist_threshold,
            topK = self.topK, 
            ctc_charVocab_file = self.ctc_charVocab_file,
            ctc_type = self.ctc_type,
            IAM_loss_type = self.IAM_loss_type,

        )

        if training and (args.report_cer or args.report_wer):
            self.error_calculator = ErrorCalculator(
                self.decoder if args.dtype == "custom" else self.dec,
                self.transducer_tasks.joint_network,
                args.char_list,
                args.sym_space,
                args.sym_blank,
                args.report_cer,
                args.report_wer,
            )
        else:
            self.error_calculator = None

        self.etype = args.etype
        self.dtype = args.dtype

        self.sos = odim - 1
        self.eos = odim - 1
        self.blank_id = blank_id
        self.ignore_id = ignore_id

        self.space = args.sym_space
        self.blank = args.sym_blank

        self.odim = odim

        self.reporter = Reporter()

        self.default_parameters(args)

        self.loss = None
        self.rnnlm = None

        self.current_epoch = 0

    def default_parameters(self, args: Namespace):
        """Initialize/reset parameters for Transducer.

        Args:
            args: Namespace containing model options.

        """
        initializer(self, args)

    def forward(
            self, feats: torch.Tensor, feats_len: torch.Tensor, labels: torch.Tensor, 
    ) -> torch.Tensor:
        """E2E forward.

        Args:
            feats: Feature sequences. (B, F, D_feats)
            feats_len: Feature sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Returns:
            loss: Transducer loss value

        """
        # 1. encoder
        feats = feats[:, : max(feats_len)]

        if self.etype == "custom":
            feats_mask = (
                make_non_pad_mask(feats_len.tolist()).to(feats.device).unsqueeze(-2)
            )

            _enc_out, _enc_out_len = self.encoder(feats, feats_mask)
        else:
            _enc_out, _enc_out_len, _ = self.enc(feats, feats_len)

        if self.use_auxiliary_enc_outputs:
            enc_out, aux_enc_out = _enc_out[0], _enc_out[1]
            enc_out_len, aux_enc_out_len = _enc_out_len[0], _enc_out_len[1]
        else:
            enc_out, aux_enc_out = _enc_out, None
            enc_out_len, aux_enc_out_len = _enc_out_len, None

        # 2. decoder
        dec_in = get_decoder_input(labels, self.blank_id, self.ignore_id)

        if self.dtype == "custom":
            self.decoder.set_device(enc_out.device)

            dec_in_mask = target_mask(dec_in, self.blank_id)
            dec_out, _ = self.decoder(dec_in, dec_in_mask)
        else:
            self.dec.set_device(enc_out.device)

            if self.future_context_lm_type.lower() == 'linear':
                dec_out = self.dec(dec_in,torch.zeros(1))
            elif self.future_context_lm_type.lower() == 'lstm':
                zero_pad = torch.nn.ConstantPad1d((0,self.transducer_tasks.joint_network.future_context_lm_kernel-1),0)
                convolved_am = self.transducer_tasks.joint_network.future_context_conv_network(zero_pad(enc_out.squeeze(2).transpose(1,2))).transpose(1,2).unsqueeze(2)
                dec_out = self.dec(dec_in,convolved_am)
            else:
                dec_out = self.dec(dec_in,torch.zeros(1))
                

        # 3. Transducer task and auxiliary tasks computation
        losses = self.transducer_tasks(
            enc_out,
            aux_enc_out,
            dec_out,
            labels,
            enc_out_len,
            aux_enc_out_len,
            char_list=self.char_list,
        )

        if self.training or self.error_calculator is None:
            cer, wer = None, None
        else:
            cer, wer = self.error_calculator(
                enc_out, self.transducer_tasks.get_target()
            )

        if not(self.acoustic_warm_start) or (self.current_epoch >= self.acoustic_warm_start_epoch):
            self.loss = sum(losses)
        else:
            self.loss = losses[6]/self.IAM_loss_weight  # 7th loss component is IAM loss
        loss_data = float(self.loss)

        if not math.isnan(loss_data):
            self.reporter.report(
                loss_data,
                *[float(loss) for loss in losses],
                cer,
                wer,
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)

        return self.loss

    def encode_custom(self, feats: numpy.ndarray) -> torch.Tensor:
        """Encode acoustic features.

        Args:
            feats: Feature sequence. (F, D_feats)

        Returns:
            enc_out: Encoded feature sequence. (T, D_enc)

        """
        feats = torch.as_tensor(feats).unsqueeze(0)
        enc_out, _ = self.encoder(feats, None)

        return enc_out.squeeze(0)

    def encode_rnn(self, feats: numpy.ndarray) -> torch.Tensor:
        """Encode acoustic features.

        Args:
            feats: Feature sequence. (F, D_feats)

        Returns:
            enc_out: Encoded feature sequence. (T, D_enc)

        """
        p = next(self.parameters())

        feats_len = [feats.shape[0]]

        feats = feats[:: self.subsample[0], :]
        feats = torch.as_tensor(feats, device=p.device, dtype=p.dtype)
        feats = feats.contiguous().unsqueeze(0)

        enc_out, _, _ = self.enc(feats, feats_len)

        return enc_out.squeeze(0)

    def recognize(
        self, feats: numpy.ndarray, beam_search: BeamSearchTransducer
    ) -> List:
        """Recognize input features.

        Args:
            feats: Feature sequence. (F, D_feats)
            beam_search: Beam search class.

        Returns:
            nbest_hyps: N-best decoding results.

        """
        self.eval()

        if self.etype == "custom":
            enc_out = self.encode_custom(feats)
        else:
            enc_out = self.encode_rnn(feats)

        nbest_hyps = beam_search(enc_out)

        return [asdict(n) for n in nbest_hyps]

    def calculate_all_attentions(
        self, feats: torch.Tensor, feats_len: torch.Tensor, labels: torch.Tensor
    ) -> numpy.ndarray:
        """E2E attention calculation.

        Args:
            feats: Feature sequences. (B, F, D_feats)
            feats_len: Feature sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Returns:
            ret: Attention weights with the following shape,
                1) multi-head case => attention weights. (B, D_att, U, T),
                2) other case => attention weights. (B, U, T)

        """
        self.eval()

        if self.etype != "custom" and self.dtype != "custom":
            return []
        else:
            with torch.no_grad():
                self.forward(feats, feats_len, labels)

            ret = dict()
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention) or isinstance(
                    m, RelPositionMultiHeadedAttention
                ):
                    ret[name] = m.attn.cpu().numpy()

        self.train()

        return ret
