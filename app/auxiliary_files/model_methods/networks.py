import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from app.auxiliary_files.model_methods.auxiliary_networks_scinet import EncoderTree
from app.auxiliary_files.model_methods.auxiliary_networks_informer import Encoder, EncoderLayer, ConvLayer
from app.auxiliary_files.model_methods.auxiliary_networks_informer import Decoder, DecoderLayer
from app.auxiliary_files.model_methods.auxiliary_networks_informer import FullAttention, ProbAttention, AttentionLayer
from app.auxiliary_files.model_methods.auxiliary_networks_informer import DataEmbedding
from app.auxiliary_files.model_methods.auxiliary_networks_autopilot import get_granular_rec, get_peak_rec, get_aggregated_signal, get_wavg_rec, get_jp_rec, get_ml_rec


def select_network(config, lag_size: int, prediction_size: int, number_features: int, device: str):
    return eval(config['name'])(lag_size, prediction_size, number_features, device, **config['args'])


# for debugging
class PrintLayer(nn.Module):
    def __init__(self, name):
        super(PrintLayer, self).__init__()
        self.name = name

    def forward(self, x):
        print(self.name + ': -> ', x.shape)
        return x


class AutopilotWindowRecommenderPeak(nn.Module):
    def __init__(
            self,
            lag_size: int,
            prediction_size: int,
            number_features: int,
            device: str,
            min_cpu: float,
            max_cpu: float,
            mean_cpu: float
    ):
        super(AutopilotWindowRecommenderPeak, self).__init__()

        num_buckets = 400
        self.resolution = 20
        self.n = 12

        self.mean_cpu = mean_cpu
        bucket_size_cpu = (max_cpu - abs(min_cpu)) / num_buckets
        self.cpu_buckets = np.linspace(min_cpu + bucket_size_cpu, max_cpu, num=num_buckets)
        self.cpu_bins = np.linspace(min_cpu, max_cpu, num=num_buckets + 1)

    def forward(self, time_series_list, y=None):
        return

    def predict(self, time_series_list):
        output = []
        for time_series in time_series_list:
            aggregated_signal = get_aggregated_signal(time_series, self.resolution, self.cpu_bins)
            output.append(get_granular_rec(get_peak_rec(aggregated_signal, self.cpu_buckets, self.n), self.mean_cpu, self.resolution)[:time_series.shape[0]])
        return output


class AutopilotWindowRecommenderWeighted(nn.Module):
    def __init__(
            self,
            lag_size: int,
            prediction_size: int,
            number_features: int,
            device: str,
            min_cpu: float,
            max_cpu: float,
            mean_cpu: float
    ):
        super(AutopilotWindowRecommenderWeighted, self).__init__()

        self.num_buckets = 400
        self.resolution = 20
        self.n = 12
        self.half_life_cpu = 12

        self.mean_cpu = mean_cpu
        bucket_size_cpu = (max_cpu - abs(min_cpu)) / self.num_buckets
        self.cpu_buckets = np.linspace(min_cpu + bucket_size_cpu, max_cpu, num=self.num_buckets)
        self.cpu_bins = np.linspace(min_cpu, max_cpu, num=self.num_buckets + 1)

    def forward(self, time_series_list, y=None):
        return

    def predict(self, time_series_list):
        output = []
        for time_series in time_series_list:
            aggregated_signal = get_aggregated_signal(time_series, self.resolution, self.cpu_bins)
            output.append(get_granular_rec(get_wavg_rec(aggregated_signal, self.cpu_buckets, self.half_life_cpu, self.num_buckets, self.n), self.mean_cpu, self.resolution)[:time_series.shape[0]])
        return output


class AutopilotWindowRecommenderPercentile(nn.Module):
    def __init__(
            self,
            lag_size: int,
            prediction_size: int,
            number_features: int,
            device: str,
            min_cpu: float,
            max_cpu: float,
            mean_cpu: float
    ):
        super(AutopilotWindowRecommenderPercentile, self).__init__()

        self.num_buckets = 400
        self.resolution = 20
        self.n = 12
        self.j_cpu = 95
        self.half_life_cpu = 12

        self.mean_cpu = mean_cpu
        bucket_size_cpu = (max_cpu - abs(min_cpu)) / self.num_buckets
        self.cpu_buckets = np.linspace(min_cpu + bucket_size_cpu, max_cpu, num=self.num_buckets)
        self.cpu_bins = np.linspace(min_cpu, max_cpu, num=self.num_buckets + 1)

    def forward(self, time_series_list, y=None):
        return

    def predict(self, time_series_list):
        output = []
        for time_series in time_series_list:
            aggregated_signal = get_aggregated_signal(time_series, self.resolution, self.cpu_bins)
            output.append(get_granular_rec(get_jp_rec(aggregated_signal, self.cpu_buckets, self.half_life_cpu, self.j_cpu, self.num_buckets), self.mean_cpu, self.resolution)[:time_series.shape[0]])
        return output


class AutopilotMLRecommender(nn.Module):
    def __init__(
            self,
            lag_size: int,
            prediction_size: int,
            number_features: int,
            device: str,
            min_cpu: float,
            max_cpu: float,
            mean_cpu: float
    ):
        super(AutopilotMLRecommender, self).__init__()

        self.num_buckets = 400
        self.resolution = 20
        self.n = 12
        self.j_cpu = 95
        self.half_life_cpu = 12

        self.mean_cpu = mean_cpu
        bucket_size_cpu = (max_cpu - abs(min_cpu)) / self.num_buckets
        self.cpu_buckets = np.linspace(min_cpu + bucket_size_cpu, max_cpu, num=self.num_buckets)
        self.cpu_bins = np.linspace(min_cpu, max_cpu, num=self.num_buckets + 1)

        self.w_o, self.w_u, self.w_delta_L, self.w_delta_m = 0.5, 0.25, 0.1, 0.1
        self.d = 0.75
        self.dm_min, self.dm_max, self.d_n_step = 0.1, 1.0, 10
        self.Mm_min, self.Mm_max, self.M_n_step = 0, 1, 2

    def forward(self, time_series_list, y=None):
        return

    def predict(self, time_series_list):
        output = []
        for time_series in time_series_list:
            aggregated_signal = get_aggregated_signal(time_series, self.resolution, self.cpu_bins)
            output.append(get_granular_rec(get_ml_rec(aggregated_signal, self.cpu_buckets, self.dm_min, self.dm_max, self.d_n_step, self.Mm_min, self.Mm_max, self.M_n_step, self.w_delta_m, self.w_delta_L, self.w_o, self.w_u, self.d), self.mean_cpu, self.resolution)[:time_series.shape[0]])
        return output


class DeepConvolutionalClassifier(nn.Module):
    def __init__(self, lag_size: int, prediction_size: int, number_features: int, device: str):
        super(DeepConvolutionalClassifier, self).__init__()
        linear_input = number_features * 2 * int(math.ceil(int(math.ceil(int(math.ceil(lag_size / 2)) / 2)) / 2))
        self.layers = nn.Sequential(
            nn.Conv1d(number_features, number_features, kernel_size=(3,), stride=(2,), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(number_features),
            nn.Conv1d(number_features, number_features * 2, kernel_size=(3,), stride=(2,), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(number_features * 2),
            nn.Conv1d(number_features * 2, number_features * 2, kernel_size=(3,), stride=(2,), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(number_features * 2),
            nn.Flatten(),
            nn.Linear(linear_input, linear_input // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(linear_input // 2),
            nn.Linear(linear_input // 2, prediction_size)
        )

    def forward(self, x, y=None):
        return self.layers(x)

    def predict(self, x):
        return self.forward(x, None)


class DeepConvolutionalClassifierInceptionBranch3NoEncBigBottleneckComposed(nn.Module):
    def __init__(self, lag_size: int, prediction_size: int, number_features: int, device: str):
        super(DeepConvolutionalClassifierInceptionBranch3NoEncBigBottleneckComposed, self).__init__()
        self.lag_size = lag_size
        self.number_features = number_features
        self.feature_reduction = nn.Sequential(
            nn.Conv1d(lag_size, lag_size * 2, kernel_size=(number_features,), stride=(3,), padding=0, groups=lag_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size * 2)
        )
        self.inception_1 = nn.Sequential(
            nn.Conv1d(lag_size * 2, lag_size, kernel_size=(1,), stride=(2,), padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size),
            nn.Conv1d(lag_size, lag_size // 2, kernel_size=(1,), stride=(2,), padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size // 2),
            nn.Flatten(),
            nn.Linear(lag_size // 2, lag_size // 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size // 8)
        )
        self.upsample_1 = nn.Sequential(
            nn.Conv1d(lag_size // 8, lag_size // 2, kernel_size=(1,), stride=(2,), padding=0)
        )
        self.inception_2 = nn.Sequential(
            nn.Conv1d(lag_size + lag_size // 2, lag_size // 2, kernel_size=(1,), stride=(2,), padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size // 2),
            nn.Conv1d(lag_size // 2, lag_size // 4, kernel_size=(1,), stride=(2,), padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size // 4),
            nn.Flatten(),
            nn.Linear(lag_size // 4, lag_size // 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size // 16)
        )
        self.upsample_2 = nn.Sequential(
            nn.Conv1d(lag_size // 16, lag_size // 4, kernel_size=(1,), stride=(2,), padding=0)
        )
        self.inception_3 = nn.Sequential(
            nn.Conv1d(lag_size // 2 + lag_size // 4, lag_size // 4, kernel_size=(1,), stride=(2,), padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size // 4),
            nn.Conv1d(lag_size // 4, lag_size // 8, kernel_size=(1,), stride=(2,), padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size // 8),
            nn.Flatten(),
            nn.Linear(lag_size // 8, lag_size // 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size // 32)
        )
        self.union = nn.Sequential(
            nn.Linear(lag_size // 8 + lag_size // 16 + lag_size // 32, prediction_size),
        )

    def forward(self, x, y=None):
        x = torch.transpose(x, 1, 2)
        x = self.feature_reduction(x)
        x1 = self.inception_1(x)
        x1_upsampled = self.upsample_1(x1.reshape((x.shape[0], self.lag_size // 8, 1)))
        x2 = torch.cat((x1_upsampled, x[:, -self.lag_size:]), 1)
        x2 = self.inception_2(x2)
        x2_upsampled = self.upsample_2(x2.reshape((x.shape[0], self.lag_size // 16, 1)))
        x3 = torch.cat((x2_upsampled, x[:, -(self.lag_size // 2):]), 1)
        x3 = self.inception_3(x3)
        x = torch.cat((x1, x2, x3), 1)
        return self.union(x)

    def predict(self, x):
        return self.forward(x, None)


class SCINet(nn.Module):
    def __init__(
            self, lag_size: int,
            prediction_size: int,
            number_features: int,
            device: str,
            hid_size=1,
            num_stacks=1,
            num_levels=3,
            concat_len=0,
            groups=1,
            kernel=5,
            dropout=0.5,
            single_step_output_One=0,
            input_len_seg=0,
            positionalE=False,
            modified=True,
            RIN=False
    ):
        super().__init__()
        output_len = prediction_size
        input_len = lag_size
        input_dim = number_features

        self.input_dim = input_dim
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_size = hid_size
        self.num_levels = num_levels
        self.groups = groups
        self.modified = modified
        self.kernel_size = kernel
        self.dropout = dropout
        self.single_step_output_One = single_step_output_One
        self.concat_len = concat_len
        self.pe = positionalE
        self.RIN = RIN

        self.blocks1 = EncoderTree(
            in_planes=self.input_dim,
            num_levels=self.num_levels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            groups=self.groups,
            hidden_size=self.hidden_size,
            INN=modified)

        if num_stacks == 2:  # we only implement two stacks at most.
            self.blocks2 = EncoderTree(
                in_planes=self.input_dim,
                num_levels=self.num_levels,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
                groups=self.groups,
                hidden_size=self.hidden_size,
                INN=modified)

        self.stacks = num_stacks

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.projection1 = nn.Conv1d(self.input_len, self.output_len, kernel_size=1, stride=1, bias=False)

        if self.single_step_output_One:  # only output the N_th timestep.
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(self.concat_len + self.output_len, 1,
                                                 kernel_size=1, bias=False)
                else:
                    self.projection2 = nn.Conv1d(self.input_len + self.output_len, 1,
                                                 kernel_size=1, bias=False)
        else:  # output the N timesteps.
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(self.concat_len + self.output_len, self.output_len,
                                                 kernel_size=1, bias=False)
                else:
                    self.projection2 = nn.Conv1d(self.input_len + self.output_len, self.output_len,
                                                 kernel_size=1, bias=False)

        # For positional encoding
        self.pe_hidden_size = input_dim
        if self.pe_hidden_size % 2 == 1:
            self.pe_hidden_size += 1

        num_timescales = self.pe_hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                max(num_timescales - 1, 1))
        temp = torch.arange(num_timescales, dtype=torch.float32)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

        ### RIN Parameters ###
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, input_dim))

    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device)  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
        temp1 = position.unsqueeze(1)  # 5 1
        temp2 = self.inv_timescales.unsqueeze(0)  # 1 256
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  # [T, C]
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)

        return signal

    def forward(self, x, y=None):
        x = torch.transpose(x, 1, 2)
        assert self.input_len % (np.power(2, self.num_levels)) == 0  # evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)
        if self.pe:
            pe = self.get_position_encoding(x)
            if pe.shape[2] > x.shape[2]:
                x += pe[:, :, :-1]
            else:
                x += self.get_position_encoding(x)

        ### activated when RIN flag is set ###
        if self.RIN:
            print('/// RIN ACTIVATED ///\r', end='')
            means = x.mean(1, keepdim=True).detach()
            # mean
            x = x - means
            # var
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            # affine
            # print(x.shape,self.affine_weight.shape,self.affine_bias.shape)
            x = x * self.affine_weight + self.affine_bias

        # the first stack
        res1 = x
        x = self.blocks1(x)
        x += res1
        x = self.projection1(x)

        if self.stacks == 1:
            ### reverse RIN ###
            if self.RIN:
                x = x - self.affine_bias
                x = x / (self.affine_weight + 1e-10)
                x = x * stdev
                x = x + means

            return x[:, :, 0]  # only first feature

        elif self.stacks == 2:
            MidOutPut = x
            if self.concat_len:
                x = torch.cat((res1[:, -self.concat_len:, :], x), dim=1)
            else:
                x = torch.cat((res1, x), dim=1)

            # the second stack
            res2 = x
            x = self.blocks2(x)
            x += res2
            x = self.projection2(x)

            ### Reverse RIN ###
            if self.RIN:
                MidOutPut = MidOutPut - self.affine_bias
                MidOutPut = MidOutPut / (self.affine_weight + 1e-10)
                MidOutPut = MidOutPut * stdev
                MidOutPut = MidOutPut + means

            if self.RIN:
                x = x - self.affine_bias
                x = x / (self.affine_weight + 1e-10)
                x = x * stdev
                x = x + means

            return x[:, :, 0]

    def predict(self, x):
        return self.forward(x, None)


class Informer(nn.Module):
    def __init__(self,
                 lag_size: int, prediction_size: int, number_features: int, device: str,
                 out_len=3, c_out=1,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='T', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 ):
        super(Informer, self).__init__()
        enc_in = number_features
        dec_in = number_features

        self.device = torch.device(device)
        self.label_len = lag_size // 2
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.final = nn.Linear(c_out * out_len, prediction_size)

    def __real_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        dec_out = dec_out.reshape((dec_out.shape[0], dec_out.shape[1] * dec_out.shape[2]))
        dec_out = dec_out[:, -self.pred_len:]
        dec_out = self.final(dec_out)
        return dec_out

    def forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark, target=None):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        dec_inp = torch.zeros([batch_y.shape[0], self.pred_len, batch_y.shape[-1]]).float().to(self.device)
        dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
        outputs = self.__real_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # batch_y = batch_y[:, -self.pred_len:, -1:].to(self.device)

        return outputs

    def predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        return self.forward(batch_x, batch_y, batch_x_mark, batch_y_mark, None)
