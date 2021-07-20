import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_electra import ElectraModel, ElectraPreTrainedModel


def get_label_weight(labels, num_label):
    number_of_labels = []

    # 입력 데이터의 전체 수를 저장하기위한 변수
    # 데이터 10개 중, 9개가 1번 라벨이고 1개가 2번 라벨일 경우
    # weight는 [0.1, 0.9]로 계산
    number_of_total = torch.zeros(size=(1,), dtype=torch.float, device=torch.device("cuda"))

    for label_index in range(0, num_label):
        # 라벨 index를 순차적으로 받아와 현재 라벨(label_index)에 해당하는 데이터 수를 계산
        number_of_label = (labels == label_index).sum(dim=-1).float()

        # 현재 라벨 분포 저장
        number_of_labels.append(number_of_label)

        # 전체 분모에 현재 라벨을 가진 데이터를 합치는 과정
        number_of_total = torch.add(number_of_total, number_of_label).float()

    # 리스트로 선언된 number_of_labels를 torch.tensor() 형태로 변환
    label_weight = torch.stack(tensors=number_of_labels, dim=0)

    # 각 라벨 분포를 전체 데이터 수로 나누어서 라벨 웨이트 계산
    label_weight = torch.ones(size=(1,), dtype=torch.float, device=torch.device("cuda")) - torch.div(label_weight,
                                                                                                     number_of_total)
    return label_weight

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def __call__(self, input_features):
        features_output1 = self.classifier1(input_features)
        features_output1 = F.gelu(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2

class AttentivePooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentivePooling, self).__init__()
        self.hidden_size = hidden_size
        self.q_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.c_projection = nn.Linear(self.hidden_size, self.hidden_size)

    def __call__(self, query, context, context_mask):
        context_mask = context_mask.masked_fill(context_mask == 0, -100)
        # query : [batch, hidden]
        # context : [batch, seq, hidden]
        # context_mask : [batch, seq, window]

        q = self.q_projection(query).unsqueeze(-1)
        c = self.c_projection(context)

        # q : [batch, hidden, 1]
        # c : [batch, seq, hidden]

        att = c.bmm(q)
        # att : [batch, seq, 1]

        expanded_att = att.expand(-1, -1, 200)
        # expanded_att : [batch, seq, window]
        masked_att = expanded_att + context_mask
        # masked_att : [batch, seq, window]


        att_alienment = F.softmax(masked_att, dim=1).transpose(1, 2)
        # att_alienment : [batch, window, seq]

        # result : [batch, window, hidden]
        result = att_alienment.bmm(c)

        return result
class ElectraForQuestionAnswering(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForQuestionAnswering, self).__init__(config)
        # 분류 해야할 라벨 개수 (start/end)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # ELECTRA 모델 선언
        self.electra = ElectraModel(config)

        # bi-gru layer 선언
        self.bi_gru = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                             num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
        self.question_encoder = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                                       num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
        # self.sent_att = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=1, bias=True)
        # bi-gru layer output을 2의 크기로 줄여주기 위한 fnn

        self.att_pool = AttentivePooling(config.hidden_size)
        self.sent_gru =  nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                             num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
        self.tok_qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # ELECTRA weight 초기화
        self.init_weights()

    def _get_question_vector(self, question_mask, sequence_outputs):
        question_mask = question_mask.unsqueeze(-1)

        encoded_question = sequence_outputs * question_mask
        encoded_question = encoded_question[:, :64, :]
        self.question_encoder.flatten_parameters()
        question_gru_outputs, question_gru_states = self.question_encoder(encoded_question)
        question_vector = torch.cat([question_gru_states[0], question_gru_states[1]], -1)
        return question_vector

    def _get_sentence_vector(self, sentence_mask, sequence_outputs, question_vector):
        # one_hot_sent_mask : [batch, 512, 200]
        one_hot_sent_mask = F.one_hot(sentence_mask, 200).float()
        # sent_output : [batch, window, hidden]
        sent_output = self.att_pool(question_vector, sequence_outputs, one_hot_sent_mask)
        return sent_output

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            sentence_mask=None,
            question_mask=None,
            inputs_embeds=None,
            long_start_positions = None,
            long_end_positions = None,
            short_start_positions=None,
            short_end_positions=None
    ):
        # outputs : [1, batch_size, seq_length, hidden_size]
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        # sequence_output : [batch_size, seq_length, hidden_size]
        sequence_output = outputs[0]
        self.bi_gru.flatten_parameters()
        # gru_output : [batch_size, seq_length, gru_hidden_size*2]
        tok_gru_output, _ = self.bi_gru(sequence_output)

        question_vector = self._get_question_vector(question_mask, tok_gru_output)

        sent_output = self._get_sentence_vector(sentence_mask, tok_gru_output, question_vector)
        self.sent_gru.flatten_parameters()
        sent_gru_output, _ = self.sent_gru(sent_output)


        sent_logits = self.tok_qa_outputs(sent_gru_output)
        sent_start_logits, sent_end_logits = sent_logits.split(1, dim=-1)
        sent_start_logits = sent_start_logits.squeeze(-1)
        sent_end_logits = sent_end_logits.squeeze(-1)

        tok_logits = self.tok_qa_outputs(tok_gru_output)
        tok_start_logits, tok_end_logits = tok_logits.split(1, dim=-1)
        tok_start_logits = tok_start_logits.squeeze(-1)
        tok_end_logits = tok_end_logits.squeeze(-1)

        norm_start_logits = torch.cat([tok_start_logits, sent_start_logits], 1)
        norm_end_logits = torch.cat([tok_end_logits, sent_end_logits], 1)
        # 학습 시
        if long_start_positions is not None:
            span_loss_fct = nn.CrossEntropyLoss()
            norm_start_positions = short_start_positions.masked_fill(short_start_positions == 0, 512) + long_start_positions
            norm_end_positions = short_end_positions.masked_fill(short_end_positions == 0, 512) + long_end_positions
            start_span_loss = span_loss_fct(norm_start_logits, norm_start_positions)
            end_span_loss = span_loss_fct(norm_end_logits, norm_end_positions)

            total_loss = (start_span_loss + end_span_loss) / 2

            return total_loss
        return sent_start_logits, sent_end_logits, tok_start_logits, tok_end_logits
