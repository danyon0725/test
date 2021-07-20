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
def _make_triu_mask(seq_len):
    mask = torch.ones([seq_len, seq_len], dtype=torch.float)
    triu_mask = torch.triu(mask)
    triu_mask = triu_mask.masked_fill(triu_mask == 0, -100)
    triu_mask = triu_mask.masked_fill(triu_mask == 1, 0)

    return triu_mask

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

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.start_projection_layer = nn.Linear(config.hidden_size, int(config.hidden_size / 4))
        self.end_projection_layer = nn.Linear(config.hidden_size, int(config.hidden_size / 4))
        self.matrix_outputs = MultiNonLinearClassifier(int(config.hidden_size/2), 1, 0.2)
        self.valid_outputs = MultiNonLinearClassifier(int(config.hidden_size/2), 2, 0.2)
        # ELECTRA weight 초기화
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            question_mask=None,
            sentence_mask=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions = None,
            end_positions = None,
            start_position=None,
            end_position=None
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


        tok_logits = self.qa_outputs(tok_gru_output)
        tok_start_logits, tok_end_logits = tok_logits.split(1, dim=-1)
        tok_start_logits = tok_start_logits.squeeze(-1)
        tok_end_logits = tok_end_logits.squeeze(-1)

        tok_start_vectors = self.start_projection_layer(tok_gru_output)
        tok_end_vectors = self.end_projection_layer(tok_gru_output)
        batch_size = tok_start_vectors.size(0)
        tok_len = tok_end_vectors.size(1)

        expanded_start_vectors = tok_start_vectors.unsqueeze(2).expand(-1, -1, tok_len, -1)
        expanded_end_vectors = tok_end_vectors.unsqueeze(1).expand(-1, tok_len, -1, -1)

        span_matrix = torch.cat([expanded_start_vectors, expanded_end_vectors], -1)
        matrix_logits = self.matrix_outputs(span_matrix).squeeze(-1)
        valid_logits = self.valid_outputs(span_matrix)
        valid_prob = F.softmax(valid_logits, dim=-1)[:, :, :, 1]

        triu_mask = _make_triu_mask(tok_len).unsqueeze(0).expand(batch_size, -1, -1).cuda()
        matrix_logits = matrix_logits + triu_mask
        # 학습 시
        if start_positions is not None and end_positions is not None:
            span_loss_fct = nn.CrossEntropyLoss()
            valid_loss_fct = nn.CrossEntropyLoss()
            matrix_loss_fct = nn.CrossEntropyLoss(reduction='none')

            row_label = torch.zeros([batch_size, tok_len], dtype=torch.long).cuda()
            col_label = torch.zeros([batch_size, tok_len], dtype = torch.long).cuda()
            valid_label = torch.zeros([batch_size, tok_len, tok_len], dtype=torch.long).cuda()
            for batch_idx in range(batch_size):
                for answer_idx in range(len(start_positions[batch_idx])):
                    if start_positions[batch_idx][answer_idx] == 0:
                        break
                    row_label[batch_idx][start_positions[batch_idx][answer_idx]] = end_positions[batch_idx][answer_idx]
                    col_label[batch_idx][end_positions[batch_idx][answer_idx]] = start_positions[batch_idx][answer_idx]
                    valid_label[batch_idx][start_positions[batch_idx][answer_idx]][end_positions[batch_idx][answer_idx]] = 1

            row_mask = torch.sum(valid_label, 2).cuda()
            col_mask = torch.sum(valid_label, 1).cuda()

            row_loss = matrix_loss_fct(matrix_logits.view(-1, tok_len),
                row_label.view(-1)).reshape(batch_size, tok_len)
            col_loss = matrix_loss_fct(matrix_logits.transpose(1, 2).reshape(batch_size * tok_len, tok_len),
                col_label.view(-1)).reshape(batch_size, tok_len)

            final_row_loss = torch.mean(torch.sum(row_loss * row_mask, 1))
            final_col_loss = torch.mean(torch.sum(col_loss * col_mask, 1))

            matrix_loss = (final_row_loss + final_col_loss) / 2

            valid_loss = valid_loss_fct(valid_logits.view(-1, 2), valid_label.view(-1,))


            start_span_loss = span_loss_fct(tok_start_logits, start_position)
            end_span_loss = span_loss_fct(tok_end_logits, end_position)

            span_loss = (start_span_loss + end_span_loss) / 2

            total_loss = (matrix_loss + valid_loss + span_loss) / 3

            return total_loss, matrix_loss, span_loss, valid_loss
        return tok_start_logits, tok_end_logits, matrix_logits, valid_prob
