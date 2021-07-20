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
# class ElectraForQuestionAnswering(ElectraPreTrainedModel):
#     def __init__(self, config):
#         super(ElectraForQuestionAnswering, self).__init__(config)
#         # 분류 해야할 라벨 개수 (start/end)
#         self.num_labels = config.num_labels
#         self.hidden_size = config.hidden_size
#
#         # ELECTRA 모델 선언
#         self.electra = ElectraModel(config)
#
#         # bi-gru layer 선언
#         self.bi_gru = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
#                              num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
#         self.question_encoder = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
#                                        num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
#         # self.sent_att = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=1, bias=True)
#         # bi-gru layer output을 2의 크기로 줄여주기 위한 fnn
#
#         self.att_pool = AttentivePooling(config.hidden_size)
#         self.sent_gru =  nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
#                              num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
#         self.tok_qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
#         self.sent_qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
#
#         self.start_projection_layer = nn.Linear(config.hidden_size, int(config.hidden_size / 4))
#         self.end_projection_layer = nn.Linear(config.hidden_size, int(config.hidden_size / 4))
#         self.matrix_outputs = MultiNonLinearClassifier(int(config.hidden_size / 2), 1, 0.2)
#         self.valid_outputs = MultiNonLinearClassifier(int(config.hidden_size / 2), 2, 0.2)
#
#         # ELECTRA weight 초기화
#         self.init_weights()
#
#     def _get_question_vector(self, question_mask, sequence_outputs):
#         question_mask = question_mask.unsqueeze(-1)
#
#         encoded_question = sequence_outputs * question_mask
#         encoded_question = encoded_question[:, :64, :]
#         question_gru_outputs, question_gru_states = self.question_encoder(encoded_question)
#         question_vector = torch.cat([question_gru_states[0], question_gru_states[1]], -1)
#         return question_vector
#
#     def _get_sentence_vector(self, sentence_mask, sequence_outputs, question_vector):
#         # one_hot_sent_mask : [batch, 512, 200]
#         one_hot_sent_mask = F.one_hot(sentence_mask, 200).float()
#         # sent_output : [batch, window, hidden]
#         sent_output = self.att_pool(question_vector, sequence_outputs, one_hot_sent_mask)
#         return sent_output
#
#     def _span_matrix_with_valid_logits(self, encoded_vectors):
#         start_vectors = self.start_projection_layer(encoded_vectors)
#         end_vectors = self.end_projection_layer(encoded_vectors)
#         batch_size = start_vectors.size(0)
#         seq_len = start_vectors.size(1)
#
#         expanded_start_vectors = start_vectors.unsqueeze(2).expand(-1, -1, seq_len, -1)
#         expanded_end_vectors = end_vectors.unsqueeze(1).expand(-1, seq_len, -1, -1)
#
#         span_matrix = torch.cat([expanded_start_vectors, expanded_end_vectors], -1)
#         matrix_logits = self.matrix_outputs(span_matrix).squeeze(-1)
#         valid_logits = self.valid_outputs(span_matrix)
#
#         triu_mask = _make_triu_mask(seq_len).unsqueeze(0).expand(batch_size, -1, -1).cuda()
#         matrix_logits = matrix_logits + triu_mask
#
#         return matrix_logits, valid_logits
#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             token_type_ids=None,
#             position_ids=None,
#             head_mask=None,
#             sentence_mask=None,
#             question_mask=None,
#             inputs_embeds=None,
#             long_start_positions = None,
#             long_end_positions = None,
#             short_start_positions=None,
#             short_end_positions=None,
#             long_start_position=None,
#             long_end_position=None,
#             short_start_position=None,
#             short_end_position=None
#     ):
#         # outputs : [1, batch_size, seq_length, hidden_size]
#         outputs = self.electra(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds
#         )
#
#         # sequence_output : [batch_size, seq_length, hidden_size]
#         sequence_output = outputs[0]
#
#         # gru_output : [batch_size, seq_length, gru_hidden_size*2]
#         tok_gru_output, _ = self.bi_gru(sequence_output)
#
#         question_vector = self._get_question_vector(question_mask, tok_gru_output)
#
#         sent_output = self._get_sentence_vector(sentence_mask, tok_gru_output, question_vector)
#
#         sent_gru_output, _ = self.sent_gru(sent_output)
#
#         tok_matrix_logits, tok_valid_logits = self._span_matrix_with_valid_logits(tok_gru_output)
#         sent_matrix_logits, sent_valid_logits = self._span_matrix_with_valid_logits(sent_gru_output)
#
#         sent_logits = self.sent_qa_outputs(sent_gru_output)
#         sent_start_logits, sent_end_logits = sent_logits.split(1, dim=-1)
#         sent_start_logits = sent_start_logits.squeeze(-1)
#         sent_end_logits = sent_end_logits.squeeze(-1)
#
#         tok_logits = self.tok_qa_outputs(tok_gru_output)
#         tok_start_logits, tok_end_logits = tok_logits.split(1, dim=-1)
#         tok_start_logits = tok_start_logits.squeeze(-1)
#         tok_end_logits = tok_end_logits.squeeze(-1)
#
#         norm_start_logits = torch.cat([tok_start_logits, sent_start_logits], 1)
#         norm_end_logits = torch.cat([tok_end_logits, sent_end_logits], 1)
#         # 학습 시
#         if long_start_positions is not None:
#             ############################    Span Loss    ############################
#             span_loss_fct = nn.CrossEntropyLoss()
#             norm_start_position = short_start_position.masked_fill(short_start_position == 0, 512) + long_start_position
#             norm_end_position = short_end_position.masked_fill(short_end_position == 0, 512) + long_end_position
#             start_span_loss = span_loss_fct(norm_start_logits, norm_start_position)
#             end_span_loss = span_loss_fct(norm_end_logits, norm_end_position)
#
#             span_loss = (start_span_loss + end_span_loss) / 2
#
#             ############################    Matrix Loss    ############################
#             ############    Short Span Matrix Loss    ############
#             batch_size = tok_logits.size(0)
#             tok_len = tok_logits.size(1)
#             sent_len = sent_logits.size(1)
#
#             matrix_loss_fct = nn.CrossEntropyLoss(reduction='none')
#             tok_row_label = torch.zeros([batch_size, tok_len], dtype=torch.long).cuda()
#             tok_col_label = torch.zeros([batch_size, tok_len], dtype=torch.long).cuda()
#             tok_valid_label = torch.zeros([batch_size, tok_len, tok_len], dtype=torch.long).cuda()
#             for batch_idx in range(batch_size):
#                 for answer_idx in range(len(short_start_positions[batch_idx])):
#                     if short_start_positions[batch_idx][answer_idx] == 0:
#                         break
#                     tok_row_label[batch_idx][short_start_positions[batch_idx][answer_idx]] = short_end_positions[batch_idx][answer_idx]
#                     tok_col_label[batch_idx][short_end_positions[batch_idx][answer_idx]] = short_start_positions[batch_idx][answer_idx]
#                     tok_valid_label[batch_idx][short_start_positions[batch_idx][answer_idx]][short_end_positions[batch_idx][answer_idx]] = 1
#
#             tok_row_mask = torch.sum(tok_valid_label, 2).cuda()
#             tok_col_mask = torch.sum(tok_valid_label, 1).cuda()
#
#             tok_row_loss = matrix_loss_fct(tok_matrix_logits.view(-1, tok_len),
#                                        tok_row_label.view(-1)).reshape(batch_size, tok_len)
#             tok_col_loss = matrix_loss_fct(tok_matrix_logits.transpose(1, 2).reshape(batch_size * tok_len, tok_len),
#                                        tok_col_label.view(-1)).reshape(batch_size, tok_len)
#
#             tok_final_row_loss = torch.mean(torch.sum(tok_row_loss * tok_row_mask, 1))
#             tok_final_col_loss = torch.mean(torch.sum(tok_col_loss * tok_col_mask, 1))
#
#             tok_matrix_loss = (tok_final_row_loss + tok_final_col_loss) / 2
#
#             ############    Long Span Matrix Loss    ############
#             sent_row_label = torch.zeros([batch_size, sent_len], dtype=torch.long).cuda()
#             sent_col_label = torch.zeros([batch_size, sent_len], dtype=torch.long).cuda()
#             sent_valid_label = torch.zeros([batch_size, sent_len, sent_len], dtype=torch.long).cuda()
#             for batch_idx in range(batch_size):
#                 for answer_idx in range(len(long_start_positions[batch_idx])):
#                     if long_start_positions[batch_idx][answer_idx] == 0:
#                         break
#                     sent_row_label[batch_idx][long_start_positions[batch_idx][answer_idx]] = \
#                         long_end_positions[batch_idx][answer_idx]
#                     sent_col_label[batch_idx][long_end_positions[batch_idx][answer_idx]] = \
#                         long_start_positions[batch_idx][answer_idx]
#                     sent_valid_label[batch_idx][long_start_positions[batch_idx][answer_idx]][
#                         long_end_positions[batch_idx][answer_idx]] = 1
#
#             sent_row_mask = torch.sum(sent_valid_label, 2).cuda()
#             sent_col_mask = torch.sum(sent_valid_label, 1).cuda()
#
#             sent_row_loss = matrix_loss_fct(sent_matrix_logits.view(-1, sent_len),
#                                             sent_row_label.view(-1)).reshape(batch_size, sent_len)
#             sent_col_loss = matrix_loss_fct(sent_matrix_logits.transpose(1, 2).reshape(batch_size * sent_len, sent_len),
#                                             sent_col_label.view(-1)).reshape(batch_size, sent_len)
#
#             sent_final_row_loss = torch.mean(torch.sum(sent_row_loss * sent_row_mask, 1))
#             sent_final_col_loss = torch.mean(torch.sum(sent_col_loss * sent_col_mask, 1))
#
#             sent_matrix_loss = (sent_final_row_loss + sent_final_col_loss) / 2
#
#             matrix_loss = (tok_matrix_loss + sent_matrix_loss) / 2
#
#             ############################    Valid Loss    ############################
#             valid_loss_fct = nn.CrossEntropyLoss()
#             tok_valid_loss = valid_loss_fct(tok_valid_logits.view(-1, 2), tok_valid_label.view(-1, ))
#             sent_valid_loss = valid_loss_fct(sent_valid_logits.view(-1, 2), sent_valid_label.view(-1, ))
#             valid_loss = tok_valid_loss + sent_valid_loss
#
#             total_loss = (span_loss + matrix_loss + valid_loss)/3
#             return total_loss, matrix_loss, span_loss, valid_loss
#         tok_valid_prob = F.softmax(tok_valid_logits, dim=-1)[:, :, :, 1]
#         sent_valid_prob = F.softmax(sent_valid_logits, dim=-1)[:, :, :, 1]
#         return sent_start_logits, sent_end_logits, sent_matrix_logits, sent_valid_prob,\
#                tok_start_logits, tok_end_logits, tok_matrix_logits, tok_valid_prob

class ElectraForQuestionAnswering_v2(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForQuestionAnswering_v2, self).__init__(config)
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
        self.sent_qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.start_projection_layer = nn.Linear(config.hidden_size, int(config.hidden_size / 4))
        self.end_projection_layer = nn.Linear(config.hidden_size, int(config.hidden_size / 4))
        self.matrix_outputs = MultiNonLinearClassifier(int(config.hidden_size / 2), 1, 0.2)
        self.start_valid_outputs = MultiNonLinearClassifier(int(config.hidden_size / 4), 2, 0.2)
        self.end_valid_outputs = MultiNonLinearClassifier(int(config.hidden_size / 4), 2, 0.2)

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

    def _span_matrix_with_valid_logits(self, encoded_vectors):
        start_vectors = self.start_projection_layer(encoded_vectors)
        end_vectors = self.end_projection_layer(encoded_vectors)
        batch_size = start_vectors.size(0)
        seq_len = start_vectors.size(1)

        expanded_start_vectors = start_vectors.unsqueeze(2).expand(-1, -1, seq_len, -1)
        expanded_end_vectors = end_vectors.unsqueeze(1).expand(-1, seq_len, -1, -1)

        span_matrix = torch.cat([expanded_start_vectors, expanded_end_vectors], -1)
        matrix_logits = self.matrix_outputs(span_matrix).squeeze(-1)
        s_valid_logits = self.start_valid_outputs(start_vectors)
        e_valid_logits = self.end_valid_outputs(end_vectors)

        triu_mask = _make_triu_mask(seq_len).unsqueeze(0).expand(batch_size, -1, -1).cuda()
        matrix_logits = matrix_logits + triu_mask

        return matrix_logits, s_valid_logits, e_valid_logits
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
            short_end_positions=None,
            long_start_position=None,
            long_end_position=None,
            short_start_position=None,
            short_end_position=None
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

        tok_matrix_logits, tok_s_valid_logits, tok_e_valid_logits = self._span_matrix_with_valid_logits(tok_gru_output)
        sent_matrix_logits, sent_s_valid_logits, sent_e_valid_logits = self._span_matrix_with_valid_logits(sent_gru_output)

        sent_logits = self.sent_qa_outputs(sent_gru_output)
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
            ############################    Span Loss    ############################
            span_loss_fct = nn.CrossEntropyLoss()
            norm_start_position = short_start_position.masked_fill(short_start_position == 0, 512) + long_start_position
            norm_end_position = short_end_position.masked_fill(short_end_position == 0, 512) + long_end_position
            start_span_loss = span_loss_fct(norm_start_logits, norm_start_position)
            end_span_loss = span_loss_fct(norm_end_logits, norm_end_position)

            span_loss = (start_span_loss + end_span_loss) / 2

            ############################    Matrix Loss    ############################
            ############    Short Span Matrix Loss    ############
            batch_size = tok_logits.size(0)
            tok_len = tok_logits.size(1)
            sent_len = sent_logits.size(1)

            matrix_loss_fct = nn.CrossEntropyLoss(reduction='none')
            tok_row_label = torch.zeros([batch_size, tok_len], dtype=torch.long).cuda()
            tok_col_label = torch.zeros([batch_size, tok_len], dtype=torch.long).cuda()
            tok_valid_label = torch.zeros([batch_size, tok_len, tok_len], dtype=torch.long).cuda()
            for batch_idx in range(batch_size):
                for answer_idx in range(len(short_start_positions[batch_idx])):
                    if short_start_positions[batch_idx][answer_idx] == 0:
                        break
                    tok_row_label[batch_idx][short_start_positions[batch_idx][answer_idx]] = short_end_positions[batch_idx][answer_idx]
                    tok_col_label[batch_idx][short_end_positions[batch_idx][answer_idx]] = short_start_positions[batch_idx][answer_idx]
                    tok_valid_label[batch_idx][short_start_positions[batch_idx][answer_idx]][short_end_positions[batch_idx][answer_idx]] = 1

            tok_row_mask = torch.sum(tok_valid_label, 2).cuda()
            tok_col_mask = torch.sum(tok_valid_label, 1).cuda()

            tok_row_loss = matrix_loss_fct(tok_matrix_logits.view(-1, tok_len),
                                       tok_row_label.view(-1)).reshape(batch_size, tok_len)
            tok_col_loss = matrix_loss_fct(tok_matrix_logits.transpose(1, 2).reshape(batch_size * tok_len, tok_len),
                                       tok_col_label.view(-1)).reshape(batch_size, tok_len)

            tok_final_row_loss = torch.mean(torch.sum(tok_row_loss * tok_row_mask, 1))
            tok_final_col_loss = torch.mean(torch.sum(tok_col_loss * tok_col_mask, 1))

            tok_matrix_loss = (tok_final_row_loss + tok_final_col_loss) / 2

            ############    Long Span Matrix Loss    ############
            sent_row_label = torch.zeros([batch_size, sent_len], dtype=torch.long).cuda()
            sent_col_label = torch.zeros([batch_size, sent_len], dtype=torch.long).cuda()
            sent_valid_label = torch.zeros([batch_size, sent_len, sent_len], dtype=torch.long).cuda()
            for batch_idx in range(batch_size):
                for answer_idx in range(len(long_start_positions[batch_idx])):
                    if long_start_positions[batch_idx][answer_idx] == 0:
                        break
                    sent_row_label[batch_idx][long_start_positions[batch_idx][answer_idx]] = \
                        long_end_positions[batch_idx][answer_idx]
                    sent_col_label[batch_idx][long_end_positions[batch_idx][answer_idx]] = \
                        long_start_positions[batch_idx][answer_idx]
                    sent_valid_label[batch_idx][long_start_positions[batch_idx][answer_idx]][
                        long_end_positions[batch_idx][answer_idx]] = 1

            sent_row_mask = torch.sum(sent_valid_label, 2).cuda()
            sent_col_mask = torch.sum(sent_valid_label, 1).cuda()

            sent_row_loss = matrix_loss_fct(sent_matrix_logits.view(-1, sent_len),
                                            sent_row_label.view(-1)).reshape(batch_size, sent_len)
            sent_col_loss = matrix_loss_fct(sent_matrix_logits.transpose(1, 2).reshape(batch_size * sent_len, sent_len),
                                            sent_col_label.view(-1)).reshape(batch_size, sent_len)

            sent_final_row_loss = torch.mean(torch.sum(sent_row_loss * sent_row_mask, 1))
            sent_final_col_loss = torch.mean(torch.sum(sent_col_loss * sent_col_mask, 1))

            sent_matrix_loss = (sent_final_row_loss + sent_final_col_loss) / 2

            matrix_loss = (tok_matrix_loss + sent_matrix_loss) / 2

            ############################    Valid Loss    ############################
            valid_loss_fct = nn.CrossEntropyLoss()
            tok_s_valid_loss = valid_loss_fct(tok_s_valid_logits.view(-1, 2), tok_row_mask.view(-1, ))
            tok_e_valid_loss = valid_loss_fct(tok_e_valid_logits.view(-1, 2), tok_col_mask.view(-1, ))

            sent_s_valid_loss = valid_loss_fct(sent_s_valid_logits.view(-1, 2), sent_row_mask.view(-1, ))
            sent_e_valid_loss = valid_loss_fct(sent_e_valid_logits.view(-1, 2), sent_col_mask.view(-1, ))

            valid_loss = tok_s_valid_loss + tok_e_valid_loss + sent_s_valid_loss + sent_e_valid_loss

            total_loss = (span_loss + matrix_loss + valid_loss)/3
            return total_loss, matrix_loss, span_loss, valid_loss
        tok_s_valid_prob = F.softmax(tok_s_valid_logits, dim=-1)[:, :, 1]
        tok_e_valid_prob = F.softmax(tok_e_valid_logits, dim=-1)[:, :, 1]
        sent_s_valid_prob = F.softmax(sent_s_valid_logits, dim=-1)[:, :, 1]
        sent_e_valid_prob = F.softmax(sent_e_valid_logits, dim=-1)[:, :, 1]
        return sent_start_logits, sent_end_logits, sent_matrix_logits, sent_s_valid_prob, sent_e_valid_prob,\
               tok_start_logits, tok_end_logits, tok_matrix_logits, tok_s_valid_prob, tok_e_valid_prob
