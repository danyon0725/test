import os
import timeit

import torch
from fastprogress.fastprogress import master_bar, progress_bar
from processor import SquadResult
from squad_metric import (
    compute_predictions_logits
)
from utils import load_examples, set_seed, to_list
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from multi_measure import evaluate_prediction_file


def get_best_index(logits, n_best_size):
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def get_best_span(valid_range, valid_dict, n_best_size, start_logits, end_logits, span_matrix, valid_prob):
    spans = []
    start_indexes = get_best_index(start_logits, n_best_size)
    end_indexes = get_best_index(end_logits, n_best_size)
    # if we could have irrelevant answers, get the min score of irrelevant

    for start_index in start_indexes:
        for end_index in end_indexes:
            if start_index >= valid_range:
                continue
            if end_index >= valid_range:
                continue
            if start_index not in valid_dict:
                continue
            if end_index not in valid_dict:
                continue
            if end_index < start_index:
                continue
            span = [start_index, end_index, start_logits[start_index], end_logits[end_index]
                , span_matrix[start_index][end_index], valid_prob[start_index][end_index]]
            spans.append(span)
    return spans

def get_best_span_binary(valid_range, valid_dict, n_best_size, start_logits, end_logits, span_matrix, s_valid_prob,
                             e_valid_prob):
    spans = []
    start_indexes = get_best_index(start_logits, n_best_size)
    end_indexes = get_best_index(end_logits, n_best_size)
    # if we could have irrelevant answers, get the min score of irrelevant

    for start_index in start_indexes:
        for end_index in end_indexes:
            if start_index >= valid_range:
                continue
            if end_index >= valid_range:
                continue
            if start_index not in valid_dict:
                continue
            if end_index not in valid_dict:
                continue
            if end_index < start_index:
                continue
            span = [start_index, end_index, start_logits[start_index], end_logits[end_index]
                , span_matrix[start_index][end_index], s_valid_prob[start_index] * e_valid_prob[end_index]]
            spans.append(span)
    return spans
def train(args, model, tokenizer, logger):
    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    mb = master_bar(range(int(args.num_train_epochs)))

    for epoch in mb:
        train_dataset = load_examples(args, tokenizer, evaluate=False, output_examples=False)
        """ Train the model """
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(args.model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Train batch size per GPU = %d", args.train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        logger.info("  Starting fine-tuning.")

        model.zero_grad()
        # Added here for reproductibility
        set_seed(args)

        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "question_mask":batch[2],
                "sentence_mask": batch[3],
                "token_type_ids": batch[4],
                "long_start_positions":batch[5],
                "long_end_positions":batch[6],
                "short_start_positions": batch[7],
                "short_end_positions": batch[8],
                "long_start_position": batch[9],
                "long_end_position": batch[10],
                "short_start_position": batch[11],
                "short_end_position": batch[12]
            }

            loss, matrix_loss, span_loss, valid_loss = model(**inputs)
            loss = loss.mean()
            matrix_loss = matrix_loss.mean()
            valid_loss = valid_loss.mean()
            # model outputs are always tuple in transformers (see doc)
            if (global_step +1) % 50 == 0:
                print("{} Processing,,,, Current Total Loss : {}".format(global_step+1, loss.item()))
                print(
                    "Matrix Loss : {} \t Span Loss : {} \t valid Loss : {}".format(matrix_loss.item(), span_loss.item(),
                                                                                   valid_loss.item()))

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    evaluate(args=args, model=model, tokenizer=tokenizer, global_step=global_step)
            if args.max_steps > 0 and global_step > args.max_steps:
                break
        mb.write("Epoch {} done".format(epoch+1))

        if args.max_steps > 0 and global_step > args.max_steps:
            break
    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix="", global_step=None, all_predict=False, logger=None):
    dataset, examples, features = load_examples(args, tokenizer, evaluate=True, output_examples=True)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)


    all_results = []
    start_time = timeit.default_timer()

    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "question_mask":batch[2],
                "sentence_mask": batch[3],
                "token_type_ids": batch[4],
            }

            feature_indices = batch[5]

            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]

            unique_id = int(eval_feature.unique_id)

            long_start_logits, long_end_logits, long_span_matrix_logits, long_s_valid_prob, long_e_valid_prob, \
            short_start_logits, short_end_logits, short_span_matrix_logits, short_s_valid_prob, short_e_valid_prob = [
                to_list(output[i]) for output in outputs]
            long_best_span = get_best_span_binary(len(eval_feature.cur_sent_to_orig_sent_map) + 1,
                                                  eval_feature.cur_sent_to_orig_sent_map, args.n_best_size,
                                                  long_start_logits, long_end_logits, long_span_matrix_logits,
                                                  long_s_valid_prob, long_e_valid_prob)
            short_best_span = get_best_span_binary(len(eval_feature.tokens), eval_feature.token_to_orig_map,
                                                   args.n_best_size,
                                                   short_start_logits, short_end_logits, short_span_matrix_logits,
                                                   short_s_valid_prob, short_e_valid_prob)

            result = SquadResult(unique_id, long_best_span, short_best_span)

            all_results.append(result)
    output_null_log_odds_file = None
    #torch.save({"features": features, "results": all_results, "examples": examples}, args.result_file)

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        # "./prediction_{}.json".format(global_step),
        None,
        None,
        None,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )
    # Write the evaluation result on file
    for k in range(10):
        print("Threshold : ", k / 10)
        evaluate_prediction_file(predictions[k], args.dev_file_path)
def only_scoring(args, tokenizer):

    results = torch.load(args.result_file)
    features, result, examples = (
        results["features"],
        results["results"],
        results["examples"],
    )


    predictions = compute_predictions_logits(
            examples,
            features,
            result,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            None,
            None,
            None,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )
    for k in range(10):
        print("Threshold : ", k/10)
        evaluate_prediction_file(predictions[k], args.dev_file_path)