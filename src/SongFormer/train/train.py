import argparse
import copy
import importlib
import os
import traceback

# monkey patch to fix issues in msaf
import scipy
import numpy as np

scipy.inf = np.inf

import hydra
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.local_sgd import LocalSGD
from accelerate.utils import LoggerType, set_seed, DistributedDataParallelKwargs

# from ema import LitEma
from ema_pytorch import EMA
from encodec.balancer import Balancer
from loguru import logger
from omegaconf import OmegaConf
from eval_infer_results_used_in_train import eval_infer_results
from vis_infer_chunk_class_used_in_train import vis_infer_chunk
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from utils.check_nan import NanInfError, check_model_param
from utils.timer import TrainTimer
from dataset.label2id import DATASET_LABEL_TO_DATASET_ID

# for lance
torch.multiprocessing.set_start_method("spawn", force=True)


def save_checkpoint(
    checkpoint_dir,
    model,
    model_ema,
    optimizer,
    scheduler,
    step,
    accelerator,
    wait_for_everyone=True,
):
    if wait_for_everyone:
        accelerator.wait_for_everyone()
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt-{}.pt".format(step))
    if accelerator.is_main_process:
        accelerator.save(
            {
                "model": accelerator.unwrap_model(model).state_dict(),
                "optimizer": accelerator.unwrap_model(optimizer).state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "model_ema": model_ema.state_dict(),
                "global_step": step,
            },
            checkpoint_path,
        )

        print("Saved checkpoint: {}".format(checkpoint_path))

        with open(os.path.join(checkpoint_dir, "checkpoint"), "w") as f:
            f.write("model.ckpt-{}.pt".format(step))
    return checkpoint_path


def attempt_to_restore(
    model,
    model_ema,
    optimizer,
    scheduler,
    checkpoint_dir,
    device,
    accelerator,
    keep_training,
    strict=True,
):
    accelerator.wait_for_everyone()

    checkpoint_list = os.path.join(checkpoint_dir, "checkpoint")

    if os.path.exists(checkpoint_list):
        checkpoint_filename = open(checkpoint_list).readline().strip()
        checkpoint_path = os.path.join(checkpoint_dir, "{}".format(checkpoint_filename))
        print("Restore from {}".format(checkpoint_path))
        checkpoint = load_checkpoint(checkpoint_path, device)
        if strict:
            accelerator.unwrap_model(model).load_state_dict(checkpoint["model"], True)
            accelerator.unwrap_model(optimizer).load_state_dict(checkpoint["optimizer"])
            if scheduler:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if model_ema and accelerator.is_main_process:
                model_ema.load_state_dict(checkpoint["model_ema"])
        else:
            accelerator.unwrap_model(model).load_state_dict(
                checkpoint["model"], False
            )  # false to change
        if keep_training:
            global_step = checkpoint["global_step"]
        else:
            global_step = 0
        del checkpoint
    else:
        global_step = 0

    return global_step


def load_checkpoint(checkpoint_path, device=None):
    if device:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    return checkpoint


def evaluate(model, eval_data_loader, accelerator, global_step):
    model.eval()
    results_by_dataset = {}
    evaluate_num = 0

    with torch.no_grad():
        with TrainTimer(
            step=global_step, name="time/eval_time", accelerator=accelerator
        ):
            for batch in tqdm(eval_data_loader, desc="Evaluating"):
                try:
                    if batch is None:
                        continue
                    batch = {
                        key: (
                            val.to(accelerator.device)
                            if isinstance(val, torch.Tensor)
                            else val
                        )
                        for key, val in batch.items()
                    }

                    assert len(batch["data_ids"]) == 1
                    dataset_id = batch["dataset_ids"].item()

                    # If multi-GPU training is used, the logic here may need to be modified.

                    # if accelerator.num_processes > 1:
                    #     result = model.module.infer_with_metrics(batch, prefix="valid_")
                    # else:
                    #     result = model.infer_with_metrics(batch, prefix="valid_")
                    result = model.ema_model.infer_with_metrics(batch, prefix="valid_")

                    if dataset_id not in results_by_dataset:
                        results_by_dataset[dataset_id] = []

                    results_by_dataset[dataset_id].append(result)
                    evaluate_num += 1

                except Exception as e:
                    logger.error(f"Error in evaluate {dataset_id}: {e}")
                    continue

    flat_result = {}

    # Average per dataset
    for dataset_id, result_list in results_by_dataset.items():
        df = pd.DataFrame(result_list)
        avg_metrics = df.mean().to_dict()
        for k, v in avg_metrics.items():
            flat_result[f"dataset_{dataset_id}_{k}"] = v

    # Overall average
    all_results = [res for results in results_by_dataset.values() for res in results]
    overall_df = pd.DataFrame(all_results)
    overall_metrics = overall_df.mean().to_dict()
    for k, v in overall_metrics.items():
        flat_result[f"overall_{k}"] = v

    return flat_result


def _resolve_primary_dataset_id(hparams):
    try:
        dataset_abstracts = hparams.train_dataset.dataset_abstracts
        if not dataset_abstracts:
            return None
        dataset_type = dataset_abstracts[0].get("dataset_type")
        if not dataset_type:
            return None
        return DATASET_LABEL_TO_DATASET_ID.get(dataset_type)
    except Exception:
        return None


def _split_cv_indices(num_items: int, folds: int, seed: int):
    if folds <= 1:
        return [np.arange(num_items)]
    rng = np.random.default_rng(seed)
    indices = np.arange(num_items)
    rng.shuffle(indices)
    return np.array_split(indices, folds)


def _get_collate_fn(dataset):
    current = dataset
    while current is not None:
        if hasattr(current, "collate_fn"):
            return current.collate_fn
        current = getattr(current, "dataset", None)
    return None


def _pick_metric(eval_res, dataset_id, metric_name):
    key = None
    value = None
    if dataset_id is not None:
        dataset_key = f"dataset_{dataset_id}_valid_{metric_name}"
        if dataset_key in eval_res:
            return eval_res[dataset_key], dataset_key
    overall_key = f"overall_valid_{metric_name}"
    if overall_key in eval_res:
        return eval_res[overall_key], overall_key
    for k, v in eval_res.items():
        if k.endswith(f"valid_{metric_name}"):
            key = k
            value = v
            break
    return value, key


def test_metrics(accelerator, model, hparams, ckpt_path, infer_dicts):
    with torch.no_grad():

        def add_dict_prefix(d, prefix: str):
            if not prefix:
                return d
            return {prefix + key: value for key, value in d.items()}

        total_results = {}

        for item in infer_dicts:
            infer_dir = ckpt_path.replace("output/", "infer_output/").replace(".pt", "")
            os.makedirs(infer_dir, exist_ok=True)
            os.makedirs(os.path.join(infer_dir, "visualisation"), exist_ok=True)

            vis_infer_chunk(
                device=accelerator.device,
                model=model,
                embedding_dir=item["embedding_dir"],
                segments_dir=item["ann_dir"],
                eval_id_scp_path=item["eval_id_scp_path"],
                visual_id_list_path=item["visual_id_list_path"],
                dataset_label=item["dataset_label"],
                dataset_ids=item["dataset_ids"],
                hparams=hparams,
                output_dir=infer_dir,
            )

            for result_type in ["normal", "prechorus2chorus", "prechorus2verse"]:
                result_dir = infer_dir.replace(
                    "infer_output", "infer_results/mannual_cbhao/" + result_type
                )
                result_type2prechorus2what = dict(
                    normal=None, prechorus2chorus="chorus", prechorus2verse="verse"
                )
                tmp_result = eval_infer_results(
                    ann_dir=item["ann_dir"],
                    est_dir=infer_dir,
                    output_dir=result_dir,
                    eval_lists_file_path=item["eval_id_scp_path"],
                    prechorus2what=result_type2prechorus2what[result_type],
                )
                tmp_result = tmp_result.to_dict(orient="records")
                assert len(tmp_result) == 1, "There should be only one record"
                tmp_result = tmp_result[0]
                tmp_result = add_dict_prefix(
                    d=tmp_result, prefix=f"{item['infer_name']}_{result_type}/"
                )
                total_results.update(tmp_result)

        total_results = add_dict_prefix(d=total_results, prefix="result_")
        return total_results


def prefix_dict(d, prefix: str):
    if prefix:
        return d
    return {prefix + key: value for key, value in d.items()}


def main(args, hparams):
    return run_training(args, hparams)


def run_training(
    args,
    hparams,
    train_dataset=None,
    eval_dataset=None,
    primary_dataset_id=None,
):
    assert hasattr(args, "init_seed"), "hparams should have seed attribute"
    set_seed(args.init_seed)
    accelerator = Accelerator(
        log_with=["wandb", LoggerType.MLFLOW],
        project_dir=os.path.join(args.checkpoint_dir, "tracker"),
        gradient_accumulation_steps=hparams.accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if os.environ.get("DDP_DUMMY_LOSS", "0") not in ("0", "false", "False", "no", "NO"):
        if accelerator.is_main_process:
            print(
                "DDP_DUMMY_LOSS is set but ignored. "
                "The dummy-loss workaround can trigger 'mark a variable ready twice' "
                "errors under DDP."
            )

    device = accelerator.device
    rank = accelerator.process_index
    local_rank = accelerator.local_process_index

    tags = []
    if args.tags:
        for tag in args.tags.split("/"):
            tags.append(tag)
    init_kwargs = {
        "wandb": {
            "resume": "allow",
            "name": args.run_name,
        },
        "mlflow": {
            "run_name": args.run_name,
        },
    }

    accelerator.init_trackers(
        "SongFormer",
        config={
            **prefix_dict(vars(copy.deepcopy(args)), "a_"),
            **prefix_dict(copy.deepcopy(hparams), "h_"),
        },
        init_kwargs=init_kwargs,
    )

    def print_rank_0(msg):
        accelerator.print(msg)

    module = importlib.import_module("models." + args.model_name)
    Model = getattr(module, "Model")
    model = Model(hparams)
    params = model.parameters()
    model_ema = None

    if accelerator.is_main_process:
        model_ema = EMA(model, include_online_model=False, **hparams.ema_kwargs)
        model_ema.to(accelerator.device)

    num_params = 0
    for param in params:
        num_params += torch.prod(torch.tensor(param.size()))

    if train_dataset is None:
        train_dataset = hydra.utils.instantiate(hparams.train_dataset)
    if eval_dataset is None:
        eval_dataset = hydra.utils.instantiate(hparams.eval_dataset)
    if primary_dataset_id is None:
        primary_dataset_id = _resolve_primary_dataset_id(hparams)

    train_collate = _get_collate_fn(train_dataset)
    eval_collate = _get_collate_fn(eval_dataset)
    data_loader = DataLoader(
        train_dataset, **hparams.train_dataloader, collate_fn=train_collate
    )
    eval_data_loader = DataLoader(
        eval_dataset, **hparams.eval_dataloader, collate_fn=eval_collate
    )

    warmup_steps = hparams.warmup_steps
    total_steps = hparams.total_steps

    balancer = Balancer(
        weights={"loss_section": 1, "loss_function": 1},
        rescale_grads=True,
        monitor=True,
    )
    use_balancer = accelerator.num_processes == 1

    optimizer = optim.Adam(
        model.parameters(),
        **hparams.optimizer,
    )

    print(f"warmup_steps: {warmup_steps}, total_steps: {total_steps}")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,  # warmup steps
        num_training_steps=total_steps,  # total training steps
    )

    model, optimizer, data_loader, scheduler = accelerator.prepare(
        model, optimizer, data_loader, scheduler
    )

    global_step = attempt_to_restore(
        model=model,
        model_ema=model_ema,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        accelerator=accelerator,
        keep_training=True,
        strict=True,
    )
    print_rank_0(
        f"-------------------------Parameters: {num_params}-----------------------"
    )

    # early stop
    best_hr05 = -float("inf")
    best_hr3 = -float("inf")
    best_score = -float("inf")
    best_eval_res = None
    no_improve_steps = 0
    early_stop_patience = (
        hparams.early_stopping_step
    )  # Stop if there is no improvement in 5 consecutive evaluations

    ran_since_savepoint_loaded = False
    max_steps = args.max_steps
    LOCAL_SGD_STEPS = 8

    eval_ckpt_lists = []
    seek_results_ckpt_lists = []

    for epoch in range(args.max_epochs):
        if no_improve_steps >= early_stop_patience:
            break

        if global_step >= max_steps:
            break
        model.train()
        if rank == 0:
            progress_bar = tqdm(range(len(data_loader)))
        with LocalSGD(
            accelerator=accelerator,
            model=model,
            local_sgd_steps=LOCAL_SGD_STEPS,
            enabled=False,
        ) as local_sgd:
            print("***data_loader length:", len(data_loader))
            for step, batch in enumerate(data_loader):
                if global_step >= max_steps:
                    break
                skip_flag = torch.tensor(
                    1 if batch is None else 0, device=accelerator.device
                )
                if accelerator.num_processes > 1:
                    skip_flag = accelerator.gather(skip_flag)
                    if skip_flag.max().item() > 0:
                        if accelerator.is_main_process:
                            print(
                                "Skipping batch because at least one rank received an invalid batch."
                            )
                        continue
                elif skip_flag.item() > 0:
                    continue
                with accelerator.accumulate(model):
                    model.train()

                    try:
                        optimizer.zero_grad()

                        with TrainTimer(global_step, "time/forward_time", accelerator):
                            logits, loss, losses = model(batch)

                        with TrainTimer(global_step, "time/backward_time", accelerator):
                            if use_balancer:
                                loss_sum = balancer.cal_mix_loss(
                                    {
                                        "loss_section": losses["loss_section"],
                                        "loss_function": losses["loss_function"],
                                    },
                                    list(model.parameters()),
                                    accelerator=accelerator,
                                )
                            else:
                                # Avoid Balancer autograd.grad path under DDP.
                                loss_sum = losses["loss"]

                            invalid_loss = (not loss_sum.requires_grad) or (
                                not torch.isfinite(loss_sum)
                            )
                            if invalid_loss:
                                flag = torch.tensor(
                                    1, device=accelerator.device, dtype=torch.int
                                )
                                if accelerator.num_processes > 1:
                                    flag = accelerator.gather(flag)
                                    if flag.max().item() > 0:
                                        if accelerator.is_main_process:
                                            data_ids = batch.get("data_ids")
                                            print(
                                                "Skipping batch due to invalid loss "
                                                f"(requires_grad={loss_sum.requires_grad}, "
                                                f"finite={bool(torch.isfinite(loss_sum))}). "
                                                f"data_ids={data_ids}"
                                            )
                                        optimizer.zero_grad(set_to_none=True)
                                        continue
                                else:
                                    data_ids = batch.get("data_ids")
                                    print(
                                        "Skipping batch due to invalid loss "
                                        f"(requires_grad={loss_sum.requires_grad}, "
                                        f"finite={bool(torch.isfinite(loss_sum))}). "
                                        f"data_ids={data_ids}"
                                    )
                                    optimizer.zero_grad(set_to_none=True)
                                    continue

                            accelerator.backward(loss_sum)

                        with TrainTimer(global_step, "time/optimize_time", accelerator):
                            if accelerator.sync_gradients:
                                optimizer.step()
                                scheduler.step()
                                local_sgd.step()

                                if accelerator.is_main_process:
                                    model_ema.update()

                        check_model_param(model, step=global_step)

                        if rank == 0 and global_step % args.log_interval == 0:
                            progress_bar.update(args.log_interval)
                            progress_bar.set_description(
                                f"epoch: {epoch:03}, step: {global_step:06}, loss_awl: {loss_sum.item():.2f}"
                            )

                        if global_step % args.log_interval == 0 and rank == 0:
                            accelerator.log(
                                {
                                    **balancer.metrics,
                                    "training/epoch": epoch,
                                    # "training/loss": loss.item(),
                                    "training/loss_awl": loss_sum.item(),
                                    "training/loss_function": losses[
                                        "loss_function"
                                    ].item(),
                                    "training/loss_section": losses[
                                        "loss_section"
                                    ].item(),
                                    "training/learning_rate": scheduler.get_lr()[0],
                                    "training/batch_size": int(
                                        hparams.train_dataloader.batch_size
                                    ),
                                    "training/local_sgd_steps": LOCAL_SGD_STEPS,
                                    "training/num_of_gpu": accelerator.num_processes,
                                },
                                step=global_step,
                            )
                        if (
                            accelerator.sync_gradients
                            and global_step % args.eval_interval == 0
                        ):
                            accelerator.wait_for_everyone()
                            if rank == 0:
                                model.eval()

                                raw_eval_res = evaluate(
                                    model=model_ema,
                                    eval_data_loader=eval_data_loader,
                                    accelerator=accelerator,
                                    global_step=global_step,
                                )

                                eval_res = prefix_dict(d=raw_eval_res, prefix="eval/")
                                accelerator.log(
                                    eval_res,
                                    step=global_step,
                                )
                                if primary_dataset_id is not None:
                                    dataset_prefix = (
                                        f"eval/dataset_{primary_dataset_id}_"
                                    )
                                    dataset_metrics = {
                                        k: v
                                        for k, v in eval_res.items()
                                        if k.startswith(dataset_prefix)
                                    }
                                    if not dataset_metrics:
                                        fallback_prefix = (
                                            f"dataset_{primary_dataset_id}_"
                                        )
                                        dataset_metrics = {
                                            k: v
                                            for k, v in eval_res.items()
                                            if k.startswith(fallback_prefix)
                                        }
                                    if dataset_metrics:
                                        print_rank_0(
                                            f"Eval metrics (dataset {primary_dataset_id}): {dataset_metrics}"
                                        )

                                hr05, hr05_key = _pick_metric(
                                    raw_eval_res,
                                    primary_dataset_id,
                                    "HitRate_0.5F",
                                )
                                hr3, hr3_key = _pick_metric(
                                    raw_eval_res,
                                    primary_dataset_id,
                                    "HitRate_3F",
                                )
                                acc, acc_key = _pick_metric(
                                    raw_eval_res, primary_dataset_id, "acc"
                                )
                                hr05 = hr05 or 0
                                hr3 = hr3 or 0
                                acc = acc or 0
                                score = 0.5 * (hr05 + hr3)

                                if score > best_score:
                                    print(
                                        "Eval improved: "
                                        f"{hr05_key or 'HitRate_0.5F'} {hr05:.4f} "
                                        f"(prev {best_hr05:.4f}), "
                                        f"{hr3_key or 'HitRate_3F'} {hr3:.4f} "
                                        f"(prev {best_hr3:.4f}), "
                                        f"score {score:.4f} (prev {best_score:.4f})"
                                    )
                                    best_hr05 = max(best_hr05, hr05)
                                    best_hr3 = max(best_hr3, hr3)
                                    best_score = max(best_score, score)
                                    no_improve_steps = 0
                                    best_eval_res = eval_res

                                    tmp_ckpt_path = save_checkpoint(
                                        checkpoint_dir=args.checkpoint_dir,
                                        model=model,
                                        model_ema=model_ema,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        step=global_step,
                                        accelerator=accelerator,
                                        wait_for_everyone=False,
                                    )
                                    eval_ckpt_lists.append(
                                        (tmp_ckpt_path, float(acc), global_step)
                                    )
                                    with open(
                                        os.path.join(
                                            args.checkpoint_dir, "early_stop.log"
                                        ),
                                        "a",
                                    ) as f:
                                        f.write(
                                            f"model.ckpt-{global_step}.pt hr0.5: {hr05}, hr3: {hr3}, score: {score}, acc: {acc}\n"
                                        )
                                    print("Saved best checkpoint at step", global_step)
                                else:
                                    logger.warning("write a ckpt not improved!")
                                    save_checkpoint(
                                        checkpoint_dir=args.checkpoint_dir,
                                        model=model,
                                        model_ema=model_ema,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        step=global_step,
                                        accelerator=accelerator,
                                        wait_for_everyone=False,
                                    )
                                    no_improve_steps += 1
                                    print(
                                        f"No improvement for {no_improve_steps} eval steps."
                                    )
                                    if no_improve_steps >= early_stop_patience:
                                        print("Early stopping triggered.")
                                        break

                        accelerator.wait_for_everyone()
                        if (
                            accelerator.sync_gradients
                            and global_step % args.save_interval == 0
                        ):
                            print("save checkpoint", global_step)
                            checkpoint_path = save_checkpoint(
                                checkpoint_dir=args.checkpoint_dir,
                                model=model,
                                model_ema=model_ema,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                step=global_step,
                                accelerator=accelerator,
                                wait_for_everyone=False,
                            )
                            if (
                                not hasattr(hparams, "infer_dicts")
                                or hparams.infer_dicts is None
                                or len(hparams.infer_dicts) == 0
                            ):
                                logger.error(
                                    "No infer_dicts provided, skipping test_metrics for ckpt",
                                )
                            else:
                                test_metrics_ret = test_metrics(
                                    accelerator=accelerator,
                                    model=model_ema.ema_model,
                                    hparams=hparams,
                                    ckpt_path=checkpoint_path,
                                    infer_dicts=hparams.infer_dicts,
                                )

                                accelerator.log(
                                    test_metrics_ret,
                                    step=global_step,
                                )
                            seek_results_ckpt_lists.append(checkpoint_path)

                        if accelerator.sync_gradients:
                            global_step += 1

                        ran_since_savepoint_loaded = True
                    except NanInfError as e:
                        print(e)
                        exit(-1)
                    except Exception:
                        traceback.print_exc()
                        # In multi-GPU DDP, swallowing an exception desynchronizes ranks and
                        # triggers "Expected to have finished reduction" on the next step.
                        # Fail fast so the real root-cause is visible.
                        if accelerator.num_processes > 1:
                            raise
                        # print(features["filenames"], features["seq_lens"], features["seqs"].shape)
                        optimizer.zero_grad()

    eval_ckpt_lists.sort(key=lambda x: x[1], reverse=True)
    seek_results_ckpt_lists = set(seek_results_ckpt_lists)

    for x in eval_ckpt_lists[:2]:
        print(f"Eval ckpt: {x[0]}, acc: {x[1]}")
        if x[0] not in seek_results_ckpt_lists:
            if (
                not hasattr(hparams, "infer_dicts")
                or hparams.infer_dicts is None
                or len(hparams.infer_dicts) == 0
            ):
                logger.error(
                    "No infer_dicts provided, skipping test_metrics for ckpt",
                    x[0],
                )
                continue
            ckpt = load_checkpoint(x[0], device=device)
            model_ema.load_state_dict(ckpt["model_ema"])

            test_metrics_ret = test_metrics(
                accelerator=accelerator,
                model=model_ema.ema_model,
                hparams=hparams,
                ckpt_path=x[0],
                infer_dicts=hparams.infer_dicts,
            )

            accelerator.log(
                test_metrics_ret,
                step=x[2],
            )

    accelerator.end_training()
    return best_eval_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script arguments")

    # ---------------- Must be specified via command line ----------------
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--init_seed", type=int, required=True, help="Random seed for initialization"
    )

    # ---------------- Optional parameters (can override configuration file) ----------------
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name (overrides config if provided)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name (overrides config if provided)",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--tags", type=str, default=None, help="Optional tags for experiment tracking"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=None, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Maximum number of training steps"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=None,
        help="Interval (in steps) to save checkpoints",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=None,
        help="Interval (in steps) to run evaluation",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=1,
        help="If >1, run K-fold CV on the training set.",
    )
    parser.add_argument(
        "--cv_seed",
        type=int,
        default=None,
        help="Random seed used to split folds.",
    )

    # ---------------- Training process control parameters ----------------
    parser.add_argument(
        "--not_keep_step",
        action="store_true",
        help="If set, do not keep the training step",
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Logging interval (in steps)"
    )

    args = parser.parse_args()

    # Read configuration file
    hp = OmegaConf.load(args.config)

    # Ensure init_seed exists
    assert hasattr(args, "init_seed"), "args should have an init_seed attribute"

    if args.run_name is None:
        args.run_name = str(hp.args.run_name) + "_" + str(args.init_seed)
    if args.model_name is None:
        args.model_name = hp.args.model_name
    if args.save_interval is None:
        args.save_interval = hp.args.save_interval
    if args.eval_interval is None:
        args.eval_interval = hp.args.eval_interval
    if args.checkpoint_dir is None:
        args.checkpoint_dir = hp.args.checkpoint_dir + "_" + str(args.init_seed)
    if args.max_epochs is None:
        args.max_epochs = hp.args.max_epochs
    if args.max_steps is None:
        args.max_steps = hp.args.max_steps
    if args.tags is None:
        args.tags = hp.args.tags
    args.keep_step = not args.not_keep_step

    if args.cv_folds <= 1:
        main(args, hp)
    else:
        primary_dataset_id = _resolve_primary_dataset_id(hp)
        full_dataset = hydra.utils.instantiate(hp.train_dataset)
        num_items = len(full_dataset)
        if num_items == 0:
            raise SystemExit("Training dataset is empty; cannot run CV.")

        cv_seed = args.cv_seed if args.cv_seed is not None else args.init_seed
        fold_indices = _split_cv_indices(num_items, args.cv_folds, cv_seed)
        fold_metrics = []

        for fold_idx, val_idx in enumerate(fold_indices):
            train_idx = np.setdiff1d(np.arange(num_items), val_idx, assume_unique=False)
            train_subset = Subset(full_dataset, train_idx.tolist())
            eval_subset = Subset(full_dataset, val_idx.tolist())

            fold_args = copy.deepcopy(args)
            fold_args.init_seed = cv_seed + fold_idx
            if fold_args.run_name:
                fold_args.run_name = f"{fold_args.run_name}_fold{fold_idx}"
            if fold_args.checkpoint_dir:
                fold_args.checkpoint_dir = os.path.join(
                    fold_args.checkpoint_dir, f"fold{fold_idx}"
                )

            print(
                f"Starting CV fold {fold_idx + 1}/{args.cv_folds}: "
                f"train={len(train_subset)} eval={len(eval_subset)} "
                f"seed={fold_args.init_seed}"
            )

            best_eval = run_training(
                fold_args,
                hp,
                train_dataset=train_subset,
                eval_dataset=eval_subset,
                primary_dataset_id=primary_dataset_id,
            )
            if best_eval:
                fold_metrics.append(best_eval)

        if fold_metrics:
            keys = fold_metrics[0].keys()
            cv = {}
            for k in keys:
                cv[k] = float(
                    sum(m.get(k, 0.0) for m in fold_metrics) / len(fold_metrics)
                )
            print(f"CV metrics (mean over {len(fold_metrics)} folds): {cv}")
