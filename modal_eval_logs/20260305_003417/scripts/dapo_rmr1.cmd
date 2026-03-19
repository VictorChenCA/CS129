@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONIOENCODING=utf-8
set MODAL_GPU=A10G
modal run DAPO_RMR1_eval/Best_of_N_selection_eval_rm_r1.py --args "--dapo_model_path gs://checkpoints-cs224n/non-curriculum/non_curriculum_phase3_dapo --rmr1_model_path gaotang/RM-R1-DeepSeek-Distilled-Qwen-7B --output_dir /tmp/dapo_rmr1_20260305_003417 --results_gcs_prefix gs://checkpoints-cs224n/non-curriculum/evaluation_results_modal_20260305_003417/dapo_rmr1 --dataset_jsonl /root/combined_holdout_50_each.jsonl --limit -1 --num_candidates 8 --gcp_project cs224n-project --dataset_gcp_project cs224n-project-data"
