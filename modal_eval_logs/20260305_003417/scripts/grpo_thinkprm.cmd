@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONIOENCODING=utf-8
set MODAL_GPU=A10G
modal run DAPO_ThinkPRM_Eval/Best_of_N_selection_eval.py --args "--dapo_model_path gs://checkpoints-cs224n/non-curriculum/non_curriculum_phase2_grpo --thinkprm_model_path launch/ThinkPRM-1.5B --output_dir /tmp/grpo_thinkprm_20260305_003417 --results_gcs_prefix gs://checkpoints-cs224n/non-curriculum/evaluation_results_modal_20260305_003417/grpo_thinkprm --dataset_jsonl /root/combined_holdout_50_each.jsonl --limit -1 --num_candidates 8 --gcp_project cs224n-project --dataset_gcp_project cs224n-project-data --skip_baseline_eval"
