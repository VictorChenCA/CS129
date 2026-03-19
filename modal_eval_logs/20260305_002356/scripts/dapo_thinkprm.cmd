@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONIOENCODING=utf-8
modal run DAPO_ThinkPRM_Eval/Best_of_N_selection_eval.py --args "--dapo_model_path gs://checkpoints-cs224n/non-curriculum/non_curriculum_phase3_dapo --thinkprm_model_path launch/ThinkPRM-1.5B --output_dir /tmp/dapo_thinkprm_20260305_002356 --results_gcs_prefix gs://checkpoints-cs224n/non-curriculum/evaluation_results_modal_20260305_002356/dapo_thinkprm --dataset_jsonl /root/combined_holdout_50_each.jsonl --limit -1 --num_candidates 8 --gcp_project cs224n-project --dataset_gcp_project cs224n-project-data"
