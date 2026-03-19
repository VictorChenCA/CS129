$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:PYTHONIOENCODING='utf-8'
& modal run DAPO_RMR1_eval/Best_of_N_selection_eval_rm_r1.py --args "--dapo_model_path gs://checkpoints-cs224n/non-curriculum/non_curriculum_phase3_dapo --rmr1_model_path gaotang/RM-R1-DeepSeek-Distilled-Qwen-7B --output_dir /tmp/rmr1_probe_20260305_004818 --dataset_jsonl /root/combined_holdout_50_each.jsonl --limit 2 --num_candidates 8 --gcp_project cs224n-project --dataset_gcp_project cs224n-project-data" 2>&1 | ForEach-Object { "2026-03-05T00:48:18.5314524-08:00 $_" }
