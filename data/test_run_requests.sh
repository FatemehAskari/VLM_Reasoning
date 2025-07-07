echo "Testing counting task with colored 2D sprite stimuli."
python run_vlm_requests.py --task='counting' \
                           --task_dir='data/vlm/counting_colored' \
                           --task_file='data/vlm/counting_colored/metadata.csv' \
                           --task_prompt_path='prompts/run/counting.txt' \
                           --parse_prompt_path='prompts/parse/counting.txt' \
                           --results_file='output/vlm_counting_colored.csv' \
                           --n_trials=1
echo "DONE"


echo "Testing counting task with black 2D sprite stimuli."
python run_vlm_requests.py --task='counting' \
                           --task_dir='data/vlm/counting_black' \
                           --task_file='data/vlm/counting_black/metadata.csv' \
                           --task_prompt_path='prompts/run/counting.txt' \
                           --parse_prompt_path='prompts/parse/counting.txt' \
                           --results_file='output/vlm_counting_black.csv' \
                           --n_trials=1
echo "DONE"


echo "Testing popout task with 2D sprite stimuli."
python run_vlm_requests.py --task='popout' \
                           --task_dir='data/vlm/popout' \
                           --task_file='data/vlm/popout/metadata.csv' \
                           --task_prompt_path='prompts/run/popout.txt' \
                           --parse_prompt_path='prompts/parse/popout.txt' \
                           --results_file='output/vlm_popout.csv' \
                           --n_trials=1
echo "DONE"


echo "Testing popout task with CLEVR stimuli."
python run_vlm_requests.py --task='popout' \
                           --task_dir='data/vlm/clevr_popout' \
                           --task_file='data/vlm/clevr_popout/metadata.csv' \
                           --task_prompt_path='prompts/run/popout.txt' \
                           --parse_prompt_path='prompts/parse/popout.txt' \
                           --results_file='output/vlm_clevr_popout.csv' \
                           --n_trials=1
echo "DONE"


echo "Testing serial search task with 2D sprite stimuli."
python run_vlm_requests.py --task='search' \
                           --task_dir='data/vlm/search' \
                           --task_file='data/vlm/search/metadata.csv' \
                           --task_prompt_path='prompts/run/search_letters.txt' \
                           --parse_prompt_path='prompts/parse/search_letters.txt' \
                           --results_file='output/vlm_search_letters.csv' \
                           --n_trials=1
                           
echo "DONE"


echo "Testing serial search task with CLEVR stimuli."
python run_vlm_requests.py --task='search' \
                           --task_dir='data/vlm/clevr_search' \
                           --task_file='data/vlm/clevr_search/metadata.csv' \
                           --task_prompt_path='prompts/run/search_clevr.txt' \
                           --parse_prompt_path='prompts/parse/search_clevr.txt' \
                           --results_file='output/vlm_clevr_search.csv' \
                           --n_trials=1
echo "DONE"


echo "Testing binding task with 2D sprite stimuli."
python run_vlm_requests.py --task='binding' \
                           --task_dir='data/vlm/binding' \
                           --task_file='data/vlm/binding/metadata.csv' \
                           --task_prompt_path='prompts/run/binding.txt' \
                           --parse_prompt_path='prompts/parse/binding.txt' \
                           --results_file='output/vlm_binding.csv' \
                            --n_trials=1
echo "DONE"
                           

echo "Testing RMTS feature decoding task (decomposed)."
python run_vlm_requests.py --task='rmts_decomposed' \
                           --task_dir='data/vlm/rmts' \
                           --task_file='data/vlm/rmts/feature_task_metadata.csv' \
                           --task_prompt_path='prompts/run/rmts_features_decomposed.txt' \
                           --parse_prompt_path='prompts/parse/rmts_feature_task.txt' \
                           --results_file='output/rmts_decomposed_feature_decoding.csv' \
                           --n_trials=1
echo "DONE"


echo "Testing RMTS feature decoding task (unified)."
python run_vlm_requests.py --task='rmts_unified' \
                           --task_dir='data/vlm/rmts' \
                           --task_file='data/vlm/rmts/feature_task_metadata.csv' \
                           --task_prompt_path='prompts/run/rmts_features_unified.txt' \
                           --parse_prompt_path='prompts/parse/rmts_feature_task.txt' \
                           --results_file='output/rmts_unified_feature_decoding.csv' \
                           --n_trials=1
echo "DONE"


echo "Testing RMTS relation decoding task (decomposed)."
python run_vlm_requests.py --task='rmts_decomposed' \
                           --task_dir='data/vlm/rmts' \
                           --task_file='data/vlm/rmts/relation_task_metadata.csv' \
                           --task_prompt_path='prompts/run/rmts_relations_decomposed.txt' \
                           --parse_prompt_path='prompts/parse/rmts_relation_task.txt' \
                           --results_file='output/rmts_decomposed_relation_decoding.csv' \
                           --n_trials=1
echo "DONE"


echo "Testing RMTS relation decoding task (unified)."
python run_vlm_requests.py --task='rmts_unified' \
                           --task_dir='data/vlm/rmts' \
                           --task_file='data/vlm/rmts/relation_task_metadata.csv' \
                           --task_prompt_path='prompts/run/rmts_relations_unified.txt' \
                           --parse_prompt_path='prompts/parse/rmts_relation_task.txt' \
                           --results_file='output/rmts_unified_relation_decoding.csv' \
                           --n_trials=1
echo "DONE"


echo "Testing full RMTS task (decomposed)."
python run_vlm_requests.py --task='rmts_decomposed' \
                           --task_dir='data/vlm/rmts' \
                           --task_file='data/vlm/rmts/full_task_metadata.csv' \
                           --task_prompt_path='prompts/run/rmts_full_decomposed.txt' \
                           --parse_prompt_path='prompts/parse/rmts_full_task.txt' \
                           --results_file='output/rmts_decomposed_full_task.csv' \
                           --n_trials=1
echo "DONE"


echo "Testing full RMTS task (unified)."
python run_vlm_requests.py --task='rmts_unified' \
                           --task_dir='data/vlm/rmts' \
                           --task_file='data/vlm/rmts/full_task_metadata.csv' \
                           --task_prompt_path='prompts/run/rmts_full_unified.txt' \
                           --parse_prompt_path='prompts/parse/rmts_full_task.txt' \
                           --results_file='output/rmts_unified_full_task.csv' \
                           --n_trials=1
echo "DONE"


echo "Testing TTI counting task."
python run_tti_requests.py --task=counting \
                           --task_file='data/tti/counting/metadata.csv' \
                           --results_file='output/tti_counting.csv' \
                           --task_payload='payloads/dalle3.json' \
                           --api='dalle' \
                           --n_trials=1
echo "DONE"


echo "Testing TTI binding task."
python run_tti_requests.py --task=binding \
                           --task_file='data/tti/binding/metadata.csv' \
                           --results_file='output/tti_binding.csv' \
                           --task_payload='payloads/dalle3.json' \
                           --api='dalle' \
                           --n_trials=1
echo "DONE"