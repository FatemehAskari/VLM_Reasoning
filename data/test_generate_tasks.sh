# Generate the counting task stimuli when the objects are colored.
echo "Generating counting (colored) trials."
python task_scripts/gen_vlm_counting.py --object_inds=37 \
                                        --n_trials=100 \
                                        --sigma=1 \
                                        --n_shapes 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
                                        --size=40 \
                                        --output_dir=data/vlm/counting_colored

# Generate the counting task stimuli when the objects are all black.
echo "Generating counting (black) trials."
python task_scripts/gen_vlm_counting.py --object_inds=37 \
                                        --n_trials=100 \
                                        --n_shapes 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
                                        --size=40 \
                                        --all_black=True \
                                        --output_dir=data/vlm/counting_black

# Generate popout trials with only green and (one) red circles.
echo "Generating popout trials."
python task_scripts/gen_vlm_popout.py --n_objects 5 10 15 20 25 30 35 40 45 50 \
                                      --n_trials=100 \
                                      --size=24 \
                                      --colors green red \
                                      --shape_inds 37 \
                                      --output_dir=data/vlm/popout

# Generate serial search trials with green/red L's and T's.
echo "Generating serial search trials."
python task_scripts/gen_vlm_search.py --n_objects 5 10 15 20 25 30 35 40 45 50 \
                                      --n_trials=100 \
                                      --size=22 \
                                      --use_letters=True \
                                      --colors green red \
                                      --output_dir=data/vlm/search

# Generate binding trials with 2-10 objects.
echo "Generating binding trials."
python task_scripts/gen_vlm_binding.py --n_objects 2 3 4 5 6 7 8 9 10 \
                                       --n_trials=100 \
                                       --size=42 \
                                       --color_names red green blue gold purple saddlebrown gray black cyan darkorange \
                                       --shape_names airplane triangle cloud cross umbrella scissors heart star circle square \
                                       --shape_inds 6 9 21 24 34 60 96 98 100 101 \
                                       --output_dir=data/vlm/binding

# Generate the RMTS trials.
echo "Generating RMTS trials."
python task_scripts/gen_vlm_rmts.py --n_trials=200 \
                                    --color_names red green blue purple saddlebrown black \
                                    --shape_names triangle star heart cross circle square \
                                    --shape_inds 9 98 96 24 100 101 \
                                    --output_dir='data/vlm/rmts'

# Generate the TTI counting trials.
echo "Generating TTI counting trials."
python task_scripts/gen_tti_counting.py --n_shapes 2 3 4 5 6 7 8 9 10 \
                                        --objects sphere cone cube cylinder \
                                        --n_trials=100 \
                                        --output_dir='data/tti/counting'

# Generate the TTI binding trials.
echo "Generating TTI binding trials."
python task_scripts/gen_tti_binding.py --n_objects 2 3 4 \
                                       --object_names teacup balloon candle hat pen \
                                       --color_names red green blue yellow black \
                                       --n_trials=100 \
                                       --output_dir='data/tti/binding'