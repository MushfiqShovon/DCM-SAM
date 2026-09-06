python3 scripts/evaluate.py --checkpoint checkpoints/dam_sam/best.pt --split gan-generated/test1 --apply_clahe --clahe_clip_limit 0.03 --clahe_grid_size 8   --gamma 0.7 --threshold_pore 0.25 --threshold_incl 0.30  --tta  --morph_size_pore 0 --morph_size_incl 1   --min_blob_pore 2 --max_blob_pore 300   --min_blob_incl 2 --max_blob_incl 0
python3 scripts/evaluate.py --checkpoint checkpoints/dam_sam/best.pt --split gan-generated/test2 --threshold_pore 0.35 --threshold_incl 0.60 --morph_size_pore 1 --morph_size_incl 7 --min_blob_pore 2 --max_blob_pore 0 --min_blob_incl 20 --max_blob_incl 0

python3 scripts/evaluate.py  --checkpoint checkpoints/dam_sam/best.pt  --split gan-generated/test4 --pore_only
python3 scripts/evaluate.py  --checkpoint checkpoints/dam_sam/best.pt --split gan-generated/test5 --pore_only --threshold_pore 0.60 --morph_size_pore 1 --min_blob_pore 12
python3 scripts/evaluate.py  --checkpoint checkpoints/dam_sam/best.pt --split gan-generated/test6 --threshold_pore 0.15 --threshold_incl 0.75 --morph_size_pore 0 --morph_size_incl 1 --min_blob_pore 2 --max_blob_pore 200 --min_blob_incl 20


########### nist
python3 scripts/evaluate.py --checkpoint checkpoints/dam_sam/best.pt --split nist/test2 --threshold_pore 0.90 --morph_size_pore 2 --erode_pore 1 --min_blob_pore 3 
python3 scripts/evaluate.py --checkpoint checkpoints/dam_sam/best.pt --split nist/test3 
python3 scripts/evaluate.py --checkpoint checkpoints/dam_sam/best.pt --split nist/test4 --threshold_pore 0.90 --morph_size_pore 2 --erode_pore 1 --min_blob_pore 3
python3 scripts/evaluate.py --checkpoint checkpoints/dam_sam/best.pt --split nist/test5 
python3 scripts/evaluate.py --checkpoint checkpoints/dam_sam/best.pt --split nist/test6 --threshold_pore 0.50 --morph_size_pore 0 --invert_pore --valid_region_mode nonzero
