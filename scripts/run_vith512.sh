# DAM-SAM ViT-H @ 512x512 evaluation.
# Two variants:
#   A) "raw"    - the exact run.sh flags, for direct comparability with the 1024 baseline.
#   B) "scaled" - post-processing rescaled for the halved resolution:
#                 blob AREA thresholds /4, morphology LINEAR kernels /2.
#                 At 512 a given physical defect covers 1/4 the pixels, so the 1024-tuned
#                 area filters are wrong by construction.
CKPT=checkpoints/dam_sam_vith_512/best.pt
COMMON="--checkpoint $CKPT --checkpoint_name facebook/sam-vit-huge --image_size 512 --no_save_images"

if [ "$1" = "scaled" ]; then
  OUT=outputs_vith512_scaled
  python3 scripts/evaluate.py $COMMON --outputs_root $OUT --split gan-generated/test1 --apply_clahe --clahe_clip_limit 0.03 --clahe_grid_size 8 --gamma 0.7 --threshold_pore 0.25 --threshold_incl 0.30 --tta --morph_size_pore 0 --morph_size_incl 1 --min_blob_pore 1 --max_blob_pore 75 --min_blob_incl 1 --max_blob_incl 0
  python3 scripts/evaluate.py $COMMON --outputs_root $OUT --split gan-generated/test2 --threshold_pore 0.35 --threshold_incl 0.60 --morph_size_pore 1 --morph_size_incl 3 --min_blob_pore 1 --max_blob_pore 0 --min_blob_incl 5 --max_blob_incl 0
  python3 scripts/evaluate.py $COMMON --outputs_root $OUT --split gan-generated/test4 --pore_only
  python3 scripts/evaluate.py $COMMON --outputs_root $OUT --split gan-generated/test5 --pore_only --threshold_pore 0.60 --morph_size_pore 1 --min_blob_pore 3
  python3 scripts/evaluate.py $COMMON --outputs_root $OUT --split gan-generated/test6 --threshold_pore 0.15 --threshold_incl 0.75 --morph_size_pore 0 --morph_size_incl 1 --min_blob_pore 1 --max_blob_pore 50 --min_blob_incl 5
else
  OUT=outputs_vith512_raw
  python3 scripts/evaluate.py $COMMON --outputs_root $OUT --split gan-generated/test1 --apply_clahe --clahe_clip_limit 0.03 --clahe_grid_size 8 --gamma 0.7 --threshold_pore 0.25 --threshold_incl 0.30 --tta --morph_size_pore 0 --morph_size_incl 1 --min_blob_pore 2 --max_blob_pore 300 --min_blob_incl 2 --max_blob_incl 0
  python3 scripts/evaluate.py $COMMON --outputs_root $OUT --split gan-generated/test2 --threshold_pore 0.35 --threshold_incl 0.60 --morph_size_pore 1 --morph_size_incl 7 --min_blob_pore 2 --max_blob_pore 0 --min_blob_incl 20 --max_blob_incl 0
  python3 scripts/evaluate.py $COMMON --outputs_root $OUT --split gan-generated/test4 --pore_only
  python3 scripts/evaluate.py $COMMON --outputs_root $OUT --split gan-generated/test5 --pore_only --threshold_pore 0.60 --morph_size_pore 1 --min_blob_pore 12
  python3 scripts/evaluate.py $COMMON --outputs_root $OUT --split gan-generated/test6 --threshold_pore 0.15 --threshold_incl 0.75 --morph_size_pore 0 --morph_size_incl 1 --min_blob_pore 2 --max_blob_pore 200 --min_blob_incl 20
fi
