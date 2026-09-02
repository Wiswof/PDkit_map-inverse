python train_glp1r_pytorch.py \
  Gen2_Table3_molecular_formula_with_SMILES.csv \
  --assay 1 \
  --epochs 20 \
  --batch-size 16 \
  --pca-variance 0.95 \
  --output-dir model_output_torch_pca

  python train_glp1r_pytorch.py \
  Gen2_Table3_molecular_formula_with_SMILES.csv \
  --assay 1 \
  --epochs 10 \
  --batch-size 16 \
  --pca-variance 0.95 \
  --output-dir model_output_torch_pca_fin