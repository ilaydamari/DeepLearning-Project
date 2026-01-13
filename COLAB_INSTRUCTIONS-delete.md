# Google Colab Setup Instructions for Assignment 3

## Step 1: Open Google Colab
1. Go to https://colab.research.google.com
2. Sign in with your Google account
3. Create a new notebook

## Step 2: Enable GPU
1. In the menu: Runtime → Change runtime type
2. Select "GPU" as Hardware accelerator
3. Click "Save"

## Step 3: Upload and Run Script
1. Copy the entire content from COLAB_FULL_SCRIPT.py
2. Paste it into Colab cells (split by "# CELL X:" comments)
3. **IMPORTANT**: Update the repository URL in Cell 3:
   ```python
   REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
   ```

## Step 4: Execute Cells in Order
Run each cell sequentially:
- Cell 1: GPU check
- Cell 2: Install packages  
- Cell 3: Clone repo (UPDATE THE URL FIRST!)
- Cell 4: Data preparation
- Cell 5: Train baseline
- Cell 6: Train melody models
- Cell 7: Generate 30 songs
- Cell 8: Run evaluation
- Cell 9: Create Excel table
- Cell 10: Package results
- Cell 11: Final verification

## Expected Runtime
- Total time: ~2-3 hours
- Training: ~1 hour
- Generation: ~30 minutes
- Evaluation: ~15 minutes

## What You'll Get
1. `all_generated_songs_table.xlsx` - For your report appendix
2. 30 generated songs in `experiment_results/`
3. Complete evaluation in `evaluation_results/`
4. Trained models in `models/`
5. TensorBoard logs in `runs/`

## Download Results
At the end, download `assignment3_complete_results.zip` which contains everything you need for the report.

## Troubleshooting
- If any cell fails, restart runtime and run from beginning
- If GPU runs out of memory, reduce batch_size in training commands
- If Word2Vec download fails, training will continue with random embeddings

## For Report Writing
Use the generated Excel table and songs for qualitative analysis. Focus on:
1. Melody influence on lyrics
2. Seed word consistency
3. Model comparison (concatenation vs conditioning)
4. Song structure quality