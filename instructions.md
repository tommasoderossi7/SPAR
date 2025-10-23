create env where to execute r package from:

# 1) Clone your current env
conda create --name spar-r --clone spar
conda activate spar-r

# 2) Prefer conda-forge in *this env only*
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict

# 3) Try staying on Python 3.13 first (works on many setups)
conda install -y r-base rpy2

# 4) Install BEAST (Rbeast) *inside the env’s R*
R -q -e "install.packages('Rbeast', repos='https://cloud.r-project.org')"

# 5) Set R_HOME correctly
export R_HOME="$(R RHOME)"
echo "$R_HOME"   # should look like /home/.../envs/spar-r/lib/R

# 6) Make this persistent for the env by creating small activation scripts:
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/r-home.sh" <<'EOF'
export R_HOME="$(R RHOME)"
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/r-home.sh" <<'EOF'
unset R_HOME
EOF

7) Sanity checks
# R can load BEAST?
Rscript -e "library(Rbeast); cat('Rbeast OK\n')"

# rpy2 sees this R?
python - <<'PY'
import rpy2.robjects as ro
import rpy2.situation as s
print("R_HOME seen by rpy2:", s.get_r_home())
ro.r('library(Rbeast)')
print("Rbeast OK in rpy2")
PY



Run the comparison:

# 1) Run token script (with the code additions) → produces:
#    math_rollouts/<model>/samples_<S>_topk_<K>_prob_<p>/problem_XXX/rollout_analysis.json
python -m forking_tokens.generate_rollout -atk 10 -amp 0.05 -sps 30

# 2) Run your chunk pipeline as before (correct &/or incorrect), e.g.:
python -m thought_anchors.analyze_rollouts

# 3) Compare (point to the sentence rollouts (not the analysis dir) so we can read base_solution.json and chunks.json)
python comparison.py
  --token_root math_rollouts/deepseek_deepseek-r1-distill-qwen-14b/samples_30_topk_10_prob_0.05
  --chunk_correct_root math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution
  --chunk_incorrect_root math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/incorrect_base_solution
  --top_tokens 0
  --top_sentences 5
  --out comparison_out