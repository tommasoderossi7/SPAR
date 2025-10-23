
Next tasks:

- allow filter by text substring the math problem --- DONE
- execute full reduced pipeline on one problem: --- DONE
    $python -m utils.generate_rollout --problem-substring "3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))" --alternate-top-k 2 --alternate-min-prob 0.45 --samples-per-fork 2
- control that the functions to extract logprobs work correctly both in the base rollout and in the forking case --- DONE
- understand the source of the problem with the token "Äł" --- DONE (the forced tokens need to be transformed in strings before getting concatenated to the prompt)
- minimize the rollout_analysis.json content (there are redundant fields now probably) --- DONE
- check chat (https://chatgpt.com/c/68e61c8d-e318-8330-9f41-d54cf95e6ec2) to fix weird tokens --- DONE
- finish adjustment of distributions calculation part of the script --- DONE
- consider sampling 30 completions for the base outcome/path as well to compute Oo --- DONE
- fix Sampling alternative branches progress bar --- DONE
- estimate the average sampling costs per problem with the forking tokens methodology --- DONE
- estimate the average sampling costs per problem with the thought anchors methodology --- DONE
- test interruption and resume of rollout_generation --- DONE

- execute full sampling on one problem --- DONE
    -  problem: distribution empty for alternative tokens 
    -  problem: distribution key __empty__ in o0

    
- run public thought anchors code on the same problem selected for forking tokens
    - adapt generate_rollouts.py --- DONE
    - adapt analyze_rollouts.py to work both on the locally generated data and on the hugging face dataset --- DONE
        - make LLM based labeling work with openrouter and openrouter spar api key
        - how to use the new script generally? which command line arguments do what? How to analyze one single problem?
    - test analyze_rollouts_v3.py
        - execute analyze_rollouts_v3.py on the data (problem_330) coming from hugging face dataset (check below: CALL Thought anchors analyzer of hugging face datasets) --- DONE
        - testing without any LLM-based labeling --- DONE
        - run v5 and handle the too many requests rate limit --- DONE
        - navigate the results
            - useful plots for now (excluding all the by category analyses):
                - explore/chunk_accuracy_by_position.png (prob forced accuracy by position, to investigate more)
                - plots/importance_by_position.png (chunk importance by position)
                - variance_analysis/chunk_variance.txt (chunks sorted by importance, chunks of the base greedy completion or base not interveened  condition completion (which temperature?))
                - variance_analysis/within_problem_variance.txt most important chunks for every problem
            - things to investigate:
                - counterfactual accuracy (probably top1 accuracy, against correct answer or base completion answer?)
                - counterfactual kl (which 2 distributions is the kl computed for? I presume base completion distribution over final answers and cunterfactual completions distribution over final answers)
                - overdeterminedness (what is overdeterminedness and from which input is it computed?)
                - explore/chunk_accuracy_by_position.png (the accuracy is forced accuracy, accuracy against correct answer or base completion answer?)
                - difference between the correct_base_solution and incorrect_base_solution analyses (are they just 2 base completions sampled with some temperature different from 0, where one is correct answer and the other is incorrect answer?)

            - questions after investigation:
                - why is the next chunk accuracy used? And why it is not used instead the accuracy of current chunk similar completions?
                    - because the counterfactual accuracy is meant to measure the differences between deviations from current completion and the current completion, and the current completion distribution in the case the current chunk is kept is the distribution over answers at the next chunk completions sampling, not between dissimar and similar completions
                - in 4) the resempling accuracy against ground truth is it also considering similar alternative chunks completions or only dissimilar alternative chunks completions?
                - in 5) why then problem_330 it is both inside correct_base_solution and incorrect_base_solution?


    - take a look at sampled completions from forking paths 
    - ensure the information saved is sufficient for all later analysis (degree of overlap and threshold based CoT decomposition)
    - start the full sampling (topk = 10 - minp = 0.05)
    - test generate_rollouts.py

    - compute precise estimation of costs for problem_330 and compare with costs of forking tokens on the same problem


- empirically calculate the sampling cost of forking tokens
- empirically calculate the sampling cost of thought anchors

- 1) compute the degree of overlap/correlation between the forking indices (sorted by importance (magnitude of the drift)) and most counterfactually important units(sentences) on the same set of 3 problems.
- 2) integrate bayesian CPD to infer the forking indices
    - use the obtained forking indices as decompositions boundaries and apply thought anchors black box attribution method on such a decomposition, automatically label each unit (LLM-based functional labeling) and compare with the results obtained in the thought anchors paper

- consider ways to reduce the costs of the forking paths approach













Substring to match
3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))

CALL Thought anchors analyzer of hugging face datasets
python -m thought_anchors.analyze_rollouts_v5  --correct_rollouts_dir hf://uzaymacar/math-rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution  --incorrect_rollouts_dir hf://uzaymacar/math-rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/incorrect_base_solution  --llm_provider none  --importance_metric counterfactual_importance_accuracy  --use_existing_metrics  --problems "330"






1) How to use the new analysis script

You can run the script in two modes:

Local-rollouts mode (use the rollouts produced by generate_rollouts.py).

Hugging Face dataset mode (use the authors’ published dataset at uzaymacar/math-rollouts).

Below is a quick map of the most useful flags, then some copy-paste examples (including “analyze one single problem”).

Most useful CLI flags (plain-English)

Input sources

--correct_rollouts_dir / --incorrect_rollouts_dir
Path to your locally generated rollouts (e.g. math_rollouts/.../correct_base_solution).
If you want to analyze only correct or only incorrect runs, you can pass only one of them.

--correct_forced_answer_rollouts_dir / --incorrect_forced_answer_rollouts_dir
Path to the forced-answer rollouts (optional). If present, the script will compute and plot the “forced importance” metrics too.

Scope filtering

--problems "12,77,105"
Only analyze these problem indices.

--max_problems 50
Cap how many problems to analyze (after any filtering).

Labeling / metadata

--force_relabel
Re-run the LLM labeling even if chunks_labeled.json already exists.

--force_metadata
Re-generate chunk summaries and problem nicknames even if they exist.

Importance metrics & knobs

--importance_metric {counterfactual_importance_accuracy|counterfactual_importance_kl|resampling_importance_accuracy|resampling_importance_kl|forced_importance_accuracy|forced_importance_kl}
Choose which metric the plots/tables center on.

--absolute
Use absolute value when computing importance deltas (handy if you only care about magnitude).

--similarity_threshold 0.8
Cosine similarity threshold to decide “similar vs dissimilar” resampled chunks.

--use_similar_chunks/--no-use_similar_chunks
Whether to include “similar” resamples when building the comparison distribution for KL metrics.

--use_prob_true (store_true)
If on, KL is computed over P(correct) (binary). If off, KL is over the full answer distribution.

Embeddings / performance

--sentence_model all-MiniLM-L6-v2
SBERT model for chunk embeddings.

--batch_size 8192
Embedding batch size (per encode step).

--num_processes <N>
CPU processes for importance computations (defaults to up to 100).

Token-frequency analysis

--token_analysis_source {dag|rollouts}
If dag, set --dag_dir <path> to point at DAG-improved chunks. If rollouts, it uses the rollouts you’re analyzing.

--get_token_frequencies
Turn on n-gram plots per category.

Output & plotting

--output_dir analysis/basic
Where all plots and JSON summaries go.

--max_chunks_to_show 100
Limit “accuracy-by-position” plots to the first N sentences to keep charts readable.

The dataset you cited lives here and includes a “card” with usage instructions; we’re using the standard 🤗 datasets loader under the hood.