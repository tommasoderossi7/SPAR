
Next tasks:

- allow filter by text substring the math problem --- DONE
- execute full reduced pipeline on one problem: --- DONE
    $python -m utils.generate_rollout --problem-substring "3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))" --alternate-top-k 2 --alternate-min-prob 0.45 --samples-per-fork 2
- control that the functions to extract logprobs work correctly both in the base rollout and in the forking case --- DONE
- understand the source of the problem with the token "Äł" --- DONE (the forced tokens need to be transformed in strings before getting concatenated to the prompt)
- minimize the rollout_analysis.json content (there are redundant fields now probably) --- DONE
- check chat (https://chatgpt.com/c/68e61c8d-e318-8330-9f41-d54cf95e6ec2) to fix weird tokens --- DONE
- finish adjustment of distributions calculation part of the script --- DONE
- consider sampling 30 completions for the base outcome/path as well to compute Oo
- fix Sampling alternative branches progress bar
- execute full pipeline on one problem
- integrate bayesian CPD to infer the forking indices
- estimate the average sampling costs per problem:
    - case 1: only sampling with importance estimation based on distribution drift
    - case 2: full pipeline with bayesian CPD estimation too
- estimate the average sampling costs per problem with the thought anchors methodology
- compare the average cost per problem of the 2 approaches
- consider ways to reduce the costs of the forking paths approach
- compute the degree of overlap between the forking indices (sorted by importance (magnitude of the drift)) and most counterfactually influential units on the same set of 3 problems
- use the obtained forking indices as decompositions boundaries and apply thought anchors black box attribution method on such a decomposition, automatically label each unit (LLM-based functional labeling) and compare with the results obtained in the thought anchors paper


Substring to match
3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))
