
Next tasks:

- allow filter by text substring the math problem
- execute full reduced pipeline on one problem:
    $python -m utils.generate_rollout --problem-substring "3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))" --alternate-top-k 2 --alternate-min-prob 0.45 --samples-per-fork 2
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


First task to do after the meeting:
- ensure that is not possible to get logits for reaoning tokens through API providers


Substring to match
3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))
