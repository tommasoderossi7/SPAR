
Next tasks:

- allow filter by text substring the math problem --- DONE
- execute full reduced pipeline on one problem: --- DONE
    $python -m utils.generate_rollout --problem-substring "3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))" --alternate-top-k 2 --alternate-min-prob 0.45 --samples-per-fork 2
- control that the functions to extract logprobs work correctly both in the base rollout and in the forking case --- DONE
- understand the source of the problem with the token "Äł"
- minimize the rollout_analysis.json content (there are redundant fields now probably)
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

I found Nanda suggesting to use nebius as the provider for CoT interventions experiments.
I experimented with it and indeed nebius provides reasoning tokens logprobs. The thing is that it serves only reasoning models such as QwQ 32b and Deepseek r1, it doesn't serve distilled versions such as the specific model from which the paper results come from "deepseek/deepseek-r1-distill-qwen-14b". 
I also tested fireworks endpoints (without passing through the openrouter wrapper) for whether they allow to get reasoning tokens logprobs and that's the case but also fireworks doesn't serve the specific model used to obtain the paper experiments results. 
I also tested whether novita endpoints, and it looks the best api provider: it returns reasoning tokens logprobs of the distilled model used in the paper. (the only minor drawback is that they don't serve QwQ 32b which is listed between the models one can chose in the open source interface of thought anchors paper (all the other 4 models are served by novita)).
It can be the case that also together ai endpoints return logprobs of reasoning tokens and serve distilled versions of deepseek as well as other popular reasoning models. I didn't test it myself since they do not provide any free credits upon subscription and novita already proved to be complete enough in terms of logprobs of resoning tokens and supported models.


Substring to match
3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))
