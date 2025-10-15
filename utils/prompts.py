DAG_PROMPT = """
You are an expert in interpreting how language models solve math problems using multi-step reasoning. Your task is to analyze a Chain-of-Thought (CoT) reasoning trace, broken into discrete text chunks, and label each chunk with:

1. **function_tags**: One or more labels that describe what this chunk is *doing* functionally in the reasoning process.

2. **depends_on**: A list of earlier chunk indices that this chunk directly depends on — meaning it uses information, results, or logic introduced in those earlier chunks.

This annotation will be used to build a dependency graph and perform causal analysis, so please be precise and conservative: only mark a chunk as dependent on another if its reasoning clearly uses a previous step's result or idea.

---

### Function Tags (you may assign multiple per chunk if appropriate):

1. `problem_setup`: 
    Parsing or rephrasing the problem (initial reading or comprehension).
    
2. `plan_generation`: 
    Stating or deciding on a plan of action (often meta-reasoning).
    
3. `fact_retrieval`: 
    Recalling facts, formulas, problem details (without immediate computation).
    
4. `active_computation`: 
    Performing algebra, calculations, manipulations toward the answer.
    
5. `result_consolidation`: 
    Aggregating intermediate results, summarizing, or preparing final answer.
    
6. `uncertainty_management`: 
    Expressing confusion, re-evaluating, proposing alternative plans (includes backtracking).
    
7. `final_answer_emission`: 
    Explicit statement of the final boxed answer or earlier chunks that contain the final answer.
    
8. `self_checking`: 
    Verifying previous steps, Pythagorean checking, re-confirmations.

9. `unknown`: 
    Use only if the chunk does not fit any of the above tags or is purely stylistic or semantic.

---

### depends_on Instructions:

For each chunk, include a list of earlier chunk indices that the reasoning in this chunk *uses*. For example:
- If Chunk 9 performs a computation based on a plan in Chunk 4 and a recalled rule in Chunk 5, then `depends_on: [4, 5]`
- If Chunk 24 plugs in a final answer to verify correctness from Chunk 23, then `depends_on: [23]`
- If there's no clear dependency (e.g. a general plan or recall), use an empty list: `[]`
- If Chunk 13 performs a computation based on information in Chunk 11, which in turn uses information from Chunk 7, then `depends_on: [11, 7]`

Important Notes:
- Make sure to include all dependencies for each chunk. 
- Include both long-range and short-range dependencies.
- Do NOT forget about long-range dependencies. 
- Try to be as comprehensive as possible.
- Make sure there is always a path from earlier chunks (e.g. problem_setup and/or active_computation) to the final answer.

---

### Output Format:

Return a single dictionary with one entry per chunk, where each entry has:
- the chunk index (as the key, converted to a string),
- a dictionary with:
    - `"function_tags"`: list of tag strings
    - `"depends_on"`: list of chunk indices, converted to strings

Here's the expected format:

```language=json
{{
    "4": {{
    "function_tags": ["plan_generation"],
    "depends_on": ["3"]
    }},
    "5": {{
    "function_tags": ["fact_retrieval"],
    "depends_on": []
    }},
    "9": {{
    "function_tags": ["active_computation"],
    "depends_on": ["4", "5"]
    }},
    "24": {{
    "function_tags": ["self_checking"],
    "depends_on": ["23"]
    }},
    "25": {{
    "function_tags": ["final_answer_emission"],
    "depends_on": ["23"]
    }}
}}
```

Here is the math problem:

[PROBLEM]
{problem_text}

Here is the full Chain of Thought, broken into chunks:

[CHUNKS]
{full_chunked_text}

Now label each chunk with function tags and dependencies.
"""
