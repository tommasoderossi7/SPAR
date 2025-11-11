import json
from argparse import ArgumentParser

parser = ArgumentParser(
    description="Token-level vs sentence-level counterfactual importance comparison utility"
)


parser.add_argument(
    "--tokenlvl_file_path",
    type=str,
    required=False,
    default="C:\\Users\\Tommaso Derossi\\OneDrive\\Desktop\\SPAR\\SPAR\\math_rollouts\\deepseek_deepseek-r1-distill-qwen-14b",
    help="Path to the token-level data file",
)
parser.add_argument(
    "--tokenlvl_file_name",
    type=str,
    required=False,
    default="rollout_analysis.json",
    help="Path to the token-level data file",
)
parser.add_argument(
    "--sentencelvl_file_path",
    type=str,
    required=False,
    default="C:\\Users\\Tommaso Derossi\\OneDrive\\Desktop\\SPAR\\SPAR\\thought_anchors\\.cache\\math_rollouts_hf\\deepseek-r1-distill-qwen-14b\\temperature_0.6_top_p_0.95",
    help="Path to the sentence-level data file",
)
parser.add_argument(
    "--sentencelvl_file_name",
    type=str,
    required=False,
    default="chunks_labeled.json",
    help="Name of the sentence-level data file",
)

parser.add_argument(
    "--comparison_output_file",
    type=str,
    required=False,
    help="Path to save the comparison results",
)
parser.add_argument(
    "--importance_measure",
    type=str,
    required=False,
    default="kl_true",
    choices=["delta_acc", "kl_true", "kl_full"],
    help="Unit importance measure to use",
)
parser.add_argument(
    "--base_sol_type",
    type=str,
    required=False,
    default="correct",
    choices=["correct", "incorrect"],
    help="Base solution type to use",
)
parser.add_argument(
    "--problems_to_compare",
    type=str,
    nargs="+",
    default=["330"],
    required=False,
    help="List of problems to compare",
)
args = parser.parse_args()

tkn_to_snt_importance_measure = {
    "delta_acc": "counterfactual_importance_accuracy",
    # "kl_true": "counterfactual_importance_kl", # this when thought_anchors.analyze_rollouts_v2.py was run with --use_prob_true
    "kl_full": "counterfactual_importance_kl",
}


def main():
    for problem in args.problems_to_compare:
        tkn_lvl_file_w_path = (
            args.tokenlvl_file_path
            + f"\problem_{problem}\samples_30_topk_10_prob_0.05\{args.tokenlvl_file_name}"
        )

        snt_lvl_file_w_path = f"{args.sentencelvl_file_path}\{args.base_sol_type}_base_solution\problem_{problem}\{args.sentencelvl_file_name}"

        with open(snt_lvl_file_w_path, "r", encoding="utf-8") as f:
            sent_data = json.load(f)

        with open(tkn_lvl_file_w_path, "r", encoding="utf-8") as f:
            token_data = json.load(f)

        # Further processing and comparison logic goes here
        tok_importance_offsets = []
        for tok in token_data["token_steps"]:
            tok_importance_offsets.append(
                (
                    tok["token"],
                    tok["cf_metrics"][args.importance_measure]
                    if "cf_metrics" in tok
                    else 0,
                    tok["text_offset"],
                    tok["text_offset"] + len(tok["token"]),
                )
            )

        sent_importance_offsets = []
        curr_offset = 0
        for chunk in sent_data:
            sent_importance_offsets.append(
                (
                    chunk["chunk"],
                    chunk[tkn_to_snt_importance_measure[args.importance_measure]],
                    curr_offset,
                    curr_offset + len(chunk["chunk"]),
                )
            )
            curr_offset += len(chunk["chunk"])

        # 1
        # now for every sentence we compute the total/max/avg importance of the tokens that at least partially fall within its offsets
        sent_token_importance = []
        for sent, sent_imp, sent_start, sent_end in sent_importance_offsets:
            overlapping_token_importances = []
            for tok, tok_imp, tok_start, tok_end in tok_importance_offsets:
                # check for overlap
                if not (tok_end <= sent_start or tok_start >= sent_end):
                    overlapping_token_importances.append(tok_imp)
            if overlapping_token_importances:
                avg_imp = sum(overlapping_token_importances) / len(
                    overlapping_token_importances
                )
                # max can be positive or negative, the value to insert should be the one with the highest absolute value
                max_imp = max(overlapping_token_importances, key=abs)
                total_imp = sum(overlapping_token_importances)
            else:
                avg_imp = 0
                max_imp = 0
                total_imp = 0
            sent_token_importance.append((sent, sent_imp, avg_imp, max_imp, total_imp))

        # now we want to compute correlation metrics between sent_imp and avg_imp/max_imp/total_imp and the rank correlation between the sentence importance and the token importance rankings
        import scipy.stats as stats

        sent_importances = [x[1] for x in sent_token_importance]
        avg_token_importances = [x[2] for x in sent_token_importance]
        max_token_importances = [x[3] for x in sent_token_importance]
        total_token_importances = [x[4] for x in sent_token_importance]
        pearson_avg = stats.pearsonr(sent_importances, avg_token_importances)
        spearman_avg = stats.spearmanr(sent_importances, avg_token_importances)
        pearson_max = stats.pearsonr(sent_importances, max_token_importances)
        spearman_max = stats.spearmanr(sent_importances, max_token_importances)
        pearson_total = stats.pearsonr(sent_importances, total_token_importances)
        spearman_total = stats.spearmanr(sent_importances, total_token_importances)

        # PLOTTING
        import matplotlib.pyplot as plt

        # PLOT 1 (token importance scatter plots)
        plt.figure(figsize=(18, 6))
        plt.scatter(
            range(len(tok_importance_offsets)),
            [tok_importance_offsets[i][1] for i in range(len(tok_importance_offsets))],
            color="lightgreen",
            label="Token Importance (Max)",
            alpha=0.5,
        )
        plt.xlabel("Token Index")
        plt.ylabel("Importance")
        plt.title("Token Importance (Max) per index")
        plt.legend()
        plt.show()

        # PLOT 2 (sentence importance scatter plot)
        plt.figure(figsize=(18, 6))
        plt.scatter(
            range(len(sent_importances)),
            sent_importances,
            color="blue",
            label="Sentence Importance",
        )
        plt.xlabel("Sentence Index")
        plt.ylabel("Importance")
        plt.title("Sentence Importance")
        plt.legend()
        plt.show()

        # PLOT 3 (token importance per sentence scatter plot)
        plt.figure(figsize=(18, 6))
        plt.scatter(
            range(len(max_token_importances)),
            max_token_importances,
            color="lightgreen",
            label="Token Importance (Max)",
            alpha=0.5,
        )
        plt.xlabel("Sentence Index")
        plt.ylabel("Importance")
        plt.title("Token Importance (Max) per Sentence")
        plt.legend()
        plt.show()

        # PLOT 4 (token vs sentence importances)
        # now I want to plot these importances in a scatter plot where the x axis is the sentence index and the y axis is the importance, in blue the sentence importance and in different greens the token-based importance (avg/max/total)

        plt.figure(figsize=(18, 6))
        plt.scatter(
            range(len(sent_importances)),
            sent_importances,
            color="blue",
            label="Sentence Importance",
        )
        """
        plt.scatter(
            range(len(avg_token_importances)),
            avg_token_importances,
            color="green",
            label="Token Importance (Avg)",
            alpha=0.5,
        )
        """
        plt.scatter(
            range(len(max_token_importances)),
            max_token_importances,
            color="lightgreen",
            label="Token Importance (Max)",
            alpha=0.5,
        )
        """
        plt.scatter(
            range(len(total_token_importances)),
            total_token_importances,
            color="darkgreen",
            label="Token Importance (Total)",
            alpha=0.9,
        )
        """
        plt.xlabel("Sentence Index")
        plt.ylabel("Importance")
        plt.title("Sentence vs Token Importance")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()


"""
    # 2
    # now we need to compute the rank correlation between the two importance lists based on offsets
    # first thing would be to take the first K most important tokens (where K is the number of sentences)
    K = len(sent_importance_offsets)
    most_important_tokens = sorted(tok_importance_offsets, key=lambda x: x[1], reverse=True)[:K]
    
    # now let's sort sentences by their importance
    sorted_sentences = sorted(sent_importance_offsets, key=lambda x: x[1], reverse=True)
    
    # now the idea is to measure the correlation between the two rankings based on overlap of offsets
"""
