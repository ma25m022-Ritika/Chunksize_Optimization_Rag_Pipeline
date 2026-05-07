PROMPT_TEMPLATES = {

    # -------------------------
    # Chain-of-Thought (CoT)
    # -------------------------
    "cot_prompt": """Answer the following question step by step.

Question:
{question}

{options_block}

Answer:""",

    # -------------------------
    # MedRAG / MoG Prompt
    # -------------------------
    "medrag_prompt": """You are a medical expert.

Use the following context to answer the question.

Context:
{context}

Question:
{question}

{options_block}

Answer:"""
}


def build_options_block(options):
    """
    Convert MCQ options dict into text block
    """
    if not options:
        return ""

    lines = []
    for k, v in options.items():
        lines.append(f"{k}: {v}")

    return "\n".join(lines)


def get_prompt(prompt_type, question, context=None, options=None):
    """
    Build final prompt
    """

    template = PROMPT_TEMPLATES[prompt_type]

    options_block = build_options_block(options)

    return template.format(
        question=question,
        context=context if context else "",
        options_block=options_block
    )