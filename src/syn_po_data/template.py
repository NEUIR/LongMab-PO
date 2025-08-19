PROMPT_DICT = {
    "gen_probe": (
        "You are provided with a long document, a complex logical reasoning question, and the correct answer. Your task is to read the document and perform step-by-step reasoning and finally reach the correct answer.\nInstructions:1.Each reasoning step should explicitly refer to the document.\n2.End your reasoning with `The answer is` followed by the correct answer.\nDocument: {context}\nQuestion: {question}\nAnswer: {answer}"
    ),
    "cot_answer": (
        "You are provided with a long document and a complex logical reasoning question. Read the document and follow my instructions to process it.\n\nDocument: {context}\nQuestion: {question}\nInstructions:\n#####\n1. Provide a reasoning process: You should first understand the complex problem and make a plan to solve it, then carry out the plan and solve the problem step-by-step and finally deduce the answer. You could perform reasoning with reflecting, verifying, and revising when encountering uncertain or contradictory information.\n2. Provide an answer: Based on your reasoning process, give a short answer to the question. Your answer should be concise and do not include any reasoning process.\n#####\nYour output should follow this format:\nReasoning: reasoning process here.\nAnswer: answer here."
    )
}
