prompt = """Take a question calculate the answer"""
examples = [
    {"question": "What is interest rate?",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"},
    {"question": "Can you please provide the value of interest rate?",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"},
    {"question": "Provide the value of interest rate?",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"},
    {"question": "Show me the value of interest rate?",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"},
    {"question": "Give the value of interest rate?",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"},
    {"question": "Could you provide a more extensive overview of interest rates?",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"},
    {"question": "Could you go into greater detail about the interest rate?",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"},
    {"question": "What is the value of interest rate?",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"},
    {"question": "Can you please provide the value of interest rate?",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"},
    {"question": "Provide the value of interest rate?",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"},
    {"question": "Could you provide a more extensive overview of interest rates?",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"},
    {"question": "Could you go into greater detail about the interest rate?",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"},
    {"question": "Share more details about the interest rate, please.?",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"},
    {"question": "What does the interest rate amount to?",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"},
    {"question": "Tell me the numerical value of the interest rate?",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"},
    {"question": "Can you please provide the value of interest rate?",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"},
    {"question": "Could you share the interest rate value, please?",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"},
    {"question": "Kindly disclose the interest rate value?",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"},
    {"question": "can you provide the interest rate value??",
     "answer": "MATCH (a:FinancialTerm)-[b:SUM_OF]->(c:FinancialTerm) OPTIONAL MATCH (c:FinancialTerm)-["
               "d:Initial_Benchmark]->(e:FinancialTerm) OPTIONAL MATCH (a:FinancialTerm)-[f:SUM_OF]->(g:FinancialTerm "
               "{name: 'Spread'}) OPTIONAL MATCH (l:FinancialTerm)-[r:MINIMUM_VALUE]->(ifr:FinancialTerm {name: "
               "'Index Floor Rate'}) OPTIONAL MATCH (ir:FinancialTerm)-[calculated_from:CALCULATED_FROM]->("
               "dr:FinancialTerm {name: 'Default Rate'}) RETURN a, b, c, d, e, g, l, r, ifr, ir, calculated_from, dr"}

]


# prompt_with_examples = prompt + "\n".join([f"{example['question']}\n{example['answer']}" for example in examples])

# Create the prompt with examples
# prompt_with_examples = create_prompt_with_examples(examples)

# Print the output
# print(prompt_with_examples)
def create_prompt_with_examples():
    """
    """
    prompt = """Take a question and calculate the answer"""
    prompt_with_examples = prompt + "\n".join([f"{example['question']}\n{example['answer']}" for example in examples])
    return prompt_with_examples
