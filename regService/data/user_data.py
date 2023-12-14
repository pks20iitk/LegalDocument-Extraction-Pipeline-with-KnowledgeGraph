from .prompt_reader import get_prompts

LOAN_AGREEMENT_QUESTIONS =  [
  {
    "Question": "What is the name of the borrower in the loan agreement?",
    "key": 'Name of Borrower',
  },
  {
    "Question":  "What is the name of the lender in the loan agreement?",
    "key": 'Name of Lender',
  },
  {
    "Question": "What is the maximum principal amount of loan in the loan agreement and the initial closing date advance on loan amount?",
    "key": 'Maximum Principal Amount of Loan',
  },
  {
    "Question": "what amount shall remain unfunded on the closing date in an unfunded reserve for payment and completion of approved initial capex? Answer with the amount and the purpose.",
    "key": 'If loan was not fully disbursed at closing, amount of future advances:',
  },
  {
    "Question": "Describe use of future advances in the loan agreement. Explain in detailed bulleted points, all the conditions under 'Initial CapEx Reserve' of the loan agreement. Strictly do not summarize.",
    "key": 'Describe uses of future advances',
  },
  {
    "Question": "What is the closing date of the loan agreement?",
    "key": 'Closing Date',
  },
  {
    "Question": "What is the Stated Maturity Date of the loan in the agreement?",
    "key": 'Initial Maturity Date',
  },
  {
    "Question": "Describe the Extension Options in the loan agreement under Extension Options subheading? Explain in detailed bulleted points, all the conditions of extension under 'Extension Conditions' of the loan agreement. State all the conditions and do not summarize.",
    "key": 'Extension Options',
  },
  {
    "Question": "Is the loan prepayable, answer yes or no? If yes, mention the under which circumstances and what are the conditions of prepayment? What is 'Exit Fee' and the conditions related to it? What is 'Prepayment Premium' and the conditions related to it? Explain in detailed bulleted points, all the conditions of prepayment under 'Optional Prepayments' of the loan agreement. Strictly do not summarize.",
    "key": 'Is Loan Prepayable?',
  },
  {
    "Question": "What is the spread and what is the initial benchmark and its relation with the spread? What is the index floor percentage in this agreement?",
    "key": 'Interest Rate',
  }
];


PROMPTS = get_prompts()

MSA_AGREEMENT_QUESTIONS = [
  {
    "Question": "what is the customer name in the MSA",
    "key": "Customer Name"
  },
  {
    "Question": "what is the service provider name in the MSA",
    "key": "Service provider"
  },
  {
    "Question":"what is the Subscription start date in MSA present in the Subscription in the form of date",
    "key": "Contract start date"
  },
  {
    "Question":  "what is the governing law mentioned in the MSA contract only mention the governing law",
    "key": "Governing Law"
  },
  {
    "Question": "what is the data Disclosure notice or data breach notice or unintentional Disclosure notice or the duration to which Disclosing party in hours mentioned in the MSA contract? only mention the duration in numbers",
    "key": "Data breach notice"
  }
]

MSA_TYPES = ["MSA", "Master Services Agreement","Platform-as-a-Service (PaaS) ORDER FORM","Main Services Agreement","Stripe Services Agreement","Stripe Services Agreement","Master Subscription Agreement","AWS Customer Agreement"]

CREDIT_TYPES= ["Loan Agreement", "Credit Agreement","TERM LOAN CREDIT AGREEMENT"]

KNOWLEDGE_GRAPH_PROMPT = """Given a financial agreement that includes terms like 'Spread,' 'Initial Benchmark,' 'Index Floor Rate,' 'Interest Rate,' and their respective relationships represented as {relations}, generate a question to extract relevant information. For instance, you can ask questions like:

'What is the spread, and what is its relationship with the initial benchmark in this agreement?'
'How is the interest rate calculated based on the spread and the initial benchmark?'
'What is the percentage of the index floor rate in this financial arrangement?'
Now, let's generate a single question: 'What is the spread, and how does it impact the calculation of the interest rate based on the initial benchmark in this financial agreement?"""




DEFAULT_CREDIT_AGREEMENT_QUESTIONS = [
    "What is the principal amount of the loan for this agreement?",
    "What is the interest rate according to this agreement?",
    "What is the stated maturity date for this agreement?",
    "What are the available extension options?",
    "Is the loan prepayable? "
]