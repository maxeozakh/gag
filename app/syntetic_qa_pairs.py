from tqdm.auto import tqdm # type: ignore
import pandas as pd
from typing import Optional, List, Tuple
import json
import datasets # type: ignore

from dotenv import load_dotenv
import os
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

# Retrieve the token from environment variables
# hf_token = os.getenv('HF_TOKEN')

# Log in using the token
# login(token=hf_token)


from langchain_community.document_loaders.csv_loader import CSVLoader

# Initialize the CSVLoader with your file path


# Verify the number of documents loaded

def load_dataset(path):
    loader = CSVLoader(file_path=path)

    # Load the data into a list of Document objects
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    return documents

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
import pandas as pd
from huggingface_hub import InferenceClient

class QAPairs:
    def __init__(self, path):
        # Load the dataset using CSVLoader
        loader = CSVLoader(file_path=path)
        self.ds = loader.load()

        # Convert loaded documents to LangchainDocument format
        langchain_docs = [
            LangchainDocument(page_content=doc.page_content, metadata=doc.metadata)
            for doc in tqdm(self.ds, desc="Processing documents")
        ]

        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            add_start_index=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        # Split documents into chunks
        self.docs_processed = []
        for doc in langchain_docs:
            self.docs_processed.extend(text_splitter.split_documents([doc]))

        print('Docs processed', len(self.docs_processed))




        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

        llm_client = InferenceClient(
            model=repo_id,
            timeout=120,
        )


        def call_llm(inference_client: InferenceClient, prompt: str):
            response = inference_client.text_generation(prompt, max_new_tokens=1000)
            return response


        
        call_llm(llm_client, "who let the dogs out")

        QA_generation_prompt = """
        Your task is to write a factoid question, an answer, and a key fact given a context.
        Your factoid question should be answerable with a specific, concise piece of factual information from the context.
        Your factoid question should be formulated in the same style as questions users could ask in a search engine.
        This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

        IMPORTANT: You must ALWAYS extract a key fact that represents the main product being discussed.
        The key fact should be a single, concise phrase (2-5 words) that captures the main idea of the context.
        Examples of good key facts:
        - "Logo baseball hat"
        - "Material: nylon"
        - "Cotton t-shirt"

        Provide your answer in EXACTLY this format:

        Output:::
        Factoid question: (your factoid question)
        Answer: (your answer to the factoid question)
        Key fact: (your key fact)

        Now here is the context.

        Context: {context}
        Output:::"""

        import random

        N_GENERATIONS = 20  

        print(f"Generating {N_GENERATIONS} QA couples...")

        outputs = []
        counter = 0
        for sampled_context in tqdm(random.sample(self.docs_processed, N_GENERATIONS)):
            counter += 1
            # Generate QA couple
            output_QA_couple = call_llm(
                llm_client, QA_generation_prompt.format(context=sampled_context.page_content)
            )
            try:
                # Split the output into parts
                output_parts = output_QA_couple.split("\n")
                question = ""
                answer = ""
                key_fact = ""
                
                for part in output_parts:
                    part = part.strip()  # Remove leading/trailing whitespace
                    if part.startswith("Factoid question:"):
                        question = part.replace("Factoid question:", "").strip()
                    elif part.startswith("Answer:"):
                        answer = part.replace("Answer:", "").strip()
                    elif part.startswith("Key fact:"):
                        key_fact = part.replace("Key fact:", "").strip()
                
                assert len(answer) < 300, "Answer is too long"
                if key_fact == "":
                    print('key fact is missing', output_parts)
                    continue
                
                outputs.append({
                    "context": sampled_context.page_content,
                    "question": question,
                    "answer": answer,
                    "key_fact": key_fact,
                    "source_doc": sampled_context.metadata["source"],
                    "groundedness_score": 1,
                    "relevance_score": 1,
                    "standalone_score": 1,
                    "groundedness_eval": "",
                    "relevance_eval": "",
                    "standalone_eval": "",
                })
                print('successfully generated QA pair', counter)
            except Exception as e:
                print('failed to generate QA pair at index', counter, str(e))
                continue

        question_groundedness_critique_prompt = """
        You will be given a context and a question.
        Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
        Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

        Provide your answer as follows:

        Answer:::
        Evaluation: (your rationale for the rating, as a text)
        Total rating: (your rating, as a number between 1 and 5)

        You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

        Now here are the question and context.

        Question: {question}\n
        Context: {context}\n
        Answer::: """

        question_relevance_critique_prompt = """
        You will be given a question.
        Your task is to provide a 'total rating' representing how useful this question can be to machine learning developers building NLP applications with the Hugging Face ecosystem.
        Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

        Provide your answer as follows:

        Answer:::
        Evaluation: (your rationale for the rating, as a text)
        Total rating: (your rating, as a number between 1 and 5)

        You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

        Now here is the question.

        Question: {question}\n
        Answer::: """

        question_standalone_critique_prompt = """
        You will be given a question.
        Your task is to provide a 'total rating' representing how context-independent this question is.
        Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
        For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
        The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

        For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independent from the context.

        Provide your answer as follows:

        Answer:::
        Evaluation: (your rationale for the rating, as a text)
        Total rating: (your rating, as a number between 1 and 5)

        You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

        Now here is the question.

        Question: {question}\n
        Answer::: """


        print("Generating critique for each QA couple...")
        for idx, output in enumerate(tqdm(outputs)):
            print(f'generating critique for QA pair at index {idx + 1}')
            try:
                evaluations = {
                    "groundedness": call_llm(
                        llm_client,
                        question_groundedness_critique_prompt.format(
                            context=output["context"], question=output["question"]
                        ),
                    ),
                    "relevance": call_llm(
                        llm_client,
                        question_relevance_critique_prompt.format(question=output["question"]),
                    ),
                    "standalone": call_llm(
                        llm_client,
                        question_standalone_critique_prompt.format(question=output["question"]),
                    ),
                }

                # Try to parse each evaluation
                for criterion, evaluation in evaluations.items():
                    try:
                        if "Total rating:" in evaluation and "Evaluation:" in evaluation:
                            # Get everything after "Total rating:" but only the first line
                            rating_part = evaluation.split("Total rating:")[-1].split('\n')[0].strip()
                            # Get the evaluation part between "Evaluation:" and "Total rating:"
                            eval_part = evaluation.split("Evaluation:")[-1].split("Total rating:")[0].strip()
                            
                            
                            score = extract_rating(rating_part)
                            output[f"{criterion}_score"] = score
                            output[f"{criterion}_eval"] = eval_part
                    except Exception as parse_error:
                        print(f'Failed to parse {criterion} critique: {parse_error}')
                        continue

            except Exception as e:
                print(f'Failed to generate critique for QA pair {idx + 1}: {e}')
                continue

        # Move DataFrame creation outside the loop
        if not outputs:
            print("No QA pairs were generated successfully. Exiting...")
            return

        generated_questions = pd.DataFrame.from_dict(outputs)

        # Only proceed with evaluation if we have data
        if len(generated_questions) > 0:
            print("Evaluation dataset before filtering:")
            print(
                generated_questions[
                    [
                        "groundedness_score",
                        "relevance_score",
                        "standalone_score",
                    ]
                ]
            )

            # Filter and save operations
            filtering_threshold = 5
            generated_questions = generated_questions.loc[
                (generated_questions["groundedness_score"] + 
                generated_questions["relevance_score"] + 
                generated_questions["standalone_score"]) >= filtering_threshold
            ]
        else:
            print("No data to evaluate. Exiting...")
            return

        print("============================================")
        print("Final evaluation dataset:")
        print(
            generated_questions[
                [
                    "question",
                    "groundedness_score",
                    "relevance_score",
                    "standalone_score",
                ]
            ]
        )

        # Save to CSV
        columns_to_save = [
            "id",
            "question",
            "answer",
            "key_fact"
        ]

        # Create a DataFrame with the required columns
        generated_questions['id'] = range(1, len(generated_questions) + 1)  # Add sequential IDs

        print('start to save csv')
        output_csv_path = "data/generated_qa_pairs.csv"
        generated_questions[columns_to_save].to_csv(output_csv_path, index=False)
        print(f"QA pairs saved to {output_csv_path}")

        # eval_dataset = datasets.Dataset.from_pandas(
        #     generated_questions, split="train", preserve_index=False
        # )

def extract_rating(rating_text: str) -> int:
    # First try to get the first number from the string
    import re
    numbers = re.findall(r'\d+', rating_text.split('\n')[0])
    if numbers:
        rating = int(numbers[0])
        if 1 <= rating <= 5:
            return rating
    raise ValueError(f"Failed to extract valid rating (1-5) from text: {rating_text}")