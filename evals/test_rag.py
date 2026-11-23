from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    GEval
)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from graph import graph
from pathlib import Path
import pytest
from deepeval import assert_test
#=======
# Load goldens
#=======

dataset = EvaluationDataset()

dataset.add_goldens_from_json_file(
    file_path=str(Path(__file__).resolve().parent / "goldens.json")
)

# print(f"Loaded {len(dataset.goldens)} goldens")

MODEL="gpt-4o-mini"
#====
# GEval - Build your own custom metric
#=====
correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Compare the facts in actual output with the facts in expected output",
        "Penalize facts that are missing or contradict the expected output"
    ],
    evaluation_params=[
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT
    ],
    threshold=0.7,
    model=MODEL
)


#====
# Metrics
#====

metrics = [
    FaithfulnessMetric(threshold=0.8, model=MODEL), # Is answer grounded in the chunks -> input + actual_output + retrieved_context
    AnswerRelevancyMetric(threshold=0.7, model=MODEL), # Was the answer an answer to the question -> input + actual_output
    ContextualRecallMetric(threshold=0.7, model=MODEL), # Did the chunks contain everything needed for the answer. -> input + expected_output + retrieved_context
    ContextualPrecisionMetric(threshold=0.6, model=MODEL), # Are the best or relevant chunks at the top of the array. -> input + expected_output + retrieved_context
    ContextualRelevancyMetric(threshold=0.5, model=MODEL), # Was the chunks full of junk or worth it. -> input + retrieved_context,
    correctness
]



# ====
# Run the eval test
# ====

for index,golden in enumerate(dataset.goldens[0:4]):
    print("Question:",index+1)
    
    
    state_after_graph_invoke = graph.invoke({"question": golden.input})
    
    # LLM answer written by the write_answer_node
    llm_answer = state_after_graph_invoke.get("answer"," ")
    
    # The retrieved chunks used
    retrieved_chunks = state_after_graph_invoke.get("chunks",[])
    
    
    # =====
    #  Build the test case
    # =====
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=llm_answer,
        retrieval_context=[chunk["text"] for chunk in  retrieved_chunks],
        expected_output=golden.expected_output
    )
    
    for metric in metrics:
        metric.measure(test_case=test_case)
        status = "PASS" if metric.is_successful() else "FAIL"
        
        print(f"{metric.__name__}: {status}")
        print(f"Score: {metric.score:.2f}")
        print(f"Cost: $ {metric.evaluation_cost:.6f}")
        
        if not metric.is_successful():
            print(f"Reason: {metric.reason}")
        print("------------------------------------------------------------")
        
    
    print("\n")
    
    
    
# @pytest.mark.parametrize("golden",dataset.goldens[0:4])
# def test_rag(golden):
#     state = graph.invoke({"question": golden.input})
#     answer = state.get("answer","")
#     chunks = state.get("chunks",[])
    
#     test_case = LLMTestCase(
#         input=golden.input,
#         actual_output=answer,
#         retrieval_context=[chunk["text"] for chunk in chunks],
#         expected_output=golden.expected_output
#     )
#     assert_test(test_case=test_case, metrics=metrics)