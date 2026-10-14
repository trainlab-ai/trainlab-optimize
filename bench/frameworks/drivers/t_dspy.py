# reply: [[ ## reasoning ## ]]\nr\n\n[[ ## answer ## ]]\nok\n\n[[ ## completed ## ]]
from tl import BASE
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_base=f"{BASE}/v1", cache=False))
qa = dspy.Predict("question -> answer")
qa(question="hi")  # @call
dspy.ChainOfThought("question -> answer")(question="why")  # @call
