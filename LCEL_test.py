from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

### PROMPT ###
# Define the prompt
prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
prompt

# Define the value to input into the prompt
prompt_value = prompt.invoke({"topic": "ice cream"})
prompt_value
prompt_value.to_messages()
prompt_value.to_string()

### MODEL ###
# defines the model to use
model = ChatOpenAI(model="gpt-3.5-turbo-0125")

# prompt_value is passed to the model
message = model.invoke(prompt_value)
message

### OUTPUT PARSER ###
# define the output parser
output_parser = StrOutputParser()

# message3 is passed to the output parser 
output_parser.invoke(message)

chain = prompt | model | output_parser

chain.invoke({"topic": "ice cream"})

prompt.invoke({"topic": "ice cream"})