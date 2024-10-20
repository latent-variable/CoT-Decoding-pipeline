
# CoT-Decoding Pipeline

A pipeline that implements **[Chain-of-Thought (CoT) Decoding](https://arxiv.org/abs/2402.10200)** using the Ollama API within [Open-WebUI](https://github.com/open-webui/open-webui).

## Description

Waiting on Ollama team to impelent [logprobs](https://github.com/ollama/ollama/pull/1640) to have an original implementation. 
Currently this will prompt the model to use the options as an inner monologue. 

This pipeline leverages the Ollama API to perform Chain-of-Thought decoding, which involves:

- **Extracting alternative top-𝑘 decoding paths.**
- **Calculating confidence metrics for each decoding path.**
- **Selecting the most reliable path based on confidence.**

By generating multiple responses to a given prompt and selecting the most confident one, the pipeline aims to improve the accuracy and reliability of the generated answers.

## Features

- **Chain-of-Thought Decoding**: Implements CoT decoding to enhance reasoning capabilities without explicit prompting.
- **Customizable Parameters**: Adjust `k`, `temperature`, `max_tokens`, and other parameters via configuration.
- **Debugging Output**: Optionally enable detailed debugging information to see generated responses and confidence scores.

## Requirements

- **Python Libraries**:
  - `requests` (install via `pip install requests`)
  - `pydantic` (should be available in the Open-WebUI environment)

- **Ollama API**:
  - Ensure the Ollama API is running and accessible.
  - Default API URL: `http://localhost:11434/api/generate`

## Installation

1. **Place the Script**:

   - Save the script as `cot_decoding_pipeline.py` (or any appropriate name) in the pipelines directory of Open-WebUI.

2. **Install Required Libraries**:

   ```bash
   pip install requests
   ```

## Configuration

### Valves Parameters

The pipeline uses a `Valves` class for configuration, allowing you to adjust parameters without modifying the code.

- **pipelines**: List of pipelines to connect to (default: `["*"]`).
- **priority**: Pipeline execution priority (default: `0`).
- **ollama_api_url**: URL of the Ollama API (default: `http://localhost:11434/api/generate`).
- **model**: Name of the model to use (must be set).
- **k**: Number of top-k alternatives to consider (default: `10`).
- **temperature**: Sampling temperature for response generation (default: `0.7`).
- **max_tokens**: Maximum number of tokens to generate (default: `256`).
- **debug**: Enable debugging output (default: `True`).

### Environment Variables

You can also set configuration parameters using environment variables:

- `OLLAMA_API_URL`: Overrides `ollama_api_url`.
- `COT_DECODING_K`: Overrides `k`.
- `COT_DECODING_TEMPERATURE`: Overrides `temperature`.
- `COT_DECODING_MAX_TOKENS`: Overrides `max_tokens`.
- `COT_DECODING_DEBUG`: Set to `"True"` or `"False"` to enable or disable debugging output.

### Setting the Model

Ensure that the `model` parameter in the `Valves` configuration is set to the name of the model you wish to use.

```python
self.valves = self.Valves(
    **{
        "model": "your_model_name_here",
        # other parameters...
    }
)
```

## Usage

The pipeline works within Open-WebUI and interacts with user inputs to generate responses. Here's how it operates:

1. **User Input**: The user provides a message or question.

2. **Prompt Formatting**: The pipeline formats the conversation history into a prompt in the standard Q&A format.

3. **Generating Responses**:

   - It generates `k` alternative responses using the Ollama API.
   - Each response is generated with a different random seed to ensure diversity.

4. **Calculating Confidence**:

   - For each response, a confidence score is calculated.
   - The confidence metric used is tokens per second (`eval_count / eval_duration`).

5. **Selecting the Best Response**:

   - The response with the highest confidence score is selected as the final answer.

6. **Debugging Output** (if enabled):

   - The pipeline prints the formatted prompt, all generated responses, their confidence scores, and the selected best response.

### Sample Debug Output

```
Formatted Prompt:
Q: What is the capital of France?
A:

Generated Responses and Confidence Scores:
Response 1:
Content: The capital of France is Paris.
Confidence: 0.000023

Response 2:
Content: Paris is the capital city of France.
Confidence: 0.000025

... (additional responses)

Selected Best Response:
Content: Paris is the capital city of France.
Confidence: 0.000025
```

## Notes

- **Confidence Metric**:

  - The current confidence calculation is simplified. Waiting on Ollama team to impelent [logprobs](https://github.com/ollama/ollama/pull/1640) passthrough to have a fully working version
  - For more accurate confidence metrics, consider implementing a method based on token probabilities if available from the API.

- **Debugging**:

  - Debugging output is helpful during development and testing.
  - In a production environment, consider setting `debug` to `False` to avoid exposing internal details.

- **Error Handling**:

  - The pipeline includes basic error handling for API calls.
  - You may want to enhance error handling based on your specific use case.

## Troubleshooting

- **No Response Generated**:

  - Ensure the `model` parameter is correctly set.
  - Verify that the Ollama API is running and accessible.

- **API Errors**:

  - Check the API URL and network connectivity.
  - Review error messages printed in the console or logs.

- **Debug Information Not Displayed**:

  - Confirm that `debug` is set to `True` in the configuration.
  - Ensure that the environment variable `COT_DECODING_DEBUG` is not overriding the `debug` setting.

## Contributing

Contributions and improvements to this pipeline are welcome. Feel free to fork the repository and submit pull requests.

## License

This project is licensed under the MIT License.

## Acknowledgments

- **Open-WebUI**: For providing the framework to build and integrate this pipeline.
- **Ollama**: For the API used to generate model responses.

---

If you have any questions or need further assistance, please feel free to reach out.
