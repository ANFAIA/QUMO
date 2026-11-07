"""Where the model lives: an OpenAI-compatible chat-completions endpoint (OpenRouter)."""


@fieldwise_init
struct ModelEndpoint(Copyable):
    var base_url: String           # "https://openrouter.ai/api/v1"
    var api_key_env: String        # "OPENROUTER_API_KEY"
    var header_names: List[String]
    var header_values: List[String]
    var timeout_s: Int             # 300, per connect/read/write operation (A5)
    var max_retries: Int           # 2 (A6)
