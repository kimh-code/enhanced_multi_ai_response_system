import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 키
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# 활성화된 모델 설정
ACTIVE_CLAUDE_MODEL = os.getenv("ACTIVE_CLAUDE_MODEL", "claude-3-sonnet-20240229")
ACTIVE_GPT_MODEL = os.getenv("ACTIVE_GPT_MODEL", "gpt-3.5-turbo")
ACTIVE_PERPLEXITY_MODEL = os.getenv("ACTIVE_PERPLEXITY_MODEL", "sonar")

# 토큰 비용 설정 (USD per 1K tokens)
MODEL_COSTS = {
    # Claude 모델들
    "claude-3-opus-20240229": {
        "prompt": float(os.getenv("CLAUDE_OPUS_PROMPT_COST", "0.015")),
        "completion": float(os.getenv("CLAUDE_OPUS_COMPLETION_COST", "0.075"))
    },
    "claude-3-sonnet-20240229": {
        "prompt": float(os.getenv("CLAUDE_SONNET_PROMPT_COST", "0.008")),
        "completion": float(os.getenv("CLAUDE_SONNET_COMPLETION_COST", "0.024"))
    },
    "claude-3-7-sonnet-20250219": {  # 새로운 모델 추가
        "prompt": float(os.getenv("CLAUDE_3_7_SONNET_PROMPT_COST", "0.008")),
        "completion": float(os.getenv("CLAUDE_3_7_SONNET_COMPLETION_COST", "0.024"))
    },
    "claude-3-haiku-20240307": {
        "prompt": float(os.getenv("CLAUDE_HAIKU_PROMPT_COST", "0.00025")),
        "completion": float(os.getenv("CLAUDE_HAIKU_COMPLETION_COST", "0.00125"))
    },
    "claude-2.1": {
        "prompt": float(os.getenv("CLAUDE_2_1_PROMPT_COST", "0.008")),
        "completion": float(os.getenv("CLAUDE_2_1_COMPLETION_COST", "0.024"))
    },
    "claude-2.0": {
        "prompt": float(os.getenv("CLAUDE_2_0_PROMPT_COST", "0.008")),
        "completion": float(os.getenv("CLAUDE_2_0_COMPLETION_COST", "0.024"))
    },
    
    # OpenAI 모델들
    "gpt-4o": {
        "prompt": float(os.getenv("GPT4O_PROMPT_COST", "0.005")),
        "completion": float(os.getenv("GPT4O_COMPLETION_COST", "0.015"))
    },
    "gpt-4-turbo": {
        "prompt": float(os.getenv("GPT4_TURBO_PROMPT_COST", "0.01")),
        "completion": float(os.getenv("GPT4_TURBO_COMPLETION_COST", "0.03"))
    },
    "gpt-4": {
        "prompt": float(os.getenv("GPT4_PROMPT_COST", "0.03")),
        "completion": float(os.getenv("GPT4_COMPLETION_COST", "0.06"))
    },
    "gpt-3.5-turbo": {
        "prompt": float(os.getenv("GPT35_PROMPT_COST", "0.0005")),
        "completion": float(os.getenv("GPT35_COMPLETION_COST", "0.0015"))
    },  # 여기 쉼표 추가됨
    
    # Perplexity 모델들
    "sonar": {
        "prompt": float(os.getenv("PERPLEXITY_SONAR_PROMPT_COST", "0.0080")),
        "completion": float(os.getenv("PERPLEXITY_SONAR_COMPLETION_COST", "0.0240"))
    },
    "pplx-7b-online": {
        "prompt": float(os.getenv("PERPLEXITY_7B_PROMPT_COST", "0.0006")),
        "completion": float(os.getenv("PERPLEXITY_7B_COMPLETION_COST", "0.0012"))
    },
    "pplx-70b-online": {
        "prompt": float(os.getenv("PERPLEXITY_70B_PROMPT_COST", "0.0030")),
        "completion": float(os.getenv("PERPLEXITY_70B_COMPLETION_COST", "0.0090"))
    },
    "mistral-7b": {
        "prompt": float(os.getenv("PERPLEXITY_MISTRAL_PROMPT_COST", "0.0006")),
        "completion": float(os.getenv("PERPLEXITY_MISTRAL_COMPLETION_COST", "0.0012"))
    },
    "llama-2-70b": {
        "prompt": float(os.getenv("PERPLEXITY_LLAMA_PROMPT_COST", "0.0030")),
        "completion": float(os.getenv("PERPLEXITY_LLAMA_COMPLETION_COST", "0.0090"))
    }
}  # 닫는 괄호 추가됨

# 기본 비용 (찾지 못할 경우 사용)
DEFAULT_CLAUDE_COST = {
    "prompt": 0.008,
    "completion": 0.024
}

DEFAULT_GPT_COST = {
    "prompt": 0.0005,
    "completion": 0.0015
}

DEFAULT_PERPLEXITY_COST = {
    "prompt": 0.0080,
    "completion": 0.0240
}

def get_model_cost(model_name):
    """모델 이름으로 비용 정보 반환"""
    # 정확한 모델 이름으로 찾기
    if model_name in MODEL_COSTS:
        return MODEL_COSTS[model_name]
    
    # 모델 이름의 일부만 일치하는 경우 (부분 일치)
    for model in MODEL_COSTS:
        if model in model_name:
            return MODEL_COSTS[model]
    
    # Claude 계열인지 확인
    if "claude" in model_name.lower():
        return DEFAULT_CLAUDE_COST
    
    # GPT 계열인지 확인
    if "gpt" in model_name.lower():
        return DEFAULT_GPT_COST
    
    # Perplexity 계열인지 확인
    if any(name in model_name.lower() for name in ["perplexity", "pplx", "sonar", "mistral", "llama"]):
        return DEFAULT_PERPLEXITY_COST
    
    # 기본값 (Claude)
    return DEFAULT_CLAUDE_COST

# 활성화된 모델의 비용 정보
ACTIVE_CLAUDE_COST = get_model_cost(ACTIVE_CLAUDE_MODEL)
ACTIVE_GPT_COST = get_model_cost(ACTIVE_GPT_MODEL)
ACTIVE_PERPLEXITY_COST = get_model_cost(ACTIVE_PERPLEXITY_MODEL)