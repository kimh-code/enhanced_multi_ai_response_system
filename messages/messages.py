"""
메시지 번역 정의
한국어 및 영어에 대한 모든 UI 텍스트 정의
"""

from .message_keys import MK

# 모든 UI 텍스트를 언어별로 정의
MESSAGES = {
    "ko": {
        # 앱 제목 및 기본 UI
        MK.APP_TITLE: "Enhanced AI 응답 시스템",
        MK.APP_SUBTITLE: "더 나은 AI 응답을 위한 대화형 시스템",
        MK.SETTINGS: "설정",
        MK.SYSTEM_SETTINGS: "시스템 설정",
        MK.LANGUAGE: "언어 / Language",
        
        # 미들웨어 초기화
        MK.CLAUDE_CLIENT_INIT_FAILED: "Claude 클라이언트 초기화 실패: {0}",
        MK.OPENAI_CLIENT_INIT_FAILED: "OpenAI 클라이언트 초기화 실패: {0}",
        MK.PERPLEXITY_CLIENT_INIT_FAILED: "Perplexity 클라이언트 초기화 실패: {0}",
        MK.MIDDLEWARE_INIT_HELP: "API 연결 문제가 있을 때 사용하세요",
        MK.MIDDLEWARE_INIT_SUCCESS: "미들웨어가 성공적으로 초기화되었습니다.",
        MK.CHECK_API_STATUS_HELP: "각 API 서비스의 연결 상태를 확인합니다",
        MK.API_STATUS_NORMAL: "정상",
        MK.API_STATUS_NOT_INITIALIZED: "초기화되지 않음",
        MK.API_STATUS_ERROR: "오류:",
        MK.MIDDLEWARE_NOT_INITIALIZED: "미들웨어가 초기화되지 않았습니다.",
        MK.MIDDLEWARE_STATUS_INITIALIZED: "초기화됨",
        MK.MIDDLEWARE_STATUS_NOT_INITIALIZED: "초기화되지 않음",

        # 설정 및 옵션
        MK.BUDGET_LIMIT: "예산 제한 ($)",
        MK.PERPLEXITY_ENABLED: "Perplexity AI 사용",
        MK.RESET_TOKEN_USAGE_HELP: "현재 세션의 토큰 사용량 기록을 초기화합니다",
        MK.RESET_SESSION_HELP: "모든 대화 내역과 쿼리 기록을 초기화합니다",
        MK.CLAUDE_MODEL_HELP: "사용할 Claude 모델을 선택하세요",
        MK.GPT_MODEL_HELP: "사용할 GPT 모델을 선택하세요",
        MK.PERPLEXITY_MODEL_HELP: "사용할 Perplexity 모델을 선택하세요",
        MK.MODEL_SETTINGS_APPLIED: "모델 설정이 적용되었습니다. 미들웨어를 초기화합니다...",

        # 쿼리 및 응답 관련
        MK.USER_LABEL: "사용자",
        MK.QUERY_NUMBER: "쿼리 #{0}",
        MK.QUESTION_LABEL: "질문:",
        MK.PROCESSING_TIME_LABEL: "처리 시간:",
        MK.COST_LABEL: "비용:",
        MK.SECONDS_UNIT: "초",
        MK.NEW_QUESTION: "새 질문 입력",
        MK.QUESTION_INPUT: "질문 입력:",
        MK.SUBMIT_QUESTION: "질문 제출",
        MK.CONTINUE_CONVERSATION: "대화 계속하기",
        MK.NO_RESPONSE_TO_SELECT: "선택할 응답이 없습니다",
        MK.MODEL_RESPONSE: "{0} 응답",
        MK.RESPONSE_NUMBER: "응답 #{0}",
        MK.RESPONSE_LABEL: "응답",
        MK.INITIAL_RESPONSE_LABEL: "초기 응답",

        # 결과 표시
        MK.QUESTION: "질문",
        MK.CURRENT_QUERY_RESULTS: "현재 쿼리 결과",
        MK.CONVERSATION_HISTORY: "대화 기록",
        MK.PREVIOUS_QUERY_HISTORY: "이전 쿼리 기록",
        MK.NO_PREVIOUS_HISTORY: "아직 이전 쿼리 기록이 없습니다.",
        MK.PREVIOUS_QUESTION: "이전 질문",
        MK.QUESTION_OR_INSTRUCTION_INPUT: "질문 또는 지시사항 입력:",

        # 토큰 및 비용
        MK.AI_MODEL_COLUMN: "AI 모델",
        MK.INPUT_TOKENS_COLUMN: "입력 토큰",
        MK.OUTPUT_TOKENS_COLUMN: "출력 토큰",
        MK.COST_COLUMN: "비용",
        MK.TOKEN_USAGE_BY_MODEL_TITLE: "AI 모델별 토큰 사용량",
        MK.COST_DISTRIBUTION_TITLE: "비용 분포",
        MK.TOKEN_USAGE: "토큰 사용량",
        MK.TOTAL_COST: "총 비용",
        MK.NO_TOKENS_USED: "현재 사용된 토큰이 없습니다.",
        
        # 응답 표시 설정
        MK.RESPONSE_DISPLAY_SETTINGS: "응답 표시 설정",
        MK.SELECT_RESPONSE_TYPES: "응답 표시 유형 선택 (다중 선택 가능):",
        MK.APPLY_DISPLAY_SETTINGS: "응답 표시 설정 적용",
        MK.SELECTED_OPTIONS: "선택된 옵션:",
        MK.DISPLAY_SETTINGS_SAVED: "✅ {0}개의 응답 표시 설정이 저장되었습니다. 다음 질문부터 적용됩니다.",
        
        # 탭 및 섹션
        MK.IMPROVED_RESPONSES: "개선된 응답",
        MK.INITIAL_RESPONSES: "초기 응답",
        MK.ANALYSIS: "분석",
        MK.FOLLOW_UP_QUESTIONS: "후속 질문",
        MK.DISPLAY_SETTINGS: "응답 표시 설정",
        
        # 응답 타입 옵션
        MK.DISPLAY_OPTIONS_TYPE: "옵션 표시 방식:",
        MK.DISPLAY_OPTIONS_HELP: "기본 옵션은 간단한 선택을, 상세 옵션은 더 세분화된 선택을 제공합니다",
        MK.BASIC_OPTIONS: "기본 옵션",
        MK.SEE_DETAILED_OPTIONS: "더 상세한 옵션을 보려면 '상세 옵션'을 선택하세요.",
        MK.DETAILED_OPTIONS: "상세 옵션",
        MK.RETURN_TO_BASIC_OPTIONS: "간단한 옵션으로 돌아가려면 '기본 옵션'을 선택하세요.",
        MK.SELECT_RESPONSE_TYPES_HELP: "다음 질문부터 어떤 유형의 개선된 응답을 표시할지 선택하세요. 여러 유형을 동시에 선택할 수 있습니다.",
        MK.BASIC_DISPLAY_OPTIONS: "기본 표시 옵션:",
        MK.CURRENT_SETTINGS: "현재 적용된 설정",
        MK.SELECTED_DISPLAY_TYPES: "선택된 응답 표시 유형: {0}",
        MK.SETTINGS_APPLY_NEXT: "이 설정은 다음 질문부터 적용됩니다.",

        # 응답 표시 제목 관련
        MK.SELF_ANALYSIS_TITLE: "{0} (자체 분석)",
        MK.MULTIPLE_ANALYSIS_CLAUDE: "{0} (GPT+Perplexity 분석)",
        MK.MULTIPLE_ANALYSIS_GPT: "{0} (Claude+Perplexity 분석)",
        MK.MULTIPLE_ANALYSIS_PERPLEXITY: "{0} (Claude+GPT 분석)",
        MK.MULTIPLE_ANALYSIS_GENERIC: "{0} (다중 AI 분석)",
        MK.EXTERNAL_ANALYSIS_TITLE: "{0} ({1} 분석)",
        MK.IMPROVED_RESPONSE_SUFFIX: " 개선된 응답",

        # 버튼 및 액션
        MK.SAVE_RESULTS: "결과 저장",
        MK.RESULTS_SAVED: "결과가 저장되었습니다: {0}",
        MK.CONTINUE_WITH: "{0}로 계속하기",
        MK.SELECT_TO_CONTINUE: "특정 AI의 응답을 선택하여 대화를 계속할 수 있습니다.",
        
        # 프롬프트 관련
        MK.ANALYSIS_PROMPT_PERPLEXITY: """
        사용자 질문과 Perplexity의 응답을 분석해 주세요.
        
        사용자 질문: {0}
        
        Perplexity 응답: {1}
        
        다음 3가지 항목에 대해 분석해주세요:
        1. 응답에서 더 자세히 설명하면 좋을 부분
        2. Perplexity가 놓친 중요한 정보나 관점
        3. 사용자에게 더 도움이 될 만한 질문이나 제안
        
        반드시 아래 JSON 형식만 사용해서 응답해주세요. 다른 설명이나 추가 텍스트 없이 JSON만 반환하세요.
        
        {{
        "improvements": ["개선점1", "개선점2", "개선점3"],
        "missing_information": ["놓친 정보1", "놓친 정보2", "놓친 정보3"],
        "follow_up_questions": ["후속 질문1", "후속 질문2", "후속 질문3"]
        }}
        """,
        
        MK.ANALYSIS_PROMPT_GENERAL: """
        아래는 사용자 질문과 {0}의 응답입니다:

        사용자: {1}

        {0}: {2}

        이 응답을 분석해주세요:
        1. 이 응답에서 더 자세히 설명하면 좋을 부분이 있나요?
        2. {0}가 놓친 중요한 정보나 관점이 있나요?
        3. 사용자에게 더 도움이 될 만한 질문이나 제안은 무엇인가요?

        {0}의 응답을 어떻게 개선할 수 있을지 **정확히 다음 JSON 형식으로만** 제안해주세요:

        ```json
        {{
        "improvements": ["개선점1", "개선점2"],
        "missing_information": ["놓친 정보1", "놓친 정보2"],
        "follow_up_questions": ["후속 질문1", "후속 질문2"]
        }}
        ```
        """,
        
        MK.IMPROVEMENT_PROMPT: """
        당신의 이전 응답: "{0}"
        
        이 응답을 다음과 같이 개선해주세요 ({1}의 분석 기반):
        
        개선할 점: {2}
        추가할 정보: {3}
        
        사용자가 관심을 가질 만한 후속 질문도 1-2개 자연스럽게 포함해주세요.
        가능한 후속 질문: {4}
        
        더 자연스럽고, 사려깊으며, 도움이 되는 응답을 작성해주세요.
        """,
        
        MK.PERPLEXITY_SYSTEM_MESSAGE: "당신은 AI 응답을 분석하고 개선하는 전문가입니다.",
        MK.NONE: "없음",
        MK.GENERATE_APPROPRIATE_QUESTIONS: "적절한 후속 질문을 생성해주세요",

        # 분석 관련
        MK.SELECT_RESPONSE_TYPES_MULTIPLE_HELP: "어떤 종류의 개선된 응답을 보고 싶은지 선택하세요 (여러 개 선택 가능)",
        MK.APPLY_MODEL_SETTINGS_HELP: "선택한 모델 설정을 적용합니다",
        MK.ERROR_TEXT: "오류",
        MK.UNKNOWN_ERROR_TEXT: "알 수 없는 오류",
        MK.ANALYSIS_LABEL: "분석",        
        MK.SELF_ANALYSIS_LABEL: "자체 분석",
        MK.ANALYSIS_RESULTS: "분석 결과",
        MK.ANALYSIS_INFO: "분석 결과에는 두 가지 유형이 있습니다:\n1. **자체 분석**: 각 AI가 자신의 응답을 분석한 결과\n2. **외부 분석**: 다른 AI가 다른 AI의 응답을 분석한 결과\n\n현재 선택된 표시 유형에서:\n- 응답 생성 모델: {0}\n- 분석 전용 모델: {1}\n- 분석 작업: {2}",
        MK.RESPONSE_TYPE_SELF_ANALYSIS: "응답 타입 (1/4): 자체 분석 옵션",
        MK.RESPONSE_TYPE_EXTERNAL_ANALYSIS: "응답 타입 (2/4): 외부 분석 옵션",
        MK.RESPONSE_TYPE_MULTIPLE_ANALYZERS: "응답 타입 (3/4): 다중 분석자 옵션",
        MK.RESPONSE_TYPE_INITIAL_RESPONSE: "응답 타입 (4/4): 초기 응답 옵션",
        MK.IMPROVEMENT_SUGGESTIONS: "개선 제안",
        MK.MISSING_INFORMATION: "누락된 정보",
        MK.FOLLOW_UP_SUGGESTIONS: "후속 질문 제안",
        MK.SELECT_ANALYSIS_HELP: "아래에서 표시할 분석을 선택하세요. 분석 항목을 숨기더라도 데이터는 유지됩니다.",
        MK.SELECT_ANALYSIS: "표시할 분석 선택",
        MK.SHOW_ALL_ANALYSES: "모든 분석 표시",
        MK.NO_ANALYSIS_SELECTED: "선택된 분석이 없습니다. '모든 분석 표시' 버튼을 클릭하여 모든 분석을 복원할 수 있습니다.",
        MK.RESTORE_ALL_ANALYSES: "모든 분석 복원",
        MK.SELF_ANALYSIS_RESULTS: "자체 분석 결과",
        MK.EXTERNAL_ANALYSIS_RESULTS: "외부 분석 결과",
        MK.NO_IMPROVEMENT_SUGGESTIONS: "개선 제안이 없습니다.",
        MK.NO_MISSING_INFORMATION: "누락된 정보가 없습니다.",
        MK.NO_FOLLOW_UP_SUGGESTIONS: "후속 질문 제안이 없습니다.",
        MK.SELF_ANALYSIS_DESCRIPTION: "{0}의 자체 분석",
        MK.EXTERNAL_ANALYSIS_DESCRIPTION: "{0}에 대한 {1}의 분석",

        # 질문 관련
        MK.EXTRACTED_QUESTIONS_LABEL: "추출 질문",
        MK.SUGGESTED_QUESTIONS_LABEL: "제안 질문",
        MK.NO_FOLLOW_UP_QUESTIONS: "후속 질문이 없습니다.",

        # 고급 설정
        MK.ADVANCED_SETTINGS: "고급 설정",
        MK.ADVANCED_SETTINGS_WARNING: "고급 설정은 개발자 또는 문제 해결이 필요한 경우에만 사용하세요.",
        MK.API_SERVICES: "API 서비스",
        MK.MIDDLEWARE_INIT: "미들웨어 초기화",
        MK.CHECK_API_STATUS: "API 상태 확인",
        
        # 처리 정보
        MK.PROCESSING_TIME: "처리 시간: {0}초",
        MK.TOTAL_COST_VALUE: "총 비용: ${0}",
        MK.DISPLAY_TYPES_USED_COUNT: "사용된 응답 표시 유형 ({0}개):",
        MK.DISPLAY_TYPES_USED: "사용된 응답 표시 유형: {0}",
        MK.ANALYSIS_TASKS_PERFORMED: "수행된 분석 작업 ({0}개):",

        # 오류 및 경고
        MK.ERROR_OCCURRED: "오류 발생: {0}",
        MK.NO_RESPONSES: "선택한 응답 표시 유형에 맞는 {0}이 없습니다.", # {0} = 응답 유형 (개선된 응답, 초기 응답답 등)
        MK.MIN_ONE_SELECTION: "최소 하나 이상의 유형을 선택해야 합니다.",
        MK.UNKNOWN: "알 수 없음",

        # 미들웨어 오류 메시지
        MK.UNSUPPORTED_MODEL_ERROR: "지원하지 않는 모델: {0}",
        MK.PERPLEXITY_CLIENT_NOT_INITIALIZED: "Perplexity 클라이언트가 초기화되지 않았습니다. API 키를 확인하세요.",
        MK.MODEL_UNAVAILABLE: "{0} 모델을 사용할 수 없습니다.",
        MK.NO_VALID_RESPONSE: "{0} 모델의 유효한 응답이 없습니다.",
        MK.NO_RESPONSE_FROM_ALL_MODELS: "선택된 모든 AI 모델에서 응답을 생성하지 못했습니다.",
        MK.BUDGET_LIMIT_REACHED: "예산 제한(${0:.2f})에 도달했습니다. 현재 비용: ${1:.2f}",
        MK.JSON_PARSING_FAILED_WITH_DETAILS: "JSON 파싱 실패: {0}",
        MK.JSON_PARSING_FAILED: "JSON 파싱 실패",
        MK.JSON_FORMAT_NOT_FOUND: "JSON 형식 찾기 실패",

        # 상태 메시지
        MK.PROCESSING: "처리 중... 잠시만 기다려주세요.",
        MK.MIDDLEWARE_INITIALIZED: "미들웨어가 자동으로 초기화되었습니다.",
        
        # 후속 질문 섹션
        MK.EXTRACTED_QUESTIONS: "개선된 응답에서 추출된 질문",
        MK.SUGGESTED_QUESTIONS: "제안된 질문",
        MK.NO_EXTRACTED_QUESTIONS: "추출된 질문이 없습니다.",
        MK.NO_SUGGESTED_QUESTIONS: "제안된 질문이 없습니다.",
        
        # 모델 설정
        MK.MODEL_SETTINGS: "모델 설정",
        MK.CLAUDE_MODEL: "Claude 모델",
        MK.GPT_MODEL: "GPT 모델",
        MK.PERPLEXITY_MODEL: "Perplexity 모델",
        MK.APPLY_MODEL_SETTINGS: "모델 설정 적용",
        
        # 데이터 및 설정
        MK.DATA_SETTINGS: "데이터 및 설정",
        MK.RESET_TOKEN_USAGE: "토큰 사용량 리셋",
        MK.RESET_SESSION: "세션 초기화",
        MK.SESSION_RESET: "세션이 초기화되었습니다.",
        MK.TOKEN_USAGE_RESET: "토큰 사용량이 초기화되었습니다.",
        
        # 시스템 정보
        MK.SYSTEM_INFO: "시스템 정보",
        MK.APP_START_TIME: "앱 시작 시간: {0}",
        MK.MIDDLEWARE_STATUS: "현재 미들웨어 상태: {0}",
        MK.STREAMLIT_VERSION: "Streamlit 버전: {0}",
        
        # 응답 개선 유형 설명
        MK.ResponseTypes.INITIAL_ONLY: "초기 응답만",
        MK.ResponseTypes.INITIAL_ONLY_DESC: "초기 응답만 표시 (개선 과정 없음)",
        MK.ResponseTypes.CLAUDE_BY_GPT: "Claude를 GPT가 분석",
        MK.ResponseTypes.CLAUDE_BY_GPT_DESC: "Claude를 GPT가 분석하여 개선한 결과 표시",
        MK.ResponseTypes.GPT_BY_CLAUDE: "GPT를 Claude가 분석",
        MK.ResponseTypes.GPT_BY_CLAUDE_DESC: "GPT를 Claude가 분석하여 개선한 결과 표시",
        MK.ResponseTypes.ALL_SELF_ANALYSIS: "모든 자체 분석",
        MK.ResponseTypes.ALL_SELF_ANALYSIS_DESC: "모든 자체 분석하여 개선한 결과 표시",
        MK.ResponseTypes.ALL_COMBINATIONS: "모든 분석 조합",
        MK.ResponseTypes.ALL_COMBINATIONS_DESC: "모든 분석 조합하여 개선한 결과 표시 (가장 많은 처리 시간과 비용 소모)",
        
        # 자세한 응답 유형 설명
        MK.ResponseTypes.CLAUDE_SELF: "Claude의 자체 분석",
        MK.ResponseTypes.CLAUDE_SELF_DESC: "Claude가 자신의 응답을 분석하여 개선한 결과 표시",
        MK.ResponseTypes.GPT_SELF: "GPT의 자체 분석",
        MK.ResponseTypes.GPT_SELF_DESC: "GPT가 자신의 응답을 분석하여 개선한 결과 표시",
        MK.ResponseTypes.PERPLEXITY_SELF: "Perplexity의 자체 분석",
        MK.ResponseTypes.PERPLEXITY_SELF_DESC: "Perplexity가 자신의 응답을 분석하여 개선한 결과 표시",
        
        MK.ResponseTypes.CLAUDE_BY_PERPLEXITY: "Claude를 Perplexity가 분석",
        MK.ResponseTypes.CLAUDE_BY_PERPLEXITY_DESC: "Claude를 Perplexity가 분석하여 개선한 결과 표시",
        MK.ResponseTypes.GPT_BY_PERPLEXITY: "GPT를 Perplexity가 분석",
        MK.ResponseTypes.GPT_BY_PERPLEXITY_DESC: "GPT를 Perplexity가 분석하여 개선한 결과 표시",
        MK.ResponseTypes.PERPLEXITY_BY_CLAUDE: "Perplexity를 Claude가 분석",
        MK.ResponseTypes.PERPLEXITY_BY_CLAUDE_DESC: "Perplexity를 Claude가 분석하여 개선한 결과 표시",
        MK.ResponseTypes.PERPLEXITY_BY_GPT: "Perplexity를 GPT가 분석",
        MK.ResponseTypes.PERPLEXITY_BY_GPT_DESC: "Perplexity를 GPT가 분석하여 개선한 결과 표시",
        
        MK.ResponseTypes.CLAUDE_BY_MULTIPLE: "Claude를 여러 AI가 분석",
        MK.ResponseTypes.CLAUDE_BY_MULTIPLE_DESC: "Claude의 응답을 GPT와 Perplexity가 함께 분석하여 개선한 결과 표시",
        MK.ResponseTypes.GPT_BY_MULTIPLE: "GPT를 여러 AI가 분석",
        MK.ResponseTypes.GPT_BY_MULTIPLE_DESC: "GPT의 응답을 Claude와 Perplexity가 함께 분석하여 개선한 결과 표시",
        MK.ResponseTypes.PERPLEXITY_BY_MULTIPLE: "Perplexity를 여러 AI가 분석",
        MK.ResponseTypes.PERPLEXITY_BY_MULTIPLE_DESC: "Perplexity의 응답을 Claude와 GPT가 함께 분석하여 개선한 결과 표시",
        
        MK.ResponseTypes.CLAUDE_INITIAL: "Claude 초기 응답",
        MK.ResponseTypes.CLAUDE_INITIAL_DESC: "Claude의 초기 응답만 표시 (개선 과정 없음)",
        MK.ResponseTypes.GPT_INITIAL: "GPT 초기 응답",
        MK.ResponseTypes.GPT_INITIAL_DESC: "GPT의 초기 응답만 표시 (개선 과정 없음)",
        MK.ResponseTypes.PERPLEXITY_INITIAL: "Perplexity 초기 응답",
        MK.ResponseTypes.PERPLEXITY_INITIAL_DESC: "Perplexity의 초기 응답만 표시 (개선 과정 없음)",
    },
    "en": {
        # App Title and Basic UI
        MK.APP_TITLE: "Enhanced AI Response System",
        MK.APP_SUBTITLE: "Interactive System for Better AI Responses",
        MK.SETTINGS: "Settings",
        MK.SYSTEM_SETTINGS: "System Settings",
        MK.LANGUAGE: "Language / 언어",
        
        # Middleware Initialization
        MK.CLAUDE_CLIENT_INIT_FAILED: "Claude client initialization failed: {0}",
        MK.OPENAI_CLIENT_INIT_FAILED: "OpenAI client initialization failed: {0}",
        MK.PERPLEXITY_CLIENT_INIT_FAILED: "Perplexity client initialization failed: {0}",
        MK.MIDDLEWARE_INIT_HELP: "Use this when there are API connection issues",
        MK.MIDDLEWARE_INIT_SUCCESS: "Middleware has been successfully initialized.",
        MK.CHECK_API_STATUS_HELP: "Check the connection status of each API service",
        MK.API_STATUS_NORMAL: "Normal",
        MK.API_STATUS_NOT_INITIALIZED: "Not initialized",
        MK.API_STATUS_ERROR: "Error:",
        MK.MIDDLEWARE_NOT_INITIALIZED: "Middleware has not been initialized.",
        MK.MIDDLEWARE_STATUS_INITIALIZED: "Initialized",
        MK.MIDDLEWARE_STATUS_NOT_INITIALIZED: "Not initialized",

        # Settings and Options
        MK.BUDGET_LIMIT: "Budget Limit ($)",
        MK.PERPLEXITY_ENABLED: "Use Perplexity AI",
        MK.RESET_TOKEN_USAGE_HELP: "Reset the token usage record for the current session",
        MK.RESET_SESSION_HELP: "Reset all conversation history and query records",
        MK.CLAUDE_MODEL_HELP: "Select the Claude model to use",
        MK.GPT_MODEL_HELP: "Select the GPT model to use",
        MK.PERPLEXITY_MODEL_HELP: "Select the Perplexity model to use",
        MK.MODEL_SETTINGS_APPLIED: "Model settings applied. Initializing middleware...",

        # Query and Response Related
        MK.USER_LABEL: "User",
        MK.QUERY_NUMBER: "Query #{0}",
        MK.QUESTION_LABEL: "Question:",
        MK.PROCESSING_TIME_LABEL: "Processing Time:",
        MK.COST_LABEL: "Cost:",
        MK.SECONDS_UNIT: "s",
        MK.NEW_QUESTION: "Enter New Question",
        MK.QUESTION_INPUT: "Enter your question:",
        MK.SUBMIT_QUESTION: "Submit Question",
        MK.CONTINUE_CONVERSATION: "Continue Conversation",
        MK.NO_RESPONSE_TO_SELECT: "No response to select",
        MK.MODEL_RESPONSE: "{0} Response",
        MK.RESPONSE_NUMBER: "Response #{0}",
        MK.RESPONSE_LABEL: "Response",        
        MK.INITIAL_RESPONSE_LABEL: "Initial Response",

        # Result Display
        MK.QUESTION: "Question",
        MK.CURRENT_QUERY_RESULTS: "Current Query Results",
        MK.CONVERSATION_HISTORY: "Conversation History",
        MK.PREVIOUS_QUERY_HISTORY: "Previous Query History",
        MK.NO_PREVIOUS_HISTORY: "No previous query history yet.",
        MK.PREVIOUS_QUESTION: "Previous question",
        MK.QUESTION_OR_INSTRUCTION_INPUT: "Enter question or instruction:",

        # Token and Cost
        MK.AI_MODEL_COLUMN: "AI Model",
        MK.INPUT_TOKENS_COLUMN: "Input Tokens",
        MK.OUTPUT_TOKENS_COLUMN: "Output Tokens",
        MK.COST_COLUMN: "Cost",
        MK.TOKEN_USAGE_BY_MODEL_TITLE: "Token Usage by AI Model",
        MK.COST_DISTRIBUTION_TITLE: "Cost Distribution",
        MK.TOKEN_USAGE: "Token Usage",
        MK.TOTAL_COST: "Total Cost",
        MK.NO_TOKENS_USED: "No tokens have been used yet.",
        
        # Response Display Settings
        MK.RESPONSE_DISPLAY_SETTINGS: "Response Display Settings",
        MK.SELECT_RESPONSE_TYPES: "Select Response Display Types (Multiple Choices Allowed):",
        MK.APPLY_DISPLAY_SETTINGS: "Apply Display Settings",
        MK.SELECTED_OPTIONS: "Selected Options:",
        MK.DISPLAY_SETTINGS_SAVED: "✅ {0} response display settings saved. They will be applied to the next question.",
        
        # Tabs and Sections
        MK.IMPROVED_RESPONSES: "Improved Responses",
        MK.INITIAL_RESPONSES: "Initial Responses",
        MK.ANALYSIS: "Analysis",
        MK.FOLLOW_UP_QUESTIONS: "Follow-up Questions",
        MK.DISPLAY_SETTINGS: "Display Settings",
        
        # Response Type Options
        MK.DISPLAY_OPTIONS_TYPE: "Display options mode:",
        MK.DISPLAY_OPTIONS_HELP: "Basic options provide simple choices, while detailed options offer more fine-grained settings",
        MK.BASIC_OPTIONS: "Basic Options",
        MK.SEE_DETAILED_OPTIONS: "Select 'Detailed Options' to see more advanced settings.",
        MK.DETAILED_OPTIONS: "Detailed Options",
        MK.RETURN_TO_BASIC_OPTIONS: "Select 'Basic Options' to return to simplified options.",
        MK.SELECT_RESPONSE_TYPES_HELP: "Select which types of improved responses to display for the next question. Multiple types can be selected.",
        MK.BASIC_DISPLAY_OPTIONS: "Basic display options:",
        MK.CURRENT_SETTINGS: "Current Applied Settings",
        MK.SELECTED_DISPLAY_TYPES: "Selected response display types: {0}",
        MK.SETTINGS_APPLY_NEXT: "These settings will be applied to the next question.",

        # 응답 표시 제목 관련
        MK.SELF_ANALYSIS_TITLE: "{0} (Self-Analysis)",
        MK.MULTIPLE_ANALYSIS_CLAUDE: "{0} (GPT+Perplexity Analysis)",
        MK.MULTIPLE_ANALYSIS_GPT: "{0} (Claude+Perplexity Analysis)",
        MK.MULTIPLE_ANALYSIS_PERPLEXITY: "{0} (Claude+GPT Analysis)",
        MK.MULTIPLE_ANALYSIS_GENERIC: "{0} (Multiple AI Analysis)",
        MK.EXTERNAL_ANALYSIS_TITLE: "{0} ({1} Analysis)",
        MK.IMPROVED_RESPONSE_SUFFIX: " Improved Response",

        # Buttons and Actions
        MK.SAVE_RESULTS: "Save Results",
        MK.RESULTS_SAVED: "Results saved: {0}",
        MK.CONTINUE_WITH: "Continue with {0}",
        MK.SELECT_TO_CONTINUE: "Select a specific AI's response to continue the conversation.",
        
        # Prompt Related
        MK.ANALYSIS_PROMPT_PERPLEXITY: """
        Please analyze the user question and Perplexity's response.
        
        User question: {0}
        
        Perplexity response: {1}
        
        Please analyze these 3 aspects:
        1. Parts of the response that could use more detailed explanation
        2. Important information or perspectives missing from Perplexity's response
        3. Follow-up questions or suggestions that could help the user
        
        Please respond ONLY in the JSON format below. Do not include any other text or explanation:
        
        {{
        "improvements": ["improvement 1", "improvement 2", "improvement 3"],
        "missing_information": ["missing info 1", "missing info 2", "missing info 3"],
        "follow_up_questions": ["follow-up question 1", "follow-up question 2", "follow-up question 3"]
        }}
        """,
        
        MK.ANALYSIS_PROMPT_GENERAL: """
        Below is a user question and {0}'s response:
        
        User: {1}
        
        {0}: {2}
        
        Please analyze this response:
        1. Are there parts that could benefit from more detailed explanation?
        2. Is there any important information or perspective that {0} missed?
        3. What follow-up questions or suggestions might be helpful to the user?
        
        Please suggest how to improve {0}'s response using **exactly the following JSON format only**:
        
        ```json
        {{
        "improvements": ["improvement 1", "improvement 2"],
        "missing_information": ["missing info 1", "missing info 2"],
        "follow_up_questions": ["follow-up question 1", "follow-up question 2"]
        }}
        ```
        """,
        
        MK.IMPROVEMENT_PROMPT: """
        Your previous response: "{0}"
        
        Please improve this response based on {1}'s analysis:
        
        Points to improve: {2}
        Information to add: {3}
        
        Please naturally include 1-2 follow-up questions that might interest the user.
        Possible follow-up questions: {4}
        
        Please write a more natural, thoughtful, and helpful response.
        """,
        
        MK.PERPLEXITY_SYSTEM_MESSAGE: "You are an expert at analyzing and improving AI responses.",
        MK.NONE: "none",
        MK.GENERATE_APPROPRIATE_QUESTIONS: "please generate appropriate follow-up questions",

        # Analysis Related
        MK.SELECT_RESPONSE_TYPES_MULTIPLE_HELP: "Select which types of improved responses you want to see (multiple selections allowed)",
        MK.APPLY_MODEL_SETTINGS_HELP: "Apply the selected model settings",
        MK.ERROR_TEXT: "error",
        MK.UNKNOWN_ERROR_TEXT: "unknown error",
        MK.ANALYSIS_LABEL: "Analysis",
        MK.SELF_ANALYSIS_LABEL: "Self-Analysis",
        MK.ANALYSIS_RESULTS: "Analysis Results",
        MK.ANALYSIS_INFO: "Analysis results include two types:\n1. **Self-Analysis**: Each AI's analysis of its own response\n2. **External Analysis**: Analysis of one AI's response by another AI\n\nIn the currently selected display type:\n- Response generation models: {0}\n- Analysis-only models: {1}\n- Analysis tasks: {2}",
        MK.RESPONSE_TYPE_SELF_ANALYSIS: "Response Type (1/4): Self-Analysis Options",
        MK.RESPONSE_TYPE_EXTERNAL_ANALYSIS: "Response Type (2/4): External Analysis Options",
        MK.RESPONSE_TYPE_MULTIPLE_ANALYZERS: "Response Type (3/4): Multiple Analyzer Options",
        MK.RESPONSE_TYPE_INITIAL_RESPONSE: "Response Type (4/4): Initial Response Options",
        MK.IMPROVEMENT_SUGGESTIONS: "Improvement Suggestions",
        MK.MISSING_INFORMATION: "Missing Information",
        MK.FOLLOW_UP_SUGGESTIONS: "Follow-up Question Suggestions",
        MK.SELECT_ANALYSIS_HELP: "Select analyses to display below. Data is preserved even when analysis items are hidden.",
        MK.SELECT_ANALYSIS: "Select Analyses to Display",
        MK.SHOW_ALL_ANALYSES: "Show All Analyses",
        MK.NO_ANALYSIS_SELECTED: "No analyses selected. Click the 'Show All Analyses' button to restore all analyses.",
        MK.RESTORE_ALL_ANALYSES: "Restore All Analyses",
        MK.SELF_ANALYSIS_RESULTS: "Self-Analysis Results",
        MK.EXTERNAL_ANALYSIS_RESULTS: "External Analysis Results",
        MK.NO_IMPROVEMENT_SUGGESTIONS: "No improvement suggestions.",
        MK.NO_MISSING_INFORMATION: "No missing information.",
        MK.NO_FOLLOW_UP_SUGGESTIONS: "No follow-up question suggestions.",
        MK.SELF_ANALYSIS_DESCRIPTION: "{0}'s Self-Analysis",
        MK.EXTERNAL_ANALYSIS_DESCRIPTION: "{1}'s Analysis of {0}",

        # Question Related
        MK.EXTRACTED_QUESTIONS_LABEL: "Extracted Questions",
        MK.SUGGESTED_QUESTIONS_LABEL: "Suggested Questions",
        MK.NO_FOLLOW_UP_QUESTIONS: "No follow-up questions."
,
        # Advanced Settings
        MK.ADVANCED_SETTINGS: "Advanced Settings",
        MK.ADVANCED_SETTINGS_WARNING: "Advanced settings should only be used by developers or for troubleshooting.",
        MK.API_SERVICES: "API Services",
        MK.MIDDLEWARE_INIT: "Initialize Middleware",
        MK.CHECK_API_STATUS: "Check API Status",
        
        # Processing Information
        MK.PROCESSING_TIME: "Processing Time: {0} seconds",
        MK.TOTAL_COST_VALUE: "Total cost: ${0}",
        MK.DISPLAY_TYPES_USED_COUNT: "Response display types used ({0}):",
        MK.DISPLAY_TYPES_USED: "Response display types used: {0}",
        MK.ANALYSIS_TASKS_PERFORMED: "Analysis tasks performed ({0}):",

        # Errors and Warnings
        MK.ERROR_OCCURRED: "Error Occurred: {0}",
        MK.NO_RESPONSES: "No {0} matching your selected display types.", # {0} = 응답 유형 (improved_responses, initial_responses 등)
        MK.MIN_ONE_SELECTION: "You must select at least one type.",
        MK.UNKNOWN: "Unknown",

        # 미들웨어 오류 메시지
        MK.UNSUPPORTED_MODEL_ERROR: "Unsupported model: {0}",
        MK.PERPLEXITY_CLIENT_NOT_INITIALIZED: "Perplexity client is not initialized. Please check your API key.",
        MK.MODEL_UNAVAILABLE: "Model {0} is unavailable.",
        MK.NO_VALID_RESPONSE: "No valid response from model {0}.",
        MK.NO_RESPONSE_FROM_ALL_MODELS: "Failed to generate response from any of the selected AI models.",
        MK.BUDGET_LIMIT_REACHED: "Budget limit (${0:.2f}) reached. Current cost: ${1:.2f}",
        MK.JSON_PARSING_FAILED_WITH_DETAILS: "JSON parsing failed: {0}",
        MK.JSON_PARSING_FAILED: "JSON parsing failed",
        MK.JSON_FORMAT_NOT_FOUND: "Failed to find JSON format",

        # Status Messages
        MK.PROCESSING: "Processing... Please wait.",
        MK.MIDDLEWARE_INITIALIZED: "Middleware has been automatically initialized.",
        
        # Follow-up Questions Section
        MK.EXTRACTED_QUESTIONS: "Questions Extracted from Improved Responses",
        MK.SUGGESTED_QUESTIONS: "Suggested Questions",
        MK.NO_EXTRACTED_QUESTIONS: "No extracted questions.",
        MK.NO_SUGGESTED_QUESTIONS: "No suggested questions.",
        
        # Model Settings
        MK.MODEL_SETTINGS: "Model Settings",
        MK.CLAUDE_MODEL: "Claude Model",
        MK.GPT_MODEL: "GPT Model",
        MK.PERPLEXITY_MODEL: "Perplexity Model",
        MK.APPLY_MODEL_SETTINGS: "Apply Model Settings",
        
        # Data and Settings
        MK.DATA_SETTINGS: "Data and Settings",
        MK.RESET_TOKEN_USAGE: "Reset Token Usage",
        MK.RESET_SESSION: "Reset Session",
        MK.SESSION_RESET: "Session has been reset.",
        MK.TOKEN_USAGE_RESET: "Token usage has been reset.",
        
        # System Information
        MK.SYSTEM_INFO: "System Information",
        MK.APP_START_TIME: "App Start Time: {0}",
        MK.MIDDLEWARE_STATUS: "Current Middleware Status: {0}",
        MK.STREAMLIT_VERSION: "Streamlit Version: {0}",
        
        # Response Improvement Type Descriptions
        MK.ResponseTypes.INITIAL_ONLY: "Initial Responses Only",
        MK.ResponseTypes.INITIAL_ONLY_DESC: "Display only initial responses from all AIs (without improvement process)",
        MK.ResponseTypes.CLAUDE_BY_GPT: "Claude Analyzed by GPT",
        MK.ResponseTypes.CLAUDE_BY_GPT_DESC: "Display results where GPT analyzes and improves Claude's response",
        MK.ResponseTypes.GPT_BY_CLAUDE: "GPT Analyzed by Claude",
        MK.ResponseTypes.GPT_BY_CLAUDE_DESC:  "Display results where Claude analyzes and improves GPT's response",
        MK.ResponseTypes.ALL_SELF_ANALYSIS: "All Self-Analysis",
        MK.ResponseTypes.ALL_SELF_ANALYSIS_DESC: "Display results where all AIs analyze and improve their own responses",
        MK.ResponseTypes.ALL_COMBINATIONS: "All Analysis Combinations",
        MK.ResponseTypes.ALL_COMBINATIONS_DESC: "Display all types of analysis and improved responses (requires the most processing time and cost)",
        
        # Detailed Response Type Descriptions
        MK.ResponseTypes.CLAUDE_SELF: "Claude's Self-Analysis",
        MK.ResponseTypes.CLAUDE_SELF_DESC: "Display results where Claude analyzes and improves its own response",
        MK.ResponseTypes.GPT_SELF: "GPT's Self-Analysis",
        MK.ResponseTypes.GPT_SELF_DESC: "Display results where GPT analyzes and improves its own response",
        MK.ResponseTypes.PERPLEXITY_SELF: "Perplexity's Self-Analysis",
        MK.ResponseTypes.PERPLEXITY_SELF_DESC: "Display results where Perplexity analyzes and improves its own response",
        
        MK.ResponseTypes.CLAUDE_BY_PERPLEXITY: "Claude Analyzed by Perplexity",
        MK.ResponseTypes.CLAUDE_BY_PERPLEXITY_DESC: "Display results where Perplexity analyzes and improves Claude's response",
        MK.ResponseTypes.GPT_BY_PERPLEXITY: "GPT Analyzed by Perplexity",
        MK.ResponseTypes.GPT_BY_PERPLEXITY_DESC: "Display results where GPT analyzes and improves Perplexity's response",
        MK.ResponseTypes.PERPLEXITY_BY_CLAUDE: "Perplexity Analyzed by Claude",
        MK.ResponseTypes.PERPLEXITY_BY_CLAUDE_DESC: "Display results where Claude analyzes and improves Perplexity's response",
        MK.ResponseTypes.PERPLEXITY_BY_GPT: "Perplexity Analyzed by GPT",
        MK.ResponseTypes.PERPLEXITY_BY_GPT_DESC: "Display results where GPT analyzes and improves Perplexity's response",
        
        MK.ResponseTypes.CLAUDE_BY_MULTIPLE: "Claude Analyzed by Multiple AIs",
        MK.ResponseTypes.CLAUDE_BY_MULTIPLE_DESC: "Display results where GPT and Perplexity jointly analyze and improve Claude's response",
        MK.ResponseTypes.GPT_BY_MULTIPLE: "GPT Analyzed by Multiple AIs",
        MK.ResponseTypes.GPT_BY_MULTIPLE_DESC: "Display results where Claude and Perplexity jointly analyze and improve GPT's response",
        MK.ResponseTypes.PERPLEXITY_BY_MULTIPLE: "Perplexity Analyzed by Multiple AIs",
        MK.ResponseTypes.PERPLEXITY_BY_MULTIPLE_DESC: "Display results where Claude and GPT jointly analyze and improve Perplexity's response",
        
        MK.ResponseTypes.CLAUDE_INITIAL: "Claude Initial Response",
        MK.ResponseTypes.CLAUDE_INITIAL_DESC: "Display only Claude's initial response (without improvement process)",
        MK.ResponseTypes.GPT_INITIAL: "GPT Initial Response",
        MK.ResponseTypes.GPT_INITIAL_DESC: "Display only GPT's initial response (without improvement process)",
        MK.ResponseTypes.PERPLEXITY_INITIAL: "Perplexity Initial Response",
        MK.ResponseTypes.PERPLEXITY_INITIAL_DESC: "Display only Perplexity's initial response (without improvement process)",
    }
}

def get_message(key, lang="ko", *args):
    """
    지정된 키의 메시지를 현재 언어로 가져옵니다.
    포맷 인자가 제공되면 메시지에 적용합니다.
    
    Args:
        key: 메시지 키
        lang: 언어 코드 (기본값: "ko")
        *args: 포맷팅에 사용할 인자들
        
    Returns:
        str: 번역된 메시지
    """
    # 지원하지 않는 언어인 경우 한국어로 폴백
    if lang not in MESSAGES:
        lang = "ko"
    
    # 메시지가 없으면 키 자체를 반환
    message = MESSAGES[lang].get(key, key)
    
    # 포맷 인자가 있으면 적용
    if args:
        try:
            message = message.format(*args)
        except Exception as e:
            # 포맷팅 오류 시 원본 메시지 반환
            print(f"Message formatting error for key '{key}': {e}")
    
    return message