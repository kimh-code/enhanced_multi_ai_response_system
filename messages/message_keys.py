"""
메시지 키 상수 정의
모든 UI 텍스트의 키를 상수로 정의하여 코드의 일관성과 유지보수성 향상
"""

class MK:  # MessageKeys의 약자
    # 앱 제목 및 기본 UI
    APP_TITLE = "app_title"
    APP_SUBTITLE = "app_subtitle"
    SETTINGS = "settings"
    SYSTEM_SETTINGS = "system_settings"
    LANGUAGE = "language"
    
    # 미들웨어 초기화
    CLAUDE_CLIENT_INIT_FAILED = "claude_client_init_failed"
    OPENAI_CLIENT_INIT_FAILED = "openai_client_init_failed"
    PERPLEXITY_CLIENT_INIT_FAILED = "perplexity_client_init_failed"
    MIDDLEWARE_INIT_HELP = "middleware_init_help"
    MIDDLEWARE_INIT_SUCCESS = "middleware_init_success"
    CHECK_API_STATUS_HELP = "check_api_status_help"
    API_STATUS_NORMAL = "api_status_normal"
    API_STATUS_NOT_INITIALIZED = "api_status_not_initialized"
    API_STATUS_ERROR = "api_status_error"
    MIDDLEWARE_NOT_INITIALIZED = "middleware_not_initialized"
    MIDDLEWARE_STATUS_INITIALIZED = "middleware_status_initialized"
    MIDDLEWARE_STATUS_NOT_INITIALIZED = "middleware_status_not_initialized"

    # 설정 및 옵션
    BUDGET_LIMIT = "budget_limit"
    PERPLEXITY_ENABLED = "perplexity_enabled"
    RESET_TOKEN_USAGE_HELP = "reset_token_usage_help"
    RESET_SESSION_HELP = "reset_session_help"
    CLAUDE_MODEL_HELP = "claude_model_help"
    GPT_MODEL_HELP = "gpt_model_help"
    PERPLEXITY_MODEL_HELP = "perplexity_model_help"
    MODEL_SETTINGS_APPLIED = "model_settings_applied"

    # 쿼리 및 응답 관련
    USER_LABEL = "user_label"
    QUERY_NUMBER = "query_number"
    QUESTION_LABEL = "question_label"
    PROCESSING_TIME_LABEL = "processing_time_label"
    COST_LABEL = "cost_label"
    SECONDS_UNIT = "seconds_unit"
    NEW_QUESTION = "new_question"
    QUESTION_INPUT = "question_input"
    SUBMIT_QUESTION = "submit_question"
    CONTINUE_CONVERSATION = "continue_conversation"
    NO_RESPONSE_TO_SELECT = "no_response_to_select"
    MODEL_RESPONSE = "model_response"
    RESPONSE_NUMBER = "response_number"
    RESPONSE_LABEL = "response_label"
    INITIAL_RESPONSE_LABEL = "initial_response_label"

    # 결과 표시
    QUESTION = "question"
    CURRENT_QUERY_RESULTS = "current_query_results"
    CONVERSATION_HISTORY = "conversation_history"
    PREVIOUS_QUERY_HISTORY = "previous_query_history"
    NO_PREVIOUS_HISTORY = "no_previous_history"
    PREVIOUS_QUESTION = "previous_question"
    QUESTION_OR_INSTRUCTION_INPUT = "question_or_instruction_input"

    # 토큰 및 비용
    AI_MODEL_COLUMN = "ai_model_column"
    INPUT_TOKENS_COLUMN = "input_tokens_column"
    OUTPUT_TOKENS_COLUMN = "output_tokens_column" 
    COST_COLUMN = "cost_column"
    TOKEN_USAGE_BY_MODEL_TITLE = "token_usage_by_model_title"
    COST_DISTRIBUTION_TITLE = "cost_distribution_title"
    TOKEN_USAGE = "token_usage"
    TOTAL_COST = "total_cost"
    NO_TOKENS_USED = "no_tokens_used"
    
    # 응답 표시 설정
    RESPONSE_DISPLAY_SETTINGS = "response_display_settings"
    SELECT_RESPONSE_TYPES = "select_response_types"
    APPLY_DISPLAY_SETTINGS = "apply_display_settings"
    SELECTED_OPTIONS = "selected_options"
    DISPLAY_SETTINGS_SAVED = "display_settings_saved"
    
    # 탭 및 섹션
    IMPROVED_RESPONSES = "improved_responses"
    INITIAL_RESPONSES = "initial_responses"
    ANALYSIS = "analysis"
    FOLLOW_UP_QUESTIONS = "follow_up_questions"
    DISPLAY_SETTINGS = "display_settings"
    
    # 응답 타입 옵션
    DISPLAY_OPTIONS_TYPE = "display_options_type"
    DISPLAY_OPTIONS_HELP = "display_options_help"
    BASIC_OPTIONS = "basic_options"
    SEE_DETAILED_OPTIONS = "see_detailed_options"
    DETAILED_OPTIONS = "detailed_options"
    RETURN_TO_BASIC_OPTIONS = "return_to_basic_options"
    SELECT_RESPONSE_TYPES_HELP = "select_response_types_help"
    BASIC_DISPLAY_OPTIONS = "basic_display_options"
    CURRENT_SETTINGS = "current_settings"
    SELECTED_DISPLAY_TYPES = "selected_display_types"
    SETTINGS_APPLY_NEXT = "settings_apply_next"

    # 응답 표시 제목 관련
    SELF_ANALYSIS_TITLE = "self_analysis_title"
    MULTIPLE_ANALYSIS_CLAUDE = "multiple_analysis_claude"
    MULTIPLE_ANALYSIS_GPT = "multiple_analysis_gpt"
    MULTIPLE_ANALYSIS_PERPLEXITY = "multiple_analysis_perplexity"
    MULTIPLE_ANALYSIS_GENERIC = "multiple_analysis_generic"
    EXTERNAL_ANALYSIS_TITLE = "external_analysis_title"
    IMPROVED_RESPONSE_SUFFIX = "improved_response_suffix"

    # 버튼 및 액션
    SAVE_RESULTS = "save_results"
    RESULTS_SAVED = "results_saved"
    CONTINUE_WITH = "continue_with"
    SELECT_TO_CONTINUE = "select_to_continue"
    
    # 프롬롬프트 관련
    ANALYSIS_PROMPT_PERPLEXITY = "analysis_prompt_perplexity"
    ANALYSIS_PROMPT_GENERAL = "analysis_prompt_general"
    IMPROVEMENT_PROMPT = "improvement_prompt"
    PERPLEXITY_SYSTEM_MESSAGE = "perplexity_system_message"
    NONE = "none"
    GENERATE_APPROPRIATE_QUESTIONS = "generate_appropriate_questions"

    # 분석 관련
    SELECT_RESPONSE_TYPES_MULTIPLE_HELP = "select_response_types_multiple_help"
    APPLY_MODEL_SETTINGS_HELP = "apply_model_settings_help"
    ERROR_TEXT = "error_text"
    UNKNOWN_ERROR_TEXT = "unknown_error_text"  # 추가된 속성
    ANALYSIS_LABEL = "analysis_label"
    SELF_ANALYSIS_LABEL = "self_analysis_label"
    ANALYSIS_RESULTS = "analysis_results"
    ANALYSIS_INFO = "analysis_info"
    RESPONSE_TYPE_SELF_ANALYSIS = "response_type_self_analysis"
    RESPONSE_TYPE_EXTERNAL_ANALYSIS = "response_type_external_analysis"
    RESPONSE_TYPE_MULTIPLE_ANALYZERS = "response_type_multiple_analyzers"
    RESPONSE_TYPE_INITIAL_RESPONSE = "response_type_initial_response"
    IMPROVEMENT_SUGGESTIONS = "improvement_suggestions"
    MISSING_INFORMATION = "missing_information"
    FOLLOW_UP_SUGGESTIONS = "follow_up_suggestions"
    SELECT_ANALYSIS_HELP = "select_analysis_help"
    SELECT_ANALYSIS = "select_analysis"
    SHOW_ALL_ANALYSES = "show_all_analyses"
    NO_ANALYSIS_SELECTED = "no_analysis_selected"
    RESTORE_ALL_ANALYSES = "restore_all_analyses"
    SELF_ANALYSIS_RESULTS = "self_analysis_results"
    EXTERNAL_ANALYSIS_RESULTS = "external_analysis_results"
    NO_IMPROVEMENT_SUGGESTIONS = "no_improvement_suggestions"
    NO_MISSING_INFORMATION = "no_missing_information"
    NO_FOLLOW_UP_SUGGESTIONS = "no_follow_up_suggestions"
    SELF_ANALYSIS_DESCRIPTION = "self_analysis_description"
    EXTERNAL_ANALYSIS_DESCRIPTION = "external_analysis_description"

    # 질문 관련
    EXTRACTED_QUESTIONS_LABEL = "extracted_questions_label"
    SUGGESTED_QUESTIONS_LABEL = "suggested_questions_label"

    # 고급 설정
    ADVANCED_SETTINGS = "advanced_settings"
    ADVANCED_SETTINGS_WARNING = "advanced_settings_warning"
    API_SERVICES = "api_services"
    MIDDLEWARE_INIT = "middleware_init"
    CHECK_API_STATUS = "check_api_status"
    
    # 처리 정보
    PROCESSING_TIME = "processing_time"
    TOTAL_COST_VALUE = "total_cost_value"
    DISPLAY_TYPES_USED_COUNT = "display_types_used_count"
    DISPLAY_TYPES_USED = "display_types_used"
    ANALYSIS_TASKS_PERFORMED = "analysis_tasks_performed"

    # 오류 및 경고
    ERROR_OCCURRED = "error_occurred"
    NO_RESPONSES = "no_responses"
    MIN_ONE_SELECTION = "min_one_selection"
    UNKNOWN = "unknown"
    NO_FOLLOW_UP_QUESTIONS = "no_follow_up_questions"
    
    # 미들웨어 오류 메시지
    UNSUPPORTED_MODEL_ERROR = "unsupported_model_error"
    PERPLEXITY_CLIENT_NOT_INITIALIZED = "perplexity_client_not_initialized"
    MODEL_UNAVAILABLE = "model_unavailable"
    NO_VALID_RESPONSE = "no_valid_response"
    NO_RESPONSE_FROM_ALL_MODELS = "no_response_from_all_models"
    BUDGET_LIMIT_REACHED = "budget_limit_reached"
    JSON_PARSING_FAILED_WITH_DETAILS = "json_parsing_failed_with_details"
    JSON_PARSING_FAILED = "json_parsing_failed"
    JSON_FORMAT_NOT_FOUND = "json_format_not_found"

    # 상태 메시지
    PROCESSING = "processing"
    MIDDLEWARE_INITIALIZED = "middleware_initialized"
    
    # 후속 질문 섹션
    EXTRACTED_QUESTIONS = "extracted_questions"
    SUGGESTED_QUESTIONS = "suggested_questions"
    NO_EXTRACTED_QUESTIONS = "no_extracted_questions"
    NO_SUGGESTED_QUESTIONS = "no_suggested_questions"
    
    # 모델 설정
    MODEL_SETTINGS = "model_settings"
    CLAUDE_MODEL = "claude_model"
    GPT_MODEL = "gpt_model"
    PERPLEXITY_MODEL = "perplexity_model"
    APPLY_MODEL_SETTINGS = "apply_model_settings"
    
    # 데이터 및 설정
    DATA_SETTINGS = "data_settings"
    RESET_TOKEN_USAGE = "reset_token_usage"
    RESET_SESSION = "reset_session"
    SESSION_RESET = "session_reset"
    TOKEN_USAGE_RESET = "token_usage_reset"
    
    # 시스템 정보
    SYSTEM_INFO = "system_info"
    APP_START_TIME = "app_start_time"
    MIDDLEWARE_STATUS = "middleware_status"
    STREAMLIT_VERSION = "streamlit_version"
    
    # 응답 개선 유형 설명
    class ResponseTypes:
        INITIAL_ONLY = "initial_only"
        INITIAL_ONLY_DESC = "initial_only_desc"
        
        CLAUDE_BY_GPT = "claude_by_gpt"
        CLAUDE_BY_GPT_DESC = "claude_by_gpt_desc"
        GPT_BY_CLAUDE = "gpt_by_claude"
        GPT_BY_CLAUDE_DESC = "gpt_by_claude_desc"
        ALL_SELF_ANALYSIS = "all_self_analysis"
        ALL_SELF_ANALYSIS_DESC = "all_self_analysis_desc"
        ALL_COMBINATIONS = "all_combinations"
        ALL_COMBINATIONS_DESC = "all_combinations_desc"

        # 자체 분석 옵션
        CLAUDE_SELF = "claude_self"
        CLAUDE_SELF_DESC = "claude_self_desc"
        GPT_SELF = "gpt_self"
        GPT_SELF_DESC = "gpt_self_desc"
        PERPLEXITY_SELF = "perplexity_self"
        PERPLEXITY_SELF_DESC = "perplexity_self_desc"

        # 외부 분석 옵션 
        CLAUDE_BY_PERPLEXITY = "claude_by_perplexity"
        CLAUDE_BY_PERPLEXITY_DESC = "claude_by_perplexity_desc"
        GPT_BY_PERPLEXITY = "gpt_by_perplexity"
        GPT_BY_PERPLEXITY_DESC = "gpt_by_perplexity_desc"
        PERPLEXITY_BY_CLAUDE = "perplexity_by_claude"
        PERPLEXITY_BY_CLAUDE_DESC = "perplexity_by_claude_desc"
        PERPLEXITY_BY_GPT = "perplexity_by_gpt"
        PERPLEXITY_BY_GPT_DESC = "perplexity_by_gpt_desc"

        # 다중 분석 옵션
        CLAUDE_BY_MULTIPLE = "claude_by_multiple"
        CLAUDE_BY_MULTIPLE_DESC = "claude_by_multiple_desc"
        GPT_BY_MULTIPLE = "gpt_by_multiple"
        GPT_BY_MULTIPLE_DESC = "gpt_by_multiple_desc"
        PERPLEXITY_BY_MULTIPLE = "perplexity_by_multiple"
        PERPLEXITY_BY_MULTIPLE_DESC = "perplexity_by_multiple_desc"

        # 초기 응답 옵션
        CLAUDE_INITIAL = "claude_initial"
        CLAUDE_INITIAL_DESC = "claude_initial_desc"
        GPT_INITIAL = "gpt_initial"
        GPT_INITIAL_DESC = "gpt_initial_desc"
        PERPLEXITY_INITIAL = "perplexity_initial"
        PERPLEXITY_INITIAL_DESC = "perplexity_initial_desc"