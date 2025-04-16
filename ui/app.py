"""
Enhanced AI 응답 시스템 Streamlit UI - 개선된 구조 및 명명 체계
다국어 지원 (한국어/영어) 추가
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import sys
import time
from datetime import datetime
import anthropic
import openai

# 상위 디렉토리 경로 추가하여 모듈 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ANTHROPIC_API_KEY, OPENAI_API_KEY, PERPLEXITY_API_KEY
from config import ACTIVE_CLAUDE_MODEL, ACTIVE_GPT_MODEL, ACTIVE_PERPLEXITY_MODEL
from enhanced_middleware import EnhancedMultiAIMiddleware, AnalysisTask, ImprovementPlan
from utils.token_tracker import TokenUsageTracker
from messages import MK, get_message

def init_session_state():
    """세션 상태 초기화"""
    # 언어 설정 추가
    if "language" not in st.session_state:
        st.session_state.language = "ko"  # 기본 언어: 한국어
        
    if "middleware" not in st.session_state:
        st.session_state.middleware = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "perplexity_enabled" not in st.session_state:
        st.session_state.perplexity_enabled = False
    # 현재 결과 저장을 위한 상태 추가
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    # 선택된 응답 모델 저장
    if "selected_response_model" not in st.session_state:
        st.session_state.selected_response_model = None
    # 대화 모드 여부
    if "conversation_mode" not in st.session_state:
        st.session_state.conversation_mode = False
    # 표시할 개선 응답 유형 설정 (기본값: "claude_analyzed_by_openai")
    if "display_improvement_types" not in st.session_state:
        st.session_state.display_improvement_types = ["claude_analyzed_by_openai"]  # 리스트로 저장
    # 선택된 표시 유형을 저장하는 상태 변수 추가
    if "selected_display_types" not in st.session_state:
        st.session_state.selected_display_types = ["claude_analyzed_by_openai"]  # 기본값    
    # UI에서 마지막 선택한 옵션 추적
    if "detailed_last_selected" not in st.session_state:
        st.session_state.detailed_last_selected = "col1"
    # 토큰 트래커 초기화 확인 로그 추가
    print("DEBUG - 세션 상태 초기화 중...")
    if "token_tracker_initialized" not in st.session_state:
        st.session_state.token_tracker_initialized = False
        print("DEBUG - 토큰 트래커가 아직 초기화되지 않았습니다.")
    else:
        print(f"DEBUG - 토큰 트래커 초기화 상태: {st.session_state.token_tracker_initialized}")
        
    # 옵션 표시 방식 선택 상태 저장 (추가된 부분)
    if "display_option_type" not in st.session_state:
        st.session_state.display_option_type = "basic"  # 기본값: 기본 옵션
        
    # 체크박스 상태 저장을 위한 변수 추가 (추가된 부분)
    if "sidebar_checkbox_states" not in st.session_state:
        st.session_state.sidebar_checkbox_states = {}
    
    # 설정 탭에서의 체크박스 상태 저장 (추가된 부분)
    if "tab_basic_checkbox_states" not in st.session_state:
        st.session_state.tab_basic_checkbox_states = {}
    if "tab_detailed_checkbox_states" not in st.session_state:
        st.session_state.tab_detailed_checkbox_states = {}

def initialize_middleware():
    """미들웨어 초기화"""
    try:
        current_lang = st.session_state.get("language", "ko")
        print("DEBUG - 미들웨어 초기화 시작...")
        
        # Claude 클라이언트 초기화
        try:
            claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            print("DEBUG - Claude 클라이언트 초기화됨")
        except Exception as e:
            claude_client = None
            st.warning(get_message(MK.CLAUDE_CLIENT_INIT_FAILED, current_lang, str(e))) # "Claude 클라이언트 초기화 실패: {str(e)}"
            print(f"DEBUG - Claude 클라이언트 초기화 실패: {str(e)}")
        
        # OpenAI 클라이언트 초기화  
        try:
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            print("DEBUG - OpenAI 클라이언트 초기화됨")
        except Exception as e:
            openai_client = None
            st.warning(get_message(MK.OPENAI_CLIENT_INIT_FAILED, current_lang, str(e))) # "OpenAI 클라이언트 초기화 실패: {str(e)}"
            print(f"DEBUG - OpenAI 클라이언트 초기화 실패: {str(e)}")
        
        # Perplexity 클라이언트 초기화
        try:
            perplexity_client = openai.OpenAI(
                api_key=PERPLEXITY_API_KEY,
                base_url="https://api.perplexity.ai"
            )
            print("DEBUG - Perplexity 클라이언트가 OpenAI 인터페이스로 초기화되었습니다")
        except Exception as e:
            perplexity_client = None
            st.warning(get_message(MK.PERPLEXITY_CLIENT_INIT_FAILED, current_lang, str(e))) # "Perplexity 클라이언트 초기화 실패: {str(e)}"
            print(f"DEBUG - Perplexity 클라이언트 초기화 실패: {str(e)}")

        # 미들웨어 초기화
        st.session_state.middleware = EnhancedMultiAIMiddleware(
            claude_client, 
            openai_client, 
            perplexity_client
        )
        
        # 미들웨어의 언어 설정 추가
        st.session_state.middleware.language = current_lang

        # 토큰 트래커 초기화 확인
        st.session_state.token_tracker_initialized = True
        print("DEBUG - 토큰 트래커가 초기화되었습니다.")
        
        # 예산 제한 설정
        budget = st.session_state.get("budget", 5.0)
        st.session_state.middleware.set_budget_limit(budget)
        
        return True
    except Exception as e:
        st.error(get_message(MK.ERROR_OCCURRED, current_lang, str(e)))
        print(f"DEBUG - 미들웨어 초기화 오류: {str(e)}")
        return False

def display_token_usage():
    """토큰 사용량 표시"""
    current_lang = st.session_state.get("language", "ko")
    
    if not st.session_state.middleware:
        return
    
    usage = st.session_state.middleware.get_usage_summary()
    
    # 총 비용
    st.metric(get_message(MK.TOTAL_COST, current_lang), f"${usage['combined']['estimated_cost']:.4f}")
    
    # 토큰 사용량 데이터프레임 생성 - 빈 리스트로 시작
    token_data = {
        get_message(MK.AI_MODEL_COLUMN, current_lang): [], # "AI 모델"
        get_message(MK.INPUT_TOKENS_COLUMN, current_lang): [], # "입력 토큰"
        get_message(MK.OUTPUT_TOKENS_COLUMN, current_lang): [], # "출력 토큰"
        get_message(MK.COST_COLUMN, current_lang): [] # "비용"
    }
    
    # Claude가 실제로 사용된 경우만 추가
    ai_model_col = get_message(MK.AI_MODEL_COLUMN, current_lang) # "AI 모델"
    input_tokens_col = get_message(MK.INPUT_TOKENS_COLUMN, current_lang) # "입력 토큰"
    output_tokens_col = get_message(MK.OUTPUT_TOKENS_COLUMN, current_lang) # "출력 토큰"
    cost_col = get_message(MK.COST_COLUMN, current_lang) # "비용"

    if usage['claude_usage']['total_tokens'] > 0:
        token_data[ai_model_col].append('Claude')
        token_data[input_tokens_col].append(usage['claude_usage']['prompt_tokens'])
        token_data[output_tokens_col].append(usage['claude_usage']['completion_tokens'])
        token_data[cost_col].append(usage['claude_usage']['estimated_cost'])

    # GPT가 실제로 사용된 경우만 추가
    ai_model_col = get_message(MK.AI_MODEL_COLUMN, current_lang) # "AI 모델"
    input_tokens_col = get_message(MK.INPUT_TOKENS_COLUMN, current_lang) # "입력 토큰"
    output_tokens_col = get_message(MK.OUTPUT_TOKENS_COLUMN, current_lang) # "출력 토큰"
    cost_col = get_message(MK.COST_COLUMN, current_lang) # "비용"

    if usage['openai_usage']['total_tokens'] > 0:
        token_data[ai_model_col].append('openai')
        token_data[input_tokens_col].append(usage['openai_usage']['prompt_tokens'])
        token_data[output_tokens_col].append(usage['openai_usage']['completion_tokens'])
        token_data[cost_col].append(usage['openai_usage']['estimated_cost'])
    
    # Perplexity가 실제로 사용된 경우만 추가
    ai_model_col = get_message(MK.AI_MODEL_COLUMN, current_lang) # "AI 모델"
    input_tokens_col = get_message(MK.INPUT_TOKENS_COLUMN, current_lang) # "입력 토큰"
    output_tokens_col = get_message(MK.OUTPUT_TOKENS_COLUMN, current_lang) # "출력 토큰"
    cost_col = get_message(MK.COST_COLUMN, current_lang) # "비용"

    if usage['perplexity_usage']['total_tokens'] > 0:
        token_data[ai_model_col].append('perplexity')
        token_data[input_tokens_col].append(usage['perplexity_usage']['prompt_tokens'])
        token_data[output_tokens_col].append(usage['perplexity_usage']['completion_tokens'])
        token_data[cost_col].append(usage['perplexity_usage']['estimated_cost'])

    # 데이터가 있는 경우에만 그래프 표시
    if token_data[ai_model_col]:  # 수정: 변수 사용
        df = pd.DataFrame(token_data)
        
        # 토큰 사용량 차트
        fig = px.bar(
            df, 
            x=ai_model_col, # "AI 모델"
            y=[input_tokens_col, output_tokens_col], # "입력 토큰", "출력 토큰"
            title=get_message(MK.TOKEN_USAGE_BY_MODEL_TITLE, current_lang), # "AI 모델별 토큰 사용량"
            barmode='stack'
        )

        st.plotly_chart(fig, use_container_width=True)
        
        # 비용 차트
        fig2 = px.pie(
            df, 
            values=cost_col, # "비용"
            names=ai_model_col, # "AI 모델"
            title=get_message(MK.COST_DISTRIBUTION_TITLE, current_lang) # "비용 분포"
        )

        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info(get_message(MK.NO_TOKENS_USED, current_lang))

def display_conversation_history():
    """대화 기록 표시"""
    current_lang = st.session_state.get("language", "ko")
    
    if not st.session_state.conversation_history:
        return
    
    # 대화 기록 표시
    for i, message in enumerate(st.session_state.conversation_history):
        if message["role"] == "user":
            st.write(f"**{get_message(MK.USER_LABEL, current_lang)}:** {message['content']}") # "사용자:"
        else:
            # display_title이 있으면 사용, 없으면 기본 title 사용
            title = message.get("display_title", get_message(MK.RESPONSE_NUMBER, current_lang, i+1)) # "응답 #"
            with st.expander(f"{title}", expanded=True):
                st.write(message["content"])

def display_query_history():
    """쿼리 기록 표시 (현재 쿼리 제외)"""
    current_lang = st.session_state.get("language", "ko")
    
    if not st.session_state.query_history or len(st.session_state.query_history) <= 1:
        st.info(get_message(MK.NO_PREVIOUS_HISTORY, current_lang))
        return
    
    # 현재 쿼리(마지막 쿼리)를 제외한 이전 쿼리만 표시
    for i, query in enumerate(st.session_state.query_history[:-1]):
        with st.expander(f"{get_message(MK.QUERY_NUMBER, current_lang, i+1)}: {query['query'][:50]}...", expanded=False): # "쿼리 #{i+1}"
            st.write(f"**{get_message(MK.QUESTION_LABEL, current_lang)}** {query['query']}") # "질문"
            st.write(f"**{get_message(MK.PROCESSING_TIME_LABEL, current_lang)}** {query['time']:.2f}{get_message(MK.SECONDS_UNIT, current_lang)}") # "처리 시간"
            st.write(f"**{get_message(MK.COST_LABEL, current_lang)}** ${query['cost']:.4f}") # "비용"   
                
            # 응답 표시를 위한 탭
            tabs = st.tabs([
                get_message(MK.INITIAL_RESPONSES, current_lang),
                get_message(MK.IMPROVED_RESPONSES, current_lang),
                get_message(MK.FOLLOW_UP_QUESTIONS, current_lang)
            ])
            
            # 초기 응답 탭
            with tabs[0]:
                for ai, response in query["initial_responses"].items():
                    st.subheader(ai)
                    st.write(response)
            
            # 개선된 응답 탭
            with tabs[1]:
                for ai, response in query["improved_responses"].items():
                    st.subheader(ai)
                    st.write(response)
            
            # 후속 질문 탭
            with tabs[2]:
                for ai, questions in query["follow_up_questions"].items():
                    st.subheader(ai)
                    if isinstance(questions, dict):
                        extracted = questions.get("extracted", [])
                        if extracted:
                            for q in enumerate(extracted, 1):
                                st.write(f"- {q}")
                        else:
                            st.write(get_message(MK.NO_EXTRACTED_QUESTIONS, current_lang))
                    else:
                        st.write(questions)  # 오류 메시지

def process_query(query, compare_mode=True, conversation_mode=False):
    """쿼리 처리"""
    current_lang = st.session_state.get("language", "ko")
    
    if not st.session_state.middleware:
        middleware_error = get_message(MK.MIDDLEWARE_NOT_INITIALIZED, current_lang)
        st.error(get_message(MK.ERROR_OCCURRED, current_lang, middleware_error)) # "미들웨어가 초기화되지 않았습니다"
        return
    
    with st.spinner(get_message(MK.PROCESSING, current_lang)):
        start_time = time.time()
        
        # 대화 모드라면 이전 대화 내역을 포함하여 전달
        if conversation_mode and st.session_state.conversation_history:
            # 미들웨어에 대화 내역 전달 기능을 추가해야 함
            # (현재 코드에는 이 기능이 없으므로 일단 대화 내역 저장만 구현)
            pass
            
        # 표시할 개선 응답 유형 가져오기 (다중 선택 지원)
        display_improvement_types = st.session_state.display_improvement_types
        
        # 미들웨어에 쿼리 전달
        result = st.session_state.middleware.process_query(
            query, 
            show_comparison=compare_mode,
            display_improvement_types=display_improvement_types
        )
        process_time = time.time() - start_time
    
    if "error" in result:
        st.error(get_message(MK.ERROR_OCCURRED, current_lang, result['error']))
        return
    
    # 결과 포맷팅
    formatted = st.session_state.middleware.format_comparison_result(result)
    
    # 쿼리 기록에 추가
    st.session_state.query_count += 1
    st.session_state.query_history.append({
        "query": query,
        "time": process_time,
        "initial_responses": formatted["comparison"]["initial_responses"],
        "improved_responses": formatted["comparison"]["improved_responses"],
        "follow_up_questions": formatted["comparison"]["follow_up_questions"],
        "cost": formatted["processing_info"]["token_usage"]["combined"]["estimated_cost"],
        "display_improvement_types": display_improvement_types,  # 선택된 표시 유형도 저장
        "response_models": formatted["comparison"].get("response_models", []),
        "analysis_models": formatted["comparison"].get("analysis_models", []),
        "analysis_only_models": formatted["comparison"].get("analysis_only_models", []),
        "analysis_tasks": formatted["comparison"].get("analysis_tasks", [])
    })
    
    # 현재 결과를 세션 상태에 저장
    st.session_state.current_result = formatted
    
    # 대화 기록에 사용자 질문 추가
    st.session_state.conversation_history.append({"role": "user", "content": query})
    
    # 대화 모드 활성화
    st.session_state.conversation_mode = True

    return formatted

def select_response(ai_model, response_text, display_title=None):
    """특정 AI의 응답을 선택하여 대화 계속"""
    current_lang = st.session_state.get("language", "ko")
    
    if not response_text:
        no_response_error = get_message(MK.NO_RESPONSE_TO_SELECT, current_lang) # "선택할 응답이 없습니다"
        st.error(get_message(MK.ERROR_OCCURRED, current_lang, no_response_error))
        return
    
    # 선택된 응답 모델 저장
    st.session_state.selected_response_model = ai_model
    
    # 대화 기록에 추가 (display_title 포함)
    st.session_state.conversation_history.append({
        "role": "assistant", 
        "content": response_text,
        "model": ai_model,
        "display_title": display_title or get_message(MK.MODEL_RESPONSE, current_lang, ai_model) # f"{ai_model} 응답"
    })
    
    # 대화 모드 활성화
    st.session_state.conversation_mode = True
    
    # 화면 새로고침 (스크롤 위치 조정을 위해)
    st.rerun()

def display_improvement_settings_sidebar():
    """사이드바에 개선 응답 유형 설정 UI 표시"""
    current_lang = st.session_state.get("language", "ko")
    
    st.sidebar.header(get_message(MK.RESPONSE_DISPLAY_SETTINGS, current_lang))
    st.sidebar.write(get_message(MK.SELECT_TO_CONTINUE, current_lang))
    
    # 옵션 표시 방식 선택 (기본/상세) - 세션 상태에서 값 읽어오기 (수정된 부분)
    # 기본값 설정
    display_option_index = 0
    if st.session_state.get("display_option_type") == "detailed":
        display_option_index = 1
    
    display_option_type = st.sidebar.radio(
        get_message(MK.DISPLAY_OPTIONS_TYPE, current_lang),
        [get_message(MK.BASIC_OPTIONS, current_lang), get_message(MK.DETAILED_OPTIONS, current_lang)],
        index=display_option_index,
        help=get_message(MK.DISPLAY_OPTIONS_HELP, current_lang)
    )
    
    # 세션 상태에 현재 선택된 옵션 타입 저장 (추가된 부분)
    if display_option_type == get_message(MK.BASIC_OPTIONS, current_lang):
        st.session_state.display_option_type = "basic"
    else:
        st.session_state.display_option_type = "detailed"
    
    # 기본 옵션
    basic_improvement_display_options = [
        {"id": "initial_only", "name": get_message(MK.ResponseTypes.INITIAL_ONLY, current_lang), 
        "description": get_message(MK.ResponseTypes.INITIAL_ONLY_DESC, current_lang)},
        {"id": "claude_analyzed_by_openai", "name": get_message(MK.ResponseTypes.CLAUDE_BY_GPT, current_lang), 
        "description": get_message(MK.ResponseTypes.CLAUDE_BY_GPT_DESC, current_lang)},
        {"id": "openai_analyzed_by_claude", "name": get_message(MK.ResponseTypes.GPT_BY_CLAUDE, current_lang), 
        "description": get_message(MK.ResponseTypes.GPT_BY_CLAUDE_DESC, current_lang)},
        {"id": "all_self_analysis", "name": get_message(MK.ResponseTypes.ALL_SELF_ANALYSIS, current_lang), 
        "description": get_message(MK.ResponseTypes.ALL_SELF_ANALYSIS_DESC, current_lang)},
        {"id": "all", "name": get_message(MK.ResponseTypes.ALL_COMBINATIONS, current_lang), 
        "description": get_message(MK.ResponseTypes.ALL_COMBINATIONS_DESC, current_lang)}
    ]

    # 상세 옵션 - 그룹으로 분류
    detailed_improvement_display_options = [
        # 자체 분석 옵션
        {"id": "claude_analyzed_by_self", "name": get_message(MK.ResponseTypes.CLAUDE_SELF, current_lang), 
        "description": get_message(MK.ResponseTypes.CLAUDE_SELF_DESC, current_lang), "group": 1},
        {"id": "openai_analyzed_by_self", "name": get_message(MK.ResponseTypes.GPT_SELF, current_lang), 
        "description": get_message(MK.ResponseTypes.GPT_SELF_DESC, current_lang), "group": 1},
        {"id": "perplexity_analyzed_by_self", "name": get_message(MK.ResponseTypes.PERPLEXITY_SELF, current_lang), 
        "description": get_message(MK.ResponseTypes.PERPLEXITY_SELF_DESC, current_lang), "group": 1},
        
        # 단일 외부 분석 옵션
        {"id": "claude_analyzed_by_openai", "name": get_message(MK.ResponseTypes.CLAUDE_BY_GPT, current_lang), 
        "description": get_message(MK.ResponseTypes.CLAUDE_BY_GPT_DESC, current_lang), "group": 2},
        {"id": "claude_analyzed_by_perplexity", "name": get_message(MK.ResponseTypes.CLAUDE_BY_PERPLEXITY, current_lang), 
        "description": get_message(MK.ResponseTypes.CLAUDE_BY_PERPLEXITY_DESC, current_lang), "group": 2},
        {"id": "openai_analyzed_by_claude", "name": get_message(MK.ResponseTypes.GPT_BY_CLAUDE, current_lang), 
        "description": get_message(MK.ResponseTypes.GPT_BY_CLAUDE_DESC, current_lang), "group": 2},
        {"id": "openai_analyzed_by_perplexity", "name": get_message(MK.ResponseTypes.GPT_BY_PERPLEXITY, current_lang), 
        "description": get_message(MK.ResponseTypes.GPT_BY_PERPLEXITY_DESC, current_lang), "group": 2},
        {"id": "perplexity_analyzed_by_claude", "name": get_message(MK.ResponseTypes.PERPLEXITY_BY_CLAUDE, current_lang), 
        "description": get_message(MK.ResponseTypes.PERPLEXITY_BY_CLAUDE_DESC, current_lang), "group": 2},
        {"id": "perplexity_analyzed_by_openai", "name": get_message(MK.ResponseTypes.PERPLEXITY_BY_GPT, current_lang), 
        "description": get_message(MK.ResponseTypes.PERPLEXITY_BY_GPT_DESC, current_lang), "group": 2},
        
        # 다중 분석자 옵션
        {"id": "claude_analyzed_by_multiple", "name": get_message(MK.ResponseTypes.CLAUDE_BY_MULTIPLE, current_lang), 
        "description": get_message(MK.ResponseTypes.CLAUDE_BY_MULTIPLE_DESC, current_lang), "group": 3},
        {"id": "openai_analyzed_by_multiple", "name": get_message(MK.ResponseTypes.GPT_BY_MULTIPLE, current_lang), 
        "description": get_message(MK.ResponseTypes.GPT_BY_MULTIPLE_DESC, current_lang), "group": 3},
        {"id": "perplexity_analyzed_by_multiple", "name": get_message(MK.ResponseTypes.PERPLEXITY_BY_MULTIPLE, current_lang), 
        "description": get_message(MK.ResponseTypes.PERPLEXITY_BY_MULTIPLE_DESC, current_lang), "group": 3},
        
        # 초기 응답 옵션
        {"id": "claude_initial", "name": get_message(MK.ResponseTypes.CLAUDE_INITIAL, current_lang), 
        "description": get_message(MK.ResponseTypes.CLAUDE_INITIAL_DESC, current_lang), "group": 4},
        {"id": "openai_initial", "name": get_message(MK.ResponseTypes.GPT_INITIAL, current_lang), 
        "description": get_message(MK.ResponseTypes.GPT_INITIAL_DESC, current_lang), "group": 4},
        {"id": "perplexity_initial", "name": get_message(MK.ResponseTypes.PERPLEXITY_INITIAL, current_lang), 
        "description": get_message(MK.ResponseTypes.PERPLEXITY_INITIAL_DESC, current_lang), "group": 4},

        # 특수 옵션
        {"id": "all", "name": get_message(MK.ResponseTypes.ALL_COMBINATIONS, current_lang), 
        "description": get_message(MK.ResponseTypes.ALL_COMBINATIONS_DESC, current_lang), "group": 0}
    ]
    
    # 선택된 옵션 표시 방식에 따라 다른 옵션 표시
    if st.session_state.display_option_type == "basic":
        display_options = basic_improvement_display_options
        st.sidebar.info(get_message(MK.SEE_DETAILED_OPTIONS, current_lang))
    else:
        display_options = detailed_improvement_display_options
        st.sidebar.info(get_message(MK.RETURN_TO_BASIC_OPTIONS, current_lang))

    # 세션 상태에 다중 선택 저장
    if "selected_display_types" not in st.session_state:
        st.session_state.selected_display_types = ["claude_analyzed_by_openai"]  # 기본값
    
    # 현재 선택된 값들이 표시되는 옵션 목록에 있는지 확인
    selected_types = [t for t in st.session_state.selected_display_types 
                      if any(opt["id"] == t for opt in display_options)]
    
    if not selected_types:  # 선택된 것이 하나도 없으면 기본값 설정
        selected_types = ["claude_analyzed_by_openai"] if st.session_state.display_option_type == "basic" else ["claude_analyzed_by_openai"]
    
    # 다중 선택 UI
    # 상세 옵션인 경우 그룹 속성 무시하고 모든 옵션 표시
    if st.session_state.display_option_type == "detailed":
        # 사이드바용 옵션 목록에서는 그룹 구분 없이 단순 리스트로 표시
        sidebar_options = [
            {"id": opt["id"], "name": opt["name"], "description": opt["description"]} 
            for opt in display_options
        ]
        
        # 체크박스 상태를 세션 상태에서 초기화 (추가된 부분)
        for opt in sidebar_options:
            option_id = opt["id"]
            if option_id not in st.session_state.sidebar_checkbox_states:
                # 기본값: selected_types에 있으면 True, 없으면 False
                st.session_state.sidebar_checkbox_states[option_id] = option_id in selected_types
        
        # 체크박스 대신 직접 구현 (수정된 부분)
        display_types = []
        st.sidebar.write(get_message(MK.SELECT_RESPONSE_TYPES, current_lang))
        for option in sidebar_options:
            option_id = option["id"]
            option_name = option["name"]
            
            # 체크박스 상태를 세션에서 가져오기
            checkbox_value = st.sidebar.checkbox(
                option_name,
                value=st.session_state.sidebar_checkbox_states.get(option_id, False),
                key=f"sidebar_opt_{option_id}",
                help=option["description"]
            )
            
            # 체크박스 상태 업데이트
            st.session_state.sidebar_checkbox_states[option_id] = checkbox_value
            
            if checkbox_value:
                display_types.append(option_id)
    else:
        # 기본 옵션일 경우 기존 코드 유지하되 체크박스 상태 추적 추가
        display_types = []
        st.sidebar.write(get_message(MK.SELECT_RESPONSE_TYPES, current_lang))
        for option in display_options:
            option_id = option["id"]
            option_name = option["name"]
            
            # 체크박스 상태를 세션에서 가져오기
            if option_id not in st.session_state.sidebar_checkbox_states:
                # 기본값: selected_types에 있으면 True, 없으면 False
                st.session_state.sidebar_checkbox_states[option_id] = option_id in selected_types
                
            checkbox_value = st.sidebar.checkbox(
                option_name,
                value=st.session_state.sidebar_checkbox_states.get(option_id, False),
                key=f"sidebar_opt_{option_id}",
                help=option["description"]
            )
            
            # 체크박스 상태 업데이트
            st.session_state.sidebar_checkbox_states[option_id] = checkbox_value
            
            if checkbox_value:
                display_types.append(option_id)
    
    # 선택된 것이 없으면 기본값 적용
    if not display_types:
        display_types = ["claude_analyzed_by_openai"] if st.session_state.display_option_type == "basic" else ["claude_analyzed_by_openai"]
        st.sidebar.warning(get_message(MK.MIN_ONE_SELECTION, current_lang))
    
    # 선택된 옵션에 대한 설명 표시
    st.sidebar.subheader(get_message(MK.SELECTED_OPTIONS, current_lang))
    for display_type in display_types:
        selected_option = next((option for option in display_options if option["id"] == display_type), None)
        if selected_option:
            st.sidebar.info(f"**{selected_option['name']}**: {selected_option['description']}")
    
    # 선택 사항 저장
    if st.sidebar.button(get_message(MK.APPLY_DISPLAY_SETTINGS, current_lang), key="apply_display_settings", type="primary"):
        st.session_state.selected_display_types = display_types
        # 리스트로 저장
        st.session_state.display_improvement_types = display_types
        st.sidebar.success(get_message(MK.DISPLAY_SETTINGS_SAVED, current_lang, len(display_types)))
        st.rerun()

# display_result 함수에서 메타데이터 참조 방식 수정
# 3. UI에서 결과 표시 시 동적 제목 생성 및 메타데이터 활용
def display_result(result, user_query):
    """결과 표시 - 동적 제목 생성 지원"""
    current_lang = st.session_state.get("language", "ko")
    
    if not result:
        return
    
    # 사용자 질문 표시
    st.header(get_message(MK.QUESTION, current_lang))
    st.write(user_query)
    
    # 미들웨어 객체 가져오기
    middleware = st.session_state.middleware
    
    # 현재 선택된 개선 유형 가져오기
    current_improvement_types = result.get("comparison", {}).get("display_improvement_types", "claude_analyzed_by_openai")
    if isinstance(current_improvement_types, str):
        if "," in current_improvement_types:
            current_improvement_types = [t.strip() for t in current_improvement_types.split(",")]
        else:
            current_improvement_types = [current_improvement_types]
    elif not isinstance(current_improvement_types, list):
        current_improvement_types = [current_improvement_types]

    # 초기 응답 옵션인지 확인
    is_initial_only = any(t.endswith("_initial") or t == "initial_only" for t in current_improvement_types)

    # 개선 응답 옵션도 있는지 확인
    has_improved_types = any(not (t.endswith("_initial") or t == "initial_only") for t in current_improvement_types)

    # 탭 생성 - 두 가지 경우를 명확히 구분
    if is_initial_only and not has_improved_types:
        tabs = st.tabs([
            get_message(MK.INITIAL_RESPONSES, current_lang),
            get_message(MK.ANALYSIS, current_lang),
            get_message(MK.FOLLOW_UP_QUESTIONS, current_lang),
            get_message(MK.DISPLAY_SETTINGS, current_lang)
        ])
        initial_resp_tab_idx = 0
        improved_resp_tab_idx = None
    else:
        tabs = st.tabs([
            get_message(MK.IMPROVED_RESPONSES, current_lang),
            get_message(MK.INITIAL_RESPONSES, current_lang),
            get_message(MK.ANALYSIS, current_lang),
            get_message(MK.FOLLOW_UP_QUESTIONS, current_lang),
            get_message(MK.DISPLAY_SETTINGS, current_lang)
        ])
        improved_resp_tab_idx = 0
        initial_resp_tab_idx = 1
    
    # 개선된 응답 탭 (있는 경우에만)
    if improved_resp_tab_idx is not None:
        with tabs[improved_resp_tab_idx]:
            st.header(get_message(MK.IMPROVED_RESPONSES, current_lang))
            
            st.info(get_message(MK.SELECT_TO_CONTINUE, current_lang))
            
            if not result["comparison"]["improved_responses"]:
                st.info(get_message(MK.NO_RESPONSES, current_lang, get_message(MK.IMPROVED_RESPONSES, current_lang).lower()))
            
            # 개선된 응답 표시 - 동적 제목 생성 적용
            for original_title, response_data in result["comparison"]["improved_responses"].items():
                # response_data가 dict이고 metadata가 있는 경우 (신규 형식)
                if isinstance(response_data, dict) and "metadata" in response_data:
                    response_text = response_data.get("text", "")
                    metadata = response_data.get("metadata", {})
                    
                    # 현재 언어로 동적 제목 생성
                    if middleware:
                        dynamic_title = middleware.generate_display_title_from_metadata(metadata, current_lang)
                    else:
                        # 미들웨어가 없는 경우 원래 제목 사용
                        dynamic_title = original_title
                    
                    with st.expander(f"{dynamic_title}", expanded=True):
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.write(response_text)
                        with col2:
                            # 응답 표시 이름에서 모델 이름 추출
                            base_model = None
                            for model in ["Claude", "GPT", "Perplexity"]:
                                if model.lower() in dynamic_title.lower():
                                    base_model = model
                                    break
                                    
                            if not base_model and "analyzed_model" in metadata:
                                model_display_names = {"claude": "Claude", "openai": "GPT", "perplexity": "Perplexity"}
                                base_model = model_display_names.get(metadata["analyzed_model"], metadata["analyzed_model"].capitalize())
                            
                            if not base_model:
                                base_model = dynamic_title.split(' ')[0]
                            
                            # 이 응답으로 대화 계속하기 버튼 
                            if st.button(get_message(MK.CONTINUE_WITH, current_lang, base_model), key=f"improved_{dynamic_title}"):
                                # 메타데이터 포함하여 전달 - 현재 언어로 생성된 제목 사용
                                select_response(dynamic_title, response_text, dynamic_title)
                else:
                    # 기존 형식 호환성 유지 (단순 텍스트인 경우)
                    response_text = response_data
                    with st.expander(f"{original_title}", expanded=True):
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.write(response_text)
                        with col2:
                            base_model = original_title.split(' ')[0]
                            if st.button(get_message(MK.CONTINUE_WITH, current_lang, base_model), key=f"improved_{original_title}"):
                                select_response(original_title, response_text, original_title)

    # 초기 응답 탭도 비슷하게 수정 (동적 제목 생성 적용)
    with tabs[initial_resp_tab_idx]:
        st.header(get_message(MK.INITIAL_RESPONSES, current_lang))
        
        if not result["comparison"]["initial_responses"]:
            st.info(get_message(MK.NO_RESPONSES, current_lang, get_message(MK.INITIAL_RESPONSES, current_lang).lower()))
        
        for original_title, response_data in result["comparison"]["initial_responses"].items():
            # 신규 형식 (메타데이터 포함)
            if isinstance(response_data, dict) and "metadata" in response_data:
                response_text = response_data.get("text", "")
                metadata = response_data.get("metadata", {})
                
                # 초기 응답 옵션 필터링 적용
                show_this_model = True
                if is_initial_only and not has_improved_types:
                    analyzed_model = metadata.get("analyzed_model", "").lower()
                    if analyzed_model == "claude" and not any(t == "claude_initial" or t == "initial_only" for t in current_improvement_types):
                        show_this_model = False
                    elif analyzed_model == "openai" and not any(t == "openai_initial" or t == "initial_only" for t in current_improvement_types):
                        show_this_model = False
                    elif analyzed_model == "perplexity" and not any(t == "perplexity_initial" or t == "initial_only" for t in current_improvement_types):
                        show_this_model = False
                
                if show_this_model:
                    # 현재 언어로 동적 제목 생성
                    if middleware:
                        dynamic_title = middleware.generate_display_title_from_metadata(metadata, current_lang)
                        if metadata.get("is_initial_response", False):
                            # 초기 응답임을 명시적으로 표시
                            dynamic_title += f" {get_message(MK.INITIAL_RESPONSE_LABEL, current_lang)}"
                    else:
                        dynamic_title = original_title
                    
                    with st.expander(f"{dynamic_title}", expanded=True):
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.write(response_text)
                        with col2:
                            # 모델명 추출
                            model_name = None
                            if "analyzed_model" in metadata:
                                model_display_names = {"claude": "Claude", "openai": "GPT", "perplexity": "Perplexity"}
                                model_name = model_display_names.get(metadata["analyzed_model"], metadata["analyzed_model"].capitalize())
                            else:
                                model_name = dynamic_title.split(' ')[0]  
                            
                            button_text = get_message(MK.CONTINUE_WITH, current_lang, model_name)
                            if st.button(button_text, key=f"initial_{dynamic_title}"):
                                select_response(dynamic_title, response_text, dynamic_title)
            else:
                # 기존 형식 호환성 유지
                response_text = response_data
                show_this_model = True
                if is_initial_only and not has_improved_types:
                    if "Claude" in original_title and not any(t == "claude_initial" or t == "initial_only" for t in current_improvement_types):
                        show_this_model = False
                    elif "GPT" in original_title and not any(t == "openai_initial" or t == "initial_only" for t in current_improvement_types):
                        show_this_model = False
                    elif "Perplexity" in original_title and not any(t == "perplexity_initial" or t == "initial_only" for t in current_improvement_types):
                        show_this_model = False
                
                if show_this_model:
                    with st.expander(f"{original_title}", expanded=True):
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.write(response_text)
                        with col2:
                            model_name = original_title.split(' ')[0]
                            button_text = get_message(MK.CONTINUE_WITH, current_lang, model_name)
                            if st.button(button_text, key=f"initial_{original_title}"):
                                select_response(original_title, response_text, original_title)

    # 분석 탭도 동적 제목 생성 적용
    with tabs[2 if not is_initial_only or has_improved_types else 1]:
        st.header(get_message(MK.ANALYSIS_RESULTS, current_lang))
        
        # 응답 생성 모델과 분석 모델 정보 가져오기
        response_models = result["comparison"].get("response_models", [])
        analysis_models = result["comparison"].get("analysis_models", [])
        analysis_only_models = result["comparison"].get("analysis_only_models", [])
        analysis_tasks = result["comparison"].get("analysis_tasks", [])
        
        # 분석 정보 설명 개선
        response_models_str = ', '.join([m.capitalize() for m in response_models])
        analysis_only_models_str = ', '.join([m.capitalize() for m in analysis_only_models]) if analysis_only_models else get_message(MK.NONE, current_lang) 
        analysis_tasks_str = ', '.join(analysis_tasks) if analysis_tasks else get_message(MK.NONE, current_lang)
        
        analysis_info = get_message(MK.ANALYSIS_INFO, current_lang, 
                                    response_models_str, 
                                    analysis_only_models_str, 
                                    analysis_tasks_str)
        st.info(analysis_info)

        # 상태 관리를 위한 키 설정
        if "selected_analyses" not in st.session_state:
            st.session_state.selected_analyses = None
        
        # 동적으로 현재 언어에 맞는 분석 제목 생성
        analyses_with_metadata = []
        title_to_key_map = {}
        key_to_title_map = {}

        for key, analysis_data in result["comparison"]["ai_analyses"].items():
            if isinstance(analysis_data, dict) and "metadata" in analysis_data:
                # 현재 언어로 동적 제목 생성
                display_title = middleware.generate_display_title_from_metadata(analysis_data["metadata"], current_lang)
                analyses_with_metadata.append({
                    "original_key": key,         # 원래 결과 접근용 키
                    "display_title": display_title,  # 현재 언어 표시 텍스트
                    "metadata": analysis_data["metadata"]
                })
                # 매핑 추가
                title_to_key_map[display_title] = key
                key_to_title_map[key] = display_title
            else:
                # 메타데이터가 없는 경우 원래 키 사용
                analyses_with_metadata.append({
                    "original_key": key,
                    "display_title": key,
                })
                title_to_key_map[key] = key
                key_to_title_map[key] = key

        # 현재 언어 기준으로 정렬된 표시 제목들
        display_titles = [item["display_title"] for item in analyses_with_metadata]

        # 자체 분석과 교차 분석 구분 (현재 언어 기준)
        self_analysis_text = get_message(MK.SELF_ANALYSIS_LABEL, current_lang)
        self_analyses = [title for title in display_titles if self_analysis_text in title]
        external_analyses = [title for title in display_titles if self_analysis_text not in title]

        # 초기 값 설정 - 저장된 키를 현재 언어 제목으로 변환
        if "selected_analyses" not in st.session_state or st.session_state.selected_analyses is None:
            # None일 경우 모든 분석 항목 선택
            st.session_state.selected_analyses = [item["original_key"] for item in analyses_with_metadata]
            selected_display_titles = display_titles
        else:
            # 기존에 선택된 키들을 현재 언어의 표시 제목으로 변환
            selected_display_titles = [key_to_title_map.get(key, key) for key in st.session_state.selected_analyses if key in key_to_title_map]

        # 선택 목록이 비어있는 경우 모든 항목 선택
        if not selected_display_titles:
            selected_display_titles = display_titles

        # 설명 추가
        st.info(get_message(MK.SELECT_ANALYSIS_HELP, current_lang))

        col1, col2 = st.columns([3, 1])
        
        # 수정된 콜백 함수
        def update_selections():
            # 선택된 표시 제목들을 원래 키로 변환하여 저장
            st.session_state.selected_analyses = [title_to_key_map.get(title, title) for title in st.session_state.analysis_multiselect]

        def show_all_analyses():
            st.session_state.analysis_multiselect = display_titles
            st.session_state.selected_analyses = [item["original_key"] for item in analyses_with_metadata]

        with col1:
            # 현재 언어의 표시 제목으로 선택 UI 제공
            selected_display_titles = st.multiselect(
                get_message(MK.SELECT_ANALYSIS, current_lang),
                options=display_titles,
                default=selected_display_titles,
                key="analysis_multiselect",
                on_change=update_selections
            )

        with col2:
            # 모든 분석 복원 버튼
            st.button(get_message(MK.SHOW_ALL_ANALYSES, current_lang), 
                    type="primary", 
                    key="show_all_button", 
                    on_click=show_all_analyses)

        # selected_display_titles가 비어있을 때 처리
        if not selected_display_titles:
            st.warning(get_message(MK.NO_ANALYSIS_SELECTED, current_lang))
            st.button(get_message(MK.RESTORE_ALL_ANALYSES, current_lang), key="restore_all_analyses", type="primary", on_click=show_all_analyses)

        # 자체 분석 섹션 - 동적 제목과 키 매핑 사용
        self_analysis_text = get_message(MK.SELF_ANALYSIS_LABEL, current_lang)  # 현재 언어로 필터링
        self_titles_selected = [title for title in selected_display_titles if self_analysis_text in title]
        if self_titles_selected:
            st.subheader(get_message(MK.SELF_ANALYSIS_RESULTS, current_lang))
            for display_title in self_titles_selected:
                # 표시 제목을 원래 키로 변환하여 데이터 접근
                original_key = title_to_key_map.get(display_title, display_title)
                if original_key in result["comparison"]["ai_analyses"]:
                    analysis_data = result["comparison"]["ai_analyses"][original_key]
                    
                    # 신규 형식: 메타데이터가 있는 경우
                    if isinstance(analysis_data, dict) and "metadata" in analysis_data:
                        analysis = analysis_data
                        
                        # 이미 생성된 표시 제목을 사용 (동적 재생성은 불필요)
                        with st.expander(f"{display_title}", expanded=True):
                            if analysis.get("has_error", False):
                                st.error(analysis.get("error_text", "Error"))
                            else:
                                st.subheader(get_message(MK.IMPROVEMENT_SUGGESTIONS, current_lang))
                                if analysis.get("improvements", []):
                                    for i, item in enumerate(analysis["improvements"], 1):
                                        st.write(f"{i}. {item}")
                                else:
                                    st.write(get_message(MK.NO_IMPROVEMENT_SUGGESTIONS, current_lang))
                                
                                st.subheader(get_message(MK.MISSING_INFORMATION, current_lang))
                                if analysis.get("missing_information", []):
                                    for i, item in enumerate(analysis["missing_information"], 1):
                                        st.write(f"{i}. {item}")
                                else:
                                    st.write(get_message(MK.NO_MISSING_INFORMATION, current_lang))
                                
                                st.subheader(get_message(MK.FOLLOW_UP_SUGGESTIONS, current_lang))
                                if analysis.get("follow_up_questions", []):
                                    for i, item in enumerate(analysis["follow_up_questions"], 1):
                                        st.write(f"{i}. {item}")
                                else:
                                    st.write(get_message(MK.NO_FOLLOW_UP_SUGGESTIONS, current_lang))
                    else:
                        # 기존 형식 호환성 유지
                        analysis = analysis_data
                        with st.expander(f"{display_title}", expanded=True):
                            error_text = get_message(MK.ERROR_TEXT, current_lang)
                            if isinstance(analysis, str) and error_text in analysis:
                                st.error(analysis)
                            else:
                                st.subheader(get_message(MK.IMPROVEMENT_SUGGESTIONS, current_lang))
                                if analysis["improvements"]:
                                    for i, item in enumerate(analysis["improvements"], 1):
                                        st.write(f"{i}. {item}")
                                else:
                                    st.write(get_message(MK.NO_IMPROVEMENT_SUGGESTIONS, current_lang))
                                
                                st.subheader(get_message(MK.MISSING_INFORMATION, current_lang))
                                if analysis["missing_information"]:
                                    for i, item in enumerate(analysis["missing_information"], 1):
                                        st.write(f"{i}. {item}")
                                else:
                                    st.write(get_message(MK.NO_MISSING_INFORMATION, current_lang))
                                
                                st.subheader(get_message(MK.FOLLOW_UP_SUGGESTIONS, current_lang))
                                if analysis["follow_up_questions"]:
                                    for i, item in enumerate(analysis["follow_up_questions"], 1):
                                        st.write(f"{i}. {item}")
                                else:
                                    st.write(get_message(MK.NO_FOLLOW_UP_SUGGESTIONS, current_lang))

        # 외부 분석 섹션 - 동적 제목과 키 매핑 사용
        self_analysis_text = get_message(MK.SELF_ANALYSIS_LABEL, current_lang)  # 현재 언어로 필터링
        external_titles_selected = [title for title in selected_display_titles if self_analysis_text not in title]
        if external_titles_selected:
            st.subheader(get_message(MK.EXTERNAL_ANALYSIS_RESULTS, current_lang))
            for display_title in external_titles_selected:
                # 표시 제목을 원래 키로 변환하여 데이터 접근
                original_key = title_to_key_map.get(display_title, display_title)
                if original_key in result["comparison"]["ai_analyses"]:
                    analysis_data = result["comparison"]["ai_analyses"][original_key]
                    
                    # 신규 형식: 메타데이터가 있는 경우
                    if isinstance(analysis_data, dict) and "metadata" in analysis_data:
                        analysis = analysis_data
                        
                        # 이미 생성된 표시 제목을 사용 (동적 재생성은 불필요)
                        with st.expander(f"{display_title}", expanded=True):
                            if analysis.get("has_error", False):
                                st.error(analysis.get("error_text", "Error"))
                            else:
                                st.subheader(get_message(MK.IMPROVEMENT_SUGGESTIONS, current_lang))
                                if analysis.get("improvements", []):
                                    for i, item in enumerate(analysis["improvements"], 1):
                                        st.write(f"{i}. {item}")
                                else:
                                    st.write(get_message(MK.NO_IMPROVEMENT_SUGGESTIONS, current_lang))
                                
                                st.subheader(get_message(MK.MISSING_INFORMATION, current_lang))
                                if analysis.get("missing_information", []):
                                    for i, item in enumerate(analysis["missing_information"], 1):
                                        st.write(f"{i}. {item}")
                                else:
                                    st.write(get_message(MK.NO_MISSING_INFORMATION, current_lang))
                                
                                st.subheader(get_message(MK.FOLLOW_UP_SUGGESTIONS, current_lang))
                                if analysis.get("follow_up_questions", []):
                                    for i, item in enumerate(analysis["follow_up_questions"], 1):
                                        st.write(f"{i}. {item}")
                                else:
                                    st.write(get_message(MK.NO_FOLLOW_UP_SUGGESTIONS, current_lang))
                    else:
                        # 기존 형식 호환성 유지
                        analysis = analysis_data
                        with st.expander(f"{display_title}", expanded=True):
                            error_text = get_message(MK.ERROR_TEXT, current_lang)
                            if isinstance(analysis, str) and error_text in analysis:
                                st.error(analysis)
                            else:
                                st.subheader(get_message(MK.IMPROVEMENT_SUGGESTIONS, current_lang))
                                if analysis["improvements"]:
                                    for i, item in enumerate(analysis["improvements"], 1):
                                        st.write(f"{i}. {item}")
                                else:
                                    st.write(get_message(MK.NO_IMPROVEMENT_SUGGESTIONS, current_lang))
                                
                                st.subheader(get_message(MK.MISSING_INFORMATION, current_lang))
                                if analysis["missing_information"]:
                                    for i, item in enumerate(analysis["missing_information"], 1):
                                        st.write(f"{i}. {item}")
                                else:
                                    st.write(get_message(MK.NO_MISSING_INFORMATION, current_lang))
                                
                                st.subheader(get_message(MK.FOLLOW_UP_SUGGESTIONS, current_lang))
                                if analysis["follow_up_questions"]:
                                    for i, item in enumerate(analysis["follow_up_questions"], 1):
                                        st.write(f"{i}. {item}")
                                else:
                                    st.write(get_message(MK.NO_FOLLOW_UP_SUGGESTIONS, current_lang))

    # 후속 질문 탭도 동적 제목 생성 적용
    with tabs[3 if not is_initial_only or has_improved_types else 2]:
        st.header(get_message(MK.FOLLOW_UP_QUESTIONS, current_lang))
        
        # 두 열 생성
        col1, col2 = st.columns(2)
        
        # 접미사 가져오기
        extracted_suffix = get_message(MK.EXTRACTED_QUESTIONS_LABEL, current_lang)
        suggested_suffix = get_message(MK.SUGGESTED_QUESTIONS_LABEL, current_lang)
        improved_response_suffix = get_message(MK.IMPROVED_RESPONSE_SUFFIX, current_lang)
        
        with col1:
            st.subheader(get_message(MK.EXTRACTED_QUESTIONS, current_lang))
            extracted_count = 0
            
            # 개선된 응답에서 추출된 질문만 표시 - 동적 제목 생성 적용
            for original_title, questions_data in result["comparison"]["follow_up_questions"].items():
                # 신규 형식: 메타데이터가 있는 경우
                if isinstance(questions_data, dict) and "metadata" in questions_data:
                    # section_type으로 필터링 - extracted 타입만 추출 질문 섹션에 표시
                    if questions_data.get("section_type") == "extracted":
                        extracted_count += 1
                        
                        # 동적 제목 생성
                        if middleware:
                            metadata = questions_data["metadata"]
                            display_title = middleware.generate_display_title_from_metadata(metadata, current_lang)
                            display_title = f"{display_title}{improved_response_suffix} {extracted_suffix}"
                        else:
                            display_title = original_title
                        
                        with st.expander(f"{display_title}", expanded=True):
                            if questions_data.get("extracted", []):
                                for i, q in enumerate(questions_data["extracted"], 1):
                                    if st.button(f"{q}", key=f"extracted_q_{display_title}_{i}"):
                                        # 이 질문으로 새 쿼리 제출
                                        process_query(q, compare_mode=True, conversation_mode=True)
                                        st.rerun()
                            else:
                                st.write(get_message(MK.NO_EXTRACTED_QUESTIONS, current_lang))
                # 기존 형식 호환성 유지
                elif isinstance(questions_data, dict) and questions_data.get("section_type") == "extracted":
                    extracted_count += 1
                    with st.expander(f"{original_title}", expanded=True):
                        if questions_data.get("extracted", []):
                            for i, q in enumerate(questions_data["extracted"], 1):
                                if st.button(f"{q}", key=f"extracted_q_{original_title}_{i}"):
                                    process_query(q, compare_mode=True, conversation_mode=True)
                                    st.rerun()
                        else:
                            st.write(get_message(MK.NO_EXTRACTED_QUESTIONS, current_lang))
            
            # 추출된 질문이 없을 경우
            if extracted_count == 0:
                st.info(get_message(MK.NO_EXTRACTED_QUESTIONS, current_lang))
        
        with col2:
            st.subheader(get_message(MK.SUGGESTED_QUESTIONS, current_lang))
            suggested_count = 0
            
            # 모든 항목의 제안된 질문 표시 - 동적 제목 생성 적용
            for original_title, questions_data in result["comparison"]["follow_up_questions"].items():
                # 신규 형식: 메타데이터가 있는 경우
                if isinstance(questions_data, dict) and "metadata" in questions_data:
                    # 제안 질문이 있는 항목만 표시
                    if questions_data.get("suggested", []):
                        suggested_count += 1
                        
                        # 동적 제목 생성
                        if middleware:
                            metadata = questions_data["metadata"]
                            display_title = middleware.generate_display_title_from_metadata(metadata, current_lang)
                            display_title = f"{display_title} {suggested_suffix}"
                        else:
                            display_title = original_title
                        
                        with st.expander(f"{display_title}", expanded=True):
                            for i, q in enumerate(questions_data["suggested"], 1):
                                if st.button(f"{q}", key=f"suggested_q_{display_title}_{i}"):
                                    # 이 질문으로 새 쿼리 제출
                                    process_query(q, compare_mode=True, conversation_mode=True)
                                    st.rerun()
                # 기존 형식 호환성 유지
                elif isinstance(questions_data, dict) and questions_data.get("suggested", []):
                    suggested_count += 1
                    with st.expander(f"{original_title}", expanded=True):
                        for i, q in enumerate(questions_data["suggested"], 1):
                            if st.button(f"{q}", key=f"suggested_q_{original_title}_{i}"):
                                process_query(q, compare_mode=True, conversation_mode=True)
                                st.rerun()
            
            # 제안된 질문이 없을 경우
            if suggested_count == 0:
                st.info(get_message(MK.NO_SUGGESTED_QUESTIONS, current_lang))

    # 응답 표시 설정 탭 - 인덱스 조정
    with tabs[4 if not is_initial_only or has_improved_types else 3]:
        st.header(get_message(MK.RESPONSE_DISPLAY_SETTINGS, current_lang))
        st.write(get_message(MK.SELECT_RESPONSE_TYPES_HELP, current_lang))
        
        # 기본 옵션과 상세 옵션 탭 생성
        display_option_tabs = st.tabs([
            get_message(MK.BASIC_OPTIONS, current_lang), 
            get_message(MK.DETAILED_OPTIONS, current_lang)
        ])
        
        # 현재 선택된 표시 유형 가져오기
        current_display_types = st.session_state.get("selected_display_types", ["claude_analyzed_by_openai"])
        if isinstance(current_display_types, str):
            if "," in current_display_types:
                current_display_types = [t.strip() for t in current_display_types.split(",")]
            else:
                current_display_types = [current_display_types]
        
        # 기본 옵션 탭
        with display_option_tabs[0]:
            basic_improvement_display_options = [
                {"id": "claude_analyzed_by_openai", "name": get_message(MK.ResponseTypes.CLAUDE_BY_GPT, current_lang), "description": get_message(MK.ResponseTypes.CLAUDE_BY_GPT_DESC, current_lang)},
                {"id": "openai_analyzed_by_claude", "name": get_message(MK.ResponseTypes.GPT_BY_CLAUDE, current_lang), "description": get_message(MK.ResponseTypes.GPT_BY_CLAUDE_DESC, current_lang)},
                {"id": "all_self_analysis", "name": get_message(MK.ResponseTypes.ALL_SELF_ANALYSIS, current_lang), "description": get_message(MK.ResponseTypes.ALL_SELF_ANALYSIS_DESC, current_lang)},
                {"id": "all", "name": get_message(MK.ResponseTypes.ALL_COMBINATIONS, current_lang), "description": get_message(MK.ResponseTypes.ALL_COMBINATIONS_DESC, current_lang)}
            ]
            
            st.write(get_message(MK.BASIC_DISPLAY_OPTIONS, current_lang))
            
            # 체크박스 상태 초기화
            for option in basic_improvement_display_options:
                option_id = option["id"]
                if option_id not in st.session_state.tab_basic_checkbox_states:
                    # 기본값: current_display_types에 있으면 True, 없으면 False
                    st.session_state.tab_basic_checkbox_states[option_id] = option_id in current_display_types
            
            # 체크박스로 변경하여 다중 선택 지원
            selected_basic_types = []
            for option in basic_improvement_display_options:
                option_id = option["id"]
                
                checkbox_value = st.checkbox(
                    option["name"],
                    value=st.session_state.tab_basic_checkbox_states.get(option_id, False),
                    help=option["description"],
                    key=f"tab_basic_opt_{option_id}"
                )
                
                # 체크박스 상태 업데이트
                st.session_state.tab_basic_checkbox_states[option_id] = checkbox_value
                
                if checkbox_value:
                    selected_basic_types.append(option_id)
            
            # 선택된 것이 없으면 경고 표시
            if not selected_basic_types:
                st.warning(get_message(MK.MIN_ONE_SELECTION, current_lang))
            
            # 적용 버튼
            if st.button(get_message(MK.APPLY_DISPLAY_SETTINGS, current_lang), key="tab_apply_basic_option", type="primary"):
                if not selected_basic_types:
                    st.error(get_message(MK.MIN_ONE_SELECTION, current_lang))
                else:
                    st.session_state.selected_display_types = selected_basic_types
                    st.session_state.display_improvement_types = selected_basic_types
                    
                    # 사이드바 체크박스 상태도 업데이트 (추가된 부분)
                    for option_id in st.session_state.sidebar_checkbox_states:
                        st.session_state.sidebar_checkbox_states[option_id] = option_id in selected_basic_types
                    
                    st.success(get_message(MK.DISPLAY_SETTINGS_SAVED, current_lang, len(selected_basic_types)))
                    st.rerun()
        
        # 상세 옵션 탭
        with display_option_tabs[1]:
            # 상세 옵션 - 3개 그룹으로 분류
            detailed_improvement_display_options = [
                # 자체 분석 옵션
                {"id": "claude_analyzed_by_self", "name": get_message(MK.ResponseTypes.CLAUDE_SELF, current_lang), 
                "description": get_message(MK.ResponseTypes.CLAUDE_SELF_DESC, current_lang), "group": 1}, 
                {"id": "openai_analyzed_by_self", "name": get_message(MK.ResponseTypes.GPT_SELF, current_lang), 
                "description": get_message(MK.ResponseTypes.GPT_SELF_DESC, current_lang), "group": 1}, 
                {"id": "perplexity_analyzed_by_self", "name": get_message(MK.ResponseTypes.PERPLEXITY_SELF, current_lang), 
                "description": get_message(MK.ResponseTypes.PERPLEXITY_SELF_DESC, current_lang), "group": 1}, 
                
                # 단일 외부 분석 옵션
                {"id": "claude_analyzed_by_openai", "name": get_message(MK.ResponseTypes.CLAUDE_BY_GPT, current_lang), 
                "description": get_message(MK.ResponseTypes.CLAUDE_BY_GPT_DESC, current_lang), "group": 2}, 
                {"id": "claude_analyzed_by_perplexity", "name": get_message(MK.ResponseTypes.CLAUDE_BY_PERPLEXITY, current_lang), 
                "description": get_message(MK.ResponseTypes.CLAUDE_BY_PERPLEXITY_DESC, current_lang), "group": 2}, 
                {"id": "openai_analyzed_by_claude", "name": get_message(MK.ResponseTypes.GPT_BY_CLAUDE, current_lang), 
                "description": get_message(MK.ResponseTypes.GPT_BY_CLAUDE_DESC, current_lang), "group": 2}, 
                {"id": "openai_analyzed_by_perplexity", "name": get_message(MK.ResponseTypes.GPT_BY_PERPLEXITY, current_lang), 
                "description": get_message(MK.ResponseTypes.GPT_BY_PERPLEXITY_DESC, current_lang), "group": 2},  
                {"id": "perplexity_analyzed_by_claude", "name": get_message(MK.ResponseTypes.PERPLEXITY_BY_CLAUDE, current_lang), 
                "description": get_message(MK.ResponseTypes.PERPLEXITY_BY_CLAUDE_DESC, current_lang), "group": 2}, 
                {"id": "perplexity_analyzed_by_openai", "name": get_message(MK.ResponseTypes.PERPLEXITY_BY_GPT, current_lang), 
                "description": get_message(MK.ResponseTypes.PERPLEXITY_BY_GPT_DESC, current_lang), "group": 2}, 
                
                # 다중 분석자 옵션
                {"id": "claude_analyzed_by_multiple", "name": get_message(MK.ResponseTypes.CLAUDE_BY_MULTIPLE, current_lang), 
                "description": get_message(MK.ResponseTypes.CLAUDE_BY_MULTIPLE_DESC, current_lang), "group": 3}, 
                {"id": "openai_analyzed_by_multiple", "name": get_message(MK.ResponseTypes.GPT_BY_MULTIPLE, current_lang), 
                "description": get_message(MK.ResponseTypes.GPT_BY_MULTIPLE_DESC, current_lang), "group": 3}, 
                {"id": "perplexity_analyzed_by_multiple", "name": get_message(MK.ResponseTypes.PERPLEXITY_BY_MULTIPLE, current_lang), 
                "description": get_message(MK.ResponseTypes.PERPLEXITY_BY_MULTIPLE_DESC, current_lang), "group": 3}, 
                
                # 초기 응답 옵션
                {"id": "claude_initial", "name": get_message(MK.ResponseTypes.CLAUDE_INITIAL, current_lang), 
                "description": get_message(MK.ResponseTypes.CLAUDE_INITIAL_DESC, current_lang), "group": 4}, 
                {"id": "openai_initial", "name": get_message(MK.ResponseTypes.GPT_INITIAL, current_lang), 
                "description": get_message(MK.ResponseTypes.GPT_INITIAL_DESC, current_lang), "group": 4}, 
                {"id": "perplexity_initial", "name": get_message(MK.ResponseTypes.PERPLEXITY_INITIAL, current_lang), 
                "description": get_message(MK.ResponseTypes.PERPLEXITY_INITIAL_DESC, current_lang), "group": 4}, 
                
                # 모든 응답 옵션
                {"id": "all", "name": get_message(MK.ResponseTypes.ALL_COMBINATIONS, current_lang), 
                "description": get_message(MK.ResponseTypes.ALL_COMBINATIONS_DESC, current_lang), "group": 0} 
            ]
            
            # 체크박스 상태 초기화
            for option in detailed_improvement_display_options:
                option_id = option["id"]
                if option_id not in st.session_state.tab_detailed_checkbox_states:
                    # 기본값: current_display_types에 있으면 True, 없으면 False
                    st.session_state.tab_detailed_checkbox_states[option_id] = option_id in current_display_types
            
            # 옵션을 그룹별로 표시
            selected_detailed_types = []

            # 그룹별로 옵션 분류
            group1_options = [opt for opt in detailed_improvement_display_options if opt["group"] == 1]
            group2_options = [opt for opt in detailed_improvement_display_options if opt["group"] == 2]
            group3_options = [opt for opt in detailed_improvement_display_options if opt["group"] == 3]
            group4_options = [opt for opt in detailed_improvement_display_options if opt["group"] == 4]
            group0_options = [opt for opt in detailed_improvement_display_options if opt["group"] == 0]

            # 그룹 1: 자체 분석 옵션
            st.write(get_message(MK.RESPONSE_TYPE_SELF_ANALYSIS, current_lang))
            for option in group1_options:
                option_id = option["id"]
                
                checkbox_value = st.checkbox(
                    option["name"],
                    value=st.session_state.tab_detailed_checkbox_states.get(option_id, False),
                    help=option["description"],
                    key=f"tab_detailed_opt1_{option_id}"
                )
                
                # 체크박스 상태 업데이트
                st.session_state.tab_detailed_checkbox_states[option_id] = checkbox_value
                
                if checkbox_value:
                    selected_detailed_types.append(option_id)

            st.divider()

            # 그룹 2: 외부 분석 옵션
            st.write(get_message(MK.RESPONSE_TYPE_EXTERNAL_ANALYSIS, current_lang))
            col1, col2 = st.columns(2)
            half_len = len(group2_options) // 2
            with col1:
                for option in group2_options[:half_len]:
                    option_id = option["id"]
                    
                    checkbox_value = st.checkbox(
                        option["name"],
                        value=st.session_state.tab_detailed_checkbox_states.get(option_id, False),
                        help=option["description"],
                        key=f"tab_detailed_opt2a_{option_id}"
                    )
                    
                    # 체크박스 상태 업데이트
                    st.session_state.tab_detailed_checkbox_states[option_id] = checkbox_value
                    
                    if checkbox_value:
                        selected_detailed_types.append(option_id)
                        
            with col2:
                for option in group2_options[half_len:]:
                    option_id = option["id"]
                    
                    checkbox_value = st.checkbox(
                        option["name"],
                        value=st.session_state.tab_detailed_checkbox_states.get(option_id, False),
                        help=option["description"],
                        key=f"tab_detailed_opt2b_{option_id}"
                    )
                    
                    # 체크박스 상태 업데이트
                    st.session_state.tab_detailed_checkbox_states[option_id] = checkbox_value
                    
                    if checkbox_value:
                        selected_detailed_types.append(option_id)

            st.divider()

            # 그룹 3: 다중 분석자 옵션
            st.write(get_message(MK.RESPONSE_TYPE_MULTIPLE_ANALYZERS, current_lang))
            for option in group3_options:
                option_id = option["id"]
                
                checkbox_value = st.checkbox(
                    option["name"],
                    value=st.session_state.tab_detailed_checkbox_states.get(option_id, False),
                    help=option["description"],
                    key=f"tab_detailed_opt3_{option_id}"
                )
                
                # 체크박스 상태 업데이트
                st.session_state.tab_detailed_checkbox_states[option_id] = checkbox_value
                
                if checkbox_value:
                    selected_detailed_types.append(option_id)

            st.divider()

            # 그룹 4: 초기 응답 옵션
            st.write(get_message(MK.RESPONSE_TYPE_INITIAL_RESPONSE, current_lang))
            for option in group4_options:
                option_id = option["id"]
                
                checkbox_value = st.checkbox(
                    option["name"],
                    value=st.session_state.tab_detailed_checkbox_states.get(option_id, False),
                    help=option["description"],
                    key=f"tab_detailed_opt4_{option_id}"
                )
                
                # 체크박스 상태 업데이트
                st.session_state.tab_detailed_checkbox_states[option_id] = checkbox_value
                
                if checkbox_value:
                    selected_detailed_types.append(option_id)

            # 모든 응답 옵션
            st.divider()
            for option in group0_options:
                option_id = option["id"]
                
                checkbox_value = st.checkbox(
                    option["name"],
                    value=st.session_state.tab_detailed_checkbox_states.get(option_id, False),
                    help=option["description"],
                    key=f"tab_detailed_opt0_{option_id}"
                )
                
                # 체크박스 상태 업데이트
                st.session_state.tab_detailed_checkbox_states[option_id] = checkbox_value
                
                if checkbox_value:
                    selected_detailed_types.append(option_id)

            # 선택된 것이 없으면 경고 표시
            if not selected_detailed_types:
                st.warning(get_message(MK.MIN_ONE_SELECTION, current_lang))
            
            # 선택한 옵션들의 설명 표시
            if selected_detailed_types:
                st.subheader(get_message(MK.SELECTED_OPTIONS, current_lang))
                for display_type in selected_detailed_types:
                    selected_option = next((option for option in detailed_improvement_display_options if option["id"] == display_type), None)
                    if selected_option:
                        st.info(f"**{selected_option['name']}**: {selected_option['description']}")
            
            # 적용 버튼
            if st.button(get_message(MK.APPLY_DISPLAY_SETTINGS, current_lang), key="tab_apply_detailed_option", type="primary"):
                if not selected_detailed_types:
                    st.error(get_message(MK.MIN_ONE_SELECTION, current_lang))
                else:
                    st.session_state.selected_display_types = selected_detailed_types
                    st.session_state.display_improvement_types = selected_detailed_types
                    
                    # 사이드바 체크박스 상태도 업데이트 (추가된 부분)
                    for option_id in st.session_state.sidebar_checkbox_states:
                        st.session_state.sidebar_checkbox_states[option_id] = option_id in selected_detailed_types
                    
                    st.success(get_message(MK.DISPLAY_SETTINGS_SAVED, current_lang, len(selected_detailed_types)))
                    st.rerun()
        
        # 현재 선택된 응답 표시 유형 정보
        st.divider()
        st.subheader(get_message(MK.CURRENT_SETTINGS, current_lang))
        st.write(get_message(MK.SELECTED_DISPLAY_TYPES, current_lang, ', '.join(current_display_types)))
        st.write(get_message(MK.SETTINGS_APPLY_NEXT, current_lang))

    # 처리 정보
    st.write("---")
    st.write(get_message(MK.PROCESSING_TIME, current_lang, f"{result['processing_info']['time']:.2f}"))
    cost = result['processing_info']['token_usage']['combined']['estimated_cost']
    st.write(get_message(MK.TOTAL_COST_VALUE, current_lang, f"{cost:.4f}")) # "총 비용: ${cost:.4f}"

    # 사용된 응답 표시 유형 정보 추가
    # 현재 세션에 저장된 선택된 유형 먼저 확인
    display_types = st.session_state.selected_display_types
    if not display_types:  # 세션에 없는 경우 결과에서 가져오기
        display_types = result.get("comparison", {}).get("display_improvement_types", "claude_analyzed_by_openai")
    if isinstance(display_types, str) and "," in display_types:
        display_types_list = [t.strip() for t in display_types.split(",")]
        st.write(get_message(MK.DISPLAY_TYPES_USED_COUNT, current_lang, len(display_types_list))) # "사용된 응답 표시 유형 ({len(display_types_list)}개)
        for i, display_type in enumerate(display_types_list, 1):
            st.write(f"{i}. {display_type}")
    else:
        st.write(get_message(MK.DISPLAY_TYPES_USED, current_lang, display_types)) # "사용된 응답 표시 유형: {display_types}"

    # 분석 작업 정보 표시
    analysis_tasks = result.get("comparison", {}).get("analysis_tasks", [])
    if analysis_tasks:
        st.write(get_message(MK.ANALYSIS_TASKS_PERFORMED, current_lang, len(analysis_tasks))) # "수행된 분석 작업 ({len(analysis_tasks)}개):"
        for i, task in enumerate(analysis_tasks, 1):
            st.write(f"{i}. {task}")

def save_results():
    """현재 결과 저장"""
    current_lang = st.session_state.get("language", "ko")
    
    if not st.session_state.middleware:
        middleware_error = get_message(MK.MIDDLEWARE_NOT_INITIALIZED, current_lang)
        st.error(get_message(MK.ERROR_OCCURRED, current_lang, middleware_error)) # "미들웨어가 초기화되지 않았습니다."
        return
    
    try:
        st.session_state.middleware.save_usage_data()
        
        # 쿼리 기록 저장
        if st.session_state.query_history:
            if not os.path.exists("logs"):
                os.makedirs("logs")
            
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/query_history_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.query_history, f, ensure_ascii=False, indent=2)
            
            # 대화 기록 저장
            conversation_filename = f"logs/conversation_{timestamp}.json"
            with open(conversation_filename, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.conversation_history, f, ensure_ascii=False, indent=2)
            
            st.success(get_message(MK.RESULTS_SAVED, current_lang, filename))
    except Exception as e:
        st.error(get_message(MK.ERROR_OCCURRED, current_lang, str(e)))

def run_streamlit_app():
    """Streamlit 앱 실행"""
    # 언어 설정을 먼저 가져오기
    current_lang = st.session_state.get("language", "ko")
    
    st.set_page_config(
        page_title=get_message(MK.APP_TITLE, current_lang), # "Enhanced AI 응답 시스템"
        page_icon="🤖",
        layout="wide"
    )
    
    # 세션 상태 초기화
    init_session_state()
    current_lang = st.session_state.get("language", "ko")
    
    # 다국어 지원 적용 (get_message 함수 활용)
    st.title(get_message(MK.APP_TITLE, current_lang))
    st.subheader(get_message(MK.APP_SUBTITLE, current_lang))
    
    # 사이드바 설정
    with st.sidebar:
        st.header(get_message(MK.SETTINGS, current_lang))
        
        # 기본 설정
        budget = st.number_input(get_message(MK.BUDGET_LIMIT, current_lang), min_value=0.0, value=5.0, step=0.5)
        
        # Perplexity는 항상 사용 가능하도록 설정
        st.session_state.perplexity_enabled = True
        st.session_state.budget = budget
            
        # 미들웨어가 초기화되지 않은 경우 자동 초기화
        if not st.session_state.middleware:
            initialized = initialize_middleware()
            if initialized:
                st.success(get_message(MK.MIDDLEWARE_INITIALIZED, current_lang))
        
        # 개선 응답 유형 설정 UI 표시
        display_improvement_settings_sidebar()
        
        # 토큰 사용량 표시
        st.header(get_message(MK.TOKEN_USAGE, current_lang))
        display_token_usage()
                
        # 결과 저장 버튼
        if st.button(get_message(MK.SAVE_RESULTS, current_lang)):
            save_results()

        # 고급 설정 섹션
        with st.expander(get_message(MK.ADVANCED_SETTINGS, current_lang), expanded=False):
            st.info(get_message(MK.ADVANCED_SETTINGS_WARNING, current_lang))
            
            # 두 열로 UI 구성
            col1, col2 = st.columns(2)
            
            with col1:
                # API 서비스 섹션
                st.subheader(get_message(MK.API_SERVICES, current_lang))
                
                # 미들웨어 초기화 버튼
                if st.button(get_message(MK.MIDDLEWARE_INIT, current_lang), help=get_message(MK.MIDDLEWARE_INIT_HELP, current_lang)): # "API 연결 문제가 있을 때 사용하세요"):
                    initialized = initialize_middleware()
                    if initialized:
                        st.success(get_message(MK.MIDDLEWARE_INIT_SUCCESS, current_lang)) # "미들웨어가 성공적으로 초기화되었습니다.")

                # API 상태 확인 버튼
                if st.button(get_message(MK.CHECK_API_STATUS, current_lang), help=get_message(MK.CHECK_API_STATUS_HELP, current_lang)): # "각 API 서비스의 연결 상태를 확인합니다"):
                    if st.session_state.middleware:
                        api_status = {}
                        
                        # Claude API 확인
                        try:
                            claude_client = st.session_state.middleware.claude_client
                            if claude_client:
                                # 간단한 API 호출로 연결 확인
                                claude_client.messages.create(
                                    model=ACTIVE_CLAUDE_MODEL,
                                    max_tokens=10,
                                    messages=[{"role": "user", "content": "Hello"}]
                                )
                                api_status["Claude API"] = get_message(MK.API_STATUS_NORMAL, current_lang) # "정상"
                            else:
                                api_status["Claude API"] = get_message(MK.API_STATUS_NOT_INITIALIZED, current_lang) # "초기화되지 않음"
                        except Exception as e:
                            api_status["Claude API"] = f"{get_message(MK.API_STATUS_ERROR, current_lang)} {str(e)[:50]}..." # f"오류: {str(e)[:50]}..."
                        
                        # OpenAI API 확인
                        try:
                            openai_client = st.session_state.middleware.openai_client
                            if openai_client:
                                openai_client.chat.completions.create(
                                    model=ACTIVE_GPT_MODEL,
                                    messages=[{"role": "user", "content": "Hello"}]
                                )
                                api_status["OpenAI API"] = get_message(MK.API_STATUS_NORMAL, current_lang) # "정상"
                            else:
                                api_status["OpenAI API"] = get_message(MK.API_STATUS_NOT_INITIALIZED, current_lang) # "초기화되지 않음"
                        except Exception as e:
                            api_status["OpenAI API"] = f"{get_message(MK.API_STATUS_ERROR, current_lang)} {str(e)[:50]}..." # f"오류: {str(e)[:50]}..."
                        
                        # Perplexity API 확인
                        try:
                            perplexity_client = st.session_state.middleware.perplexity_client
                            if perplexity_client:
                                perplexity_client.chat.completions.create(
                                    model=ACTIVE_PERPLEXITY_MODEL,
                                    messages=[{"role": "user", "content": "Hello"}]
                                )
                                api_status["Perplexity API"] = get_message(MK.API_STATUS_NORMAL, current_lang) # "정상"
                            else:
                                api_status["Perplexity API"] = get_message(MK.API_STATUS_NOT_INITIALIZED, current_lang) # "초기화되지 않음"
                        except Exception as e:
                            api_status["Perplexity API"] = f"{get_message(MK.API_STATUS_ERROR, current_lang)} {str(e)[:50]}..." # f"오류: {str(e)[:50]}..."
                        
                        # 결과 표시
                        for api, status in api_status.items():
                            if "정상" in status:
                                st.success(f"{api}: {status}")
                            elif "초기화" in status:
                                st.warning(f"{api}: {status}")
                            else:
                                st.error(f"{api}: {status}")
                    else:
                        st.error(get_message(MK.MIDDLEWARE_NOT_INITIALIZED, current_lang)) # "미들웨어가 초기화되지 않았습니다."
            
            with col2:
                # 데이터 및 설정 섹션
                st.subheader(get_message(MK.DATA_SETTINGS, current_lang))
                
                # 토큰 사용량 리셋 버튼
                if st.button(get_message(MK.RESET_TOKEN_USAGE, current_lang), help=get_message(MK.RESET_TOKEN_USAGE_HELP, current_lang)): # "현재 세션의 토큰 사용량 기록을 초기화합니다"):
                    if st.session_state.middleware:
                        # 기존 미들웨어 객체 유지하면서 토큰 트래커만 초기화
                        st.session_state.middleware.token_tracker = TokenUsageTracker()
                        st.session_state.token_tracker_initialized = True
                        st.success(get_message(MK.TOKEN_USAGE_RESET, current_lang))
                    else:
                        st.error(get_message(MK.MIDDLEWARE_NOT_INITIALIZED, current_lang)) # "미들웨어가 초기화되지 않았습니다.")

                # 세션 초기화 버튼
                if st.button(get_message(MK.RESET_SESSION, current_lang), help=get_message(MK.RESET_SESSION_HELP, current_lang)): # "모든 대화 내역과 쿼리 기록을 초기화합니다"):
                    st.session_state.conversation_history = []
                    st.session_state.query_history = []
                    st.session_state.query_count = 0
                    st.session_state.current_result = None
                    st.success(get_message(MK.SESSION_RESET, current_lang))
                    st.rerun()
            
            # 모델 설정 섹션
            st.subheader(get_message(MK.MODEL_SETTINGS, current_lang))
            
            # 모델 선택 옵션들
            claude_models = ["claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307", "claude-3-7-sonnet-20250219"]
            gpt_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]
            perplexity_models = ["sonar", "pplx-7b-online", "pplx-70b-online", "mistral-7b", "llama-2-70b"]
            
            # 세 열로 UI 구성
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_claude = st.selectbox(
                    get_message(MK.CLAUDE_MODEL, current_lang),
                    options=claude_models,
                    index=claude_models.index(ACTIVE_CLAUDE_MODEL) if ACTIVE_CLAUDE_MODEL in claude_models else 0,
                    help=get_message(MK.CLAUDE_MODEL_HELP, current_lang) # "사용할 Claude 모델을 선택하세요"
                )
            
            with col2:
                selected_gpt = st.selectbox(
                    get_message(MK.GPT_MODEL, current_lang),
                    options=gpt_models,
                    index=gpt_models.index(ACTIVE_GPT_MODEL) if ACTIVE_GPT_MODEL in gpt_models else 0,
                    help=get_message(MK.GPT_MODEL_HELP, current_lang) # "사용할 GPT 모델을 선택하세요"
                )
            
            with col3:
                selected_perplexity = st.selectbox(
                    get_message(MK.PERPLEXITY_MODEL, current_lang),
                    options=perplexity_models,
                    index=perplexity_models.index(ACTIVE_PERPLEXITY_MODEL) if ACTIVE_PERPLEXITY_MODEL in perplexity_models else 0,
                    help=get_message(MK.PERPLEXITY_MODEL_HELP, current_lang) # "사용할 Perplexity 모델을 선택하세요"
                )
            
            # 모델 설정 적용 버튼
            if st.button(get_message(MK.APPLY_MODEL_SETTINGS, current_lang), help=get_message(MK.APPLY_MODEL_SETTINGS_HELP, current_lang)): # "선택한 모델 설정을 적용합니다"
                # 세션 상태에 선택된 모델 저장
                st.session_state.custom_claude_model = selected_claude
                st.session_state.custom_gpt_model = selected_gpt
                st.session_state.custom_perplexity_model = selected_perplexity
                
                # 미들웨어 다시 초기화
                st.success(get_message(MK.MODEL_SETTINGS_APPLIED, current_lang)) # "모델 설정이 적용되었습니다. 미들웨어를 초기화합니다..."
                initialized = initialize_middleware()
                if initialized:
                    st.success(get_message(MK.MIDDLEWARE_INIT_SUCCESS, current_lang)) # "미들웨어가 성공적으로 초기화되었습니다."
                st.rerun()
            
            # 시스템 정보 표시
            st.subheader(get_message(MK.SYSTEM_INFO, current_lang))
            st.text(get_message(MK.APP_START_TIME, current_lang, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            middleware_status = get_message(MK.MIDDLEWARE_STATUS_INITIALIZED, current_lang) if st.session_state.middleware else get_message(MK.MIDDLEWARE_STATUS_NOT_INITIALIZED, current_lang) # "초기화됨" if st.session_state.middleware else "초기화되지 않음"
            st.text(get_message(MK.MIDDLEWARE_STATUS, current_lang, middleware_status))
            st.text(get_message(MK.STREAMLIT_VERSION, current_lang, st.__version__))

        # 사이드바 하단에 언어 설정 배치
        with st.sidebar:
            st.divider()  # 구분선 추가
            st.subheader(get_message(MK.SYSTEM_SETTINGS, current_lang))
            
            # 언어 선택 옵션
            selected_language = st.selectbox(
                get_message(MK.LANGUAGE, current_lang),
                options=["ko", "en"],
                format_func=lambda x: "한국어" if x == "ko" else "English",
                index=0 if current_lang == "ko" else 1
            )
            
            # 언어 변경 감지
            if selected_language != current_lang:
                st.session_state.language = selected_language
                # 미들웨어 언어 설정 동기화 추가
                if st.session_state.middleware:
                    st.session_state.middleware.language = selected_language
                st.rerun()

    # 대화 기록 표시 (항상 최상단에 표시)
    if st.session_state.conversation_history:
        st.header(get_message(MK.CONVERSATION_HISTORY, current_lang))
        display_conversation_history()
    
    # 현재 쿼리 결과가 있는 경우 표시
    if st.session_state.current_result:
        # 쿼리 기록이 있다면 마지막 쿼리 가져오기
        if st.session_state.query_history:
            last_query = st.session_state.query_history[-1]["query"]
        else:
            last_query = get_message(MK.PREVIOUS_QUESTION, current_lang) # "이전 질문"
            
        st.header(get_message(MK.CURRENT_QUERY_RESULTS, current_lang))
        display_result(st.session_state.current_result, last_query)
    
    # 쿼리 기록 표시
    if st.session_state.query_history and len(st.session_state.query_history) > 1:
        st.header(get_message(MK.PREVIOUS_QUERY_HISTORY, current_lang))
        display_query_history()
    
    # 메인 입력 영역 (항상 최하단에 표시)
    st.header(get_message(MK.NEW_QUESTION, current_lang))
    
    # 대화 모드인 경우 문구 변경
    if st.session_state.conversation_mode:
        input_label = get_message(MK.QUESTION_OR_INSTRUCTION_INPUT, current_lang) # "질문 또는 지시사항 입력:"
        submit_label = get_message(MK.CONTINUE_CONVERSATION, current_lang)
    else:
        input_label = get_message(MK.QUESTION_INPUT, current_lang)
        submit_label = get_message(MK.SUBMIT_QUESTION, current_lang)
        
    user_input = st.text_area(input_label, height=150)
    
    # 제출 버튼
    submit_button = st.button(submit_label)
    
    # 쿼리 처리
    if submit_button and user_input and st.session_state.middleware:
        # 항상 비교 모드 활성화 (옵션 제거)
        use_compare = True
        
        # 쿼리 처리 및 결과 표시
        result = process_query(
            user_input, 
            compare_mode=use_compare, 
            conversation_mode=st.session_state.conversation_mode
        )
        
        if result:
            # 페이지 새로고침 (UI 업데이트를 위해)
            st.rerun()

if __name__ == "__main__":
    run_streamlit_app()