"""
AI 응답 비교 시스템 메인 스크립트
"""

import anthropic
import openai
import traceback
import json
import argparse
import os
from datetime import datetime

from config import ANTHROPIC_API_KEY, OPENAI_API_KEY
from enhanced_middleware import EnhancedMultiAIMiddleware

def setup_logging():
    """로깅 설정"""
    import logging
    
    # 로그 디렉토리 생성
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    log_file = f"logs/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 로깅 포맷 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("main")

def compare_mode():
    """AI 응답 비교 모드 실행"""
    logger = setup_logging()
    
    try:
        # 명령줄 인수 파싱
        parser = argparse.ArgumentParser(description="AI 응답 비교 시스템")
        parser.add_argument("--budget", type=float, help="사용할 최대 예산 (달러)", default=5.0)
        parser.add_argument("--perplexity", action="store_true", help="Perplexity API 사용 (API 키 필요)")
        parser.add_argument("--perplexity-key", type=str, help="Perplexity API 키")
        args = parser.parse_args()
        
        # API 클라이언트 초기화
        logger.info("API 클라이언트 초기화 중...")
        claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        perplexity_client = None
        if args.perplexity:
            perplexity_key = args.perplexity_key or os.environ.get("PERPLEXITY_API_KEY")
            if perplexity_key:
                # 여기만 수정
                perplexity_client = openai.OpenAI(
                    api_key=perplexity_key,
                    base_url="https://api.perplexity.ai"
                )
                logger.info("Perplexity 클라이언트가 OpenAI 인터페이스로 초기화됨")
            else:
                logger.warning("Perplexity API 키가 없어 Perplexity 기능이 비활성화됩니다.")

        logger.info("미들웨어 초기화 중...")
        middleware = EnhancedMultiAIMiddleware(claude_client, openai_client, perplexity_client)
        
        # 예산 제한 설정
        middleware.set_budget_limit(args.budget)
        logger.info(f"예산 제한: ${args.budget:.2f}")
        
        logger.info("AI 응답 비교 시스템 시작")
        print("\n=== AI 응답 비교 시스템 ===")
        print("- 종료하려면 'exit' 또는 'quit'을 입력하세요.")
        print("- 'usage'를 입력하면 현재 토큰 사용량을 확인할 수 있습니다.")
        print("- 'compare'를 입력하면 모든 AI 응답을 비교합니다.")
        print("- 'standard'를 입력하면 표준 모드로 전환합니다.")
        print("=============================\n")
        
        query_count = 0
        compare_mode = False  # 기본적으로 표준 모드
        
        while True:
            user_input = input("\n질문을 입력하세요: ")
            
            if user_input.lower() in ["exit", "quit"]:
                # 종료 전 사용량 저장
                middleware.save_usage_data()
                print("최종 토큰 사용량 요약:")
                middleware.print_usage_report()
                logger.info("프로그램 종료")
                print("프로그램을 종료합니다.")
                break
            
            elif user_input.lower() == "usage":
                # 현재 사용량 출력
                middleware.print_usage_report()
                continue
            
            elif user_input.lower() == "compare":
                compare_mode = True
                print("비교 모드로 전환되었습니다. 모든 AI 응답을 비교합니다.")
                continue
            
            elif user_input.lower() == "standard":
                compare_mode = False
                print("표준 모드로 전환되었습니다. Claude+GPT 개선 방식을 사용합니다.")
                continue
            
            query_count += 1
            print(f"\n----- 쿼리 #{query_count} -----")
            
            # 처리 모드에 따라 쿼리 처리
            result = middleware.process_query(user_input, show_comparison=compare_mode)

            if "error" in result:
                print(f"\n오류가 발생했습니다: {result['error']}")
            else:
                # 표준 모드 결과 표시
                if not compare_mode:
                    print("\n=== 최종 응답 ===")
                    if "final_response" in result:
                        print(result["final_response"])
                    
                    # 선택된 모델 표시 - 여기에 추가
                    if "selected_models" in result:
                        print("\n=== 선택된 모델 ===")
                        print(", ".join(result["selected_models"]))

                    # 토큰 사용량 표시
                    if "token_usage" in result:
                        print("\n=== 토큰 사용량 ===")
                        usage = result["token_usage"]
                        print(f"Claude: {usage['claude_usage']['total_tokens']:,} 토큰 (${usage['claude_usage']['estimated_cost']:.4f})")
                        print(f"OpenAI: {usage['openai_usage']['total_tokens']:,} 토큰 (${usage['openai_usage']['estimated_cost']:.4f})")
                        if 'perplexity_usage' in usage:
                            print(f"Perplexity: {usage['perplexity_usage']['total_tokens']:,} 토큰 (${usage['perplexity_usage']['estimated_cost']:.4f})")
                        print(f"총 비용: ${usage['combined']['estimated_cost']:.4f}")
                    
                    # 전체 분석 데이터 표시 (선택적)
                    if "full_analysis" in result:
                        print("\n=== 분석 데이터 ===")
                        print(json.dumps(result["full_analysis"], indent=2, ensure_ascii=False))
                    else:
                        print("분석 데이터를 표시할 수 없습니다.")
                
                # 비교 모드 결과 표시
                else:
                    formatted = middleware.format_comparison_result(result)
                    
                    # 초기 응답 표시
                    print("\n=== 초기 응답 ===")
                    for ai, response in formatted["comparison"]["initial_responses"].items():
                        print(f"\n--- {ai} 초기 응답 ---")
                        print(response[:500] + "..." if len(response) > 500 else response)
                    
                    # 개선된 응답 표시
                    print("\n=== 개선된 응답 ===")
                    for ai, response in formatted["comparison"]["improved_responses"].items():
                        print(f"\n--- {ai} 개선 응답 ---")
                        print(response[:500] + "..." if len(response) > 500 else response)
                    
                    # 후속 질문 표시
                    print("\n=== 후속 질문 ===")
                    for ai, questions in formatted["comparison"]["follow_up_questions"].items():
                        if isinstance(questions, dict):
                            print(f"\n--- {ai} 후속 질문 ---")
                            extracted = questions.get("extracted", [])
                            if extracted:
                                for i, q in enumerate(extracted, 1):
                                    print(f"{i}. {q}")
                            else:
                                print("추출된 후속 질문이 없습니다.")
                        else:
                            print(f"\n--- {ai} 후속 질문 ---")
                            print(questions)  # 오류 메시지
                    
                    # 토큰 사용량 표시
                    print("\n=== 토큰 사용량 ===")
                    usage = formatted["processing_info"]["token_usage"]
                    print(f"Claude: {usage['claude_usage']['total_tokens']:,} 토큰 (${usage['claude_usage']['estimated_cost']:.4f})")
                    print(f"OpenAI: {usage['openai_usage']['total_tokens']:,} 토큰 (${usage['openai_usage']['estimated_cost']:.4f})")
                    if 'perplexity_usage' in usage:
                        print(f"Perplexity: {usage['perplexity_usage']['total_tokens']:,} 토큰 (${usage['perplexity_usage']['estimated_cost']:.4f})")
                    print(f"총 비용: ${usage['combined']['estimated_cost']:.4f}")
                
                # 경고 표시
                if "warning" in result:
                    print(f"\n⚠️ 경고: {result['warning']}")
            
            # 5개 쿼리마다 현재 사용량 저장
            if query_count % 5 == 0:
                middleware.save_usage_data()
                print("\n현재까지의 사용량이 저장되었습니다.")
            
    except Exception as e:
        logger.error(f"초기화 중 오류 발생: {str(e)}")
        traceback.print_exc()

def streamlit_ui():
    """Streamlit UI 시작"""
    try:
        import streamlit as st
        from ui.app import run_streamlit_app
        
        run_streamlit_app()
    except ImportError:
        print("Streamlit이 설치되지 않았습니다. 'pip install streamlit'을 실행하여 설치하세요.")
    except Exception as e:
        print(f"UI 시작 중 오류 발생: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI 응답 비교 시스템")
    parser.add_argument("--ui", action="store_true", help="Streamlit UI 시작")
    
    args, remaining_args = parser.parse_known_args()
    
    if args.ui:
        # UI 모드 시작
        streamlit_ui()
    else:
        # 커맨드 라인 비교 모드 시작
        compare_mode()