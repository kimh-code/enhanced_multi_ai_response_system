"""
패키지를 직접 실행할 때 사용되는 엔트리 포인트
python -m enhanced_multi_ai_middleware 형태로 실행 가능
"""

import sys
import argparse

if __name__ == "__main__":
    # 패키지 내부에서 실행될 때와 직접 실행될 때 모두 작동하도록 조정
    try:
        from Script.program.claude_middleware.main import compare_mode, streamlit_ui
    except ImportError:
        from main import compare_mode, streamlit_ui
    
    parser = argparse.ArgumentParser(description="AI 응답 비교 시스템")
    parser.add_argument("--ui", action="store_true", help="Streamlit UI 시작")
    
    args, remaining_args = parser.parse_known_args()
    
    if args.ui:
        streamlit_ui()
    else:
        compare_mode()