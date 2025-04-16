"""
Enhanced Multi-AI Middleware: Claude, GPT, Perplexity 응답 비교 및 분석

이 미들웨어는 여러 AI 모델(Claude, GPT, Perplexity)의 응답을 비교 분석하고,
분석 결과를 바탕으로 각 응답을 개선하여 최적의 결과를 제공합니다.

구조적 개선:
- 명확한 명명 체계: 분석 관계를 직접적으로 표현하는 명명 사용
- AnalysisTask 클래스: 분석 작업을 명확하게 정의하고 실행
- 데이터 구조 개선: 결과 형식 및 메타데이터 구조화
"""

import anthropic
import openai
import json
import time
import re
import traceback
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from utils.token_tracker import TokenUsageTracker
from openai import OpenAI
from messages import MK, get_message

from config import ANTHROPIC_API_KEY, OPENAI_API_KEY, PERPLEXITY_API_KEY, ACTIVE_CLAUDE_MODEL, ACTIVE_GPT_MODEL, ACTIVE_PERPLEXITY_MODEL
from config import get_model_cost

class AnalysisTask:
    """분석 작업을 명확하게 정의하고 실행하는 클래스"""
    
    def __init__(self, analyzer: str, analyzed: str, task_type: str = "external", display_type: str = None):
        """
        분석 작업 초기화
        
        Args:
            analyzer: 분석을 수행하는 모델 ("claude", "openai", "perplexity")
            analyzed: 분석 대상 모델 ("claude", "openai", "perplexity")
            task_type: 분석 유형 ("external" 또는 "self")
            display_type: UI 표시용 식별자 (선택적)
        """
        self.analyzer = analyzer      # 분석을 수행하는 모델
        self.analyzed = analyzed      # 분석 대상 모델
        self.task_type = task_type    # 분석 유형 (external/self)
        self.display_type = display_type  # 표시 식별자
        self.result = None
        self.middleware = None  # 미들웨어 참조 초기화 (나중에 설정됨)
        
    def get_task_id(self) -> str:
        """분석 작업 고유 ID 생성"""
        if self.task_type == "self":
            return f"{self.analyzed}_analyzed_by_self"
        return f"{self.analyzed}_analyzed_by_{self.analyzer}"
        
    def get_description(self) -> str:
        """이 분석 작업에 대한 설명 생성"""
        # 미들웨어 참조를 통해 get_message와 언어 설정 사용
        if hasattr(self, 'middleware') and self.middleware:
            lang = self.middleware.language
            if self.task_type == "self":
                return self.middleware.get_message(MK.SELF_ANALYSIS_DESCRIPTION, lang, self.analyzed.capitalize())
            return self.middleware.get_message(MK.EXTERNAL_ANALYSIS_DESCRIPTION, lang, 
                                            self.analyzed.capitalize(), self.analyzer.capitalize())
        
        # 미들웨어 참조가 없는 경우 기본 텍스트 반환
        if self.task_type == "self":
            return f"{self.analyzed.capitalize()}의 자체 분석"
        return f"{self.analyzed.capitalize()}에 대한 {self.analyzer.capitalize()}의 분석"

    def get_display_title(self) -> str:
        """UI 표시용 제목 생성"""
        # 모델 표시 이름 매핑
        model_display_names = {
            "claude": "Claude",
            "openai": "GPT",
            "perplexity": "Perplexity"
        }
        
        # 분석된 모델과 분석자 모델의 표시 이름 가져오기
        analyzed_display = model_display_names.get(self.analyzed, self.analyzed.capitalize())
        analyzer_display = model_display_names.get(self.analyzer, self.analyzer.capitalize())
        
        # 미들웨어 참조 확인 및 안전한 접근
        if not hasattr(self, 'middleware') or not self.middleware:
            # 미들웨어 참조가 없는 경우 기본 형식 사용
            if self.task_type == "self":
                return f"{analyzed_display} (자체 분석)"
            elif self.analyzer == "multiple":
                if self.analyzed == "claude":
                    return f"{analyzed_display} (GPT+Perplexity 분석)"
                elif self.analyzed == "openai":
                    return f"{analyzed_display} (Claude+Perplexity 분석)"
                elif self.analyzed == "perplexity":
                    return f"{analyzed_display} (Claude+GPT 분석)"
                else:
                    return f"{analyzed_display} (다중 AI 분석)"
            else:
                return f"{analyzed_display} ({analyzer_display} 분석)"
        
        # 미들웨어 참조가 있는 경우 현재 언어 사용
        lang = self.middleware.language
        if self.task_type == "self":
            return self.middleware.get_message(MK.SELF_ANALYSIS_TITLE, lang, analyzed_display)
        elif self.analyzer == "multiple":
            if self.analyzed == "claude":
                return self.middleware.get_message(MK.MULTIPLE_ANALYSIS_CLAUDE, lang, analyzed_display)
            elif self.analyzed == "openai":
                return self.middleware.get_message(MK.MULTIPLE_ANALYSIS_GPT, lang, analyzed_display)
            elif self.analyzed == "perplexity":
                return self.middleware.get_message(MK.MULTIPLE_ANALYSIS_PERPLEXITY, lang, analyzed_display)
            else:
                return self.middleware.get_message(MK.MULTIPLE_ANALYSIS_GENERIC, lang, analyzed_display)
        else:
            return self.middleware.get_message(MK.EXTERNAL_ANALYSIS_TITLE, lang, analyzed_display, analyzer_display)
        
    def get_response_display_title(self) -> str:
        """개선된 응답 표시용 제목 생성"""
        if not hasattr(self, 'middleware') or not self.middleware:
            raise AttributeError("Middleware reference not set in AnalysisTask. This is a critical bug that must be fixed.")
        
        return f"{self.get_display_title()}{self.middleware.get_message(MK.IMPROVED_RESPONSE_SUFFIX, self.middleware.language)}"

    def __str__(self) -> str:
        return self.get_task_id()
        
    def __repr__(self) -> str:
        return f"AnalysisTask({self.analyzer}, {self.analyzed}, {self.task_type})"
        
    def __eq__(self, other):
        if not isinstance(other, AnalysisTask):
            return False
        return (self.analyzer == other.analyzer and 
                self.analyzed == other.analyzed and 
                self.task_type == other.task_type)
        
    def __hash__(self):
        return hash((self.analyzer, self.analyzed, self.task_type))

class ImprovementPlan:
    """개선 계획을 정의하고 관리하는 클래스"""
    
    def __init__(self):
        self.analysis_tasks: Set[AnalysisTask] = set()
        self.response_models: Set[str] = set()
        self.analysis_models: Set[str] = set()
        self.analysis_only_models: Set[str] = set()
        
    def add_task(self, task: AnalysisTask):
        """분석 작업 추가"""
        self.analysis_tasks.add(task)
        self.analysis_models.add(task.analyzer)
        self.response_models.add(task.analyzed)
        
        # 분석만 수행하는 모델 식별
        if task.analyzer not in self.response_models:
            self.analysis_only_models.add(task.analyzer)
    
    def get_tasks_for_model(self, model: str, as_analyzer: bool = False) -> List[AnalysisTask]:
        """특정 모델이 포함된 분석 작업 반환"""
        if as_analyzer:
            return [task for task in self.analysis_tasks if task.analyzer == model]
        else:
            return [task for task in self.analysis_tasks if task.analyzed == model]
    
    def get_analyzers_for_model(self, model: str) -> List[str]:
        """특정 모델을 분석하는 분석자 목록 반환"""
        return [task.analyzer for task in self.analysis_tasks 
                if task.analyzed == model and task.task_type == "external"]
    
    def is_task_needed(self, task):
        """이 태스크가 실제로 필요한지 확인 - 중복 방지"""
        # 다중 분석이 이미 있으면 개별 분석은 불필요
        if task.task_type == "external":
            combined_task = AnalysisTask(analyzer="multiple", analyzed=task.analyzed, task_type="combined")
            if combined_task in self.analysis_tasks:
                return False
        return True

    def get_all_models(self) -> Set[str]:
        """모든 모델 반환 (응답 생성 + 분석 전용)"""
        return self.response_models.union(self.analysis_only_models)
    
    def __str__(self) -> str:
        return f"ImprovementPlan(tasks={len(self.analysis_tasks)}, response_models={self.response_models}, analysis_models={self.analysis_models})"

class EnhancedMultiAIMiddleware:
    def __init__(self, claude_client, openai_client, perplexity_client=None):
        self.claude_client = claude_client
        self.openai_client = openai_client
        
        # 기존 코드 대체
        self.perplexity_client = None
        if perplexity_client:
            # 기존의 PerplexityClient 대신 OpenAI 클라이언트 사용
            try:
                self.perplexity_client = OpenAI(
                    api_key=PERPLEXITY_API_KEY,
                    base_url="https://api.perplexity.ai"
                )
                print("Perplexity 클라이언트가 OpenAI 인터페이스로 초기화되었습니다.")
            except Exception as e:
                print(f"Perplexity 클라이언트 초기화 실패: {str(e)}")
                
        self.conversation_history = []
        self.token_tracker = TokenUsageTracker()
        self.total_cost = 0.0
        self.budget_limit = None  # 예산 제한 (옵션)
        
        # 클라이언트 매핑 사전
        self.clients = {
            "claude": self.claude_client,
            "openai": self.openai_client,
            "perplexity": self.perplexity_client
        }

        # 언어 설정 (기본값: 한국어)
        self.language = "ko" 
        
        # get_message 함수 참조
        try:
            from messages import get_message
            self.get_message = get_message
        except ImportError:
            # 메시지 모듈을 임포트할 수 없는 경우 간단한 fallback 함수 정의
            def simple_get_message(key, lang="ko", *args):
                return key.format(*args) if args else key
            self.get_message = simple_get_message

    def get_client(self, model: str):
        """지정된 모델의 API 클라이언트 반환"""
        return self.clients.get(model)

    def set_budget_limit(self, limit: float):
        """API 호출 예산 제한 설정"""
        self.budget_limit = limit
        print(f"예산 제한이 ${limit:.2f}로 설정되었습니다.")
    
    def check_budget(self) -> Tuple[bool, float]:
        """예산 제한 확인"""
        if self.budget_limit is not None:
            current_cost = self.token_tracker.get_current_session_summary()["combined"]["estimated_cost"]
            if current_cost >= self.budget_limit:
                return False, current_cost
        return True, self.token_tracker.get_current_session_summary()["combined"]["estimated_cost"]
    
    def process_query(self, user_input: str, show_comparison: bool = False, 
                      selected_models: List[str] = None, 
                      display_improvement_types: Union[str, List[str]] = "claude_analyzed_by_openai"):
        """
        사용자 쿼리 처리 - 내부적으로는 선택된 모델만 사용하여 비교 분석을 수행하고, 
        결과 표시 방식을 제어
        
        Args:
            user_input: 사용자 입력 텍스트
            show_comparison: 사용자에게 비교 분석 과정을 보여줄지 여부 (기본값: False)
            selected_models: 비교 분석에 사용할 AI 모델 목록 
                            (기본값: None, None인 경우 사용 가능한 모든 모델 사용)
                            예: ["claude", "openai"], ["claude", "perplexity"] 등
            display_improvement_types: 표시할 개선 응답 유형
                                단일 값: "claude_analyzed_by_openai", "all_self_analysis" 등
                                또는 리스트: ["claude_analyzed_by_openai", "openai_analyzed_by_claude"]
        
        Returns:
            dict: 결과 사전 (show_comparison에 따라 형식이 달라짐)
        """
        try:
            # 개선 계획 생성
            improvement_plan = self.create_improvement_plan(display_improvement_types)
            
            # 선택된 모델이 명시적으로 지정된 경우, 응답 생성 모델로 사용
            if selected_models is not None:
                # 선택된 모델 검증 및 필터링
                valid_models = [model for model in selected_models 
                            if model in ["claude", "openai", "perplexity"]]
                
                if not valid_models:
                    return {"error": self.get_message(MK.UNSUPPORTED_MODEL_ERROR, self.language, str(selected_models))} # 지원하지 않는 모델

                # 개선 계획 재조정 - 선택된 모델만 응답 생성
                response_models = set(valid_models)
                improvement_plan.response_models = response_models
                
                # 필요한 분석 작업만 유지
                improvement_plan.analysis_tasks = {task for task in improvement_plan.analysis_tasks 
                                                if task.analyzed in response_models}
            
            # Perplexity가 선택되었지만 클라이언트가 없는 경우
            if "perplexity" in improvement_plan.response_models and not self.perplexity_client:
                return {"error": self.get_message(MK.PERPLEXITY_CLIENT_NOT_INITIALIZED, self.language)} # "Perplexity 클라이언트가 초기화되지 않았습니다. API 키를 확인하세요."

            # 선택된 모델로 비교 분석 수행
            result = self._process_with_improvement_plan(
                user_input, 
                improvement_plan
            )

            # 오류가 있는 경우 그대로 반환
            if "error" in result:
                return result
                
            # 결과 처리: 비교 결과를 보여줄지 아니면 최적화된 단일 결과만 보여줄지
            if show_comparison:
                # 비교 결과 모두 포함하여 반환 (이미 result에 모든 정보가 있음)
                return result
            else:
                # 단일 결과만 필요한 경우 - 최적의 응답만 반환
                # 선택된 모델 중 우선순위에 따라 최적의 응답 선택
                best_response = self._select_best_response(result, improvement_plan.response_models)
                
                if not best_response:
                    return {"error": self.get_message(MK.NO_RESPONSE_FROM_ALL_MODELS, self.language)} # "선택된 모든 AI 모델에서 응답을 생성하지 못했습니다."

                # 결과 저장을 위해 원본 분석 결과도 포함
                return {
                    "final_response": best_response,
                    "processing_time": result["processing_time"],
                    "token_usage": result["token_usage"],
                    "selected_models": list(improvement_plan.response_models),
                    "analysis_tasks": [task.get_task_id() for task in improvement_plan.analysis_tasks],
                    "display_improvement_types": display_improvement_types,
                    "full_analysis": result  # 내부 처리용으로 전체 분석 결과 포함
                }
            
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            traceback.print_exc()
            return {"error": str(e)}

    def _select_best_response(self, result: Dict, response_models: Set[str]) -> Optional[str]:
        """결과에서 최적의 응답 선택"""
        # 우선순위 모델 리스트
        priority_models = ["claude", "openai", "perplexity"]
        
        # 1순위: 개선된 응답 먼저 탐색
        if "improved_responses" in result:
            for model in priority_models:
                if model in response_models:
                    # 먼저 모델 이름 정확히 일치하는 응답 찾기
                    for key, response in result["improved_responses"].items():
                        metadata = response.get("response_metadata", {})
                        analyzed_model = metadata.get("analyzed_model")
                        if analyzed_model == model and not response.get("error", False):
                            return response["text"]
                    
                    # 다음으로 키에 모델 이름이 포함된 첫 번째 응답 찾기
                    for key, response in result["improved_responses"].items():
                        if model in key.lower() and not response.get("error", False):
                            return response["text"]
        
        # 2순위: 초기 응답 탐색
        if "initial_responses" in result:
            for model in priority_models:
                if model in response_models and model in result["initial_responses"]:
                    if not result["initial_responses"][model].get("error", False):
                        return result["initial_responses"][model]["text"]
        
        return None

    def create_improvement_plan(self, display_improvement_types: Union[str, List[str]]) -> ImprovementPlan:
        """표시할 개선 응답 유형에 따라 필요한 분석 작업과 모델을 결정"""
        # 문자열을 리스트로 변환
        if isinstance(display_improvement_types, str):
            if "," in display_improvement_types:
                display_types = [dt.strip() for dt in display_improvement_types.split(",")]
            else:
                display_types = [display_improvement_types]
        else:
            display_types = display_improvement_types
        
        # 개선 계획 객체 생성
        plan = ImprovementPlan()
        
        # 정확히 선택된 유형만 추가
        print(f"선택된 표시 유형: {display_types}")
        
        # 초기 응답만 요청되었는지 확인
        only_initial_responses = all(dt.endswith("_initial") or dt == "initial_only" for dt in display_types if dt)
        if only_initial_responses:
            # 초기 응답만 필요한 경우 처리
            for dt in display_types:
                if dt == "initial_only":
                    plan.response_models.update(["claude", "openai", "perplexity"])
                elif dt == "claude_initial":
                    plan.response_models.add("claude")
                elif dt == "openai_initial" or dt == "gpt_initial":
                    plan.response_models.add("openai")
                elif dt == "perplexity_initial":
                    plan.response_models.add("perplexity")
            
            return plan
        
        # 각 표시 유형에 대해 필요한 분석 작업 생성
        for display_type in display_types:
            # 초기 응답 옵션 처리
            if display_type.endswith("_initial") or display_type == "initial_only":
                if display_type == "initial_only":
                    plan.response_models.update(["claude", "openai", "perplexity"])
                elif display_type == "claude_initial":
                    plan.response_models.add("claude")
                elif display_type == "openai_initial" or display_type == "gpt_initial":
                    plan.response_models.add("openai")
                elif display_type == "perplexity_initial":
                    plan.response_models.add("perplexity")
                continue
            
            # 자체 분석 옵션
            elif display_type == "all_self_analysis":
                # 모든 모델의 자체 분석
                for model in ["claude", "openai", "perplexity"]:
                    task = AnalysisTask(analyzer=model, analyzed=model, task_type="self")
                    # 미들웨어 참조 설정 - 추가된 부분
                    task.middleware = self
                    plan.add_task(task)
            
            elif display_type == "claude_analyzed_by_self":
                task = AnalysisTask(analyzer="claude", analyzed="claude", task_type="self")
                task.middleware = self  # 미들웨어 참조 설정
                plan.add_task(task)
                
            elif display_type == "openai_analyzed_by_self" or display_type == "gpt_analyzed_by_self":
                task = AnalysisTask(analyzer="openai", analyzed="openai", task_type="self")
                task.middleware = self  # 미들웨어 참조 설정
                plan.add_task(task)
                
            elif display_type == "perplexity_analyzed_by_self":
                task = AnalysisTask(analyzer="perplexity", analyzed="perplexity", task_type="self")
                task.middleware = self  # 미들웨어 참조 설정
                plan.add_task(task)
            
            # 단일 외부 분석 옵션
            elif display_type == "claude_analyzed_by_openai" or display_type == "claude_analyzed_by_gpt":
                task = AnalysisTask(analyzer="openai", analyzed="claude", task_type="external")
                task.middleware = self  # 미들웨어 참조 설정
                plan.add_task(task)
                
            elif display_type == "claude_analyzed_by_perplexity":
                task = AnalysisTask(analyzer="perplexity", analyzed="claude", task_type="external")
                task.middleware = self  # 미들웨어 참조 설정
                plan.add_task(task)
                
            elif display_type == "openai_analyzed_by_claude" or display_type == "gpt_analyzed_by_claude":
                task = AnalysisTask(analyzer="claude", analyzed="openai", task_type="external")
                task.middleware = self  # 미들웨어 참조 설정
                plan.add_task(task)
                
            elif display_type == "openai_analyzed_by_perplexity" or display_type == "gpt_analyzed_by_perplexity":
                task = AnalysisTask(analyzer="perplexity", analyzed="openai", task_type="external")
                task.middleware = self  # 미들웨어 참조 설정
                plan.add_task(task)
                
            elif display_type == "perplexity_analyzed_by_claude":
                task = AnalysisTask(analyzer="claude", analyzed="perplexity", task_type="external")
                task.middleware = self  # 미들웨어 참조 설정
                plan.add_task(task)
                
            elif display_type == "perplexity_analyzed_by_openai" or display_type == "perplexity_analyzed_by_gpt":
                task = AnalysisTask(analyzer="openai", analyzed="perplexity", task_type="external")
                task.middleware = self  # 미들웨어 참조 설정
                plan.add_task(task)
            
            # 다중 분석자 옵션
            elif display_type == "claude_analyzed_by_multiple":
                # 다중 분석을 위한 특별한 태스크 타입 사용
                task = AnalysisTask(analyzer="multiple", analyzed="claude", task_type="combined")
                task.middleware = self  # 미들웨어 참조 설정
                plan.add_task(task)
                            
            elif display_type == "openai_analyzed_by_multiple" or display_type == "gpt_analyzed_by_multiple":
                # 다중 분석을 위한 특별한 태스크 타입 사용
                task = AnalysisTask(analyzer="multiple", analyzed="openai", task_type="combined")
                task.middleware = self  # 미들웨어 참조 설정
                plan.add_task(task)
                
            elif display_type == "perplexity_analyzed_by_multiple":
                # 다중 분석을 위한 특별한 태스크 타입 사용
                task = AnalysisTask(analyzer="multiple", analyzed="perplexity", task_type="combined")
                task.middleware = self  # 미들웨어 참조 설정
                plan.add_task(task)
            
            # 기존 호환성 옵션
            elif display_type == "cross_only":
                # 모든 모델 간 교차 분석
                for analyzed in ["claude", "openai", "perplexity"]:
                    for analyzer in ["claude", "openai", "perplexity"]:
                        if analyzed != analyzer:  # 자체 분석 제외
                            task = AnalysisTask(analyzer=analyzer, analyzed=analyzed, task_type="external")
                            task.middleware = self  # 미들웨어 참조 설정
                            plan.add_task(task)
                
            elif display_type == "self_only":
                # 모든 모델 자체 분석
                for model in ["claude", "openai", "perplexity"]:
                    task = AnalysisTask(analyzer=model, analyzed=model, task_type="self")
                    task.middleware = self  # 미들웨어 참조 설정
                    plan.add_task(task)
                
            elif display_type == "all" or display_type == "all_detailed":
                # 모든 가능한 분석 (자체 + 교차)
                for analyzed in ["claude", "openai", "perplexity"]:
                    # 자체 분석
                    task = AnalysisTask(analyzer=analyzed, analyzed=analyzed, task_type="self")
                    task.middleware = self  # 미들웨어 참조 설정
                    plan.add_task(task)
                    
                    # 교차 분석
                    for analyzer in ["claude", "openai", "perplexity"]:
                        if analyzed != analyzer:
                            task = AnalysisTask(analyzer=analyzer, analyzed=analyzed, task_type="external")
                            task.middleware = self  # 미들웨어 참조 설정
                            plan.add_task(task)
            
            # 알 수 없는 옵션
            else:
                print(f"알 수 없는 표시 유형: {display_type}. 안전하게 기본 분석을 수행합니다.")
                task = AnalysisTask(analyzer="openai", analyzed="claude", task_type="external")
                task.middleware = self  # 미들웨어 참조 설정
                plan.add_task(task)
        
        # Perplexity 클라이언트가 없는 경우 Perplexity 관련 작업 제거
        if not self.perplexity_client:
            # perplexity를 분석자로 사용하는 태스크 제거
            plan.analysis_tasks = {task for task in plan.analysis_tasks if task.analyzer != "perplexity"}
            
            # perplexity를 응답 모델로 사용하는 태스크 제거
            plan.analysis_tasks = {task for task in plan.analysis_tasks if task.analyzed != "perplexity"}
            
            # 응답 모델에서 perplexity 제거
            plan.response_models.discard("perplexity")
            
            # 분석 모델 목록 업데이트
            plan.analysis_models = {task.analyzer for task in plan.analysis_tasks}
            
            # 분석 전용 모델 목록 업데이트
            plan.analysis_only_models = {model for model in plan.analysis_models if model not in plan.response_models}
        
        # 계획이 비어있을 경우 기본 계획 생성
        if not plan.analysis_tasks and not only_initial_responses:
            print("비어있는 분석 계획. 기본 계획으로 대체합니다.")
            task = AnalysisTask(analyzer="openai", analyzed="claude", task_type="external")
            task.middleware = self  # 미들웨어 참조 설정
            plan.add_task(task)
            
        print(f"생성된 개선 계획: {plan}")
        print(f"분석 작업: {[str(task) for task in plan.analysis_tasks]}")
        print(f"응답 모델: {plan.response_models}")
        print(f"분석 모델: {plan.analysis_models}")
        
        return plan

    def _process_with_improvement_plan(self, user_input: str, improvement_plan: ImprovementPlan) -> Dict:
        """
        개선 계획에 따라 사용자 쿼리 분석 및 개선 처리
        
        Args:
            user_input: 사용자 입력 텍스트
            improvement_plan: 개선 계획 객체
            
        Returns:
            dict: 처리 결과
        """
        try:   
            # 시작 시간 기록
            start_time = time.time()
                
            # 사용자 입력 기록
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # 선택된 개선 유형 가져오기 - 필요한 정보 추출
            display_improvement_types = [task.get_task_id() for task in improvement_plan.analysis_tasks]
            
            # 1. 응답 생성이 필요한 AI 모델에서만 초기 응답 가져오기
            print(f"1. 응답 생성 모델 ({', '.join(improvement_plan.response_models)})에서 초기 응답 가져오기...")
            initial_responses = self._get_initial_responses(user_input, improvement_plan.response_models)
            print(f"초기 응답 생성 완료 ({time.time() - start_time:.2f}초)")
            
            # 예산 재확인
            budget_ok, current_cost = self.check_budget()
            if not budget_ok:
                return {
                    "initial_responses": initial_responses,
                    "warning": self.get_message(MK.BUDGET_LIMIT_REACHED, self.language, self.budget_limit, current_cost)
                }
            
            # 분석 작업이 없는 경우 - 초기 응답만 반환
            if not improvement_plan.analysis_tasks:
                print("분석 작업이 없습니다. 초기 응답만 반환합니다.")
                return {
                    "initial_responses": initial_responses,
                    "improved_responses": {},
                    "analyses": {},
                    "follow_up_questions": {},
                    "processing_time": time.time() - start_time,
                    "token_usage": self.token_tracker.get_current_session_summary(),
                    "response_models": list(improvement_plan.response_models),
                    "analysis_models": [],
                    "analysis_only_models": [],
                    "analysis_tasks": []
                }
            
            # 2. 분석 작업 실행
            print(f"2. 분석 작업 실행 중... (작업 수: {len(improvement_plan.analysis_tasks)})")

            # 수정된 필터링 로직
            filtered_tasks = []
            skip_individual_analyses = {}

            # 사용자가 명시적으로 선택한 작업은 모두 포함
            for task in improvement_plan.analysis_tasks:
                task_id = task.get_task_id()
                
                # 사용자가 명시적으로 선택한 작업인 경우 무조건 포함
                if task_id in display_improvement_types:
                    filtered_tasks.append(task)
                # 명시적으로 선택되지 않은 작업에 대해서만 중복 제거 로직 적용
                elif "multiple" in task_id or task.task_type == "combined":
                    # 이 모델에 대한 다중 분석이 있고, 명시적으로 선택되지 않은 개별 분석은 건너뛰기
                    skip_individual_analyses[task.analyzed] = True
                elif task.analyzed not in skip_individual_analyses:
                    # 명시적으로 선택되지 않았지만, 다중 분석에 포함되지 않는 작업은 추가
                    filtered_tasks.append(task)

            print(f"중복 제거 후 실행할 분석 작업: {len(filtered_tasks)}개")
            for task in filtered_tasks:
                print(f"- {task.get_task_id()}")

            # 필터링된 작업으로 분석 실행
            analyses = {}
            for task in filtered_tasks:
                task_result = self._execute_single_analysis_task(user_input, initial_responses, task)
                if task_result:
                    analyses[task.get_task_id()] = task_result

            # 모든 분석 작업이 완료된 후, 캐시에서 추가 분석 결과 추출
            # 이 추가 분석 결과는 UI 분석 탭에만 표시되고 응답 개선에는 사용되지 않음
            ui_only_analyses = {}  # UI 분석 탭에만 표시할 분석 결과
            if hasattr(self, '_analysis_cache'):
                print("캐시된 개별 분석 결과 확인 중...")
                for cache_key, analysis_result in self._analysis_cache.items():
                    # 캐시 키 형식이 "{analyzer}_{analyzed_model}_analysis"인지 확인
                    if '_analysis' in cache_key and isinstance(analysis_result, dict) and "error" not in analysis_result:
                        parts = cache_key.split('_')
                        if len(parts) >= 2:
                            analyzer = parts[0]
                            analyzed_model = parts[1]
                            
                            # 모델별 표시 이름 설정
                            model_display_names = {
                                "claude": "Claude",
                                "openai": "GPT",
                                "perplexity": "Perplexity"
                            }
                            # 분석된 모델과 분석자 모델의 표시 이름 가져오기
                            analyzed_display = model_display_names.get(analyzed_model, analyzed_model.capitalize())
                            analyzer_display = model_display_names.get(analyzer, analyzer.capitalize())
                            
                            # 이미 analyses에 있는지 확인
                            individual_task_id = f"{analyzed_model}_analyzed_by_{analyzer}"
                            if individual_task_id not in analyses:
                                # 메타데이터 추가
                                analysis_copy = analysis_result.copy()
                                if "response_metadata" not in analysis_copy:
                                    analysis_copy["response_metadata"] = {
                                        "analyzed_model": analyzed_model,
                                        "analyzers": [analyzer],
                                        "analysis_type": "external",
                                        "task_id": individual_task_id,
                                        "display_name": self.get_message(MK.EXTERNAL_ANALYSIS_TITLE, self.language, analyzed_display, analyzer_display) # f"{analyzed_display} ({analyzer_display} 분석)"
                                    }
                                
                                # UI 분석 탭에만 표시할 추가 분석 결과로 저장
                                print(f"UI 분석 탭에만 표시할 추가 분석: {individual_task_id}")
                                ui_only_analyses[individual_task_id] = analysis_copy

            # 예산 재확인
            budget_ok, current_cost = self.check_budget()
            if not budget_ok:
                return {
                    "initial_responses": initial_responses,
                    "analyses": analyses,
                    "ui_only_analyses": ui_only_analyses,  # UI 분석 탭에만 표시할 추가 분석 결과
                    "warning": self.get_message(MK.BUDGET_LIMIT_REACHED, self.language, self.budget_limit, current_cost) # f"예산 제한(${self.budget_limit:.2f})에 도달했습니다. 현재 비용: ${current_cost:.2f}"
                }
            
            # 3. 분석 결과를 바탕으로 응답 개선 - display_improvement_types 전달
            # 중요: 명시적으로 선택된 분석만 응답 개선에 사용 (analyses만 사용)
            print("3. 응답 개선 중...")
            improved_responses = self._improve_responses_with_analyses(
                user_input, 
                initial_responses, 
                analyses,  # 사용자가 명시적으로 선택한 분석만 포함
                improvement_plan,
                display_improvement_types  # 여기에 표시 유형 전달
            )
            print(f"응답 개선 완료 ({time.time() - start_time:.2f}초)")

            # 4. 후속 질문 추출
            print("4. 후속 질문 추출 중...")
            follow_up_questions = self._extract_follow_up_questions(
                improved_responses, 
                improvement_plan.response_models,
                analyses,            # 명시적으로 선택한 분석
                ui_only_analyses     # UI 분석 탭에만 표시할 추가 분석
            )
            print(f"후속 질문 추출 완료 ({time.time() - start_time:.2f}초)")

            # 최종 응답 기록 (응답 생성 모델 중 우선순위가 높은 모델 사용)
            for model in ["claude", "openai", "perplexity"]:
                if model in improvement_plan.response_models:
                    # 개선된 응답 찾기
                    final_response = None
                    for key, response in improved_responses.items():
                        if model.lower() in key.lower() and not response.get("error", False):
                            final_response = response["text"]
                            break
                    
                    # 개선된 응답이 없으면 초기 응답 사용
                    if not final_response and model in initial_responses and not initial_responses[model].get("error", False):
                        final_response = initial_responses[model]["text"]
                    
                    if final_response:
                        self.conversation_history.append({"role": "assistant", "content": final_response})
                        break
            
            # 총 소요 시간 계산
            total_time = time.time() - start_time
            print(f"총 처리 시간: {total_time:.2f}초")
            
            # 토큰 사용량 요약
            usage_summary = self.token_tracker.get_current_session_summary()
            
            # 결과 반환
            return {
                "initial_responses": initial_responses,
                "analyses": analyses,  # 사용자가 선택한 분석 결과
                "ui_only_analyses": ui_only_analyses,  # UI 분석 탭에만 표시할 추가 분석 결과
                "improved_responses": improved_responses,
                "follow_up_questions": follow_up_questions,
                "processing_time": total_time,
                "token_usage": usage_summary,
                "response_models": list(improvement_plan.response_models),
                "analysis_models": list(improvement_plan.analysis_models),
                "analysis_only_models": list(improvement_plan.analysis_only_models),
                "analysis_tasks": [str(task) for task in improvement_plan.analysis_tasks],
                "display_improvement_types": display_improvement_types  # 표시 유형 정보 추가
            }
            
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            traceback.print_exc()
            return {"error": str(e)}

    def _execute_single_analysis_task(self, user_input: str, initial_responses: Dict, task: AnalysisTask) -> Dict:
        """단일 분석 작업 실행 - 캐싱 메커니즘 추가"""
        try:
            task_id = task.get_task_id()
            print(f"{task_id} 분석 작업 실행 중...")
            
            # 정적 캐시가 없으면 생성 (클래스 변수로 유지)
            if not hasattr(self, '_analysis_cache'):
                self._analysis_cache = {}

            # 다중 분석 처리
            if "multiple" in task_id or task.task_type == "combined":
                analyzed_model = task.analyzed
                if analyzed_model not in initial_responses or initial_responses[analyzed_model].get("error", False):
                    return {"error": self.get_message(MK.NO_VALID_RESPONSE, self.language, analyzed_model)} # f"{analyzed_model} 모델의 유효한 응답이 없습니다."

                # 분석 대상 텍스트
                analyzed_text = initial_responses[analyzed_model]["text"]
                
                # 모델별 표시 이름 설정
                model_display_names = {
                    "claude": "Claude",
                    "openai": "GPT",
                    "perplexity": "Perplexity"
                }
                model_display = model_display_names.get(analyzed_model, analyzed_model.capitalize())

                # 표시 이름 설정
                if analyzed_model.lower() == "claude":
                    display_name = self.get_message(MK.MULTIPLE_ANALYSIS_CLAUDE, self.language, model_display)
                elif analyzed_model.lower() == "openai":
                    display_name = self.get_message(MK.MULTIPLE_ANALYSIS_GPT, self.language, model_display)
                elif analyzed_model.lower() == "perplexity":
                    display_name = self.get_message(MK.MULTIPLE_ANALYSIS_PERPLEXITY, self.language, model_display)
                else:
                    display_name = self.get_message(MK.MULTIPLE_ANALYSIS_GENERIC, self.language, model_display)

                # 결합된 분석 정보
                combined_analysis = {
                    "improvements": [],
                    "missing_information": [],
                    "follow_up_questions": [],
                    "response_metadata": {
                        "analyzed_model": analyzed_model,
                        "analyzers": ["multiple"],
                        "analysis_type": "combined",
                        "task_id": task_id,
                        "display_name": display_name
                    }
                }
                
                # 사용 가능한 분석자들 (analyzed 제외)
                analyzers = ["claude", "openai", "perplexity"]
                if analyzed_model in analyzers:
                    analyzers.remove(analyzed_model)
                
                # 각 분석자로 분석 수행 - 캐싱 메커니즘 적용
                for analyzer in analyzers:
                    if not self.get_client(analyzer):
                        continue
                    
                    # 개별 분석 캐시 키 생성
                    cache_key = f"{analyzer}_{analyzed_model}_analysis"
                    
                    # 캐시에 있는지 확인하고 재사용
                    if cache_key in self._analysis_cache:
                        print(f"캐시된 분석 결과 재사용: {cache_key}")
                        analysis_result = self._analysis_cache[cache_key]
                    else:
                        # 새로 분석 실행
                        analysis_result = self._analyze_with_model(user_input, analyzed_model, analyzed_text, analyzer)
                        # 캐시에 저장
                        self._analysis_cache[cache_key] = analysis_result
                    
                    # 결과 결합
                    if isinstance(analysis_result, dict) and "error" not in analysis_result:
                        combined_analysis["improvements"].extend(analysis_result.get("improvements", []))
                        combined_analysis["missing_information"].extend(analysis_result.get("missing_information", []))
                        combined_analysis["follow_up_questions"].extend(analysis_result.get("follow_up_questions", []))
                
                # 중복 제거
                combined_analysis["improvements"] = list(set(combined_analysis["improvements"]))
                combined_analysis["missing_information"] = list(set(combined_analysis["missing_information"]))
                combined_analysis["follow_up_questions"] = list(set(combined_analysis["follow_up_questions"]))
                
                return combined_analysis
                    
            # 일반 분석 작업
            else:
                # 분석 대상 모델의 응답이 있는지 확인
                if task.analyzed not in initial_responses or initial_responses[task.analyzed].get("error", False):
                    print(f"{task_id} 작업 건너뛰기: {task.analyzed} 모델의 유효한 응답이 없습니다.")
                    return None
                
                # 분석 대상 텍스트
                analyzed_text = initial_responses[task.analyzed]["text"]
                
                # 캐시 키 생성
                cache_key = f"{task.analyzer}_{task.analyzed}_analysis"
                
                # 캐시에 있는지 확인하고 재사용
                if cache_key in self._analysis_cache:
                    print(f"캐시된 분석 결과 재사용: {cache_key}")
                    analysis_result = self._analysis_cache[cache_key]
                else:
                    # 새로 분석 수행
                    analysis_result = self._analyze_with_model(user_input, task.analyzed, analyzed_text, task.analyzer)
                    # 캐시에 저장
                    self._analysis_cache[cache_key] = analysis_result
                
                # 메타데이터 추가
                if isinstance(analysis_result, dict) and "error" not in analysis_result:
                    analysis_result["response_metadata"] = {
                        "analyzed_model": task.analyzed,
                        "analyzers": [task.analyzer],
                        "analysis_type": task.task_type,
                        "task_id": task_id,
                        "display_name": task.get_display_title()
                    }

                return analysis_result
                    
        except Exception as e:
            print(f"{task.get_task_id()} 분석 작업 오류: {str(e)}")
            return {"error": str(e)}

    def _analyze_with_model(self, user_input, analyzed_model, analyzed_text, analyzer_model):
        """특정 모델로 분석 수행"""
        analysis_prompt = self._create_analysis_prompt(user_input, analyzed_model, analyzed_text)
        
        if analyzer_model == "claude":
            analysis_response = self.claude_client.messages.create(
                model=ACTIVE_CLAUDE_MODEL,
                max_tokens=800,
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            self.token_tracker.track_claude_usage(analysis_response, analysis_prompt)
            return self._extract_json_data(analysis_response.content[0].text)
            
        elif analyzer_model == "openai":
            analysis_response = self.openai_client.chat.completions.create(
                model=ACTIVE_GPT_MODEL,
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            self.token_tracker.track_openai_usage(analysis_response, analysis_prompt)
            return self._extract_json_data(analysis_response.choices[0].message.content)
            
        elif analyzer_model == "perplexity" and self.perplexity_client:
            system_message = self.get_message(MK.PERPLEXITY_SYSTEM_MESSAGE, self.language)
            analysis_response = self.perplexity_client.chat.completions.create(
                model=ACTIVE_PERPLEXITY_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": analysis_prompt}
                ]
            )

            self.token_tracker.track_perplexity_usage(analysis_response, analysis_prompt)
            return self._extract_json_data(analysis_response.choices[0].message.content)
        
        return {"error": self.get_message(MK.MODEL_UNAVAILABLE, self.language, analyzer_model)} # f"분석 모델 {analyzer_model}을 사용할 수 없습니다."

    def _get_initial_responses(self, user_input: str, response_models: Set[str]) -> Dict[str, Dict]:
        """
        지정된 모델들에서 초기 응답 가져오기
        
        Args:
            user_input: 사용자 입력 텍스트
            response_models: 응답을 가져올 모델 목록 (Set)
            
        Returns:
            dict: 각 모델별 초기 응답
        """
        responses = {}
        
        # Claude 응답 (선택된 경우)
        if "claude" in response_models:
            try:
                print("Claude 초기 응답 요청 중...")
                claude_response = self.claude_client.messages.create(
                    model=ACTIVE_CLAUDE_MODEL,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": user_input}]
                )
                responses["claude"] = {
                    "text": claude_response.content[0].text,
                    "raw": claude_response,
                    "model": "claude",
                    "response_type": "initial"
                }
                # 토큰 사용량 추적
                claude_usage = self.token_tracker.track_claude_usage(claude_response, user_input)
                print(f"Claude 초기 응답 토큰: {claude_usage['total_tokens']} (비용: ${claude_usage['estimated_cost']:.4f})")
            except Exception as e:
                print(f"Claude 응답 가져오기 오류: {str(e)}")
                responses["claude"] = {"text": f"{self.get_message(MK.ERROR_TEXT, self.language)}: {str(e)}", "error": True, "model": "claude"}

        # GPT 응답 (선택된 경우)
        if "openai" in response_models:
            try:
                print("GPT 초기 응답 요청 중...")
                gpt_response = self.openai_client.chat.completions.create(
                    model=ACTIVE_GPT_MODEL,
                    messages=[{"role": "user", "content": user_input}]
                )
                responses["openai"] = {
                    "text": gpt_response.choices[0].message.content,
                    "raw": gpt_response,
                    "model": "openai",
                    "response_type": "initial"
                }
                # 토큰 사용량 추적
                gpt_usage = self.token_tracker.track_openai_usage(gpt_response, user_input)
                print(f"GPT 초기 응답 토큰: {gpt_usage['total_tokens']} (비용: ${gpt_usage['estimated_cost']:.4f})")
            except Exception as e:
                print(f"GPT 응답 가져오기 오류: {str(e)}")
                responses["openai"] = {"text": f"{self.get_message(MK.ERROR_TEXT, self.language)}: {str(e)}", "error": True, "model": "openai"}

        # Perplexity 응답 (선택된 경우)
        if "perplexity" in response_models and self.perplexity_client:
            try:
                print("Perplexity 초기 응답 요청 중...")
                # OpenAI 인터페이스 사용
                perplexity_response = self.perplexity_client.chat.completions.create(
                    model=ACTIVE_PERPLEXITY_MODEL,
                    messages=[
                        {"role": "user", "content": user_input}
                    ]
                )
                responses["perplexity"] = {
                    "text": perplexity_response.choices[0].message.content,
                    "raw": perplexity_response,
                    "model": "perplexity",
                    "response_type": "initial"
                }
                # 토큰 사용량 추적
                perplexity_usage = self.token_tracker.track_perplexity_usage(perplexity_response, user_input)
                print(f"Perplexity 초기 응답 토큰: {perplexity_usage['total_tokens']} (비용: ${perplexity_usage['estimated_cost']:.4f})")
            except Exception as e:
                print(f"Perplexity 응답 가져오기 오류: {str(e)}")
                responses["perplexity"] = {"text": f"{self.get_message(MK.ERROR_TEXT, self.language)}: {str(e)}", "error": True, "model": "perplexity"}

        return responses

    def _execute_analysis_tasks(self, user_input: str, initial_responses: Dict[str, Dict], 
                               improvement_plan: ImprovementPlan) -> Dict[str, Any]:
        """
        분석 작업 실행
        
        Args:
            user_input: 사용자 입력 텍스트
            initial_responses: 초기 응답 사전
            improvement_plan: 개선 계획 객체
            
        Returns:
            dict: 분석 결과
        """
        analyses = {}
        
        # 각 분석 작업 실행
        for task in improvement_plan.analysis_tasks:
            # 분석 대상 모델의 응답이 있는지 확인
            if task.analyzed not in initial_responses or initial_responses[task.analyzed].get("error", False):
                print(f"분석 작업 {task} 건너뛰기: {task.analyzed} 모델의 유효한 응답이 없습니다.")
                continue
            
            try:
                task_id = task.get_task_id()
                print(f"{task_id} 분석 작업 실행 중...")
                
                # 분석 대상 텍스트
                analyzed_text = initial_responses[task.analyzed]["text"]
                
                # 분석 프롬프트 생성
                analysis_prompt = self._create_analysis_prompt(user_input, task.analyzed, analyzed_text)
                
                # 분석 실행 (분석자 모델에 따라 다른 클라이언트 사용)
                if task.analyzer == "claude":
                    analysis_response = self.claude_client.messages.create(
                        model=ACTIVE_CLAUDE_MODEL,
                        max_tokens=800,
                        messages=[{"role": "user", "content": analysis_prompt}]
                    )
                    
                    # 토큰 사용량 추적
                    self.token_tracker.track_claude_usage(analysis_response, analysis_prompt)
                    
                    analysis_result = self._extract_json_data(analysis_response.content[0].text)
                    
                elif task.analyzer == "openai":
                    analysis_response = self.openai_client.chat.completions.create(
                        model=ACTIVE_GPT_MODEL,
                        messages=[{"role": "user", "content": analysis_prompt}]
                    )
                    
                    # 토큰 사용량 추적
                    self.token_tracker.track_openai_usage(analysis_response, analysis_prompt)
                    
                    analysis_result = self._extract_json_data(analysis_response.choices[0].message.content)
                    
                elif task.analyzer == "perplexity" and self.perplexity_client:
                    # query 메서드 대신 chat.completions.create 사용
                    analysis_response = self.perplexity_client.chat.completions.create(
                        model=ACTIVE_PERPLEXITY_MODEL,
                        messages=[
                            {"role": "system", "content": "당신은 AI 응답을 분석하고 개선하는 전문가입니다."},
                            {"role": "user", "content": analysis_prompt}
                        ]
                    )
                    
                    # 토큰 사용량 추적
                    self.token_tracker.track_perplexity_usage(analysis_response, analysis_prompt)
                    
                    analysis_result = self._extract_json_data(analysis_response.choices[0].message.content)
                
                # 분석 결과에 메타데이터 추가
                if isinstance(analysis_result, dict) and "error" not in analysis_result:
                    analysis_result["analyzer"] = task.analyzer
                    analysis_result["analyzed"] = task.analyzed
                    analysis_result["task_type"] = task.task_type
                    analysis_result["task_id"] = task_id
                    analysis_result["display_title"] = task.get_display_title()
                
                # 결과 저장
                analyses[task_id] = analysis_result
                print(f"{task_id} 분석 작업 완료")
                
            except Exception as e:
                print(f"{task} 분석 작업 오류: {str(e)}")
                analyses[task.get_task_id()] = {"error": str(e)}
        
        return analyses

    def _create_analysis_prompt(self, user_input: str, analyzed_model: str, analyzed_text: str) -> str:
        """분석 프롬프트 생성"""
        if analyzed_model == "perplexity":
            # Perplexity 전용 강화된 프롬프트
            return self.get_message(MK.ANALYSIS_PROMPT_PERPLEXITY, self.language, 
                                user_input, analyzed_text)
        else:
            # 일반 모델용 프롬프트
            return self.get_message(MK.ANALYSIS_PROMPT_GENERAL, self.language, 
                                analyzed_model.capitalize(), user_input, analyzed_text)

    def _improve_responses_with_analyses(self, user_input: str, initial_responses: Dict[str, Dict], 
                                   analyses: Dict[str, Dict], improvement_plan: ImprovementPlan,
                                   display_improvement_types: List[str]) -> Dict[str, Dict]:
        """
        분석 결과를 바탕으로 응답 개선
        
        Args:
            user_input: 사용자 입력 텍스트
            initial_responses: 초기 응답
            analyses: 분석 결과
            improvement_plan: 개선 계획
            display_improvement_types: 표시할 개선 응답 유형 목록
            
        Returns:
            dict: 개선된 응답
        """
        improved_responses = {}
        
        # 각 응답 모델에 대한 개선 처리
        for model in improvement_plan.response_models:
            if model not in initial_responses or initial_responses[model].get("error", False):
                print(f"{model} 모델의 응답 개선 건너뛰기: 유효한 초기 응답이 없습니다.")
                continue
            
            # 각 응답 모델에 대한 분석 작업 찾기
            model_analyses = []
            for task_id, analysis in analyses.items():
                if isinstance(analysis, dict) and analysis.get("response_metadata", {}).get("analyzed_model") == model and "error" not in analysis:
                    model_analyses.append(analysis)
            
            if not model_analyses:
                print(f"{model} 모델에 대한 유효한 분석 결과가 없습니다.")
                continue
            
            # 개선 프로세스 실행
            print(f"{model} 모델 응답 개선 중... (분석 결과 {len(model_analyses)}개 사용)")
            
            # 각 분석 결과별로 개선 응답 생성
            for analysis in model_analyses:
                try:
                    # 메타데이터 가져오기
                    metadata = analysis.get("response_metadata", {})
                    analyzer = metadata.get("analyzers", [])[0] if metadata.get("analyzers") else "unknown"
                    analyzed = metadata.get("analyzed_model")
                    task_type = metadata.get("analysis_type")
                    task_id = metadata.get("task_id")
                    
                    if not all([analyzer, analyzed, task_type, task_id]):
                        print(f"분석 메타데이터 누락: {analysis}")
                        continue
                    
                    # 고유한 응답 키 생성
                    response_key = f"{analyzed}_{analyzer}"
                    if task_type == "self":
                        response_key = f"{analyzed}_self_analysis"
                    
                    # 개선 프롬프트 생성
                    improvements = analysis.get("improvements", [])
                    missing_info = analysis.get("missing_information", [])
                    follow_ups = analysis.get("follow_up_questions", [])
                    
                    improvement_prompt = self._create_improvement_prompt(
                        user_input, 
                        initial_responses[model]["text"],
                        improvements,
                        missing_info,
                        follow_ups,
                        analyzer
                    )
                    
                    # 모델에 따라 다른 클라이언트 사용
                    if model == "claude":
                        improved_response = self.claude_client.messages.create(
                            model=ACTIVE_CLAUDE_MODEL,
                            max_tokens=1500,
                            messages=[
                                {"role": "user", "content": user_input},
                                {"role": "assistant", "content": initial_responses[model]["text"]},
                                {"role": "user", "content": improvement_prompt}
                            ]
                        )
                        
                        # 토큰 사용량 추적
                        self.token_tracker.track_claude_usage(improved_response, improvement_prompt)
                        
                        improved_responses[response_key] = {
                            "text": improved_response.content[0].text,
                            "raw": improved_response,
                            "analysis_data": analysis,
                            "response_metadata": {
                                "analyzed_model": analyzed,
                                "analyzers": [analyzer],
                                "analysis_type": task_type,
                                "task_id": task_id,
                                "display_name": metadata.get("response_display_name") or f"{metadata.get('display_name', self.get_message(MK.UNKNOWN, self.language))} {self.get_message(MK.IMPROVED_RESPONSE_SUFFIX, self.language)}" # f"{metadata.get('display_name', '알 수 없음')} 개선된 응답"
                            }
                        }
                        
                    elif model == "openai":
                        improved_response = self.openai_client.chat.completions.create(
                            model=ACTIVE_GPT_MODEL,
                            messages=[
                                {"role": "user", "content": user_input},
                                {"role": "assistant", "content": initial_responses[model]["text"]},
                                {"role": "user", "content": improvement_prompt}
                            ]
                        )
                        
                        # 토큰 사용량 추적
                        self.token_tracker.track_openai_usage(improved_response, improvement_prompt)
                        
                        improved_responses[response_key] = {
                            "text": improved_response.choices[0].message.content,
                            "raw": improved_response,
                            "analysis_data": analysis,
                            "response_metadata": {
                                "analyzed_model": analyzed,
                                "analyzers": [analyzer],
                                "analysis_type": task_type,
                                "task_id": task_id,
                                "display_name": metadata.get("response_display_name") or f"{metadata.get('display_name', self.get_message(MK.UNKNOWN, self.language))} {self.get_message(MK.IMPROVED_RESPONSE_SUFFIX, self.language)}" # f"{metadata.get('display_name', '알 수 없음')} 개선된 응답"
                            }
                        }
                        
                    elif model == "perplexity" and self.perplexity_client:
                        try:
                            # 시스템 메시지 준비
                            system_message = f"당신은 AI 응답을 개선하는 전문가입니다. {analyzer.capitalize()}의 분석을 기반으로 이전 응답을 개선해주세요."

                            # OpenAI 인터페이스로 호출
                            improved_response = self.perplexity_client.chat.completions.create(
                                model=ACTIVE_PERPLEXITY_MODEL,
                                messages=[
                                    {"role": "system", "content": system_message},
                                    {"role": "user", "content": user_input},
                                    {"role": "assistant", "content": initial_responses[model]["text"]},
                                    {"role": "user", "content": improvement_prompt}
                                ]
                            )
                            
                            # 토큰 사용량 추적
                            self.token_tracker.track_perplexity_usage(improved_response, improvement_prompt)
                            
                            improved_responses[response_key] = {
                                "text": improved_response.choices[0].message.content,
                                "raw": improved_response,
                                "analysis_data": analysis,
                                "response_metadata": {
                                    "analyzed_model": analyzed,
                                    "analyzers": [analyzer],
                                    "analysis_type": task_type,
                                    "task_id": task_id,
                                    "display_name": metadata.get("response_display_name") or f"{metadata.get('display_name', self.get_message(MK.UNKNOWN, self.language))} {self.get_message(MK.IMPROVED_RESPONSE_SUFFIX, self.language)}" # f"{metadata.get('display_name', '알 수 없음')} 개선된 응답"
                                }
                            }
                        except Exception as e:
                            print(f"Perplexity 응답 개선 오류: {str(e)}")
                            improved_responses[response_key] = {
                                "text": initial_responses[model]["text"],
                                "error": str(e),
                                "response_metadata": {
                                    "analyzed_model": model,
                                    "analyzers": [analyzer],
                                    "task_id": task_id
                                }
                            }
                    
                    print(f"{response_key} 응답 개선 완료")
                    
                except Exception as e:
                    print(f"{model} 응답 개선 오류: {str(e)}")
                    traceback.print_exc()
            
            # 다중 분석자 결합 - 조건부 실행으로 변경
            multiple_analysis_id = f"{model}_analyzed_by_multiple"
            if any(multiple_analysis_id in task_id for task_id in display_improvement_types):
                print(f"다중 분석 결합 실행: {multiple_analysis_id}")
                self._combine_multiple_analyses(model, model_analyses, improved_responses, user_input, initial_responses)
            else:
                print(f"다중 분석 결합 건너뛰기: {multiple_analysis_id} 요청되지 않음")
        
        return improved_responses

    def _create_improvement_prompt(self, user_input: str, initial_response: str, 
                             improvements: List[str], missing_info: List[str], 
                             follow_ups: List[str], analyzer: str) -> str:
        """개선 프롬프트 생성"""
        # 리스트를 문자열로 변환
        improvements_str = ", ".join(improvements) if improvements else self.get_message(MK.NONE, self.language)
        missing_info_str = ", ".join(missing_info) if missing_info else self.get_message(MK.NONE, self.language)
        follow_ups_str = ", ".join(follow_ups) if follow_ups else self.get_message(MK.GENERATE_APPROPRIATE_QUESTIONS, self.language)
        
        return self.get_message(MK.IMPROVEMENT_PROMPT, self.language, 
                            initial_response, analyzer.capitalize(), 
                            improvements_str, missing_info_str, follow_ups_str)

    def _combine_multiple_analyses(self, model: str, analyses: List[Dict], 
                             improved_responses: Dict[str, Dict], 
                             user_input: str, initial_responses: Dict[str, Dict]):
        """여러 분석 결과를 결합하여 하나의 개선 응답 생성"""
        # 동일한 모델을 분석한 여러 분석자가 있는지 확인
        analyzers = []
        for analysis in analyses:
            metadata = analysis.get("response_metadata", {})
            if metadata.get("analyzers"):
                analyzers.extend(metadata.get("analyzers", []))
        
        analyzers = list(set(analyzers))  # 중복 제거
        
        if len(analyzers) <= 1:
            # 분석자가 1개 이하면 결합 불필요
            return
        
        print(f"{model} 모델에 대한 다중 분석 결과 결합 중... (분석자: {', '.join(analyzers)})")
        
        # 결합된 분석 정보 - 개선된 구조
        combined_analysis = {
            "by_analyzer": {},  # 분석자별 결과 저장
            "improvements": [],
            "missing_information": [],
            "follow_up_questions": [],
            "response_metadata": {
                "analyzed_model": model,
                "analyzers": analyzers,
                "analysis_type": "combined",
                "task_id": f"{model}_analyzed_by_multiple"
            }
        }
        
        # 모델별 표시 이름 설정
        model_display_names = {
            "claude": "Claude",
            "openai": "GPT",
            "perplexity": "Perplexity"
        }
        model_display = model_display_names.get(model, model.capitalize())

        # 분석된 모델에 따라 분석자 조합 결정
        display_name = ""
        if model.lower() == "claude":
            display_name = self.get_message(MK.MULTIPLE_ANALYSIS_CLAUDE, self.language, model_display)
        elif model.lower() == "openai":
            display_name = self.get_message(MK.MULTIPLE_ANALYSIS_GPT, self.language, model_display)
        elif model.lower() == "perplexity":
            display_name = self.get_message(MK.MULTIPLE_ANALYSIS_PERPLEXITY, self.language, model_display)
        else:
            display_name = self.get_message(MK.MULTIPLE_ANALYSIS_GENERIC, self.language, model_display)

        # 메타데이터에 표시 이름 설정
        combined_analysis["response_metadata"]["display_name"] = display_name
        combined_analysis["response_metadata"]["response_display_name"] = f"{display_name} {self.get_message(MK.IMPROVED_RESPONSE_SUFFIX, self.language)}" # f"{display_name} 개선된 응답"

        # 분석자별로 결과 분류하고 결합
        for analysis in analyses:
            metadata = analysis.get("response_metadata", {})
            analyzer = metadata.get("analyzers", ["unknown"])[0]
            
            # 분석자별 결과 저장
            if analyzer not in combined_analysis["by_analyzer"]:
                combined_analysis["by_analyzer"][analyzer] = {
                    "improvements": [],
                    "missing_information": [],
                    "follow_up_questions": []
                }
            
            # 분석자별 결과에 추가
            combined_analysis["by_analyzer"][analyzer]["improvements"].extend(analysis.get("improvements", []))
            combined_analysis["by_analyzer"][analyzer]["missing_information"].extend(analysis.get("missing_information", []))
            combined_analysis["by_analyzer"][analyzer]["follow_up_questions"].extend(analysis.get("follow_up_questions", []))
            
            # 통합 결과에도 추가
            combined_analysis["improvements"].extend(analysis.get("improvements", []))
            combined_analysis["missing_information"].extend(analysis.get("missing_information", []))
            combined_analysis["follow_up_questions"].extend(analysis.get("follow_up_questions", []))

        # 중복 제거
        combined_analysis["improvements"] = list(set(combined_analysis["improvements"]))
        combined_analysis["missing_information"] = list(set(combined_analysis["missing_information"]))
        combined_analysis["follow_up_questions"] = list(set(combined_analysis["follow_up_questions"]))

        # 분석자별 결과도 중복 제거
        for analyzer in combined_analysis["by_analyzer"]:
            combined_analysis["by_analyzer"][analyzer]["improvements"] = list(set(combined_analysis["by_analyzer"][analyzer]["improvements"]))
            combined_analysis["by_analyzer"][analyzer]["missing_information"] = list(set(combined_analysis["by_analyzer"][analyzer]["missing_information"]))
            combined_analysis["by_analyzer"][analyzer]["follow_up_questions"] = list(set(combined_analysis["by_analyzer"][analyzer]["follow_up_questions"]))

        # 결합된 분석으로 응답 개선
        try:
            improvement_prompt = self._create_improvement_prompt(
                user_input, 
                initial_responses[model]["text"],
                combined_analysis["improvements"],
                combined_analysis["missing_information"],
                combined_analysis["follow_up_questions"],
                '+'.join(analyzers)  # 분석자 목록을 문자열로 변환
            )
            
            response_key = f"{model}_analyzed_by_multiple"
            
            # 모델에 따라 다른 클라이언트 사용
            if model == "claude":
                improved_response = self.claude_client.messages.create(
                    model=ACTIVE_CLAUDE_MODEL,
                    max_tokens=1500,
                    messages=[
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": initial_responses[model]["text"]},
                        {"role": "user", "content": improvement_prompt}
                    ]
                )
                
                # 토큰 사용량 추적
                self.token_tracker.track_claude_usage(improved_response, improvement_prompt)
                
                # 모델 타입에 따라 응답 텍스트 추출 방식 다르게 처리
                if model == "claude":
                    response_text = improved_response.content[0].text
                elif model == "openai":
                    response_text = improved_response.choices[0].message.content
                else:
                    # 기타 모델을 위한 기본 처리 (attributes 확인)
                    if hasattr(improved_response, 'choices'):
                        response_text = improved_response.choices[0].message.content
                    elif hasattr(improved_response, 'content'):
                        response_text = improved_response.content[0].text
                    else:
                        # 최후의 수단
                        response_text = str(improved_response)

                improved_responses[response_key] = {
                    "text": improved_response.content[0].text,
                    "raw": improved_response,
                    "analysis_data": combined_analysis,
                    "response_metadata": {
                        "analyzed_model": combined_analysis["response_metadata"]["analyzed_model"],
                        "analyzers": combined_analysis["response_metadata"]["analyzers"],
                        "analysis_type": combined_analysis["response_metadata"]["analysis_type"],
                        "task_id": combined_analysis["response_metadata"]["task_id"],
                        # 개선된 응답용 필드 사용
                        "display_name": combined_analysis["response_metadata"].get("response_display_name", 
                                        f"{combined_analysis['response_metadata']['display_name']} {self.get_message(MK.IMPROVED_RESPONSES, self.language)}")
                    }
                }
                
            elif model == "openai":
                improved_response = self.openai_client.chat.completions.create(
                    model=ACTIVE_GPT_MODEL,
                    messages=[
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": initial_responses[model]["text"]},
                        {"role": "user", "content": improvement_prompt}
                    ]
                )
                
                # 토큰 사용량 추적
                self.token_tracker.track_openai_usage(improved_response, improvement_prompt)
                
                improved_responses[response_key] = {
                    "text": improved_response.content[0].text,
                    "raw": improved_response,
                    "analysis_data": combined_analysis,
                    "response_metadata": {
                        "analyzed_model": combined_analysis["response_metadata"]["analyzed_model"],
                        "analyzers": combined_analysis["response_metadata"]["analyzers"],
                        "analysis_type": combined_analysis["response_metadata"]["analysis_type"],
                        "task_id": combined_analysis["response_metadata"]["task_id"],
                        # 개선된 응답용 필드 사용
                        "display_name": combined_analysis["response_metadata"].get("response_display_name", 
                                        f"{combined_analysis['response_metadata']['display_name']} {self.get_message(MK.IMPROVED_RESPONSES, self.language)}")
                    }
                }
                
            elif model == "perplexity" and self.perplexity_client:
                try:
                    # 시스템 메시지 준비
                    system_message = self.get_message(MK.PERPLEXITY_SYSTEM_MESSAGE, self.language) # f"당신은 AI 응답을 개선하는 전문가입니다. {analyzer.capitalize()}의 분석을 기반으로 이전 응답을 개선해주세요."

                    # OpenAI 인터페이스로 호출
                    improved_response = self.perplexity_client.chat.completions.create(
                        model=ACTIVE_PERPLEXITY_MODEL,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_input},
                            {"role": "assistant", "content": initial_responses[model]["text"]},
                            {"role": "user", "content": improvement_prompt}
                        ]
                    )
                    
                    # 토큰 사용량 추적
                    self.token_tracker.track_perplexity_usage(improved_response, improvement_prompt)
                    
                    improved_responses[response_key] = {
                        "text": improved_response.content[0].text,
                        "raw": improved_response,
                        "analysis_data": combined_analysis,
                        "response_metadata": {
                            "analyzed_model": combined_analysis["response_metadata"]["analyzed_model"],
                            "analyzers": combined_analysis["response_metadata"]["analyzers"],
                            "analysis_type": combined_analysis["response_metadata"]["analysis_type"],
                            "task_id": combined_analysis["response_metadata"]["task_id"],
                            # 개선된 응답용 필드 사용
                            "display_name": combined_analysis["response_metadata"].get("response_display_name", 
                                            f"{combined_analysis['response_metadata']['display_name']} {self.get_message(MK.IMPROVED_RESPONSES, self.language)}")
                        }
                    }

                except Exception as e:
                    print(f"Perplexity 결합 응답 개선 오류: {str(e)}")
                    improved_responses[response_key] = {
                        "text": initial_responses[model]["text"],
                        "error": str(e),
                        "response_metadata": {
                            "analyzed_model": model,
                            "analyzers": analyzers,
                            "task_id": response_key
                        }
                    }
            
            print(f"{response_key} 결합 응답 개선 완료")
            
        except Exception as e:
            print(f"{model} 결합 응답 개선 오류: {str(e)}")
            traceback.print_exc()

    def extract_question_paragraphs(self, text: str) -> List[str]:
        """텍스트에서 질문이 포함된 문단 추출 - 문단 단위로 처리하여 문맥 유지"""
        import re
        
        # 문단 분리 (빈 줄 기준)
        paragraphs = re.split(r'\n\s*\n', text)
        question_paragraphs = []
        
        # 현재 언어 확인
        lang = self.language if hasattr(self, 'language') else "ko"
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            contains_question = False
            
            # 1. 물음표 확인 - 모든 언어
            if '?' in paragraph:
                contains_question = True
                
            # 2. 언어별 질문 표시 확인
            elif lang == "ko":  # 한국어
                # 한국어 질문 키워드
                question_keywords = ['질문', '궁금', '알고 싶', '무엇인가', '어떻게', 
                                '왜', '언제', '어디서', '누구', '어느', '까요', '나요', 
                                '을까', '일까', '인가요', '할까요']
                
                # 한국어 문장 종결 패턴
                ending_patterns = ['까', '니', '나요', '을까요', '를까요', '인가요', '할까요', '일까요']
                
                # 키워드 확인
                for keyword in question_keywords:
                    if keyword in paragraph:
                        contains_question = True
                        break
                        
                # 종결 패턴 확인
                if not contains_question:
                    # 문장으로 분리
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    for sentence in sentences:
                        for pattern in ending_patterns:
                            if sentence.strip().endswith(pattern):
                                contains_question = True
                                break
                        if contains_question:
                            break
                            
            elif lang == "en":  # 영어
                # 영어 질문 키워드
                question_keywords = ['question', 'wonder', 'curious', 'inquiry', 'ask', 
                                'how can', 'how do', 'how would', 'what is', 'what are',
                                'where is', 'where can', 'when will', 'why does', 
                                'who is', 'would like to know']
                
                # 소문자로 변환해서 검사
                lower_p = paragraph.lower()
                for keyword in question_keywords:
                    if keyword in lower_p:
                        contains_question = True
                        break
            
            if contains_question:
                # 너무 긴 문단은 간결하게 만들기
                if len(paragraph) > 300:  # 300자 넘으면 축약
                    # 분석을 위해 문장 분리
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    
                    # 질문 문장 또는 질문 키워드가 있는 문장만 선택
                    question_sentences = []
                    for sentence in sentences:
                        if '?' in sentence:
                            question_sentences.append(sentence)
                        else:
                            # 언어별 질문 키워드 확인
                            contains_keyword = False
                            if lang == "ko":
                                for keyword in question_keywords:
                                    if keyword in sentence:
                                        contains_keyword = True
                                        break
                                for pattern in ending_patterns:
                                    if sentence.strip().endswith(pattern):
                                        contains_keyword = True
                                        break
                            elif lang == "en":
                                lower_s = sentence.lower()
                                for keyword in question_keywords:
                                    if keyword in lower_s:
                                        contains_keyword = True
                                        break
                            
                            if contains_keyword:
                                question_sentences.append(sentence)
                    
                    # 문장 결합 (최대 3개)
                    if question_sentences:
                        paragraph = ' '.join(question_sentences[:3])
                
                question_paragraphs.append(paragraph)
        
        # 결과가 없으면 기본 패턴으로 다시 시도 (물음표 있는 문장만 추출)
        if not question_paragraphs:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sentence in sentences:
                if '?' in sentence:
                    question_paragraphs.append(sentence.strip())
        
        # 중복 제거 및 최대 5개로 제한
        unique_paragraphs = []
        for p in question_paragraphs:
            if p not in unique_paragraphs:
                unique_paragraphs.append(p)
        
        return unique_paragraphs[:5]

    def _extract_follow_up_questions(self, improved_responses: Dict[str, Dict], 
                    response_models: Set[str],
                    analyses: Dict[str, Dict] = None,
                    ui_only_analyses: Dict[str, Dict] = None) -> Dict[str, Dict]:
        """
        개선된 응답에서 후속 질문 추출 및 모든 분석 결과의 제안 질문 수집
        UI에서 중복되지 않게 표시하도록 구조화된 결과 반환
        """
        # 결과 구조화: 모든 항목을 하나의 딕셔너리에 수집하되, 각 항목에 표시 정보 포함
        follow_up_questions = {}
        
        # 모델별 표시 이름 매핑
        model_display_names = {
            "claude": "Claude",
            "openai": "GPT", 
            "perplexity": "Perplexity"
        }
        
        # 추적용 세트 - 이미 처리된 항목 확인
        processed_tasks = set()
        processed_titles = set()  # 중복 제목 방지용
        
        # =========== 1. 개선된 응답에서 질문 추출 ===========
        for key, response in improved_responses.items():
            if response.get("error", False):
                continue
            
            # 메타데이터에서 모델 정보 가져오기
            metadata = response.get("response_metadata", {})
            task_id = metadata.get("task_id", "")
            processed_tasks.add(task_id)
            
            analyzed_model = metadata.get("analyzed_model", "unknown")
            analyzers = metadata.get("analyzers", [])
            analyzer = analyzers[0] if analyzers else "unknown"
            
            # 기본 모델 이름 생성
            analyzed_display = model_display_names.get(analyzed_model.lower(), analyzed_model.capitalize())
            
            # 분석 타입에 따른 기본 제목 생성
            if analyzer.lower() == analyzed_model.lower():
                base_title = self.get_message(MK.SELF_ANALYSIS_TITLE, self.language, analyzed_display)
            elif analyzer.lower() == "multiple":
                if analyzed_model.lower() == "claude":
                    base_title = self.get_message(MK.MULTIPLE_ANALYSIS_CLAUDE, self.language, analyzed_display)
                elif analyzed_model.lower() == "openai" or analyzed_model.lower() == "gpt":
                    base_title = self.get_message(MK.MULTIPLE_ANALYSIS_GPT, self.language, analyzed_display)
                elif analyzed_model.lower() == "perplexity":
                    base_title = self.get_message(MK.MULTIPLE_ANALYSIS_PERPLEXITY, self.language, analyzed_display)
                else:
                    base_title = self.get_message(MK.MULTIPLE_ANALYSIS_GENERIC, self.language, analyzed_display)
            else:
                analyzer_display = model_display_names.get(analyzer.lower(), analyzer.capitalize())
                base_title = self.get_message(MK.EXTERNAL_ANALYSIS_TITLE, self.language, analyzed_display, analyzer_display)
            
            # "Openai"를 "GPT"로 통일
            base_title = base_title.replace("Openai", "GPT")
            
            # 이미 처리된 제목은 건너뛰기
            if base_title in processed_titles:
                continue
                
            processed_titles.add(base_title)

            try:
                # 개선된 응답에서 질문 추출 (API 호출 없이)
                extracted_paragraphs = self.extract_question_paragraphs(response["text"])
                
                # 분석 데이터에서 제안된 후속 질문 가져오기
                suggested_questions = []
                if "analysis_data" in response:
                    suggested_questions = response["analysis_data"].get("follow_up_questions", [])
                
                # UI 구분을 위한 새로운 필드 추가: section_type
                # 'extracted'는 개선된 응답에서 추출된 질문 섹션임을 의미
                # 추출 질문과 제안 질문을 함께 포함하되, 섹션 구분을 위한 필드 사용
                follow_up_questions[base_title] = {
                    "extracted": extracted_paragraphs,
                    "suggested": suggested_questions,
                    "section_type": "extracted",  # UI에서 추출 질문 섹션임을 표시
                    "response_metadata": {
                        "analyzed_model": analyzed_model,
                        "analyzers": analyzers,
                        "base_title": base_title,  # 기본 제목 저장
                        "task_id": task_id
                    }
                }
                
            except Exception as e:
                print(f"{key} 후속 질문 추출 오류: {str(e)}")
        
        # =========== 2. 분석 결과에서 제안 질문만 수집 ===========
        
        def process_analyses_for_suggestions(analyses_dict):
            """분석 결과에서 제안 질문 수집 (내부 헬퍼 함수)"""
            if not analyses_dict:
                return
                
            for task_id, analysis in analyses_dict.items():
                # 이미 처리된 task_id는 건너뛰기
                if task_id in processed_tasks:
                    continue
                    
                if "error" in analysis or not isinstance(analysis, dict):
                    continue
                    
                # 메타데이터에서 모델 정보 가져오기
                metadata = analysis.get("response_metadata", {})
                analyzed_model = metadata.get("analyzed_model", "unknown")
                analyzers = metadata.get("analyzers", [])
                analyzer = analyzers[0] if analyzers else "unknown"
                
                # 기본 모델 이름 생성
                analyzed_display = model_display_names.get(analyzed_model.lower(), analyzed_model.capitalize())
                
                # 분석 타입에 따른 기본 제목 생성
                if analyzer.lower() == analyzed_model.lower():
                    base_title = self.get_message(MK.SELF_ANALYSIS_TITLE, self.language, analyzed_display)
                elif analyzer.lower() == "multiple":
                    if analyzed_model.lower() == "claude":
                        base_title = self.get_message(MK.MULTIPLE_ANALYSIS_CLAUDE, self.language, analyzed_display)
                    elif analyzed_model.lower() == "openai" or analyzed_model.lower() == "gpt":
                        base_title = self.get_message(MK.MULTIPLE_ANALYSIS_GPT, self.language, analyzed_display)
                    elif analyzed_model.lower() == "perplexity":
                        base_title = self.get_message(MK.MULTIPLE_ANALYSIS_PERPLEXITY, self.language, analyzed_display)
                    else:
                        base_title = self.get_message(MK.MULTIPLE_ANALYSIS_GENERIC, self.language, analyzed_display)
                else:
                    analyzer_display = model_display_names.get(analyzer.lower(), analyzer.capitalize())
                    base_title = self.get_message(MK.EXTERNAL_ANALYSIS_TITLE, self.language, analyzed_display, analyzer_display)
                
                # "Openai"를 "GPT"로 통일
                base_title = base_title.replace("Openai", "GPT")
                
                # 이미 처리된 제목은 건너뛰기
                if base_title in processed_titles:
                    continue
                    
                processed_titles.add(base_title)
                
                # 제안된 질문만 추가하고 'suggested' 섹션임을 표시
                follow_up_questions[base_title] = {
                    "extracted": [],  # 이 섹션에서는 extracted 필드를 비워둠
                    "suggested": analysis.get("follow_up_questions", []),
                    "section_type": "suggested",  # UI에서 제안된 질문 섹션임을 표시
                    "response_metadata": {
                        "analyzed_model": analyzed_model,
                        "analyzers": analyzers,
                        "base_title": base_title,  # 기본 제목 저장
                        "task_id": task_id
                    }
                }
        
        # 분석 결과 처리
        if analyses:
            process_analyses_for_suggestions(analyses)
        
        # UI 전용 분석 결과 처리 
        if ui_only_analyses:
            process_analyses_for_suggestions(ui_only_analyses)

        return follow_up_questions
            
    def _extract_json_data(self, text: str) -> Dict:
        """텍스트에서 JSON 데이터 추출 - 강화된 버전"""
        try:
            # 원본 응답 저장 (디버깅용)
            original_text = text
            
            # 1. 코드 블록 내 JSON 찾기 시도
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # 2. 모든 중괄호 쌍 찾기 (강화된 패턴)
                matches = []
                open_braces = 0
                start_pos = -1
                
                for i, char in enumerate(text):
                    if char == '{':
                        if open_braces == 0:
                            start_pos = i
                        open_braces += 1
                    elif char == '}':
                        open_braces -= 1
                        if open_braces == 0 and start_pos != -1:
                            matches.append(text[start_pos:i+1])
                
                if matches:
                    # 가장 큰 JSON 객체 선택
                    json_str = max(matches, key=len)
                else:
                    # 3. 단순히 첫 중괄호부터 마지막 중괄호까지 찾기
                    start = text.find('{')
                    end = text.rfind('}')
                    if start != -1 and end != -1 and start < end:
                        json_str = text[start:end+1]
                    else:
                        print(f"JSON 찾기 실패: {text[:100]}...")
                        # Perplexity 분석에 대한 폴백 처리
                        if "perplexity" in text.lower():
                            return self._fallback_extract_perplexity_analysis(text)
                        return {"error": self.get_message(MK.JSON_FORMAT_NOT_FOUND, self.language), "raw": text}
            
            # JSON 정리 단계
            # 키가 따옴표로 감싸져 있지 않은 경우 수정
            fixed_json = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
            
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                # 작은따옴표를 큰따옴표로 변환 시도
                fixed_json = fixed_json.replace("'", '"')
                try:
                    return json.loads(fixed_json)
                except json.JSONDecodeError:
                    print(self.get_message(MK.JSON_PARSING_FAILED_WITH_DETAILS, self.language, fixed_json[:100])) # f"JSON 파싱 실패: {fixed_json[:100]}..."

                    # Perplexity 분석에 대한 폴백 처리
                    if "perplexity" in text.lower():
                        return self._fallback_extract_perplexity_analysis(text)
                    
                    # 마지막 시도: 각 줄을 분석하여 수동으로 JSON 구성
                    try:
                        # 수동 JSON 파싱
                        return self._manual_json_extraction(original_text)
                    except:
                        return {"error": self.get_message(MK.JSON_PARSING_FAILED, self.language), "raw": original_text}
                    
        except Exception as e:
            print(f"JSON 추출 오류: {str(e)}")
            return {"error": str(e), "raw": text}

    def _fallback_extract_perplexity_analysis(self, text: str) -> Dict:
        """Perplexity 분석을 위한 폴백 메서드"""
        improvements = []
        missing_info = []
        follow_ups = []
        
        # 개선점 추출
        improvement_section = re.search(r'1\.(.+?)(?:2\.|$)', text, re.DOTALL)
        if improvement_section:
            improvements_text = improvement_section.group(1).strip()
            improvements = [item.strip() for item in re.split(r'[-•*]|\d+\.', improvements_text) if item.strip()]
        
        # 누락된 정보 추출
        missing_section = re.search(r'2\.(.+?)(?:3\.|$)', text, re.DOTALL)
        if missing_section:
            missing_text = missing_section.group(1).strip()
            missing_info = [item.strip() for item in re.split(r'[-•*]|\d+\.', missing_text) if item.strip()]
        
        # 후속 질문 추출
        followup_section = re.search(r'3\.(.+?)(?:$)', text, re.DOTALL)
        if followup_section:
            followup_text = followup_section.group(1).strip()
            follow_ups = [item.strip() for item in re.split(r'[-•*]|\d+\.', followup_text) if item.strip()]
        
        return {
            "improvements": improvements[:3],  # 최대 3개까지만
            "missing_information": missing_info[:3],
            "follow_up_questions": follow_ups[:3]
        }

    def _manual_json_extraction(self, text: str) -> Dict:
        """일반적인 분석 텍스트에서 수동으로 항목 추출"""
        improvements = []
        missing_info = []
        follow_ups = []
        
        # 각 줄을 확인하면서 관련 항목 추출
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 섹션 식별
            if "개선" in line or "improvement" in line.lower():
                current_section = "improvements"
                continue
            elif "놓친" in line or "missing" in line.lower():
                current_section = "missing_info"
                continue
            elif "질문" in line or "follow" in line.lower():
                current_section = "follow_ups"
                continue
                
            # 항목 추출 (번호, 불릿 등 제거)
            item = re.sub(r'^[-•*]|\d+\.\s*', '', line).strip()
            if item and len(item) > 5:  # 최소 길이로 필터링
                if current_section == "improvements":
                    improvements.append(item)
                elif current_section == "missing_info":
                    missing_info.append(item)
                elif current_section == "follow_ups":
                    follow_ups.append(item)
        
        return {
            "improvements": improvements,
            "missing_information": missing_info,
            "follow_up_questions": follow_ups
        }

    def generate_display_title(self, model: str, metadata: Dict = None) -> str:
        """
        응답 표시용 제목 생성
        
        Args:
            model: 응답 모델
            metadata: 응답 메타데이터
            
        Returns:
            str: 표시 제목
        """
        model_display_names = {
            "claude": "Claude",
            "openai": "GPT",
            "perplexity": "Perplexity"
        }
        
        model_display = model_display_names.get(model, model.capitalize())
        
        if not metadata:
            return f"{model_display} {self.get_message(MK.INITIAL_RESPONSE_LABEL, self.language)}" # 초기 응답

        analysis_type = metadata.get("analysis_type")
        analyzers = metadata.get("analyzers", [])
        
        if not analyzers:
            return f"{model_display} {self.get_message(MK.RESPONSE_LABEL, self.language)}"
        
        # 자체 분석인 경우
        if analysis_type == "self":
            base_title = self.get_message(MK.SELF_ANALYSIS_TITLE, self.language, model_display)
            return base_title + self.get_message(MK.IMPROVED_RESPONSE_SUFFIX, self.language)
        
        # 분석자가 여러 개인 경우 (결합)
        if len(analyzers) > 1 or analysis_type == "combined":
            # "multiple"이라는 일반 분석자가 있으면 실제 분석자 목록으로 대체
            if "multiple" in analyzers and len(analyzers) == 1:
                if model == "claude":
                    base_title = self.get_message(MK.MULTIPLE_ANALYSIS_CLAUDE, self.language, model_display)
                elif model == "openai":
                    base_title = self.get_message(MK.MULTIPLE_ANALYSIS_GPT, self.language, model_display)
                elif model == "perplexity":
                    base_title = self.get_message(MK.MULTIPLE_ANALYSIS_PERPLEXITY, self.language, model_display)
                else:
                    base_title = self.get_message(MK.MULTIPLE_ANALYSIS_GENERIC, self.language, model_display)
            else:
                # 각 분석자 이름을 사용자 친화적인 이름으로 변환
                analyzer_names = [model_display_names.get(a, a.capitalize()) for a in analyzers]
                # "+" 기호로 구분하여 결합
                analyzers_text = "+".join(analyzer_names)
                base_title = self.get_message(MK.EXTERNAL_ANALYSIS_TITLE, self.language, model_display, analyzers_text)
            
            return base_title + self.get_message(MK.IMPROVED_RESPONSE_SUFFIX, self.language)
        
        # 단일 외부 분석자
        analyzer_display = model_display_names.get(analyzers[0], analyzers[0].capitalize())
        base_title = self.get_message(MK.EXTERNAL_ANALYSIS_TITLE, self.language, model_display, analyzer_display)
        return base_title + self.get_message(MK.IMPROVED_RESPONSE_SUFFIX, self.language)

    def generate_display_title_from_metadata(self, metadata: dict, current_lang: str = None) -> str:
        """
        메타데이터로부터 현재 언어에 맞는 표시 제목 동적 생성
        
        Args:
            metadata: 응답 메타데이터
            current_lang: 현재 언어 (지정하지 않으면 미들웨어의 현재 언어 사용)
            
        Returns:
            str: 현재 언어로 생성된 표시 제목
        """
        # 메타데이터가 없거나 필수 필드가 없으면 기본값 반환
        if not metadata or not isinstance(metadata, dict):
            return self.get_message(MK.UNKNOWN, current_lang or self.language)
        
        # 현재 언어 설정
        lang = current_lang or self.language
        
        # 메타데이터에서 필요한 정보 추출
        analyzed_model = metadata.get("analyzed_model")
        analyzers = metadata.get("analyzers", [])
        analysis_type = metadata.get("analysis_type")
        
        if not analyzed_model:
            return self.get_message(MK.UNKNOWN, lang)
        
        # 모델별 표시 이름 매핑
        model_display_names = {
            "claude": "Claude",
            "openai": "GPT",
            "perplexity": "Perplexity"
        }
        
        # 분석된 모델의 표시 이름
        analyzed_display = model_display_names.get(analyzed_model, analyzed_model.capitalize())
        
        # 초기 응답인 경우 특별 처리 (추가된 부분)
        if analysis_type == "initial" or metadata.get("is_initial_response", False):
            return analyzed_display  # 초기 응답은 모델 이름만 표시
        
        # 분석 유형에 따른 제목 생성 (기존 로직)
        if analysis_type == "self" or "self" in metadata.get("task_id", ""):
            base_title = self.get_message(MK.SELF_ANALYSIS_TITLE, lang, analyzed_display)
        elif analysis_type == "combined" or not analyzers or "multiple" in analyzers:
            # 다중 분석 케이스
            if analyzed_model == "claude":
                base_title = self.get_message(MK.MULTIPLE_ANALYSIS_CLAUDE, lang, analyzed_display)
            elif analyzed_model == "openai":
                base_title = self.get_message(MK.MULTIPLE_ANALYSIS_GPT, lang, analyzed_display)
            elif analyzed_model == "perplexity":
                base_title = self.get_message(MK.MULTIPLE_ANALYSIS_PERPLEXITY, lang, analyzed_display)
            else:
                base_title = self.get_message(MK.MULTIPLE_ANALYSIS_GENERIC, lang, analyzed_display)
        else:
            # 단일 외부 분석 케이스
            analyzer = analyzers[0] if analyzers else "unknown"
            analyzer_display = model_display_names.get(analyzer, analyzer.capitalize())
            base_title = self.get_message(MK.EXTERNAL_ANALYSIS_TITLE, lang, analyzed_display, analyzer_display)
        
        # 개선된 응답인 경우 접미사 추가
        if metadata.get("is_improved_response", False):
            return f"{base_title}{self.get_message(MK.IMPROVED_RESPONSE_SUFFIX, lang)}"
        
        return base_title

    def format_comparison_result(self, result: Dict) -> Dict:
        """
        비교 결과를 표시용으로 포맷팅 - 메타데이터 강화 및 동적 제목 생성 지원
        
        Args:
            result: 처리 결과 데이터
            
        Returns:
            dict: 포맷팅된 결과 (메타데이터 포함)
        """
        # 디버깅 출력 추가
        print("DEBUG - 수신된 개선된 응답 키:", list(result.get("improved_responses", {}).keys()))
        
        output = {
            "comparison": {
                "initial_responses": {},
                "improved_responses": {},
                "follow_up_questions": {},
                "ai_analyses": {},
                "response_models": result.get("response_models", []),
                "analysis_models": result.get("analysis_models", []),
                "analysis_only_models": result.get("analysis_only_models", []),
                "analysis_tasks": result.get("analysis_tasks", []),
                "display_improvement_types": result.get("display_improvement_types", "claude_analyzed_by_openai")
            },
            "processing_info": {
                "time": result["processing_time"],
                "token_usage": result["token_usage"]
            }
        }
        
        # 단일 결과 모드인 경우
        if "final_response" in result:
            output["final_response"] = result["final_response"]
            
            # 전체 분석 결과가 있는 경우
            if "full_analysis" in result:
                full_analysis = result["full_analysis"]
                
                # 응답 생성 모델 정보 추가
                output["comparison"]["response_models"] = full_analysis.get("response_models", [])
                output["comparison"]["analysis_models"] = full_analysis.get("analysis_models", [])
                output["comparison"]["analysis_only_models"] = full_analysis.get("analysis_only_models", [])
                output["comparison"]["analysis_tasks"] = full_analysis.get("analysis_tasks", [])
                
                # 결과 포맷팅을 위해 full_analysis 사용
                result = full_analysis
        
        # 초기 응답 포맷팅
        if "initial_responses" in result:
            for model_key, response in result["initial_responses"].items():
                if not response.get("error", False):
                    # 메타데이터 생성 및 추가
                    initial_metadata = {
                        "analyzed_model": model_key,
                        "analyzers": [],
                        "analysis_type": "initial",
                        "task_id": f"{model_key}_initial",
                        "is_initial_response": True
                    }
                    
                    # 동적 제목 생성을 위한 메타데이터도 함께 저장
                    model_display = self.generate_display_title_from_metadata(initial_metadata)
                    output["comparison"]["initial_responses"][model_display] = {
                        "text": response["text"],
                        "metadata": initial_metadata  # 메타데이터 추가
                    }
                else:
                    # 오류 응답도 메타데이터 포함
                    error_metadata = {
                        "analyzed_model": model_key,
                        "has_error": True,
                        "error_message": response.get('error', self.get_message(MK.UNKNOWN_ERROR_TEXT, self.language))
                    }
                    model_display = self.generate_display_title_from_metadata(error_metadata)
                    output["comparison"]["initial_responses"][model_display] = {
                        "text": f"{self.get_message(MK.ERROR_TEXT, self.language)}: {response.get('error', self.get_message(MK.UNKNOWN_ERROR_TEXT, self.language))}",
                        "metadata": error_metadata
                    }
        
        # 개선된 응답 포맷팅
        if "improved_responses" in result:
            for response_key, response in result["improved_responses"].items():
                if not response.get("error", False):
                    # 메타데이터에서 표시 이름 가져오기
                    metadata = response.get("response_metadata", {})
                    
                    # 개선된 응답임을 표시
                    metadata["is_improved_response"] = True
                    
                    # 동적 제목 생성
                    display_name = self.generate_display_title_from_metadata(metadata)
                    
                    output["comparison"]["improved_responses"][display_name] = {
                        "text": response["text"],
                        "metadata": metadata  # 메타데이터 저장
                    }
                else:
                    error_metadata = {
                        "has_error": True,
                        "error_message": response.get('error', self.get_message(MK.UNKNOWN_ERROR_TEXT, self.language))
                    }
                    if "response_metadata" in response:
                        error_metadata.update(response["response_metadata"])
                    
                    display_name = self.generate_display_title_from_metadata(error_metadata)
                    output["comparison"]["improved_responses"][display_name] = {
                        "text": f"{self.get_message(MK.ERROR_TEXT, self.language)}: {response.get('error', self.get_message(MK.UNKNOWN_ERROR_TEXT, self.language))}",
                        "metadata": error_metadata
                    }

        # 후속 질문 및 분석 결과도 유사하게 메타데이터로 처리
        # 후속 질문 포맷팅
        if "follow_up_questions" in result:
            for base_title, questions_data in result["follow_up_questions"].items():
                if isinstance(questions_data, dict):
                    metadata = questions_data.get("response_metadata", {})
                    # 제목 동적 생성을 위한 메타데이터 업데이트
                    if "section_type" in questions_data:
                        metadata["section_type"] = questions_data["section_type"]
                    
                    # 현재 언어로 동적 제목 생성
                    dynamic_title = self.generate_display_title_from_metadata(metadata)
                    
                    # 섹션 타입에 따라 접미사 추가
                    if questions_data.get("section_type") == "extracted":
                        dynamic_title += f" {self.get_message(MK.EXTRACTED_QUESTIONS_LABEL, self.language)}"
                    elif questions_data.get("section_type") == "suggested":
                        dynamic_title += f" {self.get_message(MK.SUGGESTED_QUESTIONS_LABEL, self.language)}"
                    
                    # 메타데이터를 포함하여 저장
                    output["comparison"]["follow_up_questions"][dynamic_title] = {
                        "extracted": questions_data.get("extracted", []),
                        "suggested": questions_data.get("suggested", []),
                        "section_type": questions_data.get("section_type", ""),
                        "metadata": metadata
                    }
                else:
                    # 문자열 등 다른 형식으로 저장된 경우
                    output["comparison"]["follow_up_questions"][base_title] = questions_data
        
        # 분석 포맷팅
        if "analyses" in result:
            for analysis_key, analysis in result["analyses"].items():
                if "error" not in analysis and isinstance(analysis, dict):
                    metadata = analysis.get("response_metadata", {})
                    # 현재 언어로 동적 제목 생성
                    display_title = self.generate_display_title_from_metadata(metadata)
                    
                    # 메타데이터를 포함하여 저장
                    output["comparison"]["ai_analyses"][display_title] = {
                        "improvements": analysis.get("improvements", []),
                        "missing_information": analysis.get("missing_information", []),
                        "follow_up_questions": analysis.get("follow_up_questions", []),
                        "metadata": metadata
                    }
                else:
                    # 오류 케이스
                    if isinstance(analysis, dict) and "response_metadata" in analysis:
                        metadata = analysis["response_metadata"]
                        display_title = self.generate_display_title_from_metadata(metadata)
                    else:
                        display_title = analysis_key.replace("_", " ").capitalize().replace("Openai", "GPT")
                    
                    output["comparison"]["ai_analyses"][display_title] = {
                        "error_text": f"{self.get_message(MK.ERROR_TEXT, self.language)}: {analysis.get('error', self.get_message(MK.UNKNOWN_ERROR_TEXT, self.language))}",
                        "has_error": True
                    }
        
        # UI 분석 탭에만 표시할 추가 분석 결과도 동일하게 처리
        if "ui_only_analyses" in result:
            for analysis_key, analysis in result["ui_only_analyses"].items():
                if "error" not in analysis and isinstance(analysis, dict):
                    metadata = analysis.get("response_metadata", {})
                    # 현재 언어로 동적 제목 생성
                    display_title = self.generate_display_title_from_metadata(metadata)
                    
                    # 이미 있는 키인지 확인하고 중복 방지
                    if display_title not in output["comparison"]["ai_analyses"]:
                        output["comparison"]["ai_analyses"][display_title] = {
                            "improvements": analysis.get("improvements", []),
                            "missing_information": analysis.get("missing_information", []),
                            "follow_up_questions": analysis.get("follow_up_questions", []),
                            "metadata": metadata
                        }
                else:
                    # 오류 케이스 - 메타데이터로 제목 생성
                    if isinstance(analysis, dict) and "response_metadata" in analysis:
                        metadata = analysis["response_metadata"]
                        display_title = self.generate_display_title_from_metadata(metadata)
                    else:
                        display_title = analysis_key.replace("_", " ").capitalize().replace("Openai", "GPT")
                    
                    if display_title not in output["comparison"]["ai_analyses"]:
                        output["comparison"]["ai_analyses"][display_title] = {
                            "error_text": f"{self.get_message(MK.ERROR_TEXT, self.language)}: {analysis.get('error', self.get_message(MK.UNKNOWN_ERROR_TEXT, self.language))}",
                            "has_error": True
                        }

        return output

    def link_tasks_to_middleware(self, tasks=None):
        """분석 작업에 미들웨어 참조 설정
        
        Args:
            tasks: 미들웨어를 링크할 분석 작업 목록 (None일 경우 캐시된 모든 작업)
        """
        try:
            # 특정 태스크만 처리
            if tasks and isinstance(tasks, (list, set)):
                for task in tasks:
                    if isinstance(task, AnalysisTask) and not hasattr(task, 'middleware') or task.middleware is None:
                        task.middleware = self
                        print(f"태스크 '{task}' 미들웨어 링크 완료")
            
            # 캐시된 모든 분석 작업 처리
            elif hasattr(self, '_analysis_cache'):
                for key, analysis in self._analysis_cache.items():
                    if isinstance(analysis, dict) and "response_metadata" in analysis:
                        metadata = analysis["response_metadata"]
                        task_id = metadata.get("task_id")
                        
                        # 해당 태스크를 찾아서 미들웨어 링크
                        if task_id:
                            parts = task_id.split("_analyzed_by_")
                            if len(parts) == 2:
                                analyzed, analyzer = parts
                                
                                # 자체 분석 처리
                                if analyzer == "self":
                                    task = AnalysisTask(analyzer=analyzed, analyzed=analyzed, task_type="self")
                                else:
                                    task = AnalysisTask(analyzer=analyzer, analyzed=analyzed, task_type="external")
                                    
                                task.middleware = self
                                print(f"캐시된 태스크 '{task_id}' 미들웨어 링크 완료")
        except Exception as e:
            print(f"미들웨어 링크 오류: {str(e)}")
            traceback.print_exc()

    def get_usage_summary(self):
        """현재 토큰 사용량 요약 반환"""
        return self.token_tracker.get_current_session_summary()
    
    def print_usage_report(self):
        """현재 토큰 사용량 보고서 출력"""
        self.token_tracker.print_usage_summary()
    
    def save_usage_data(self):
        """토큰 사용량 데이터 저장"""
        self.token_tracker.save_usage_data()
        print("토큰 사용량 데이터가 저장되었습니다.")
    
    def close(self):
        """미들웨어 종료 시 호출"""
        self.save_usage_data()
        print("미들웨어가 종료되었습니다.")