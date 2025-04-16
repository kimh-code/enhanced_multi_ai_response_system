"""
토큰 사용량 추적 모듈 - Perplexity 지원 추가
"""

import json
import os
from datetime import datetime
from config import MODEL_COSTS, get_model_cost, ACTIVE_CLAUDE_MODEL, ACTIVE_GPT_MODEL

class TokenUsageTracker:
    def __init__(self, log_file="token_usage.json"):
        self.log_file = log_file
        self.session_usage = {
            "session_id": datetime.now().strftime("%Y%m%d%H%M%S"),
            "start_time": datetime.now().isoformat(),
            "models_used": {},
            "total_usage": {
                "claude": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost": 0.0
                },
                "openai": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost": 0.0
                },
                "perplexity": {  # Perplexity 추가
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost": 0.0
                },
                "combined": {
                    "total_tokens": 0,
                    "estimated_cost": 0.0
                }
            },
            "queries": []
        }
        self.load_previous_usage()
    
    def load_previous_usage(self):
        """이전 사용량 데이터 로드"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    self.usage_history = json.load(f)
            except json.JSONDecodeError:
                self.usage_history = {"sessions": []}
        else:
            self.usage_history = {"sessions": []}
    
    def track_claude_usage(self, response, query_text):
        """Claude API 응답에서 토큰 사용량 추적"""
        # 모델 이름 가져오기
        model_name = response.model
        
        # 이 모델이 이미 사용된 적 있는지 확인
        if model_name not in self.session_usage["models_used"]:
            # 모델에 맞는 가격 정보 가져오기
            cost_info = get_model_cost(model_name)
            self.session_usage["models_used"][model_name] = {
                "provider": "anthropic",
                "price_per_1k": cost_info
            }
        
        # 사용된 cost_info 가져오기
        cost_info = self.session_usage["models_used"][model_name]["price_per_1k"]
        
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            "model": model_name
        }
        
        # 비용 계산
        prompt_cost = (usage["prompt_tokens"] / 1000) * cost_info["prompt"]
        completion_cost = (usage["completion_tokens"] / 1000) * cost_info["completion"]
        usage["estimated_cost"] = round(prompt_cost + completion_cost, 6)
        
        # 디버깅 로그 추가
        print(f"DEBUG - Claude 토큰 사용: 입력={usage['prompt_tokens']}, 출력={usage['completion_tokens']}, 총={usage['total_tokens']}, 비용=${usage['estimated_cost']:.4f}")

        # 세션 사용량 업데이트
        self.session_usage["total_usage"]["claude"]["prompt_tokens"] += usage["prompt_tokens"]
        self.session_usage["total_usage"]["claude"]["completion_tokens"] += usage["completion_tokens"]
        self.session_usage["total_usage"]["claude"]["total_tokens"] += usage["total_tokens"]
        self.session_usage["total_usage"]["claude"]["estimated_cost"] += usage["estimated_cost"]
        
        # 통합 사용량 업데이트
        self.session_usage["total_usage"]["combined"]["total_tokens"] += usage["total_tokens"]
        self.session_usage["total_usage"]["combined"]["estimated_cost"] += usage["estimated_cost"]
        
        # 쿼리 정보 추가
        query_info = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "provider": "anthropic",
            "query": query_text[:100] + "..." if len(query_text) > 100 else query_text,
            "usage": usage
        }
        self.session_usage["queries"].append(query_info)
        
        return usage
    
    def track_openai_usage(self, response, query_text):
        """OpenAI API 응답에서 토큰 사용량 추적"""
        # 모델 이름 가져오기
        model_name = response.model
        
        # 이 모델이 이미 사용된 적 있는지 확인
        if model_name not in self.session_usage["models_used"]:
            # 모델에 맞는 가격 정보 가져오기
            cost_info = get_model_cost(model_name)
            self.session_usage["models_used"][model_name] = {
                "provider": "openai",
                "price_per_1k": cost_info
            }
        
        # 사용된 cost_info 가져오기
        cost_info = self.session_usage["models_used"][model_name]["price_per_1k"]
        
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "model": model_name
        }
        
        # 비용 계산
        prompt_cost = (usage["prompt_tokens"] / 1000) * cost_info["prompt"]
        completion_cost = (usage["completion_tokens"] / 1000) * cost_info["completion"]
        usage["estimated_cost"] = round(prompt_cost + completion_cost, 6)
        
        # 디버깅 로그 추가
        print(f"DEBUG - OpenAI 토큰 사용: 입력={usage['prompt_tokens']}, 출력={usage['completion_tokens']}, 총={usage['total_tokens']}, 비용=${usage['estimated_cost']:.4f}")
        
        # 세션 사용량 업데이트
        self.session_usage["total_usage"]["openai"]["prompt_tokens"] += usage["prompt_tokens"]
        self.session_usage["total_usage"]["openai"]["completion_tokens"] += usage["completion_tokens"]
        self.session_usage["total_usage"]["openai"]["total_tokens"] += usage["total_tokens"]
        self.session_usage["total_usage"]["openai"]["estimated_cost"] += usage["estimated_cost"]
        
        # 통합 사용량 업데이트
        self.session_usage["total_usage"]["combined"]["total_tokens"] += usage["total_tokens"]
        self.session_usage["total_usage"]["combined"]["estimated_cost"] += usage["estimated_cost"]
        
        # 쿼리 정보 추가
        query_info = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "provider": "openai",
            "query": query_text[:100] + "..." if len(query_text) > 100 else query_text,
            "usage": usage
        }
        self.session_usage["queries"].append(query_info)
        
        return usage
    
    def track_perplexity_usage(self, response, query_text):
        """OpenAI 인터페이스를 통한 Perplexity API 응답에서 토큰 사용량 추적"""
        try:
            # 모델 이름 가져오기
            model_name = response.model
            
            # 이 모델이 이미 사용된 적 있는지 확인
            if model_name not in self.session_usage["models_used"]:
                # 모델에 맞는 가격 정보 가져오기 (기본값 사용)
                cost_info = {
                    "prompt": 0.002,  # 예상 가격 - 실제 가격으로 업데이트 필요
                    "completion": 0.008  # 예상 가격 - 실제 가격으로 업데이트 필요
                }
                self.session_usage["models_used"][model_name] = {
                    "provider": "perplexity",
                    "price_per_1k": cost_info
                }
            
            # 사용된 cost_info 가져오기
            cost_info = self.session_usage["models_used"][model_name]["price_per_1k"]
            
            # OpenAI 인터페이스에서 토큰 사용량 추출
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "model": model_name
            }
            
            # 비용 계산
            prompt_cost = (usage["prompt_tokens"] / 1000) * cost_info["prompt"]
            completion_cost = (usage["completion_tokens"] / 1000) * cost_info["completion"]
            usage["estimated_cost"] = round(prompt_cost + completion_cost, 6)
            
            # 디버깅 로그 추가
            print(f"DEBUG - Perplexity 토큰 사용: 입력={usage['prompt_tokens']}, 출력={usage['completion_tokens']}, 총={usage['total_tokens']}, 비용=${usage['estimated_cost']:.4f}")

            # 세션 사용량 업데이트
            self.session_usage["total_usage"]["perplexity"]["prompt_tokens"] += usage["prompt_tokens"]
            self.session_usage["total_usage"]["perplexity"]["completion_tokens"] += usage["completion_tokens"]
            self.session_usage["total_usage"]["perplexity"]["total_tokens"] += usage["total_tokens"]
            self.session_usage["total_usage"]["perplexity"]["estimated_cost"] += usage["estimated_cost"]
            
            # 통합 사용량 업데이트
            self.session_usage["total_usage"]["combined"]["total_tokens"] += usage["total_tokens"]
            self.session_usage["total_usage"]["combined"]["estimated_cost"] += usage["estimated_cost"]
            
            # 쿼리 정보 추가
            query_info = {
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "provider": "perplexity",
                "query": query_text[:100] + "..." if len(query_text) > 100 else query_text,
                "usage": usage
            }
            self.session_usage["queries"].append(query_info)
            
            return usage
        except Exception as e:
            print(f"Perplexity 토큰 사용량 추적 오류: {str(e)}")
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost": 0.0}

    def save_usage_data(self):
        """현재 세션의 사용량 데이터 저장"""
        self.session_usage["end_time"] = datetime.now().isoformat()
        
        # 사용량 통계 추가
        model_stats = {}
        for model_name, model_info in self.session_usage["models_used"].items():
            model_stats[model_name] = {
                "total_tokens": 0,
                "total_cost": 0
            }
        
        for query in self.session_usage["queries"]:
            model = query["model"]
            if model in model_stats:
                model_stats[model]["total_tokens"] += query["usage"]["total_tokens"]
                model_stats[model]["total_cost"] += query["usage"]["estimated_cost"]
        
        self.session_usage["model_stats"] = model_stats
        
        # 히스토리에 추가
        self.usage_history["sessions"].append(self.session_usage)
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.usage_history, f, ensure_ascii=False, indent=2)
    
    def get_current_session_summary(self):
        """현재 세션의 사용량 요약 반환"""
        return {
            "session_id": self.session_usage["session_id"],
            "duration": self._calculate_duration(),
            "total_queries": len(self.session_usage["queries"]),
            "models_used": self.session_usage["models_used"],
            "claude_usage": self.session_usage["total_usage"]["claude"],
            "openai_usage": self.session_usage["total_usage"]["openai"],
            "perplexity_usage": self.session_usage["total_usage"]["perplexity"],  # Perplexity 추가
            "combined": self.session_usage["total_usage"]["combined"]
        }
    
    def _calculate_duration(self):
        """세션 지속 시간 계산"""
        start = datetime.fromisoformat(self.session_usage["start_time"])
        end = datetime.now()
        duration_seconds = (end - start).total_seconds()
        
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{int(hours)}시간 {int(minutes)}분 {int(seconds)}초"
        elif minutes > 0:
            return f"{int(minutes)}분 {int(seconds)}초"
        else:
            return f"{int(seconds)}초"
    
    def print_usage_summary(self):
        """현재 세션의 사용량 요약 출력"""
        summary = self.get_current_session_summary()
        
        print("\n===== 토큰 사용량 요약 =====")
        print(f"세션 ID: {summary['session_id']}")
        print(f"세션 지속 시간: {summary['duration']}")
        print(f"총 쿼리 수: {summary['total_queries']}")
        
        print("\n사용된 모델:")
        for model, info in summary["models_used"].items():
            cost = info["price_per_1k"]
            print(f"  - {model} ({info['provider']})")
            print(f"    가격: ${cost['prompt']}/1K 토큰(입력), ${cost['completion']}/1K 토큰(출력)")
        
        print("\nClaude 사용량:")
        print(f"  프롬프트 토큰: {summary['claude_usage']['prompt_tokens']:,}")
        print(f"  완성 토큰: {summary['claude_usage']['completion_tokens']:,}")
        print(f"  총 토큰: {summary['claude_usage']['total_tokens']:,}")
        print(f"  예상 비용: ${summary['claude_usage']['estimated_cost']:.4f}")
        
        print("\nOpenAI 사용량:")
        print(f"  프롬프트 토큰: {summary['openai_usage']['prompt_tokens']:,}")
        print(f"  완성 토큰: {summary['openai_usage']['completion_tokens']:,}")
        print(f"  총 토큰: {summary['openai_usage']['total_tokens']:,}")
        print(f"  예상 비용: ${summary['openai_usage']['estimated_cost']:.4f}")
        
        print("\nPerplexity 사용량:")
        print(f"  프롬프트 토큰: {summary['perplexity_usage']['prompt_tokens']:,}")
        print(f"  완성 토큰: {summary['perplexity_usage']['completion_tokens']:,}")
        print(f"  총 토큰: {summary['perplexity_usage']['total_tokens']:,}")
        print(f"  예상 비용: ${summary['perplexity_usage']['estimated_cost']:.4f}")
        
        print("\n총 사용량:")
        print(f"  총 토큰: {summary['combined']['total_tokens']:,}")
        print(f"  총 예상 비용: ${summary['combined']['estimated_cost']:.4f}")
        print("=============================")