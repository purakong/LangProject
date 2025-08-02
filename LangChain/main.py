import os
import json
import re
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, Any, TypedDict
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from prompt import EMAIL_GENERATION_PROMPT, ACCURACY_CHECK_PROMPT
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# 상태 정의
class EmailState(TypedDict):
    user_input: str
    parsed_data: Dict[str, str]
    generated_email: str
    accuracy_score: Dict[str, Any]
    send_status: str
    result_summary: str
    processing_time: str
    current_date: str



class EmailGenerationSystem:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """LangGraph 워크플로우 구성"""
        workflow = StateGraph(EmailState)
        
        # 노드 추가
        workflow.add_node("input_processing", self.process_input)
        workflow.add_node("email_generation", self.generate_email)
        workflow.add_node("accuracy_check", self.check_accuracy)
        workflow.add_node("email_simulation", self.simulate_email_send)
        workflow.add_node("result_output", self.output_result)
        workflow.add_node("web_update", self.update_web_page)
        workflow.add_node("revision", self.revise_email)
        
        # 엣지 연결
        workflow.set_entry_point("input_processing")
        workflow.add_edge("input_processing", "email_generation")
        workflow.add_edge("email_generation", "accuracy_check")
        
        # 조건부 라우팅
        workflow.add_conditional_edges(
            "accuracy_check",
            self.should_send_email,
            {
                "send": "email_simulation",
                "revise": "revision"
            }
        )
        
        workflow.add_edge("revision", "email_generation")
        workflow.add_edge("email_simulation", "result_output")
        workflow.add_edge("result_output", "web_update")
        workflow.add_edge("web_update", END)
        
        return workflow.compile()
    
    def process_input(self, state: EmailState) -> EmailState:
        """입력 처리 및 파싱"""
        logger.info("입력 처리 시작")
        
        user_input = state["user_input"]
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # 간단한 파싱 로직 (실제로는 더 정교하게 구현)
        parsed_data = {
            "vehicle_model": "추출된차종",
            "software_version": "추출된버전", 
            "control_board": "추출된제어보드",
            "manager_name": "김테스트",
            "distributor_name": "박배포",
            "test_result": "All Pass"
        }
        
        state.update({
            "parsed_data": parsed_data,
            "current_date": current_date,
            "processing_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        logger.info(f"파싱 완료: {parsed_data}")
        return state
    
    def generate_email(self, state: EmailState) -> EmailState:
        """이메일 생성"""
        logger.info("이메일 생성 시작")
        
        prompt = PromptTemplate.from_template(EMAIL_GENERATION_PROMPT)
        
        formatted_prompt = prompt.format(
            user_input=state["user_input"],
            current_date=state["current_date"],
            **state["parsed_data"]
        )
        
        response = self.llm.invoke(formatted_prompt)
        generated_email = response.content
        
        state["generated_email"] = generated_email
        logger.info("이메일 생성 완료")
        return state
    
    def check_accuracy(self, state: EmailState) -> EmailState:
        """정확도 검증"""
        logger.info("정확도 체크 시작")
        
        prompt = PromptTemplate.from_template(ACCURACY_CHECK_PROMPT)
        
        formatted_prompt = prompt.format(
            original_input=state["user_input"],
            generated_email=state["generated_email"]
        )
        
        response = self.llm.invoke(formatted_prompt)
        
        try:
            # JSON 파싱
            accuracy_data = json.loads(response.content)
            state["accuracy_score"] = accuracy_data
            logger.info(f"정확도 점수: {accuracy_data['overall_score']}")
        except json.JSONDecodeError:
            logger.error("정확도 체크 응답 파싱 실패")
            state["accuracy_score"] = {
                "overall_score": 0,
                "recommendation": "REVISE",
                "feedback": "응답 파싱 실패"
            }
        
        return state
    
    def should_send_email(self, state: EmailState) -> str:
        """이메일 전송 여부 결정"""
        score = state["accuracy_score"].get("overall_score", 0)
        recommendation = state["accuracy_score"].get("recommendation", "REVISE")
        
        if score == 100 and recommendation == "APPROVE":
            logger.info("정확도 100점 - 이메일 전송 승인")
            return "send"
        else:
            logger.info(f"정확도 {score}점 - 재작성 필요")
            return "revise"
    
    def revise_email(self, state: EmailState) -> EmailState:
        """이메일 재작성"""
        logger.info("이메일 재작성")
        feedback = state["accuracy_score"].get("feedback", "")
        # 실제로는 피드백을 반영한 재작성 로직 구현
        # 현재는 단순히 로그만 출력
        logger.info(f"재작성 피드백: {feedback}")
        return state
    
    def simulate_email_send(self, state: EmailState) -> EmailState:
        """이메일 전송 시뮬레이션"""
        logger.info("이메일 전송 시뮬레이션")
        
        # 시뮬레이션 로직
        state["send_status"] = "전송 완료 (시뮬레이션)"
        logger.info("이메일 전송 시뮬레이션 완료")
        return state
    
    def output_result(self, state: EmailState) -> EmailState:
        """결과 출력"""
        logger.info("결과 출력")
        
        result_summary = f"""
        📧 이메일 생성 완료 보고서
        
        ═══════════════════════════════════════
        📝 처리 결과
        - 상태: {state['send_status']}
        - 정확도: {state['accuracy_score']['overall_score']}/100
        - 처리 시간: {state['processing_time']}
        
        📋 생성된 이메일 정보
        - 입력: {state['user_input']}
        - 파싱된 데이터: {state['parsed_data']}
        
        📊 품질 평가
        - 파싱 정확성: {state['accuracy_score'].get('parsing_accuracy', 'N/A')}/40
        - 양식 준수: {state['accuracy_score'].get('format_compliance', 'N/A')}/30
        - 필수 정보: {state['accuracy_score'].get('required_info', 'N/A')}/20
        - 형식 완성도: {state['accuracy_score'].get('format_completeness', 'N/A')}/10
        
        💡 피드백: {state['accuracy_score'].get('feedback', '없음')}
        ═══════════════════════════════════════
        """
        
        state["result_summary"] = result_summary
        print(result_summary)
        return state
    
    def update_web_page(self, state: EmailState) -> EmailState:
        """웹페이지 업데이트"""
        logger.info("웹페이지 업데이트")
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>이메일 생성 결과</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ background: #007bff; color: white; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .score {{ font-size: 24px; font-weight: bold; color: #28a745; }}
                .email-content {{ background: #f8f9fa; padding: 15px; border-radius: 5px; white-space: pre-wrap; }}
                .timestamp {{ color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚗 자동차 테스트 이메일 생성 시스템</h1>
                    <p>생성 시간: {state['processing_time']}</p>
                </div>
                
                <div class="section">
                    <h2>📝 처리 결과</h2>
                    <p><strong>상태:</strong> {state['send_status']}</p>
                    <p><strong>정확도:</strong> <span class="score">{state['accuracy_score']['overall_score']}/100</span></p>
                </div>
                
                <div class="section">
                    <h2>📋 입력 정보</h2>
                    <p><strong>사용자 입력:</strong> {state['user_input']}</p>
                    <p><strong>차종:</strong> {state['parsed_data']['vehicle_model']}</p>
                    <p><strong>소프트웨어 버전:</strong> {state['parsed_data']['software_version']}</p>
                    <p><strong>제어보드:</strong> {state['parsed_data']['control_board']}</p>
                </div>
                
                <div class="section">
                    <h2>📧 생성된 이메일</h2>
                    <div class="email-content">{state['generated_email']}</div>
                </div>
                
                <div class="section">
                    <h2>📊 품질 평가</h2>
                    <p><strong>파싱 정확성:</strong> {state['accuracy_score'].get('parsing_accuracy', 'N/A')}/40</p>
                    <p><strong>양식 준수:</strong> {state['accuracy_score'].get('format_compliance', 'N/A')}/30</p>
                    <p><strong>필수 정보:</strong> {state['accuracy_score'].get('required_info', 'N/A')}/20</p>
                    <p><strong>형식 완성도:</strong> {state['accuracy_score'].get('format_completeness', 'N/A')}/10</p>
                    <p><strong>피드백:</strong> {state['accuracy_score'].get('feedback', '없음')}</p>
                </div>
                
                <div class="timestamp">
                    마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """
        
        # HTML 파일 저장
        with open("email_result.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info("웹페이지 업데이트 완료: email_result.html")
        return state
    
    def run(self, user_input: str) -> Dict[str, Any]:
        """시스템 실행"""
        initial_state = EmailState(
            user_input=user_input,
            parsed_data={},
            generated_email="",
            accuracy_score={},
            send_status="",
            result_summary="",
            processing_time="",
            current_date=""
        )
        
        result = self.graph.invoke(initial_state)
        return result

# 실행 예제
def main():
    # OpenAI API 키 설정 (환경변수 또는 직접 입력)
    api_key = os.getenv("OPENAI_API_KEY") 
    
    # 시스템 초기화
    email_system = EmailGenerationSystem(api_key)
    
    # 테스트 실행
    test_input = "소나타, v2.1.3, ECU-2024"
    
    print("🚀 이메일 생성 시스템 시작")
    print(f"입력: {test_input}")
    print("=" * 50)
    
    result = email_system.run(test_input)
    
    print("\n✅ 처리 완료!")
    print("결과 파일: email_result.html")

if __name__ == "__main__":
    main()