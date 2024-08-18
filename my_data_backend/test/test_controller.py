import requests


def test_get_agent_query_answer_with_text():
    response = requests.post(
        "http://localhost:9001/api/agent/query",
        data={"deviceId": "test_device", "query": "API 스펙 중 aNS는 어떤 것을 뜻하나요?"},
    )
    assert response.status_code == 200
    assert "answer" in response.json()
    print(response.json().get("answer"))
