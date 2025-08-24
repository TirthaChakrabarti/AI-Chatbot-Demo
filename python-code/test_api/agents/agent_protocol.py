from typing import Protocol, Any, List, Dict

class AgentProtocol(Protocol):
    def get_response(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        ...