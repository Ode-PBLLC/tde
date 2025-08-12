"""
ODE MCP Generic - Server Health Management
Implements circuit breaker pattern and health monitoring for MCP servers.
"""
import asyncio
import time
import logging
from enum import Enum
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

# Configure logging for server health
health_logger = logging.getLogger('server_health')

class ServerStatus(Enum):
    """Server health status states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  
    DOWN = "down"
    RECOVERING = "recovering"

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests due to failures
    HALF_OPEN = "half_open" # Testing if server is back

@dataclass
class ServerMetrics:
    """Metrics for a single server"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    last_request_time: float = 0
    last_success_time: float = 0
    last_failure_time: float = 0
    response_times: list = field(default_factory=list)
    
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    def average_response_time(self) -> float:
        """Calculate average response time in seconds"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 3       # Number of failures before opening circuit
    recovery_timeout: int = 30       # Seconds before attempting recovery
    half_open_max_calls: int = 1     # Max calls in half-open state
    slow_call_threshold: float = 10.0 # Seconds to consider a call slow
    success_threshold: int = 1        # Successes needed to close circuit from half-open

class ServerHealthManager:
    """Manages health status and circuit breaker for MCP servers"""
    
    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.servers: Dict[str, ServerMetrics] = {}
        self.circuit_states: Dict[str, CircuitState] = {}
        self.server_statuses: Dict[str, ServerStatus] = {}
        self.circuit_opened_times: Dict[str, float] = {}
        self.half_open_calls: Dict[str, int] = {}
        
        health_logger.info(f"ServerHealthManager initialized with failure_threshold={self.config.failure_threshold}, "
                          f"recovery_timeout={self.config.recovery_timeout}s")
    
    def register_server(self, server_name: str):
        """Register a new server for health monitoring"""
        if server_name not in self.servers:
            self.servers[server_name] = ServerMetrics()
            self.circuit_states[server_name] = CircuitState.CLOSED
            self.server_statuses[server_name] = ServerStatus.HEALTHY
            self.half_open_calls[server_name] = 0
            health_logger.info(f"Registered server '{server_name}' for health monitoring")
    
    def can_call_server(self, server_name: str) -> bool:
        """Check if server can be called based on circuit breaker state"""
        if server_name not in self.circuit_states:
            self.register_server(server_name)
            return True
        
        circuit_state = self.circuit_states[server_name]
        current_time = time.time()
        
        if circuit_state == CircuitState.CLOSED:
            return True
        
        elif circuit_state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if server_name in self.circuit_opened_times:
                time_since_opened = current_time - self.circuit_opened_times[server_name]
                if time_since_opened >= self.config.recovery_timeout:
                    # Move to half-open state
                    self._transition_to_half_open(server_name)
                    return True
            return False
        
        elif circuit_state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            return self.half_open_calls[server_name] < self.config.half_open_max_calls
        
        return False
    
    def record_success(self, server_name: str, response_time: float):
        """Record a successful server call"""
        if server_name not in self.servers:
            self.register_server(server_name)
        
        metrics = self.servers[server_name]
        current_time = time.time()
        
        # Update metrics
        metrics.total_requests += 1
        metrics.successful_requests += 1
        metrics.consecutive_failures = 0
        metrics.last_request_time = current_time
        metrics.last_success_time = current_time
        metrics.response_times.append(response_time)
        
        # Keep only last 100 response times
        if len(metrics.response_times) > 100:
            metrics.response_times = metrics.response_times[-100:]
        
        # Update circuit breaker state
        circuit_state = self.circuit_states[server_name]
        
        if circuit_state == CircuitState.HALF_OPEN:
            self.half_open_calls[server_name] += 1
            # Check if we should close the circuit
            if self.half_open_calls[server_name] >= self.config.success_threshold:
                self._close_circuit(server_name)
        
        # Update server status
        self._update_server_status(server_name)
        
        health_logger.debug(f"Server '{server_name}' call succeeded in {response_time:.3f}s")
    
    def record_failure(self, server_name: str, error: str):
        """Record a failed server call"""
        if server_name not in self.servers:
            self.register_server(server_name)
        
        metrics = self.servers[server_name]
        current_time = time.time()
        
        # Update metrics
        metrics.total_requests += 1
        metrics.failed_requests += 1
        metrics.consecutive_failures += 1
        metrics.last_request_time = current_time
        metrics.last_failure_time = current_time
        
        # Update circuit breaker state
        circuit_state = self.circuit_states[server_name]
        
        if circuit_state == CircuitState.CLOSED:
            # Check if we should open the circuit
            if metrics.consecutive_failures >= self.config.failure_threshold:
                self._open_circuit(server_name, error)
        
        elif circuit_state == CircuitState.HALF_OPEN:
            # Failure in half-open state - back to open
            self._open_circuit(server_name, error)
        
        # Update server status
        self._update_server_status(server_name)
        
        health_logger.warning(f"Server '{server_name}' call failed: {error}")
    
    def _open_circuit(self, server_name: str, error: str):
        """Open circuit breaker for server"""
        self.circuit_states[server_name] = CircuitState.OPEN
        self.circuit_opened_times[server_name] = time.time()
        self.half_open_calls[server_name] = 0
        
        health_logger.error(f"Circuit breaker OPENED for server '{server_name}': {error}")
    
    def _close_circuit(self, server_name: str):
        """Close circuit breaker for server"""
        self.circuit_states[server_name] = CircuitState.CLOSED
        self.half_open_calls[server_name] = 0
        
        if server_name in self.circuit_opened_times:
            del self.circuit_opened_times[server_name]
        
        health_logger.info(f"Circuit breaker CLOSED for server '{server_name}' - service recovered")
    
    def _transition_to_half_open(self, server_name: str):
        """Transition circuit to half-open state"""
        self.circuit_states[server_name] = CircuitState.HALF_OPEN
        self.half_open_calls[server_name] = 0
        
        # Update status immediately
        self._update_server_status(server_name)
        
        health_logger.info(f"Circuit breaker HALF-OPEN for server '{server_name}' - testing recovery")
    
    def _update_server_status(self, server_name: str):
        """Update overall server status based on metrics and circuit state"""
        circuit_state = self.circuit_states[server_name]
        metrics = self.servers[server_name]
        
        if circuit_state == CircuitState.OPEN:
            self.server_statuses[server_name] = ServerStatus.DOWN
        elif circuit_state == CircuitState.HALF_OPEN:
            self.server_statuses[server_name] = ServerStatus.RECOVERING
        elif metrics.consecutive_failures > 0:
            self.server_statuses[server_name] = ServerStatus.DEGRADED
        else:
            self.server_statuses[server_name] = ServerStatus.HEALTHY
    
    def get_server_status(self, server_name: str) -> ServerStatus:
        """Get current status of a server"""
        return self.server_statuses.get(server_name, ServerStatus.HEALTHY)
    
    def get_server_metrics(self, server_name: str) -> Optional[ServerMetrics]:
        """Get metrics for a server"""
        return self.servers.get(server_name)
    
    def get_all_server_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health information for all servers"""
        health_info = {}
        
        for server_name in self.servers.keys():
            metrics = self.servers[server_name]
            circuit_state = self.circuit_states[server_name]
            status = self.server_statuses[server_name]
            
            health_info[server_name] = {
                "status": status.value,
                "circuit_state": circuit_state.value,
                "metrics": {
                    "total_requests": metrics.total_requests,
                    "success_rate": round(metrics.success_rate(), 2),
                    "consecutive_failures": metrics.consecutive_failures,
                    "average_response_time": round(metrics.average_response_time(), 3),
                    "last_success": metrics.last_success_time,
                    "last_failure": metrics.last_failure_time
                }
            }
            
            # Add circuit-specific info
            if circuit_state == CircuitState.OPEN and server_name in self.circuit_opened_times:
                time_until_retry = max(0, self.config.recovery_timeout - (time.time() - self.circuit_opened_times[server_name]))
                health_info[server_name]["time_until_retry"] = round(time_until_retry, 1)
        
        return health_info
    
    def create_non_blocking_error(self, server_name: str, original_error: str = None) -> Dict[str, Any]:
        """Create a non-blocking error message that LLM can handle gracefully"""
        status = self.get_server_status(server_name)
        
        if status == ServerStatus.DOWN:
            message = f"SERVER {server_name} is DOWN"
        elif status == ServerStatus.DEGRADED:
            message = f"SERVER {server_name} is DEGRADED"
        elif status == ServerStatus.RECOVERING:
            message = f"SERVER {server_name} is RECOVERING"
        else:
            message = f"SERVER {server_name} temporarily unavailable"
        
        # Add original error for logging if available
        if original_error:
            health_logger.error(f"Server '{server_name}' error: {original_error}")
        
        return {
            "error": message,
            "non_blocking": True,
            "server_name": server_name,
            "server_status": status.value,
            "timestamp": time.time()
        }

def setup_health_logging():
    """Set up logging for server health monitoring"""
    health_logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler for health logs
    import os
    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler('logs/server_health.log')
    file_handler.setFormatter(formatter)
    health_logger.addHandler(file_handler)
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    health_logger.addHandler(console_handler)

# Initialize logging when module is imported
setup_health_logging()