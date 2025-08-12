#!/usr/bin/env python3
"""
Test script for server resilience and circuit breaker functionality.
Tests health monitoring, failure detection, and recovery patterns.
"""
import os
import sys
import time
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

# Add the mcp directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mcp'))

from server_health import ServerHealthManager, CircuitBreakerConfig, ServerStatus, CircuitState

def test_circuit_breaker_basic():
    """Test basic circuit breaker functionality"""
    print("🧪 Testing Basic Circuit Breaker")
    print("-" * 50)
    
    # Create health manager with small thresholds for testing
    config = CircuitBreakerConfig(
        failure_threshold=2,  # Open after 2 failures
        recovery_timeout=1,   # 1 second recovery
        half_open_max_calls=1,
        success_threshold=1
    )
    health_manager = ServerHealthManager(config)
    
    server_name = "test_server"
    health_manager.register_server(server_name)
    
    # Initially should be able to call
    assert health_manager.can_call_server(server_name), "Should be able to call initially"
    print("✅ Initial state: server callable")
    
    # Record first failure
    health_manager.record_failure(server_name, "Test failure 1")
    assert health_manager.can_call_server(server_name), "Should still be callable after 1 failure"
    print("✅ After 1 failure: server still callable")
    
    # Record second failure - should open circuit
    health_manager.record_failure(server_name, "Test failure 2")
    assert not health_manager.can_call_server(server_name), "Should not be callable after threshold failures"
    print("✅ After 2 failures: circuit opened")
    
    # Wait for recovery timeout
    print("⏳ Waiting for recovery timeout...")
    time.sleep(1.1)  # Wait longer than recovery timeout
    
    # Should now be in half-open state
    assert health_manager.can_call_server(server_name), "Should be callable in half-open state"
    print("✅ After timeout: server callable (half-open)")
    
    # Record success to close circuit
    health_manager.record_success(server_name, 0.1)
    assert health_manager.can_call_server(server_name), "Should be callable after successful recovery"
    print("✅ After success: circuit closed")
    
    return True

def test_server_status_transitions():
    """Test server status transitions"""
    print("\n🧪 Testing Server Status Transitions") 
    print("-" * 50)
    
    config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
    health_manager = ServerHealthManager(config)
    
    server_name = "status_test_server"
    health_manager.register_server(server_name)
    
    # Check initial status
    status = health_manager.get_server_status(server_name)
    assert status == ServerStatus.HEALTHY, f"Expected HEALTHY, got {status}"
    print(f"✅ Initial status: {status.value}")
    
    # Record one failure - should be degraded
    health_manager.record_failure(server_name, "Test failure")
    status = health_manager.get_server_status(server_name)
    assert status == ServerStatus.DEGRADED, f"Expected DEGRADED, got {status}"
    print(f"✅ After 1 failure: {status.value}")
    
    # Record second failure - should be down (circuit open)
    health_manager.record_failure(server_name, "Test failure 2")
    status = health_manager.get_server_status(server_name)
    assert status == ServerStatus.DOWN, f"Expected DOWN, got {status}"
    print(f"✅ After 2 failures: {status.value}")
    
    # Wait and check half-open status
    time.sleep(1.1)
    health_manager.can_call_server(server_name)  # Trigger transition to half-open
    status = health_manager.get_server_status(server_name)
    assert status == ServerStatus.RECOVERING, f"Expected RECOVERING, got {status}"
    print(f"✅ During recovery: {status.value}")
    
    # Record success - should be healthy
    health_manager.record_success(server_name, 0.1)
    status = health_manager.get_server_status(server_name)
    assert status == ServerStatus.HEALTHY, f"Expected HEALTHY, got {status}"
    print(f"✅ After recovery: {status.value}")
    
    return True

def test_metrics_tracking():
    """Test server metrics tracking"""
    print("\n🧪 Testing Metrics Tracking")
    print("-" * 50)
    
    health_manager = ServerHealthManager()
    server_name = "metrics_test_server"
    health_manager.register_server(server_name)
    
    # Record some successes and failures
    response_times = [0.1, 0.2, 0.15, 0.3, 0.25]
    for rt in response_times:
        health_manager.record_success(server_name, rt)
    
    health_manager.record_failure(server_name, "Test failure 1")
    health_manager.record_failure(server_name, "Test failure 2")
    
    # Get metrics
    metrics = health_manager.get_server_metrics(server_name)
    assert metrics is not None, "Metrics should not be None"
    
    print(f"📊 Metrics for {server_name}:")
    print(f"   Total requests: {metrics.total_requests}")
    print(f"   Successful requests: {metrics.successful_requests}")
    print(f"   Failed requests: {metrics.failed_requests}")
    print(f"   Success rate: {metrics.success_rate():.1f}%")
    print(f"   Average response time: {metrics.average_response_time():.3f}s")
    print(f"   Consecutive failures: {metrics.consecutive_failures}")
    
    # Verify calculations
    expected_total = len(response_times) + 2  # 5 successes + 2 failures
    assert metrics.total_requests == expected_total, f"Expected {expected_total} total requests"
    
    expected_success_rate = (len(response_times) / expected_total) * 100
    assert abs(metrics.success_rate() - expected_success_rate) < 0.1, "Success rate calculation error"
    
    expected_avg_time = sum(response_times) / len(response_times)
    assert abs(metrics.average_response_time() - expected_avg_time) < 0.01, "Average time calculation error"
    
    print("✅ Metrics calculations correct")
    return True

def test_non_blocking_error_creation():
    """Test non-blocking error message creation"""
    print("\n🧪 Testing Non-blocking Error Creation")
    print("-" * 50)
    
    health_manager = ServerHealthManager()
    server_name = "error_test_server"
    health_manager.register_server(server_name)
    
    # Test different server states
    test_cases = [
        (ServerStatus.HEALTHY, "temporarily unavailable"),
        (ServerStatus.DEGRADED, "is DEGRADED"),
        (ServerStatus.DOWN, "is DOWN"),
        (ServerStatus.RECOVERING, "is RECOVERING")
    ]
    
    for status, expected_text in test_cases:
        # Manually set server status for testing
        health_manager.server_statuses[server_name] = status
        
        error_result = health_manager.create_non_blocking_error(server_name, "Test error")
        
        print(f"📋 Status {status.value}: {error_result['error']}")
        
        # Verify error structure
        assert "error" in error_result, "Error message should have 'error' field"
        assert "non_blocking" in error_result, "Should have 'non_blocking' field"
        assert "server_name" in error_result, "Should have 'server_name' field"
        assert "server_status" in error_result, "Should have 'server_status' field"
        assert "timestamp" in error_result, "Should have 'timestamp' field"
        
        assert error_result["non_blocking"] is True, "Should be marked as non-blocking"
        assert error_result["server_name"] == server_name, "Server name should match"
        assert error_result["server_status"] == status.value, "Status should match"
        assert expected_text.lower() in error_result["error"].lower(), f"Error should contain '{expected_text}'"
    
    print("✅ Non-blocking error structure correct")
    return True

def test_health_info_aggregation():
    """Test health information aggregation"""
    print("\n🧪 Testing Health Info Aggregation")
    print("-" * 50)
    
    health_manager = ServerHealthManager()
    
    # Set up multiple servers in different states
    servers = ["server1", "server2", "server3"]
    for server in servers:
        health_manager.register_server(server)
    
    # server1: healthy
    health_manager.record_success("server1", 0.1)
    
    # server2: degraded (1 failure)
    health_manager.record_failure("server2", "Test failure")
    
    # server3: down (multiple failures)
    health_manager.record_failure("server3", "Failure 1")
    health_manager.record_failure("server3", "Failure 2")
    health_manager.record_failure("server3", "Failure 3")
    
    # Get aggregated health info
    health_info = health_manager.get_all_server_health()
    
    print(f"📊 Health info for {len(health_info)} servers:")
    for server_name, info in health_info.items():
        print(f"   {server_name}: {info['status']} (circuit: {info['circuit_state']})")
        print(f"      Success rate: {info['metrics']['success_rate']}%")
        print(f"      Consecutive failures: {info['metrics']['consecutive_failures']}")
    
    # Verify structure
    assert len(health_info) == 3, "Should have health info for all 3 servers"
    
    for server_name, info in health_info.items():
        assert "status" in info, f"Server {server_name} missing status"
        assert "circuit_state" in info, f"Server {server_name} missing circuit_state"
        assert "metrics" in info, f"Server {server_name} missing metrics"
        
        metrics = info["metrics"]
        assert "total_requests" in metrics, f"Server {server_name} missing total_requests"
        assert "success_rate" in metrics, f"Server {server_name} missing success_rate"
    
    print("✅ Health info aggregation correct")
    return True

def run_all_resilience_tests():
    """Run all server resilience tests"""
    print("🚀 Running Server Resilience Tests")
    print("=" * 60)
    
    tests = [
        ("Basic Circuit Breaker", test_circuit_breaker_basic),
        ("Server Status Transitions", test_server_status_transitions),
        ("Metrics Tracking", test_metrics_tracking),
        ("Non-blocking Error Creation", test_non_blocking_error_creation),
        ("Health Info Aggregation", test_health_info_aggregation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n💥 Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Server Resilience Test Summary")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All server resilience tests passed!")
        return True
    else:
        print("⚠️ Some tests failed - check output above")
        return False

def main():
    # Change to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    success = run_all_resilience_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()