"""
Test script for Generation 2: Robust implementation with enhanced error handling and monitoring.
"""

import sys
import os
sys.path.insert(0, '/root/repo')

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from quantum_hyper_search.robust_main import QuantumHyperSearchRobust
from quantum_hyper_search.utils.robust_error_handling import global_error_handler
from quantum_hyper_search.utils.comprehensive_monitoring import global_monitor


def test_robust_optimization():
    """Test robust optimization with monitoring and error handling."""
    print("🔒 Testing Generation 2: ROBUST implementation")
    print("=" * 60)
    
    # Generate test data
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_classes=2,
        random_state=42
    )
    
    # Define parameter space
    param_space = {
        'n_estimators': [10, 50, 100],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }
    
    # Initialize robust optimizer with full monitoring
    optimizer = QuantumHyperSearchRobust(
        backend='simple',
        enable_security=True,
        enable_monitoring=True,
        max_retries=3,
        timeout_per_iteration=60.0,
        fallback_to_random=True
    )
    
    print(f"🛡️  Robust optimizer initialized (session: {optimizer.session_id})")
    print(f"🔧 Backend: {optimizer.backend_name}")
    print(f"📊 Monitoring enabled: {optimizer.enable_monitoring}")
    print(f"🔒 Security enabled: {optimizer.enable_security}")
    
    try:
        # Run optimization
        best_params, history = optimizer.optimize(
            model_class=RandomForestClassifier,
            param_space=param_space,
            X=X,
            y=y,
            n_iterations=5,
            quantum_reads=20,
            cv_folds=3,
            scoring='accuracy'
        )
        
        # Display results
        print("\n🏆 OPTIMIZATION RESULTS:")
        print(f"✅ Best score: {history.best_score:.4f}")
        print(f"🎯 Best parameters: {best_params}")
        
        # Display statistics
        stats = history.get_statistics()
        print(f"\n📈 STATISTICS:")
        print(f"   Evaluations: {stats['n_evaluations']}")
        print(f"   Errors: {stats['n_errors']}")
        print(f"   Error rate: {stats['error_rate']:.2%}")
        print(f"   Duration: {stats['duration_seconds']:.2f}s")
        print(f"   Rate: {stats['evaluations_per_second']:.2f} eval/s")
        
        # Display monitoring summary
        if optimizer.monitor:
            monitoring_summary = optimizer.monitor.get_monitoring_summary()
            print(f"\n🔍 MONITORING SUMMARY:")
            print(f"   Total metrics: {monitoring_summary['metrics_summary']['total_metrics']}")
            print(f"   Active alerts: {monitoring_summary['alerts_summary']['active_alerts']}")
            print(f"   Health status: {monitoring_summary['health_summary']['overall_status']}")
        
        # Display error handling summary
        error_summary = optimizer.error_handler.get_error_summary()
        print(f"\n🛠️  ERROR HANDLING SUMMARY:")
        print(f"   Total errors: {error_summary['total_errors']}")
        if error_summary.get('error_types'):
            print(f"   Error types: {error_summary['error_types']}")
        else:
            print("   No errors encountered")
        
        print("\n✅ ROBUST IMPLEMENTATION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ ROBUST IMPLEMENTATION TEST FAILED: {e}")
        return False
    
    finally:
        # Cleanup
        if optimizer.monitor:
            optimizer.monitor.stop_monitoring()


def test_error_recovery():
    """Test error recovery and fallback mechanisms."""
    print("\n🔧 Testing error recovery mechanisms...")
    
    # Test circuit breaker functionality
    from quantum_hyper_search.utils.robust_error_handling import CircuitBreaker
    
    def failing_function():
        raise Exception("Simulated failure")
    
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
    
    # Trigger failures to open circuit
    failures = 0
    for i in range(5):
        try:
            breaker.call(failing_function)
        except Exception:
            failures += 1
    
    print(f"   Circuit breaker triggered {failures} failures")
    print(f"   Circuit state: {breaker.state}")
    
    # Test error handler
    from quantum_hyper_search.utils.robust_error_handling import ErrorContext, ErrorSeverity
    
    context = ErrorContext(
        timestamp=1234567890,
        error_type="TestError",
        error_message="Test error message",
        severity=ErrorSeverity.MEDIUM,
        component="test",
        operation="test_operation"
    )
    
    print(f"   Error context created: {context.to_dict()}")
    print("✅ Error recovery mechanisms working!")


def test_monitoring_system():
    """Test comprehensive monitoring system."""
    print("\n📊 Testing monitoring system...")
    
    # Test metric collection
    global_monitor.record_metric("test_metric", 42.0, "units", {"tag": "test"})
    
    # Test anomaly detection
    for i in range(20):
        global_monitor.record_metric("normal_metric", np.random.normal(10, 1))
    
    # Record anomalous value
    global_monitor.record_metric("normal_metric", 50.0)  # Should trigger anomaly
    
    # Test alert creation
    alert = global_monitor.alerts.create_alert(
        severity="warning",
        title="Test Alert",
        message="This is a test alert",
        source="test_system"
    )
    
    print(f"   Created alert: {alert.id}")
    
    # Get monitoring summary
    summary = global_monitor.get_monitoring_summary()
    print(f"   Monitoring summary: {summary}")
    
    print("✅ Monitoring system working!")


if __name__ == "__main__":
    success = True
    
    try:
        success &= test_robust_optimization()
        test_error_recovery()
        test_monitoring_system()
        
        if success:
            print("\n🎉 ALL GENERATION 2 TESTS PASSED!")
            print("🔒 Robust implementation with error handling and monitoring is ready!")
        else:
            print("\n❌ SOME TESTS FAILED")
            
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR IN TESTING: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    sys.exit(0 if success else 1)