# Test Fixes Required

> **Status (2026-09):** Unit and integration tests were rewritten to match the real APIs
> (plan Option 1). Keep this file as a historical mapping; do not reintroduce assumed methods.

## Summary

The tests were written based on assumed APIs. The actual implementation has different method names and return types. Here are the required fixes:

## API Mismatches

### 1. IntelligentCacheManager

**Issue:** Tests use `_compute_cache_key()` (private) and `get()/set()` with `level` parameter

**Actual API:**
```python
# Public method (no underscore)
cache.compute_cache_key(task: str, agent: str, params: Optional[Dict] = None) -> str

# Cache levels are separate objects
cache.l1_cache.get(key)
cache.l1_cache.set(key, value)
cache.l2_cache.get(key)  # if enabled
cache.l3_cache.get(key)  # if enabled

# Main manager method
await cache.get_or_compute(key, compute_func)
```

**Fix Required:**
- Change `cache._compute_cache_key(task, {})` → `cache.compute_cache_key(task, "agent")`
- Change `cache.set(key, value, level=CacheLevel.L1)` → `cache.l1_cache.set(key, value)`
- Change `cache.get(key, level=CacheLevel.L1)` → `cache.l1_cache.get(key)`
- Tests should use `get_or_compute` for main caching logic

### 2. AdaptiveRouter

**Issue:** Tests use wrong method names and expect dict returns

**Actual API:**
```python
# Returns RoutingDecision dataclass
decision: RoutingDecision = router.route_task(task, constraints)

# Access as attributes (not dict keys)
decision.selected_agents  # NOT decision['agent']
decision.task_complexity  # NOT decision['complexity']
decision.reasoning

# Method name differences
router.classify_task(task)  # NOT classify_task_type()
```

**Fix Required:**
- Change `classify_task_type()` → `classify_task()`
- Change `result['agent']` → `result.selected_agents[0]` (if accessing first agent)
- Change `'agent' in result` → `hasattr(result, 'selected_agents')`
- Remove tests for non-existent methods: `select_agent()`, `record_result()`, `_calculate_agent_score()`

### 3. Error Handler

**Issue:** Wrong method names

**Actual API:**
```python
# Main retry method
await handler.execute_with_retry(func, *args, **kwargs)

# NOT handler.retry()

# Public classify method
handler.classify_error(error)  # NOT _classify_error()

# Delay calculation is likely internal only
```

**Fix Required:**
- Change all `handler.retry()` → `handler.execute_with_retry()`
- Change `handler._classify_error()` → `handler.classify_error()`
- Remove tests for `_calculate_delay()` (internal method)

### 4. ConsensusInference

**Issue:** Wrong method name

**Actual API:**
```python
# Actual method
result = await consensus.infer_with_consensus(task, agent_funcs)

# NOT run_consensus()
```

**Fix Required:**
- Change all `consensus.run_consensus()` → `consensus.infer_with_consensus()`

### 5. Confidence Calibrator

**Issue:** Tests expect dict-like access on ConfidenceResult

**Actual API:**
```python
# Returns ConfidenceResult dataclass
result: ConfidenceResult = calibrator.calibrate_confidence(score, agent, task_type)

# Access as attributes
result.confidence  # NOT result['confidence']
result.reliability_band
```

**Fix Required:**
- Change `'confidence' in calibrated` → `hasattr(calibrated, 'confidence')`
- Change `calibrated['confidence']` → `calibrated.confidence`

### 6. AgentMonitor

**Issue:** Missing or different method names

**Actual API (needs verification):**
```python
# Likely methods:
monitor.track_inference(...)
monitor.record_metric(...)

# NOT record_request(), record_error()
```

**Fix Required:**
- Check actual monitor API
- Update test calls to match actual methods

### 7. FallbackManager

**Issue:** Return structure different, method signature issues

**Actual API:**
```python
# Returns structured result
result = await manager.execute_with_fallback(func, task)

# Result structure needs verification
# Likely: {'result': ..., 'method': 'primary/fallback', 'success': bool}

# register_fallback signature
manager.register_fallback(strategy_name, func, priority=0)
```

**Fix Required:**
- Tests expect `result['response']` but actual returns `result['result']['response']`
- `register_fallback()` doesn't accept `quality_degradation` parameter
- `fallback_chain` is a list of dicts, not list of strings

### 8. ErrorLearner

**Issue:** Initialization expects path string, not config dict

**Actual API:**
```python
# ErrorLearner expects database path
learner = ErrorLearner(db_path="/path/to/errors.db", enabled=True)

# NOT
learner = ErrorLearner(config_dict)
```

**Fix Required:**
- Pass `temp_db_path` string directly, not wrapped in config dict
- Adjust initialization in all ErrorLearner tests

## Test Files Requiring Updates

1. `test_cache_manager.py` - 19 failures
   - Fix `_compute_cache_key` → `compute_cache_key`
   - Fix `.get(key, level=L1)` → `.l1_cache.get(key)`
   - Fix `.set(key, val, level=L1)` → `.l1_cache.set(key, val)`
   - Remove `set_semantic` / `get_semantic` tests or implement these methods

2. `test_adaptive_router.py` - 21 failures
   - Fix `classify_task_type` → `classify_task`
   - Fix dict access to attribute access on RoutingDecision
   - Remove tests for non-existent methods
   - Fix comparison of TaskComplexity enums (they're being compared as objects not values)

3. `test_error_handler.py` - 17 failures
   - Fix `retry()` → `execute_with_retry()`
   - Fix `_classify_error` → `classify_error`
   - Fix ErrorLearner initialization
   - Fix FallbackManager result structure expectations

4. `test_consensus_inference.py` - 14 failures
   - Fix `run_consensus()` → `infer_with_consensus()`

5. `test_full_inference_pipeline.py` - 7 failures
   - Fix cache key computation
   - Fix confidence result access
   - Fix monitor method calls
   - Fix Prometheus duplicate registry issue

6. `test_input_validator.py` - 2 failures
   - Whitespace normalization triggering command injection
   - Prompt injection not being detected

## Quick Fix Strategy

### Option 1: Update Tests (Recommended for Learning)
Update all tests to match actual implementation. This shows understanding of the actual API.

### Option 2: Update Implementation
Modify implementations to match test expectations. This is appropriate if tests represent the desired API.

### Option 3: Hybrid
- Update tests for reasonable API differences
- Update implementation for genuine bugs

## Recommendation

Given that the implementation appears functional and the tests were written as "ideal APIs", I recommend **Option 1**: Update tests to match reality. This will:

1. Ensure tests actually test the working code
2. Provide accurate documentation of how to use the system
3. Identify any real bugs (vs API mismatches)

## Priority Fixes

1. **High Priority** - Fix basic API mismatches (method names, access patterns)
2. **Medium Priority** - Fix return type handling (dataclass vs dict)
3. **Low Priority** - Remove tests for unimplemented features (add TODO comments)

## Estimated Time

- Automated find/replace fixes: 30 minutes
- Manual fixes for logic: 1-2 hours
- Verification and cleanup: 30 minutes

**Total: 2-3 hours to fix all tests**

