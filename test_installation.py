#!/usr/bin/env python
"""
Test whether the LLMCostCut package is installed correctly.

Run:
    python test_installation.py
"""

def test_import():
    """Test basic import."""
    try:
        from llmcostcut import monitor
        print("✅ Successfully imported monitor")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_submodules():
    """Test submodule imports."""
    try:
        from llmcostcut import task, models, buffers, correctness, trainer, registry, defaults
        print("✅ Successfully imported all submodules")
        return True
    except ImportError as e:
        print(f"❌ Submodule import failed: {e}")
        return False

def test_monitor_signature():
    """Test monitor function signature."""
    try:
        from llmcostcut import monitor
        import inspect
        
        sig = inspect.signature(monitor)
        params = list(sig.parameters.keys())
        
        required_params = ['task_id2classes', 'text', 'llm_fn']
        missing = [p for p in required_params if p not in params]
        
        if missing:
            print(f"❌ monitor function is missing required parameters: {missing}")
            return False
        
        print("✅ monitor function signature is correct")
        print(f"   Parameters: {params}")
        return True
    except Exception as e:
        print(f"❌ Failed to check function signature: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("LLMCostCut Installation Test")
    print("=" * 60)
    print()
    
    results = []
    
    print("1. Testing basic import...")
    results.append(test_import())
    print()
    
    print("2. Testing submodule imports...")
    results.append(test_submodules())
    print()
    
    print("3. Testing monitor function signature...")
    results.append(test_monitor_signature())
    print()
    
    print("=" * 60)
    if all(results):
        print("✅ All tests passed! LLMCostCut installed successfully.")
        exit(0)
    else:
        print("❌ Some tests failed. Please check the installation.")
        exit(1)

