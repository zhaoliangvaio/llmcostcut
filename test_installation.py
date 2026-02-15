#!/usr/bin/env python
"""
测试 LLMCompiler 包是否正确安装

运行方式:
    python test_installation.py
"""

def test_import():
    """测试基本导入"""
    try:
        from llmcompiler import monitor
        print("✅ 成功导入 monitor")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_submodules():
    """测试子模块导入"""
    try:
        from llmcompiler import task, models, buffers, correctness, trainer, registry, defaults
        print("✅ 成功导入所有子模块")
        return True
    except ImportError as e:
        print(f"❌ 子模块导入失败: {e}")
        return False

def test_monitor_signature():
    """测试 monitor 函数签名"""
    try:
        from llmcompiler import monitor
        import inspect
        
        sig = inspect.signature(monitor)
        params = list(sig.parameters.keys())
        
        required_params = ['task_id2classes', 'text', 'llm_fn']
        missing = [p for p in required_params if p not in params]
        
        if missing:
            print(f"❌ monitor 函数缺少必需参数: {missing}")
            return False
        
        print("✅ monitor 函数签名正确")
        print(f"   参数: {params}")
        return True
    except Exception as e:
        print(f"❌ 检查函数签名失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("LLMCompiler 安装测试")
    print("=" * 60)
    print()
    
    results = []
    
    print("1. 测试基本导入...")
    results.append(test_import())
    print()
    
    print("2. 测试子模块导入...")
    results.append(test_submodules())
    print()
    
    print("3. 测试 monitor 函数签名...")
    results.append(test_monitor_signature())
    print()
    
    print("=" * 60)
    if all(results):
        print("✅ 所有测试通过！LLMCompiler 安装成功。")
        exit(0)
    else:
        print("❌ 部分测试失败，请检查安装。")
        exit(1)

